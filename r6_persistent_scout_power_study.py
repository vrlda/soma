"""Development-only power study for unchanged persistent-scout gates.

The study compares predeclared budgets 512/1024/2048 on independent
leave-one-combination-out temporal streams.  Target-present rows use the
actual held-out target reward; shuffled rows use the same frozen examples with
reward alignment permuted.  No threshold is fitted from these results and no
R6 report is touched.  ``max_cases`` supports bounded staged runs; an
incomplete case matrix is never marked as qualifying a budget.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import r6_temporal_composition_pilot as temporal
from soma.organism import Organism


PROTOCOL = "r6-persistent-scout-power-study-v1"
SEEDS = (137, 271)
BUDGETS = (512, 1024, 2048)
HELDOUT_COMBOS = tuple(range(8))
MIN_COUNT = 32
TARGET_KEY = temporal.TARGET_KEY
LESION_DROP_MIN = 0.05
TARGET_PASS_FLOOR = 0.75
NULL_RATE_MAX = 0.0
CASE_TIME_LIMIT_SECONDS = 180.0
STATE_BYTES_LIMIT = 2_000_000


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _lesion_drop(intact_skill: float, lesioned_skill: float) -> float:
    """Return positive held-out degradation caused by the lesion."""
    return float(intact_skill) - float(lesioned_skill)


def _train(seed: int, budget: int, examples: Sequence[Mapping[str, object]], reward_mode: str):
    if reward_mode not in ("target", "shuffled"):
        raise ValueError("unsupported reward mode")
    organism = temporal._new_organism(seed, "persistent_variable_order", persistent_budget=int(budget))
    state, resources = temporal._train(organism, examples, seed, shuffled=(reward_mode == "shuffled"))
    telemetry = dict(state.get("variable_order_persistent_last_telemetry", {}))
    owners = dict(state.get("variable_order_feature_owners", {}))
    target_installed = TARGET_KEY in owners
    accepted = str(state.get("variable_order_persistent_terminal_state")) == "accepted"
    winner_key = telemetry.get("winner_key")
    return state, {
        "seed": int(seed), "budget": int(budget), "reward_mode": reward_mode,
        "stream_digest": _digest(examples), "checkpoint_digest": _digest(state),
        "evidence_count": int(resources["persistent_scout"]["evidence_count"]),
        "episode_count": int(resources["persistent_scout"]["episode_count"]),
        "terminal_state": state.get("variable_order_persistent_terminal_state"),
        "accepted": accepted, "installed": int(state.get("variable_order_install_count", 0)) > 0,
        "target_winner": winner_key == TARGET_KEY, "winner_key": winner_key,
        "winner_rank": telemetry.get("winner_rank"),
        "winner_feature_rank": telemetry.get("winner_feature_rank"),
        "target_installed": target_installed,
        "target_install_reason": telemetry.get("install_reason"),
        "family_size": telemetry.get("family_size"),
        "winner_z": telemetry.get("winner_z"),
        "winner_runner_margin": telemetry.get("winner_runner_margin"),
        "state_bytes": len(json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")),
    }


def _case(seed: int, budget: int, heldout_combo: int, reward_mode: str, evaluation_examples: int):
    started = time.monotonic()
    stream = temporal._examples(int(seed) + 1000, int(heldout_combo), int(budget))
    state, row = _train(int(seed) + 2000, int(budget), stream, reward_mode)
    heldout = temporal._examples(int(seed) + 3000, int(heldout_combo), int(evaluation_examples), require_heldout=True)
    intact = temporal._frozen_score(state, heldout, int(seed) + 4000)[1]
    lesioned, changed = temporal._lesion_target_k3(state)
    lesion_metrics = temporal._frozen_score(lesioned, heldout, int(seed) + 4000)[1] if changed else None
    # Positive means the intact path scores better than its lesioned clone.
    lesion_drop = _lesion_drop(intact["skill"], lesion_metrics["skill"]) if lesion_metrics else None
    row.update({
        "heldout_combo": int(heldout_combo),
        "heldout_sign": 1 if float(heldout[0]["target"]) > 0.0 else -1,
        "evaluation_examples": int(evaluation_examples),
        "target_first": bool(row["target_winner"]),
        "target_install": bool(row["target_installed"]),
        "lesion_available": bool(changed),
        "lesion_skill_drop": lesion_drop,
        "lesion_pass": bool(changed and lesion_drop is not None and lesion_drop > LESION_DROP_MIN),
        "target_case_pass": bool(row["target_winner"] and row["target_installed"] and changed and lesion_drop is not None and lesion_drop > LESION_DROP_MIN),
        "elapsed_seconds": time.monotonic() - started,
        "resource_guard_pass": bool((time.monotonic() - started) <= CASE_TIME_LIMIT_SECONDS
                                     and row["state_bytes"] <= STATE_BYTES_LIMIT),
    })
    return row


def _aggregate(rows: Sequence[Mapping[str, object]], budget: int, expected_cases: int):
    target = [row for row in rows if row["budget"] == budget and row["reward_mode"] == "target"]
    null = [row for row in rows if row["budget"] == budget and row["reward_mode"] == "shuffled"]
    target_pass = sum(bool(row["target_case_pass"]) for row in target)
    null_accept = sum(bool(row["accepted"]) for row in null)
    null_install = sum(bool(row["installed"]) for row in null)
    complete = len(target) == expected_cases and len(null) == expected_cases
    target_rate = target_pass / float(max(1, len(target)))
    null_accept_rate = null_accept / float(max(1, len(null)))
    null_install_rate = null_install / float(max(1, len(null)))
    qualifies = bool(complete and target_rate >= TARGET_PASS_FLOOR
                     and null_accept_rate <= NULL_RATE_MAX
                     and null_install_rate <= NULL_RATE_MAX
                     and all(bool(row["resource_guard_pass"]) for row in target + null))
    return {
        "budget": int(budget), "complete": complete,
        "expected_cases": int(expected_cases), "target_cases": len(target), "null_cases": len(null),
        "target_case_pass_rate": target_rate, "target_first_rate": sum(bool(row["target_first"]) for row in target) / float(max(1, len(target))),
        "target_install_rate": sum(bool(row["target_install"]) for row in target) / float(max(1, len(target))),
        "target_lesion_rate": sum(bool(row["lesion_pass"]) for row in target) / float(max(1, len(target))),
        "null_accept_rate": null_accept_rate, "null_install_rate": null_install_rate,
        "resource_guard_all": all(bool(row["resource_guard_pass"]) for row in target + null),
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in target + null),
        "lesion_drops": [row["lesion_skill_drop"] for row in target if row["lesion_skill_drop"] is not None],
        "qualifies": qualifies,
    }


def run_power_study(seeds: Sequence[int] = SEEDS, budgets: Sequence[int] = BUDGETS,
                    heldout_combos: Sequence[int] = HELDOUT_COMBOS,
                    max_cases: Optional[int] = None, evaluation_examples: int = 32):
    started = time.monotonic()
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_budgets = tuple(int(budget) for budget in budgets)
    all_cases = [(seed, combo) for seed in normalized_seeds for combo in tuple(int(value) for value in heldout_combos)]
    cases = all_cases[:int(max_cases)] if max_cases is not None else all_cases
    rows: List[Dict[str, object]] = []
    stream_digests = {}
    for budget in normalized_budgets:
        for seed, combo in cases:
            stream = temporal._examples(seed + 1000, combo, budget)
            stream_digests["%s:%s:%s" % (budget, seed, combo)] = _digest(stream)
            for reward_mode in ("target", "shuffled"):
                rows.append(_case(seed, budget, combo, reward_mode, evaluation_examples))
    expected_cases = len(all_cases)
    by_budget = {str(budget): _aggregate(rows, budget, expected_cases) for budget in normalized_budgets}
    selected = next((budget for budget in normalized_budgets if by_budget[str(budget)]["qualifies"]), None)
    return {
        "protocol": PROTOCOL, "development_only": True, "acceptance_gating": False,
        "r6_qualification": "not_evaluated", "seeds": list(normalized_seeds),
        "budgets": list(normalized_budgets), "heldout_combos": list(heldout_combos),
        "cases_requested": len(all_cases), "cases_run": len(cases),
        "evaluation_examples": int(evaluation_examples), "min_count": MIN_COUNT,
        "target_key": TARGET_KEY, "lesion_drop_min": LESION_DROP_MIN,
        "resource_guards": {"case_time_limit_seconds": CASE_TIME_LIMIT_SECONDS,
                            "state_bytes_limit": STATE_BYTES_LIMIT},
        "predeclared_criteria": {
            "target_case_pass_floor": TARGET_PASS_FLOOR,
            "target_case_pass": "target winner first + target feature installed + heldout lesion drop > lesion_drop_min",
            "null_stream": "shuffled reward alignment over the same frozen leave-one-out examples",
            "null_accept_rate_max": NULL_RATE_MAX, "null_install_rate_max": NULL_RATE_MAX,
            "selection": "smallest budget meeting criteria only when the full requested case matrix is complete",
        },
        "stream_digests": stream_digests, "rows": rows, "by_budget": by_budget,
        "selected_budget": selected,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
            "soma/organism.py", "soma/events/bridge.py", "r6_temporal_composition_pilot.py", __file__)},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "elapsed_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="/tmp/r6-persistent-scout-power-study.json")
    parser.add_argument("--max-cases", type=int, default=2,
                        help="bounded staged case count; incomplete runs cannot qualify a budget")
    parser.add_argument("--evaluation-examples", type=int, default=32)
    args = parser.parse_args()
    report = run_power_study(max_cases=args.max_cases, evaluation_examples=args.evaluation_examples)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_budget": report["selected_budget"], "by_budget": report["by_budget"], "cases_run": report["cases_run"]}, sort_keys=True))

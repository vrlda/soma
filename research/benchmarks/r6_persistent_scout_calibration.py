"""Frozen development calibration for the opt-in persistent variable-order scout.

This protocol is separate from R6 acceptance.  It exercises the same 2,600
declared pair/triple family at predeclared budgets and records power and null
behavior before a temporal pilot budget is frozen.
"""

from __future__ import annotations


import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from soma.events import EventBridge
from soma.events.envelope import OutcomeEvent
from soma.organism import Organism
import r6_temporal_composition_pilot as temporal


PERSISTENT_EVIDENCE_BUDGET = temporal.PERSISTENT_EVIDENCE_BUDGET
TARGET_KEY = temporal.TARGET_KEY
_examples = temporal._examples
_frame = temporal._frame
_new_organism = temporal._new_organism
SignedHistoryBitsTransducer = temporal.SignedHistoryBitsTransducer


PROTOCOL = "r6-persistent-scout-calibration-v1"
SEEDS = (0, 1, 2, 3)
BUDGETS = (128, 256, 512)
ACQUISITION_EXAMPLES = 512
MIN_COUNT = 32
FAMILY_SIZE = 2600
# Declared before running: with four independent seeds, require at least
# 3/4 target-present decisions and zero null accepts/false installs.  The
# latter is the empirical upper-bound protocol (0/4 observed), not a claim of
# a formal FWER guarantee.
POWER_FLOOR = 0.75
# Fixed family-wise false-positive objective declared before execution.  With
# four independent null seeds, the acceptance rule below requires zero
# observed null accepts/installs (the small calibration is not a formal CI).
NULL_FWER_TARGET = 0.05
NULL_ACCEPT_MAX = 0.0
NULL_FALSE_INSTALL_MAX = 0.0


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _target_present_examples(seed: int) -> List[Dict[str, object]]:
    examples = _examples(seed, -1, ACQUISITION_EXAMPLES)
    if {int(example["combo"]) for example in examples} != set(range(8)):
        raise AssertionError("target-present calibration stream lacks balanced combination coverage")
    return examples


def _rank_and_stats(state: Mapping[str, object]) -> Dict[str, object]:
    organism = Organism.from_state_dict(json.loads(json.dumps(state)))
    candidates = organism._variable_order_persistent_candidates()
    target_rank = next((index + 1 for index, item in enumerate(candidates) if item[3] == TARGET_KEY), None)
    max_score = max((float(item[0]) for item in candidates), default=0.0)
    runner_score = float(candidates[1][0]) if len(candidates) > 1 else 0.0
    return {
        "target_rank": target_rank,
        "candidate_count": len(candidates),
        "max_stat": max_score,
        "runner_score": runner_score,
        "winner_key": candidates[0][3] if candidates else None,
        "winner_z": float(candidates[0][1]) if candidates else 0.0,
        "winner_runner_margin": max_score - runner_score,
    }


def _train_calibration(seed: int, budget: int, examples: Sequence[Mapping[str, object]], reward_mode: str) -> Dict[str, object]:
    if reward_mode not in ("target", "shuffled", "random"):
        raise ValueError("unsupported calibration reward mode")
    organism = _new_organism(seed, "persistent_variable_order", persistent_budget=budget)
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xE17).uniform(-1.0, 1.0) for _ in range(len(examples))))
    bridge = EventBridge(organism, SignedHistoryBitsTransducer(8), novelty=0.10, exploration=0.20)
    reward_order = list(range(len(examples)))
    random.Random(int(seed) + 0xBAD5EED).shuffle(reward_order)
    null_rng = random.Random(int(seed) + 0xC411B4)
    for index, example in enumerate(examples):
        action = bridge.ingest(_frame(example["workspace"], index))
        if reward_mode == "target":
            reward_target = float(example["target"])
            reward = 1.0 - abs(float(action["proposal"]["value"]) - reward_target)
        elif reward_mode == "shuffled":
            reward_target = float(examples[reward_order[index]]["target"])
            reward = 1.0 - abs(float(action["proposal"]["value"]) - reward_target)
        else:
            reward = null_rng.uniform(-1.0, 1.0)
        bridge.outcome(OutcomeEvent(action["correlation_id"], float(index) + 0.5, reward, "persistent-scout-calibration").to_dict())
    bridge.close()
    state = organism.state_dict()
    scout = dict(state.get("variable_order_persistent_last_telemetry", {}))
    stats = _rank_and_stats(state)
    terminal = str(state.get("variable_order_persistent_terminal_state", "active"))
    installed = int(state.get("variable_order_install_count", 0)) > 0
    return {
        "seed": int(seed),
        "budget": int(budget),
        "reward_mode": reward_mode,
        "stream_digest": _sha(examples),
        "checkpoint_digest": _sha(state),
        "state_bytes": len(json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")),
        "family_size": int(scout.get("family_size", 0)),
        "evidence_count": int(state.get("variable_order_persistent_evidence_count", 0)),
        "episode_count": int(state.get("variable_order_persistent_episode_count", 0)),
        "terminal_state": terminal,
        "accepted": terminal == "accepted",
        "false_install": installed and reward_mode != "target",
        "installed": installed,
        "decision_reason": scout.get("decision_reason"),
        "winner_key": stats["winner_key"],
        "winner_rank": scout.get("winner_rank"),
        "winner_feature_rank": scout.get("winner_feature_rank"),
        "winner_z": stats["winner_z"],
        "runner_score": stats["runner_score"],
        "winner_runner_margin": stats["winner_runner_margin"],
        "target_rank": stats["target_rank"],
        "max_stat": stats["max_stat"],
        "candidate_count": stats["candidate_count"],
    }


def _aggregate(rows: Sequence[Mapping[str, object]], reward_mode: str) -> Dict[str, object]:
    selected = [row for row in rows if row["reward_mode"] == reward_mode]
    accepted = sum(bool(row["accepted"]) for row in selected)
    installed = sum(bool(row["installed"]) for row in selected)
    return {
        "reward_mode": reward_mode,
        "n": len(selected),
        "accepted": accepted,
        "accept_rate": accepted / float(max(1, len(selected))),
        "installed": installed,
        "false_install_rate": sum(bool(row["false_install"]) for row in selected) / float(max(1, len(selected))),
        "target_winner_rate": sum(row["winner_key"] == TARGET_KEY for row in selected) / float(max(1, len(selected))),
        "target_rank": [row["target_rank"] for row in selected],
        "max_stat": [row["max_stat"] for row in selected],
        "winner_z": [row["winner_z"] for row in selected],
        "winner_runner_margin": [row["winner_runner_margin"] for row in selected],
    }


def run_calibration(seeds: Sequence[int] = SEEDS, budgets: Sequence[int] = BUDGETS) -> Dict[str, object]:
    started = time.monotonic()
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_budgets = tuple(int(budget) for budget in budgets)
    rows: List[Dict[str, object]] = []
    streams: Dict[str, str] = {}
    for budget in normalized_budgets:
        for seed in normalized_seeds:
            target_examples = _target_present_examples(seed + 1000)
            streams["target:%s:%s" % (budget, seed)] = _sha(target_examples)
            rows.append(_train_calibration(seed + 2000, budget, target_examples, "target"))
            shuffled_examples = _target_present_examples(seed + 3000)
            streams["shuffled:%s:%s" % (budget, seed)] = _sha(shuffled_examples)
            rows.append(_train_calibration(seed + 4000, budget, shuffled_examples, "shuffled"))
            random_examples = _target_present_examples(seed + 5000)
            streams["random:%s:%s" % (budget, seed)] = _sha(random_examples)
            rows.append(_train_calibration(seed + 6000, budget, random_examples, "random"))
    by_budget = {}
    selected_budget = None
    for budget in normalized_budgets:
        budget_rows = [row for row in rows if int(row["budget"]) == budget]
        target = _aggregate(budget_rows, "target")
        shuffled = _aggregate(budget_rows, "shuffled")
        random_null = _aggregate(budget_rows, "random")
        null_accept_rate = max(float(shuffled["accept_rate"]), float(random_null["accept_rate"]))
        null_false_install_rate = max(float(shuffled["false_install_rate"]), float(random_null["false_install_rate"]))
        qualifies = bool(
            float(target["accept_rate"]) >= POWER_FLOOR
            and null_accept_rate <= NULL_ACCEPT_MAX
            and null_false_install_rate <= NULL_FALSE_INSTALL_MAX
        )
        by_budget[str(budget)] = {
            "target_present": target,
            "shuffled_null": shuffled,
            "random_null": random_null,
            "null_max_accept_rate": null_accept_rate,
            "null_max_false_install_rate": null_false_install_rate,
            "qualifies": qualifies,
        }
        if selected_budget is None and qualifies:
            selected_budget = budget
    return {
        "protocol": PROTOCOL,
        "development_only": True,
        "acceptance_gating": False,
        "budgets": list(normalized_budgets),
        "seeds": list(normalized_seeds),
        "acquisition_examples": ACQUISITION_EXAMPLES,
        "min_count": MIN_COUNT,
        "family_size": FAMILY_SIZE,
        "target_key": TARGET_KEY,
        "predeclared_criteria": {
            "power_floor": POWER_FLOOR,
            "null_fwer_target": NULL_FWER_TARGET,
            "null_accept_max": NULL_ACCEPT_MAX,
            "null_false_install_max": NULL_FALSE_INSTALL_MAX,
            "null_streams": ["shuffled", "random"],
            "selection": "smallest budget meeting all criteria",
        },
        "stream_digests": streams,
        "rows": rows,
        "by_budget": by_budget,
        "selected_budget": selected_budget,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "elapsed_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="/tmp/r6-persistent-scout-calibration.json")
    args = parser.parse_args()
    report = run_calibration()
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_budget": report["selected_budget"], "by_budget": report["by_budget"]}, sort_keys=True))

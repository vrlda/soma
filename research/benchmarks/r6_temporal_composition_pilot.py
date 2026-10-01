"""Development-only temporal composition pilot over HistoryBitsTransducer.

Raw iid 3-bit symbols are exposed through an eight-deep lag workspace.  The
evaluator-owned target is the product of three nonadjacent signed taps
(``lag0.bit0 * lag2.bit1 * lag5.bit2``).  This protocol is diagnostic only and
never changes the locked R6 tier reports.
"""

from __future__ import annotations


import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from soma.events import Event, EventBridge
from soma.events.envelope import OutcomeEvent
from soma.organism import Organism
from soma.transducers.history import HistoryBitsTransducer


PROTOCOL = "r6-temporal-composition-pilot-v3"
PERSISTENT_EVIDENCE_BUDGET = 256
DEPTH = 8
TARGET_INDICES = (0, 7, 17)  # lag0.bit0, lag2.bit1, lag5.bit2
TARGET_KEY = "k3:input-000|input-007|input-017"
WRONG_COORDINATE_KEY = "k3:input-000|input-007|input-016"
SYMBOLS = tuple(range(8))
MODES = ("variable_order", "persistent_variable_order", "nonlinear_additive_context", "shuffled_reward")


class SignedHistoryBitsTransducer(HistoryBitsTransducer):
    """Existing HistoryBits event schema with evaluator-neutral signed inputs.

    The adapter still accepts exactly the existing raw bit channels and keeps
    their declared framing.  Only the core-facing representation is centered
    to {-1,+1}, which is required for a product feature to be orthogonal to
    lower-order terms; no task feature or target enters the adapter.
    """

    def pack_frame(self, frame):
        raw = super(SignedHistoryBitsTransducer, self).pack_frame(frame)
        return tuple(2.0 * float(value) - 1.0 for value in raw[:-1]) + (1.0,)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _target(bits: Sequence[int]) -> float:
    value = 1.0
    for index in TARGET_INDICES:
        value *= 1.0 if bits[index] else -1.0
    return 0.8 * value


def _bits(symbol: int) -> Tuple[int, int, int]:
    return tuple((int(symbol) >> bit) & 1 for bit in range(3))


def _examples(seed: int, heldout_combo: int, count: int, require_heldout: bool = False) -> List[Dict[str, object]]:
    # Seed stream is independent of the held-out label; leave-one-out changes
    # only which iid workspace is retained, never the raw symbol generator.
    rng = random.Random(int(seed))
    result: List[Dict[str, object]] = []
    # Raw symbols are iid draws.  Acquisition accepts every workspace except
    # the held-out target combination; evaluation accepts only that
    # combination.  The lag workspace itself therefore remains a genuine
    # temporal view of an iid stream rather than a task-coded record.
    history = [rng.randrange(8) for _ in range(DEPTH)]
    while len(result) < int(count):
        symbol = rng.randrange(8)
        history.append(symbol)
        flat: List[int] = []
        for lag in range(DEPTH):
            flat.extend(_bits(history[-1 - lag]))
        combo = flat[0] * 4 + flat[7] * 2 + flat[17]
        accepted = combo == heldout_combo if require_heldout else combo != heldout_combo
        if accepted:
            result.append({
                "workspace": tuple(flat),
                "window": tuple(history[-DEPTH:]),
                "symbol": symbol,
                "combo": combo,
                "target": _target(flat),
            })
        del history[:-DEPTH]
    return result


def _frame(workspace: Sequence[int], event_id: int) -> List[dict]:
    return [
        Event(
            "lag%dbit%d" % (index // 3, index % 3), "temporal-pilot", int(event_id), float(event_id),
            {"bit": int(value)}, {"trust": "signed-base"},
        ).to_dict()
        for index, value in enumerate(workspace)
    ]


def _new_organism(seed: int, mode: str, persistent_budget: int = PERSISTENT_EVIDENCE_BUDGET) -> Organism:
    if mode not in MODES:
        raise ValueError("unsupported temporal pilot mode: %s" % mode)
    organism = Organism.create_default(
        input_size=DEPTH * 3 + 1, hidden_size=6, output_size=1,
        seed=int(seed), max_synapses=1024, energy_per_step=100.0,
    )
    for edge in organism.graph.iter_synapses():
        if edge.destination in organism.output_ids:
            edge.strength = 0.0
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    # Match the bounded context topology used by the validated history
    # fixture; keeping four modules also makes the causal feature owner
    # deterministic enough for this small development pilot.
    organism.enable_context_modules(max_modules=4)
    organism.resources.energy_per_step = max(100.0, organism.resources.energy_per_step)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    organism.representation_learning_enabled = False
    organism.adaptive_dendritic_enabled = False
    organism.adaptive_dendritic_shadow_enabled = False
    organism.adaptive_dendritic_fingerprint_enabled = False
    organism.actor_learning_rate = 0.10
    if mode in ("variable_order", "persistent_variable_order"):
        organism.enable_variable_order_learning(int(seed) + 7919, max_order=3)
        if mode == "persistent_variable_order":
            # Frozen pilot declaration: the calibration-selected budget action outcomes and 32 per-feature
            # observations are required before any cumulative route is even
            # offered to the legacy installer.
            organism.enable_variable_order_persistent_scout(evidence_budget=int(persistent_budget), min_count=32)
    return organism


def _train(organism: Organism, examples: Sequence[Mapping[str, object]], seed: int, shuffled: bool = False) -> Tuple[dict, Dict[str, object]]:
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xE17).uniform(-1.0, 1.0) for _ in range(len(examples))))
    bridge = EventBridge(organism, SignedHistoryBitsTransducer(DEPTH), novelty=0.10, exploration=0.20)
    order = list(range(len(examples)))
    reward_order = list(order)
    if shuffled:
        random.Random(int(seed) + 0xBAD5EED).shuffle(reward_order)
    predictions: List[float] = []
    targets: List[float] = []
    for index, example in enumerate(examples):
        action = bridge.ingest(_frame(example["workspace"], index))
        prediction = float(action["proposal"]["value"])
        target = float(example["target"])
        predictions.append(prediction)
        targets.append(target)
        reward_target = float(examples[reward_order[index]]["target"]) if shuffled else target
        reward = 1.0 - abs(prediction - reward_target)
        bridge.outcome(OutcomeEvent(action["correlation_id"], float(index) + 0.5, reward, "temporal-pilot").to_dict())
    bridge.close()
    state = organism.state_dict()
    return state, {
        "acquisition_examples": len(examples),
        "acquisition_mae": sum(abs(p - t) for p, t in zip(predictions, targets)) / float(max(1, len(targets))),
        "checkpoint_state_bytes": len(json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")),
        "cells": len(state["cells"]),
        "synapses": len(state["synapses"]),
        "variable_order_install_count": int(state.get("variable_order_install_count", 0)),
        "persistent_scout": {
            "enabled": bool(state.get("variable_order_persistent_scout_enabled", False)),
            "evidence_budget": int(state.get("variable_order_persistent_evidence_budget", 0)),
            "evidence_count": int(state.get("variable_order_persistent_evidence_count", 0)),
            "episode_count": int(state.get("variable_order_persistent_episode_count", 0)),
            "telemetry": state.get("variable_order_persistent_last_telemetry", {}),
        },
    }


def _frozen_score(checkpoint: Mapping[str, object], examples: Sequence[Mapping[str, object]], seed: int) -> Tuple[Dict[str, object], Dict[str, object]]:
    source_digest = _digest(checkpoint)
    predictions: List[float] = []
    targets: List[float] = []
    for index, example in enumerate(examples):
        clone = json.loads(json.dumps(checkpoint))
        organism = Organism.from_state_dict(clone)
        organism.set_exploration_tape((random.Random(int(seed) + 0xF00D + index).uniform(-1.0, 1.0),))
        bridge = EventBridge(organism, SignedHistoryBitsTransducer(DEPTH), novelty=0.10, exploration=0.20)
        action = bridge.ingest(_frame(example["workspace"], 0))
        predictions.append(float(action["proposal"]["value"]))
        targets.append(float(example["target"]))
        if _digest(checkpoint) != source_digest:
            raise AssertionError("temporal frozen evaluation mutated source checkpoint")
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / float(max(1, len(targets)))
    zero_sse = sum(t * t for t in targets)
    skill = 1.0 - sum((p - t) ** 2 for p, t in zip(predictions, targets)) / zero_sse if zero_sse > 1e-12 else 0.0
    return {"predictions": predictions, "targets": targets}, {
        "mae": mae,
        "skill": skill,
        "bits": len(targets),
        "checkpoint_unchanged": _digest(checkpoint) == source_digest,
        "frozen_read_only": True,
    }


def _lesion_feature(state: Mapping[str, object], feature_key: str) -> Tuple[dict, bool]:
    cloned = json.loads(json.dumps(state))
    owner_id = dict(cloned.get("variable_order_feature_owners", {})).get(feature_key)
    if owner_id is None:
        return cloned, False
    owner = dict(cloned.get("motor_modules", {}).get(owner_id, {}))
    owner_cell = owner.get("cell_id")
    feature_cell = None
    try:
        feature = tuple(feature_key.split(":", 1)[1].split("|"))
    except (IndexError, ValueError):
        return cloned, False
    for cell_id, payload in dict(cloned.get("cells", {})).items():
        if payload.get("activation_type") in ("dendritic_product", "dendritic_product_n") and tuple(payload.get("dendritic_sources", ())) == feature:
            feature_cell = cell_id
            break
    if feature_cell is None or owner_cell is None:
        return cloned, False
    changed = False
    for edge in cloned.get("synapses", []):
        if edge.get("source") == feature_cell and edge.get("destination") == owner_cell:
            edge["strength"] = 0.0
            edge["plasticity"] = 0.0
            changed = True
    return cloned, changed


def _lesion_target_k3(state: Mapping[str, object]) -> Tuple[dict, bool]:
    return _lesion_feature(state, TARGET_KEY)


class SymbolSuffixActionControl:
    """Conventional acquisition-fitted suffix-to-action mean control."""

    def __init__(self, order: int):
        if order < 1:
            raise ValueError("suffix order must be positive")
        self.order = int(order)
        self.table: Dict[Tuple[int, ...], List[float]] = {}
        self.global_values: List[float] = []

    def fit(self, examples: Sequence[Mapping[str, object]]) -> None:
        for example in examples:
            window = tuple(int(value) for value in example["window"])[-self.order:]
            target = float(example["target"])
            cell = self.table.setdefault(window, [0.0, 0.0])
            cell[0] += target
            cell[1] += 1.0
            self.global_values.append(target)

    def predict(self, example: Mapping[str, object]) -> float:
        window = tuple(int(value) for value in example["window"])
        for length in range(min(self.order, len(window)), 0, -1):
            cell = self.table.get(window[-length:])
            if cell is not None:
                return cell[0] / max(1.0, cell[1])
        return sum(self.global_values) / float(max(1, len(self.global_values)))


def _control_metrics(control: SymbolSuffixActionControl, examples: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    actions = [control.predict(example) for example in examples]
    targets = [float(example["target"]) for example in examples]
    mae = sum(abs(a - t) for a, t in zip(actions, targets)) / float(max(1, len(targets)))
    sse0 = sum(t * t for t in targets)
    skill = 1.0 - sum((a - t) ** 2 for a, t in zip(actions, targets)) / sse0 if sse0 > 1e-12 else 0.0
    return {"mae": mae, "skill": skill}


def _constant_metrics(value: float, examples: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    targets = [float(example["target"]) for example in examples]
    mae = sum(abs(float(value) - target) for target in targets) / float(max(1, len(targets)))
    sse0 = sum(target * target for target in targets)
    skill = 1.0 - sum((float(value) - target) ** 2 for target in targets) / sse0 if sse0 > 1e-12 else 0.0
    return {"mae": mae, "skill": skill}


def run_temporal_pilot(
    seeds: Sequence[int] = (0, 1),
    acquisition_examples: int = 512,
    evaluation_examples: int = 16,
    persistent_budget: int = PERSISTENT_EVIDENCE_BUDGET,
) -> Dict[str, object]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or acquisition_examples < 32 or evaluation_examples < 4:
        raise ValueError("temporal pilot requires seeds and sufficient examples")
    started = time.monotonic()
    runs: Dict[str, List[Dict[str, object]]] = {mode: [] for mode in MODES}
    manifest_digests = {}
    for seed in normalized:
        cases = []
        control_cases = []
        for heldout in SYMBOLS:
            acquisition = _examples(seed, heldout, acquisition_examples)
            evaluation = _examples(seed + 10000, heldout, evaluation_examples, require_heldout=True)
            manifest_digests["%s:%s" % (seed, heldout)] = _digest({"acquisition": acquisition, "evaluation": evaluation})
            suffix3 = SymbolSuffixActionControl(3); suffix3.fit(acquisition)
            suffix5 = SymbolSuffixActionControl(5); suffix5.fit(acquisition)
            control_cases.append({
                "heldout_combo": heldout,
                "heldout_sign": 1 if _target(evaluation[0]["workspace"]) > 0 else -1,
                "suffix3": _control_metrics(suffix3, evaluation),
                "suffix5": _control_metrics(suffix5, evaluation),
                "constant_plus": _constant_metrics(0.8, evaluation),
                "constant_minus": _constant_metrics(-0.8, evaluation),
                "zero": _constant_metrics(0.0, evaluation),
            })
            for mode in MODES:
                organism = _new_organism(seed, mode, persistent_budget=persistent_budget)
                checkpoint, resources = _train(organism, acquisition, seed, shuffled=(mode == "shuffled_reward"))
                intact, intact_metrics = _frozen_score(checkpoint, evaluation, seed + heldout)
                lesion_state, target_k3_installed = _lesion_target_k3(checkpoint)
                lesion, lesion_metrics = (None, None)
                if target_k3_installed:
                    lesion, lesion_metrics = _frozen_score(lesion_state, evaluation, seed + heldout)
                wrong_state, wrong_installed = _lesion_feature(checkpoint, WRONG_COORDINATE_KEY)
                wrong_metrics = _frozen_score(wrong_state, evaluation, seed + heldout)[1] if wrong_installed else None
                installed_k3 = sorted(
                    key for key in dict(checkpoint.get("variable_order_feature_owners", {}))
                    if str(key).startswith("k3:") and str(key) != TARGET_KEY
                )
                random_key = installed_k3[(int(seed) + int(heldout)) % len(installed_k3)] if installed_k3 else None
                random_state, random_installed = _lesion_feature(checkpoint, random_key) if random_key else (checkpoint, False)
                random_metrics = _frozen_score(random_state, evaluation, seed + heldout)[1] if random_installed else None
                resumed = Organism.from_state_dict(json.loads(json.dumps(checkpoint)))
                cases.append({
                    "heldout_combo": heldout,
                    "heldout_sign": 1 if _target(evaluation[0]["workspace"]) > 0 else -1,
                    "mode": mode,
                    "acquisition_stream_digest": _digest(acquisition),
                    "evaluation_stream_digest": _digest(evaluation),
                    "resources": resources,
                    "persistent_scout": resources.get("persistent_scout", {}),
                    "persistent_evidence_count": resources.get("persistent_scout", {}).get("evidence_count", 0),
                    "persistent_episode_count": resources.get("persistent_scout", {}).get("episode_count", 0),
                    "persistent_terminal_state": resources.get("persistent_scout", {}).get("telemetry", {}).get("terminal_state"),
                    "persistent_winner_key": resources.get("persistent_scout", {}).get("telemetry", {}).get("winner_key"),
                    "persistent_winner_rank": resources.get("persistent_scout", {}).get("telemetry", {}).get("winner_rank"),
                    "persistent_winner_feature_rank": resources.get("persistent_scout", {}).get("telemetry", {}).get("winner_feature_rank"),
                    "persistent_winner_z": resources.get("persistent_scout", {}).get("telemetry", {}).get("winner_z"),
                    "persistent_runner_score": resources.get("persistent_scout", {}).get("telemetry", {}).get("runner_score"),
                    "persistent_winner_runner_margin": resources.get("persistent_scout", {}).get("telemetry", {}).get("winner_runner_margin"),
                    "persistent_family_size": resources.get("persistent_scout", {}).get("telemetry", {}).get("family_size", 0),
                    "persistent_decision_reason": resources.get("persistent_scout", {}).get("telemetry", {}).get("decision_reason"),
                    "checkpoint_digest": _digest(checkpoint),
                    "resume_digest_equal": _digest(resumed.state_dict()) == _digest(checkpoint),
                    "target_k3_installed": target_k3_installed,
                    "frozen_metrics": intact_metrics,
                    "lesion_metrics": lesion_metrics,
                    "lesion_skill_drop": (intact_metrics["skill"] - lesion_metrics["skill"]) if lesion_metrics else None,
                    "wrong_coordinate_key": WRONG_COORDINATE_KEY,
                    "wrong_coordinate_installed": wrong_installed,
                    "wrong_coordinate_metrics": wrong_metrics,
                    "random_k3_key": random_key,
                    "random_k3_installed": random_installed,
                    "random_k3_metrics": random_metrics,
                    "frozen_read_only": intact_metrics["frozen_read_only"] and (lesion_metrics is None or lesion_metrics["frozen_read_only"]),
                })
        for mode in MODES:
            mode_cases = [case for case in cases if case["mode"] == mode]
            runs[mode].append({
                "seed": seed,
                "cases": mode_cases,
                "initial_target_k3_coverage": all(case["target_k3_installed"] for case in mode_cases) if mode == "variable_order" else False,
                "frozen_read_only_all": all(case["frozen_read_only"] for case in mode_cases),
                "resume_digest_all": all(case["resume_digest_equal"] for case in mode_cases),
            })
    variable = [case for run in runs["variable_order"] for case in run["cases"]]
    persistent = [case for run in runs["persistent_variable_order"] for case in run["cases"]]
    additive = [case for run in runs["nonlinear_additive_context"] for case in run["cases"]]
    variable_lesions = [case["lesion_skill_drop"] for case in variable if case["lesion_skill_drop"] is not None]
    additive_margin = [
        additive[index]["frozen_metrics"]["skill"] - variable[index]["frozen_metrics"]["skill"]
        for index in range(len(variable))
    ]
    return {
        "protocol": PROTOCOL,
        "development_only": True,
        "acceptance_gating": False,
        "r6_qualification": "not_evaluated",
        "depth": DEPTH,
        "target_indices": list(TARGET_INDICES),
        "target_feature_key": TARGET_KEY,
        "target_definition": "product of centered lag0.bit0, lag2.bit1, lag5.bit2",
        "persistent_scout_budget": int(persistent_budget),
        "persistent_scout_calibration_protocol": "r6-persistent-scout-calibration-v1",
        "seeds": list(normalized),
        "acquisition_examples_per_case": acquisition_examples,
        "evaluation_examples_per_case": evaluation_examples,
        "manifest_digests": manifest_digests,
        "manifest_digest": _digest(manifest_digests),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": _dependency_metadata(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "event_path": {"transducer": SignedHistoryBitsTransducer(DEPTH).spec.to_dict(), "target_in_payload": False, "heldout_target_derived_outcomes_sent": False},
        "controls": {
            "suffix_orders": [3, 5],
            "fit_partition": "acquisition_only",
            "shuffled_reward_mode": "shuffled_reward",
            "wrong_coordinate_feature_key": WRONG_COORDINATE_KEY,
            "random_k3_lesion": "deterministic installed non-target k3 feature per case",
            "constant_action_baselines": ["constant_plus", "constant_minus", "zero"],
            "cases": control_cases,
        },
        "runs": runs,
        "comparison": {
            "variable_order_minus_additive_skill": [-value for value in additive_margin],
            "mean_variable_order_minus_additive_skill": -sum(additive_margin) / float(max(1, len(additive_margin))),
            "mean_target_k3_lesion_skill_drop": sum(variable_lesions) / float(max(1, len(variable_lesions))) if variable_lesions else None,
        },
        "diagnostics": {
            "balanced_leave_one_out": all(sorted(case["heldout_combo"] for case in run["cases"]) == list(SYMBOLS) for mode in MODES for run in runs[mode]),
            "both_target_signs_covered": all({case["heldout_sign"] for case in run["cases"]} == {-1, 1} for mode in MODES for run in runs[mode]),
            "frozen_read_only_all": all(run["frozen_read_only_all"] for mode in MODES for run in runs[mode]),
            "resume_digest_all": all(run["resume_digest_all"] for mode in MODES for run in runs[mode]),
            "target_k3_installed_any": bool(variable) and any(case["target_k3_installed"] for case in variable),
            "target_k3_installed_all": bool(variable) and all(case["target_k3_installed"] for case in variable),
            "target_k3_coverage_fraction": sum(1 for case in variable if case["target_k3_installed"]) / float(max(1, len(variable))),
            "target_k3_lesion_mean_positive": bool(variable_lesions) and sum(variable_lesions) / len(variable_lesions) > 0.05,
            "target_k3_lesion_consistent": bool(variable_lesions) and all(delta > 0.05 for delta in variable_lesions),
            "persistent_target_k3_installed_any": bool(persistent) and any(case["target_k3_installed"] for case in persistent),
            "persistent_target_k3_coverage_fraction": sum(1 for case in persistent if case["target_k3_installed"]) / float(max(1, len(persistent))),
            "wrong_coordinate_control_observed": any(case["wrong_coordinate_installed"] for mode_runs in runs.values() for run in mode_runs for case in run["cases"]),
            "random_k3_control_observed": any(case["random_k3_installed"] for mode_runs in runs.values() for run in mode_runs for case in run["cases"]),
            "variable_order_beats_additive_all_cases": bool(additive_margin) and all(value < 0.0 for value in additive_margin),
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def _dependency_metadata() -> Dict[str, object]:
    files = ["soma/organism.py", "soma/events/bridge.py", "soma/events/envelope.py", "soma/events/channel.py", "soma/transducers/sdk.py", "soma/transducers/history.py"]
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *files], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"sha256": hashes, "git_revision": revision, "dirty": dirty}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--acquisition-examples", type=int, default=512)
    parser.add_argument("--evaluation-examples", type=int, default=16)
    parser.add_argument("--report", default="/tmp/r6-temporal-composition-pilot.json")
    args = parser.parse_args()
    report = run_temporal_pilot(tuple(args.seeds) if args.seeds else (0, 1), args.acquisition_examples, args.evaluation_examples)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostics"], sort_keys=True))

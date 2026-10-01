#!/usr/bin/env python3
"""Differential detector check: CUSUM update, Python organism vs Rust engine.

Drives the real Organism._apply_context_reward on one motor module with
randomized rewards/features/modes, and replays the identical tape through
the Rust port. Compares every detector field within tolerance.
"""

import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOLERANCE = 1e-9
FIELDS = ("predictor", "mean", "variance", "count", "cusum", "frozen",
          "confidence", "negative_streak", "surprise_mean", "surprise_variance",
          "surprise_count", "standardized_surprise")


def build_tape(seed=0, trials=200, dim=9, n_inputs=2):
    rng = random.Random(seed + 777)
    tape = []
    for _ in range(trials):
        tape.append({
            "reward": rng.uniform(-2.0, 2.0),
            "baseline": rng.uniform(-1.0, 1.0),
            "inputs": [rng.uniform(-1.0, 1.0) for _ in range(n_inputs)],
            "pending_action": rng.uniform(-1.0, 1.0),
            "features": [rng.uniform(-1.0, 1.0) for _ in range(dim)],
            "pending_prediction": rng.uniform(-1.0, 1.0),
            "has_pending": rng.random() < 0.7,
            "mode_normal": rng.random() < 0.8,
            "pending_scale": rng.uniform(0.05, 1.0),
            "frozen_in": rng.random() < 0.2,
            "predictor_in": [rng.uniform(-1.0, 1.0) for _ in range(dim)],
            "mean_in": rng.uniform(-1.0, 1.0),
            "variance_in": rng.uniform(0.01, 1.0),
            "count_in": rng.randint(0, 100),
            "cusum_in": rng.uniform(0.0, 4.0),
            "confidence_in": rng.random(),
            "streak_in": rng.randint(0, 5),
            "surprise_mean_in": rng.uniform(-0.5, 0.5),
            "surprise_variance_in": rng.uniform(0.01, 0.5),
        })
    return tape


def run_python(tape):
    from soma.organism import Organism
    organism = Organism.create_default(input_size=2, hidden_size=4, output_size=1, seed=0)
    organism.enable_context_modules(max_modules=1)
    module_id = organism.active_motor_module
    module = organism.motor_modules[module_id]
    organism.reward_baseline = 0.0
    organism.context_probe_module = None
    results = []
    for trial in tape:
        module.baseline = trial["baseline"]
        module.detector_predictor = list(trial["predictor_in"])
        module.detector_mean = trial["mean_in"]
        module.detector_variance = trial["variance_in"]
        module.detector_count = trial["count_in"]
        module.detector_cusum = trial["cusum_in"]
        module.detector_frozen = trial["frozen_in"]
        module.confidence = trial["confidence_in"]
        module.negative_streak = trial["streak_in"]
        organism.context_surprise_mean = 0.0
        organism.context_surprise_variance = trial["surprise_variance_in"]
        organism.context_surprise_count = 0
        organism.pending_motor_module = module_id
        organism.context_pending_mode = "normal" if trial["mode_normal"] else "search"
        organism.context_pending_features = tuple(trial["features"]) if trial["has_pending"] else ()
        organism.context_pending_action = trial["pending_action"]
        for identifier, value in zip(organism.input_ids, trial["inputs"]):
            organism.cells[identifier].activation = value
        organism.context_pending_prediction = trial["pending_prediction"]
        organism.context_pending_scale = trial["pending_scale"]
        organism.context_pending_exploration_value = 0.0
        organism.context_pending_exploration_sigma = 0.0
        organism._pending_outcome = True
        organism._apply_context_reward(trial["reward"], structural_trial=False)
        results.append({
            "predictor": list(module.detector_predictor),
            "mean": module.detector_mean,
            "variance": module.detector_variance,
            "count": module.detector_count,
            "cusum": module.detector_cusum,
            "frozen": module.detector_frozen,
            "confidence": module.confidence,
            "negative_streak": module.negative_streak,
            "surprise_mean": organism.context_surprise_mean,
            "surprise_variance": organism.context_surprise_variance,
            "surprise_count": organism.context_surprise_count,
            # NOTE: standardized_surprise is local; recompute not stored.
            "standardized_surprise": None,
        })
    return results


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tape = build_tape(seed)
    rust_tape = {
        "params": {
            "predictor_rate": 0.50,
            "surprise_leak": 0.995,
            "surprise_drift": 0.05,
            "detector_drift": 0.35,
            "detector_recovery": 0.20,
            "freeze_threshold": 2.5,
        },
        "trials": [],
    }
    for trial in tape:
        rust_tape["trials"].append({
            "reward": trial["reward"],
            "inputs": trial["inputs"],
            "pending_action": trial["pending_action"],
            "features": trial["features"],
            "pending_prediction": trial["pending_prediction"],
            "has_pending": trial["has_pending"],
            "mode_normal": trial["mode_normal"],
            "pending_scale": trial["pending_scale"],
            "state": {
                "predictor": trial["predictor_in"],
                "mean": trial["mean_in"],
                "variance": trial["variance_in"],
                "count": trial["count_in"],
                "cusum": trial["cusum_in"],
                "frozen": trial["frozen_in"],
                "confidence": trial["confidence_in"],
                "negative_streak": trial["streak_in"],
                "baseline": trial["baseline"],
                "surprise_mean": 0.0,
                "surprise_variance": trial["surprise_variance_in"],
                "surprise_count": 0,
            },
        })
    binary = "./engine/soma-engine/target/release/soma-detector"
    raw = subprocess.run([binary], input=json.dumps(rust_tape), check=True,
                         capture_output=True, text=True).stdout
    rust = json.loads(raw)["trials"]
    python = run_python(tape)
    worst, mismatches = 0.0, 0
    compare = [field for field in FIELDS if field != "standardized_surprise"]
    for index, (left, right) in enumerate(zip(python, rust)):
        right_state = right["state"]
        for field in compare:
            left_value = left[field]
            right_value = right_state[field]
            if isinstance(left_value, bool) or isinstance(right_value, bool):
                if bool(left_value) != bool(right_value):
                    mismatches += 1
                continue
            if isinstance(left_value, list):
                if len(left_value) != len(right_value):
                    mismatches += 1
                    continue
                for a, b in zip(left_value, right_value):
                    gap = abs(a - b)
                    worst = max(worst, gap)
                    if gap > TOLERANCE * max(1.0, abs(a), abs(b)):
                        mismatches += 1
                continue
            gap = abs(left_value - right_value)
            worst = max(worst, gap)
            if gap > TOLERANCE * max(1.0, abs(left_value), abs(right_value)):
                mismatches += 1
                if mismatches <= 3:
                    print("mismatch trial %d field %s: %r vs %r" % (
                        index, field, left_value, right_value))
    result = {
        "seed": seed,
        "trials": len(tape),
        "worst_gap": worst,
        "mismatches": mismatches,
        "tolerance": TOLERANCE,
    }
    result["all_passed"] = mismatches == 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

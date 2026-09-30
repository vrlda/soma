#!/usr/bin/env python3
"""Differential CircuitMixingMemory: Rust soma-mixer vs Python reference.

Trains both on the same bytes under several configurations (tight budgets
force reclamation; hashed long orders; each mechanism toggled), then scores
held-out bytes frozen and compares every per-bit probability, the
arbitration weights, and the structural counters.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOLERANCE = 1e-12
BINARY = os.path.join(ROOT, "engine", "soma-engine", "target", "release", "soma-mixer")

CONFIGS = (
    {"max_circuits": 4096},
    {"max_circuits": 4096, "plasticity_tau": 0.0, "freeze_arbitration_after": 90000},
    {"max_circuits": 4096, "plasticity_tau": 250.0},
    {"max_circuits": 8192, "orders": [0, 1, 2, 4, 8, 12], "growth_threshold": 2,
     "calibration": True, "count_limit": 255},
    {"max_circuits": 4096, "arbitration": False, "calibration": False, "correction": False,
     "halve_above": 2, "count_limit": 60, "orders": [0, 1, 2, 3]},
    {"max_circuits": 16384, "gate_partial": False, "gate_bit_position": True,
     "growth_threshold": 0, "learning_rate": 0.015, "correction": True},
)


def corpus(seed):
    with open(os.path.join(ROOT, "data", "e0", "alice.txt"), "rb") as handle:
        data = handle.read()
    start = (seed * 7919) % max(1, len(data) - 30000)
    return data[start:start + 24000], data[start + 24000:start + 27000]


def run_pair(config, train, held_out, directory):
    from soma.memory.mixing import CircuitMixingMemory
    train_path = os.path.join(directory, "train.bin")
    eval_path = os.path.join(directory, "eval.bin")
    with open(train_path, "wb") as handle:
        handle.write(train)
    with open(eval_path, "wb") as handle:
        handle.write(held_out)
    job = {"config": config, "train": [train_path], "eval": {"held_out": eval_path},
           "trace": {"held_out": 8 * len(held_out)}}
    rust = json.loads(subprocess.run([BINARY], input=json.dumps(job), check=True,
                                     capture_output=True, text=True).stdout)
    memory = CircuitMixingMemory(**config)
    memory.reset_history()
    memory.observe_bytes(train, learn=True)
    memory.reset_history()
    trace = []
    score = memory.score_bytes(held_out, learn=False, trace=trace)
    gaps = [abs(a - b) for a, b in zip(trace, rust["trace"]["held_out"])]
    weight_gap = max(abs(a - b) for mine, theirs in zip(memory.weights, rust["weights"])
                     for a, b in zip(mine, theirs))
    counters_match = (
        len(memory.circuits) == rust["circuits"]
        and memory.circuits_created == rust["circuits_created"]
        and memory.circuits_reclaimed == rust["circuits_reclaimed"]
        and len(trace) == len(rust["trace"]["held_out"]))
    return {
        "config": config,
        "bits_per_bit": score,
        "rust_bits_per_bit": rust["bits_per_bit"]["held_out"],
        "worst_probability_gap": max(gaps) if gaps else 0.0,
        "worst_weight_gap": weight_gap,
        "circuits_reclaimed": memory.circuits_reclaimed,
        "counters_match": counters_match,
    }


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train, held_out = corpus(seed)
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        for config in CONFIGS:
            cases.append(run_pair(config, train, held_out, directory))
    result = {"seed": seed, "tolerance": TOLERANCE, "cases": cases}
    result["all_passed"] = all(
        case["counters_match"] and case["worst_probability_gap"] <= TOLERANCE
        and case["worst_weight_gap"] <= TOLERANCE for case in cases)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

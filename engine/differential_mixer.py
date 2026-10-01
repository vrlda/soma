#!/usr/bin/env python3
"""Differential CircuitMixingMemory: Rust soma-mixer vs Python reference.

Trains both on the same bytes under several configurations (tight budgets
force reclamation; hashed long orders; each mechanism toggled; the second
document at trust weight 3), then scores held-out bytes frozen and compares
every per-bit probability, the arbitration weights, the structural counters,
the saved SOMAMIX1 state bytes, Rust scoring from Python's saved state, and
the state after both continue learning from it.
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
    {"max_circuits": 4096, "growth_pressure": 2, "reclaim_fraction": 0.03},
)


def corpus(seed):
    with open(os.path.join(ROOT, "data", "e0", "alice.txt"), "rb") as handle:
        data = handle.read()
    start = (seed * 7919) % max(1, len(data) - 30000)
    return data[start:start + 24000], data[start + 24000:start + 27000]


def _observe_weighted(memory, data, weight):
    for byte in data:
        for shift in range(7, -1, -1):
            memory.predict()
            memory.observe((byte >> shift) & 1, learn=True, weight=weight)


def run_pair(config, train, held_out, directory):
    """Train both engines on two documents (the second at trust weight 3),
    then compare frozen predictions, weights, counters, the saved SOMAMIX1
    bytes, and Rust scoring from Python's saved state."""
    from soma.memory.mixing import CircuitMixingMemory
    half = len(train) // 2
    first_path = os.path.join(directory, "train-1.bin")
    second_path = os.path.join(directory, "train-2.bin")
    eval_path = os.path.join(directory, "eval.bin")
    rust_state = os.path.join(directory, "rust.somamix")
    python_state = os.path.join(directory, "python.somamix")
    for path, data in ((first_path, train[:half]), (second_path, train[half:]),
                       (eval_path, held_out)):
        with open(path, "wb") as handle:
            handle.write(data)
    job = {"config": config, "train": [first_path, second_path], "train_weights": [1, 3],
           "eval": {"held_out": eval_path}, "trace": {"held_out": 8 * len(held_out)},
           "save_state": rust_state}
    rust = json.loads(subprocess.run([BINARY], input=json.dumps(job), check=True,
                                     capture_output=True, text=True).stdout)
    memory = CircuitMixingMemory(**config)
    memory.reset_history()
    _observe_weighted(memory, train[:half], 1)
    memory.reset_history()
    _observe_weighted(memory, train[half:], 3)
    state = memory.dumps()
    with open(python_state, "wb") as handle:
        handle.write(state)
    with open(rust_state, "rb") as handle:
        state_identical = handle.read() == state
    reloaded = CircuitMixingMemory.loads(state)
    memory.reset_history()
    trace = []
    score = memory.score_bytes(held_out, learn=False, trace=trace)
    reloaded.reset_history()
    reloaded_trace = []
    reloaded.score_bytes(held_out, learn=False, trace=reloaded_trace)
    cross_job = {"load_state": python_state, "train": [], "eval": {"held_out": eval_path},
                 "trace": {"held_out": 8 * len(held_out)}}
    cross = json.loads(subprocess.run([BINARY], input=json.dumps(cross_job), check=True,
                                      capture_output=True, text=True).stdout)
    # Continued learning after a load: both sides resume Python's saved state.
    resumed_state = os.path.join(directory, "resumed.somamix")
    subprocess.run([BINARY], input=json.dumps({"load_state": python_state, "train": [eval_path],
                                               "eval": {}, "save_state": resumed_state}),
                   check=True, capture_output=True, text=True)
    resumed = CircuitMixingMemory.loads(state)
    resumed.reset_history()
    _observe_weighted(resumed, held_out, 1)
    with open(resumed_state, "rb") as handle:
        resume_identical = handle.read() == resumed.dumps()
    gaps = [abs(a - b) for a, b in zip(trace, rust["trace"]["held_out"])]
    cross_gaps = [abs(a - b) for a, b in zip(trace, cross["trace"]["held_out"])]
    weight_gap = max(abs(a - b) for mine, theirs in zip(memory.weights, rust["weights"])
                     for a, b in zip(mine, theirs))
    counters_match = (
        len(memory.circuits) == rust["circuits"]
        and memory.circuits_created == rust["circuits_created"]
        and memory.circuits_reclaimed == rust["circuits_reclaimed"]
        and len(trace) == len(rust["trace"]["held_out"]) == len(cross["trace"]["held_out"]))
    return {
        "config": config,
        "bits_per_bit": score,
        "rust_bits_per_bit": rust["bits_per_bit"]["held_out"],
        "worst_probability_gap": max(gaps) if gaps else 0.0,
        "worst_cross_load_gap": max(cross_gaps) if cross_gaps else 0.0,
        "worst_weight_gap": weight_gap,
        "state_bytes": len(state),
        "state_identical": state_identical,
        "resume_identical": resume_identical,
        "python_reload_identical": reloaded_trace == trace,
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
        and case["worst_weight_gap"] <= TOLERANCE and case["worst_cross_load_gap"] <= TOLERANCE
        and case["state_identical"] and case["python_reload_identical"]
        and case["resume_identical"] for case in cases)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

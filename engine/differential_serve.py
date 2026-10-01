#!/usr/bin/env python3
"""Differential serving: EngineMixingMemory (soma-mixer-serve) vs the Python reference.

A seeded random tape of learning, frozen observation, trust weights, stream
resets, and predictions drives both memories. Every prediction and evidence
order must match exactly, and the engine's saved SOMAMIX1 file must equal
the Python reference's bytes. Also reports chat-shaped latency: condition
on a 200-byte prompt, then generate 24 bytes one bit at a time.
"""

import json
import os
import random
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from soma.memory.engine import EngineMixingMemory  # noqa: E402
from soma.memory.mixing import CircuitMixingMemory  # noqa: E402


def run_tape(seed, config):
    with open(os.path.join(ROOT, "data", "e0", "alice.txt"), "rb") as handle:
        text = handle.read()
    rng = random.Random(seed)
    reference = CircuitMixingMemory(**config)
    mismatches, predictions = 0, 0
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "brain.somamix")
        engine = EngineMixingMemory(config=config, path=path)
        try:
            for _ in range(60):
                action = rng.random()
                if action < 0.1:
                    reference.reset_history()
                    engine.reset_history()
                    continue
                start = rng.randrange(0, len(text) - 400)
                chunk = text[start:start + rng.randrange(1, 300)]
                learn = rng.random() < 0.7
                weight = rng.choice((1, 1, 1, 2, 5))
                for byte in chunk:
                    for shift in range(7, -1, -1):
                        bit = (byte >> shift) & 1
                        if rng.random() < 0.02:
                            predictions += 1
                            if reference.distribution() != engine.distribution():
                                mismatches += 1
                        reference.observe(bit, learn=learn, weight=weight)
                        engine.observe(bit, learn=learn, weight=weight)
                predictions += 1
                if reference.distribution() != engine.distribution():
                    mismatches += 1
            engine.save()
            with open(path, "rb") as handle:
                state_identical = handle.read() == reference.dumps()
            reloaded = EngineMixingMemory.from_state_dict(engine.state_dict(), base_dir=directory)
            reloaded_equal = reloaded.distribution() == reference.distribution()
            reloaded.close()
        finally:
            engine.close()
    return {"seed": seed, "config": config, "predictions": predictions,
            "mismatches": mismatches, "state_identical": state_identical,
            "reloaded_equal": reloaded_equal}


def latency():
    with open(os.path.join(ROOT, "data", "e0", "alice.txt"), "rb") as handle:
        text = handle.read()
    engine = EngineMixingMemory(config={"max_circuits": 1 << 18})
    try:
        engine.observe_bytes(text[:60000])
        engine.distribution()
        started = time.perf_counter()
        engine.reset_history()
        engine.observe_bytes(text[70000:70200], learn=False)
        engine.distribution()
        prompt_ms = 1000.0 * (time.perf_counter() - started)
        started = time.perf_counter()
        for _ in range(24 * 8):
            distribution, _ = engine.distribution()
            engine.observe(1 if distribution[1] >= 0.5 else 0, learn=False)
        generate_ms = 1000.0 * (time.perf_counter() - started)
    finally:
        engine.close()
    return {"prompt_200_bytes_ms": prompt_ms, "generate_24_bytes_ms": generate_ms,
            "per_bit_round_trip_us": 1000.0 * generate_ms / (24 * 8)}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cases = [run_tape(seed, {"max_circuits": 4096}),
             run_tape(seed + 1, {"max_circuits": 16384, "orders": [0, 1, 2, 4, 8],
                                 "calibration": True, "correction": True, "count_limit": 255})]
    result = {"cases": cases, "latency": latency()}
    result["all_passed"] = all(case["mismatches"] == 0 and case["state_identical"]
                               and case["reloaded_equal"] for case in cases)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

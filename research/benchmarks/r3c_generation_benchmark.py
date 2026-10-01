#!/usr/bin/env python3
"""Frozen R3C generation gates: validity, history dependence, lesion, silence.

G1: constrained sampled validity is 1.0 (encoding law enforced).
G2: within-horizon prefixes change continuation likelihood (spread > 0.05)
    while beyond-horizon prefixes do not (spread < 0.01).
G3: circuit lesion degrades continuation likelihood by 0.10 (causal).
G4: generation never trains protected state (counts frozen).
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json

from soma.evaluation.english import (
    load_corpus, partition_documents, split_chapters, bit_stream,
)
from soma.evaluation.generate import (
    continuation_nll, horizon_sensitivity, run_generation_suite,
)
from soma.evaluation.backgrounds import (
    BackgroundFactory, add_memory_argument, learned_state, report_path,
)
from soma.memory import SequenceCircuitMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    add_memory_argument(parser)
    args = parser.parse_args()
    if args.memory != "suffix":
        return run_mixing(args)

    data = load_corpus(args.corpus)
    acquisition, validation, _, manifest = partition_documents(split_chapters(data), 0)

    def factory():
        memory = SequenceCircuitMemory((0, 1), max_order=16, max_circuits=131072,
                                       min_support=2, prior=0.5)
        bits, _ = bit_stream(acquisition)
        for symbol in bits[:-1]:
            memory.observe(symbol)
        return memory

    suite = run_generation_suite(factory, max_bytes=50, samples=4)
    spread_within, spread_beyond = horizon_sensitivity(factory, bytes(validation))
    intact = continuation_nll(factory(), bytes(validation[:60]), bytes(validation[60:120]))
    lesioned_memory = factory()
    for circuit in lesioned_memory.circuits.values():
        circuit["counts"] = [0] * len(circuit["counts"])
    lesioned = continuation_nll(lesioned_memory, bytes(validation[:60]), bytes(validation[60:120]))
    probe = factory()
    before = [sum(circuit["counts"]) for circuit in probe.circuits.values()]
    from soma.evaluation.generate import generate_constrained
    generate_constrained(probe, b"The ", 20, deterministic=False, seed=3)
    after = [sum(circuit["counts"]) for circuit in probe.circuits.values()]
    result = {
        "protocol": "r3c-generation-v1",
        "corpus": args.corpus,
        "manifest": manifest,
        "suite": {key: value for key, value in suite.items() if key != "deterministic_outputs"},
        "spread_within_horizon": spread_within,
        "spread_beyond_horizon": spread_beyond,
        "continuation_nll_intact": intact,
        "continuation_nll_lesioned": lesioned,
        "self_training_silent": before == after,
        "gates": {
            "validity": suite["sampled_valid_rate"] == 1.0,
            "history_dependence": spread_within > 0.05 and spread_beyond < 0.01,
            "lesion_degrades": lesioned >= intact + 0.10,
            "no_self_training": before == after,
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    with open("research/reports/r3c-generation.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
    return 0 if result["all_passed"] else 1


def run_mixing(args):
    """The same four gates on the engine-served circuit-mixing memory.

    Horizon: 12 bytes (its longest context) instead of 2. Lesion: a memory
    with no learned circuits or weights. Self-training: the saved learned
    state must be byte-identical before and after generation.
    """
    from soma.evaluation.generate import generate_constrained
    data = load_corpus(args.corpus)
    acquisition, validation, _, manifest = partition_documents(split_chapters(data), 0)
    factory = BackgroundFactory(args.memory, acquisition)
    try:
        suite = run_generation_suite(factory, max_bytes=50, samples=4)
        spread_within, spread_beyond = horizon_sensitivity(factory, bytes(validation),
                                                           horizon_bytes=12)
        intact = continuation_nll(factory(), bytes(validation[:60]), bytes(validation[60:120]))
        lesioned = continuation_nll(factory.untrained(), bytes(validation[:60]),
                                    bytes(validation[60:120]))
        probe = factory()
        before = learned_state(probe)
        generate_constrained(probe, b"The ", 20, deterministic=False, seed=3)
        after = learned_state(probe)
    finally:
        factory.close()
    result = {
        "protocol": "r3c-generation-v1-mixing",
        "memory": "circuit-mixing (engine-served)",
        "corpus": args.corpus,
        "manifest": manifest,
        "suite": suite,
        "spread_within_horizon": spread_within,
        "spread_beyond_horizon": spread_beyond,
        "horizon_bytes": 12,
        "continuation_nll_intact": intact,
        "continuation_nll_lesioned": lesioned,
        "self_training_silent": before == after,
        "gates": {
            "validity": suite["sampled_valid_rate"] == 1.0,
            "history_dependence": spread_within > 0.05 and spread_beyond < 0.01,
            "lesion_degrades": lesioned >= intact + 0.10,
            "no_self_training": before == after,
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    with open(report_path("research/reports/r3c-generation.json", args.memory), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

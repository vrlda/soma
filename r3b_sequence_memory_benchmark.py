#!/usr/bin/env python3
"""Run the bounded sequence-circuit E0 experiment and causal lesion."""

import argparse
import json

from soma.evaluation.english import (
    byte_unigram_cross_bits,
    load_corpus,
    partition_documents,
    run_english_sequence_memory,
    split_chapters,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    parser.add_argument("--max-order", type=int, default=16)
    parser.add_argument("--max-circuits", type=int, default=131072)
    parser.add_argument("--acquisition-bytes", type=int, default=None)
    args = parser.parse_args()

    data = load_corpus(args.corpus)
    acquisition, validation, _, manifest = partition_documents(split_chapters(data), 0)
    if args.acquisition_bytes is not None:
        if args.acquisition_bytes < 1:
            parser.error("--acquisition-bytes must be positive")
        acquisition = acquisition[:args.acquisition_bytes]
    reports, memory = run_english_sequence_memory(
        (acquisition, validation), args.max_order, args.max_circuits)
    lesion_reports, _ = run_english_sequence_memory(
        (acquisition, validation), args.max_order, args.max_circuits, lesion=True)
    bar = byte_unigram_cross_bits(acquisition, validation)
    validation_bpb = reports[1]["bits_per_bit"]
    lesion_bpb = lesion_reports[1]["bits_per_bit"]
    result = {
        "protocol": "r3b-e0-sequence-circuit-v2",
        "corpus": args.corpus,
        "manifest": manifest,
        "acquisition_bytes_used": len(acquisition),
        "max_order": args.max_order,
        "max_circuits": args.max_circuits,
        "byte_unigram_bits_per_bit": bar,
        "reports": reports,
        "lesion_reports": lesion_reports,
        "memory": {
            "circuits": len(memory.circuits),
            "circuits_created": memory.circuits_created,
            "circuits_reclaimed": memory.circuits_reclaimed,
        },
        "gates": {
            "beats_byte_unigram": validation_bpb < bar,
            "causal_lesion_fails_gate": lesion_bpb >= bar,
            "bounded": len(memory.circuits) <= args.max_circuits,
            "state_valid": memory.validate(),
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

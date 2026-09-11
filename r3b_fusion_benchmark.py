#!/usr/bin/env python3
"""Frozen fusion gates: E0 parity, composite fallback, lesion causality.

G1: fused E0 held-out within 0.02 of memory-alone (no regression).
G2: fused composite-probe msb bits beat memory-alone by 0.05 (fallback).
G3: memory lesion degrades fused to motor level by 0.10 (causal).
"""

import argparse
import json

from soma.evaluation.english import (
    byte_unigram_bits,
    composite_probe_bytes,
    load_corpus,
    partition_documents,
    run_english_brain_memory,
    run_english_fused,
    split_chapters,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    parser.add_argument("--probe-seed", type=int, default=7)
    args = parser.parse_args()

    data = load_corpus(args.corpus)
    acquisition, validation, _, manifest = partition_documents(split_chapters(data), 0)
    probe = composite_probe_bytes(args.probe_seed)
    fused, _ = run_english_fused([acquisition, validation], 0)
    mem, _ = run_english_brain_memory([acquisition, validation], 0)
    lesion, _ = run_english_fused([acquisition, validation], 0, lesion_memory=True)
    fused_probe, _ = run_english_fused([acquisition, probe], 0)
    mem_probe, _ = run_english_brain_memory([acquisition, probe], 0)
    bar = byte_unigram_bits(validation) / 8.0
    result = {
        "protocol": "r3b-fused-v1",
        "corpus": args.corpus,
        "manifest": manifest,
        "byte_unigram_bits_per_bit": bar,
        "fused_e0": fused[1]["bits_per_bit"],
        "memory_e0": mem[1]["bits_per_bit"],
        "lesion_e0": lesion[1]["bits_per_bit"],
        "fused_probe_msb": fused_probe[1]["msb_bits_per_bit"],
        "memory_probe_msb": mem_probe[1]["msb_bits_per_bit"],
        "gates": {
            "parity": fused[1]["bits_per_bit"] <= mem[1]["bits_per_bit"] + 0.02,
            "fallback": fused_probe[1]["msb_bits_per_bit"] <= mem_probe[1]["msb_bits_per_bit"] - 0.05,
            "causal": lesion[1]["bits_per_bit"] >= fused[1]["bits_per_bit"] + 0.10,
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    with open("reports/r3b-fusion.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

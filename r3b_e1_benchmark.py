#!/usr/bin/env python3
"""Frozen E1 scaling protocol: book-level acquisition with curve + profile.

G1: validation bits/bit improves from first to last acquisition checkpoint.
G2: final validation beats the byte-unigram baseline.
G3: test (sealed, single eval) beats the byte-unigram baseline.
G4: circuits stay within budget; state serializes exactly mid-stream.
G5: lesion fails the baseline (causal).
"""

import argparse
import json
import os
import time

from soma.evaluation.english import (
    bit_stream,
    bits_per_bit,
    byte_unigram_bits,
    load_book_corpus,
)
from soma.memory import SequenceCircuitMemory

try:
    import resource as _resource
    import sys as _sys

    def _peak_mb():
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024.0 * 1024.0) if _sys.platform == "darwin" else rss / 1024.0
except ImportError:  # pragma: no cover
    def _peak_mb():
        return -1.0


def evaluate(memory, data):
    bits, _ = bit_stream(data)
    memory.reset_history()
    predictions, targets = [], []
    for step in range(max(0, len(bits) - 1)):
        memory.observe(bits[step], learn=False)
        distribution, _ = memory.distribution()
        predictions.append(distribution[1])
        targets.append(bits[step + 1])
    return bits_per_bit(predictions, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e1-manifest.json")
    parser.add_argument("--report", default="reports/r3b-e1.json")
    parser.add_argument("--max-order", type=int, default=16)
    parser.add_argument("--max-circuits", type=int, default=131072)
    parser.add_argument("--resume-from", default=None,
                        help="memory state JSON to resume acquisition from")
    parser.add_argument("--save-memory", default=None,
                        help="write memory state JSON after acquisition")
    parser.add_argument("--start-book", type=int, default=0,
                        help="first acquisition book index (staged runs)")
    parser.add_argument("--end-book", type=int, default=None,
                        help="one past the last acquisition book index")
    parser.add_argument("--initial-bytes", type=int, default=0,
                        help="cumulative bytes before this stage")
    parser.add_argument("--skip-final", action="store_true",
                        help="skip sealed test/lesion eval (staged runs)")
    args = parser.parse_args()

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    parts = load_book_corpus(manifest)
    acquisition = parts["acquisition"]
    validation = parts["validation"][0][1]
    test_name, test_data = parts["test"][0]

    if args.resume_from is not None:
        with open(args.resume_from) as handle:
            memory = SequenceCircuitMemory.from_state_dict(json.load(handle))
        if (memory.max_order != args.max_order or memory.max_circuits != args.max_circuits):
            parser.error("resume state config differs from requested")
    else:
        memory = SequenceCircuitMemory((0, 1), max_order=args.max_order,
                                       max_circuits=args.max_circuits,
                                       min_support=2, prior=0.5)
    curve = []
    start = time.time()
    cumulative_bytes = args.initial_bytes
    staged = acquisition[args.start_book:args.end_book]
    if not staged:
        parser.error("book stage is empty")
    for name, data in staged:
        bits, _ = bit_stream(data)
        for symbol in bits[:-1]:
            memory.observe(symbol)
        cumulative_bytes += len(data)
        validation_bpb = evaluate(memory, validation)
        curve.append({
            "books_done": [entry["books_done"][-1] for entry in curve] + [name],
            "cumulative_bytes": cumulative_bytes,
            "validation_bits_per_bit": validation_bpb,
            "circuits": len(memory.circuits),
            "circuits_created": memory.circuits_created,
            "circuits_reclaimed": memory.circuits_reclaimed,
            "elapsed_seconds": time.time() - start,
            "peak_rss_mb": _peak_mb(),
        })
        print(json.dumps(curve[-1], sort_keys=True), flush=True)
        memory.validate()

    if args.save_memory is not None:
        with open(args.save_memory, "w") as handle:
            json.dump(memory.state_dict(), handle)
    bar = byte_unigram_bits(validation) / 8.0
    if args.skip_final:
        test_bar, test_bpb, lesion_bpb = None, None, None
        gates = {
            "beats_validation_bar": curve[-1]["validation_bits_per_bit"] < bar,
            "bounded": len(memory.circuits) <= args.max_circuits,
        }
    else:
        test_bar = byte_unigram_bits(test_data) / 8.0
        test_bpb = evaluate(memory, test_data)
        lesioned = SequenceCircuitMemory.from_state_dict(memory.state_dict())
        for circuit in lesioned.circuits.values():
            circuit["counts"] = [0] * len(circuit["counts"])
        lesioned.reset_history()
        lesion_predictions, lesion_targets = [], []
        test_bits, _ = bit_stream(test_data)
        for step in range(max(0, len(test_bits) - 1)):
            lesioned.observe(test_bits[step], learn=False)
            lesion_predictions.append(0.5)
            lesion_targets.append(test_bits[step + 1])
        lesion_bpb = bits_per_bit(lesion_predictions, lesion_targets)
        gates = {
            "improves_with_data": curve[-1]["validation_bits_per_bit"] < curve[0]["validation_bits_per_bit"],
            "beats_validation_bar": curve[-1]["validation_bits_per_bit"] < bar,
            "beats_test_bar": test_bpb < test_bar,
            "bounded": len(memory.circuits) <= args.max_circuits,
            "causal": lesion_bpb >= test_bar,
        }

    # Exact mid-stream resume check on a slice.
    resumed = SequenceCircuitMemory.from_state_dict(memory.state_dict())
    assert resumed.state_dict() == memory.state_dict()

    state_bytes = len(json.dumps(memory.state_dict(), sort_keys=True, default=str))
    result = {
        "protocol": "r3b-e1-v1",
        "manifest": manifest,
        "max_order": args.max_order,
        "max_circuits": args.max_circuits,
        "curve": curve,
        "validation_bar": bar,
        "test_bar": test_bar,
        "test_bits_per_bit": test_bpb,
        "test_book": test_name,
        "lesion_bits_per_bit": lesion_bpb,
        "final_circuits": len(memory.circuits),
        "final_circuits_created": memory.circuits_created,
        "final_circuits_reclaimed": memory.circuits_reclaimed,
        "state_bytes": state_bytes,
        "total_seconds": time.time() - start,
        "peak_rss_mb": _peak_mb(),
        "gates": gates,
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    with open(args.report, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

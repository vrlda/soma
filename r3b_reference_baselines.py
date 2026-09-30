#!/usr/bin/env python3
"""Reference baselines for the E-series English numbers (not a gate).

SOMA's E2 score (``r3b_e1_benchmark.py``) is frozen next-bit cross-entropy
after sequential acquisition. The gate bar is a byte-unigram model, which
any context model beats. This script puts the SOMA score next to the
references a skeptical reader will check first:

- ``ngram``: a frozen, interpolated Witten-Bell byte n-gram trained on the
  same acquisition bytes and scored on the same validation/test bytes with
  no adaptation. This matches SOMA's evaluation protocol exactly.
- ``compressors``: gzip, bz2, and xz. These adapt online to the scored bytes
  (a protocol advantage over frozen scoring), so they are context, not
  like-for-like. ``xz primed`` reports the conditional cost
  ``size(acquisition + eval) - size(acquisition)``.

Byte-level bits are divided by 8 to give bits/bit; by the chain rule this
is the same unit as SOMA's MSB-first bit cross-entropy.
"""

import argparse
import bz2
import gzip
import json
import lzma
import math
import time
from collections import Counter

from soma.evaluation.english import load_verified_book_corpus

ALPHABET = 256


class WittenBellByteModel(object):
    """Interpolated Witten-Bell byte n-gram, backing off to uniform bytes."""

    def __init__(self, max_order):
        if max_order < 0:
            raise ValueError("max order must be nonnegative")
        self.max_order = int(max_order)
        self.grams = [Counter() for _ in range(self.max_order + 2)]
        self.context_totals = [Counter() for _ in range(self.max_order + 1)]
        self.context_types = [Counter() for _ in range(self.max_order + 1)]

    def fit(self, documents):
        for data in documents:
            data = bytes(data)
            for length in range(1, self.max_order + 2):
                grams = self.grams[length]
                for start in range(len(data) - length + 1):
                    grams[data[start:start + length]] += 1
        for order in range(self.max_order + 1):
            totals, types = self.context_totals[order], self.context_types[order]
            for gram, count in self.grams[order + 1].items():
                context = gram[:order]
                totals[context] += count
                types[context] += 1
        return self

    def add(self, data):
        """Incrementally learn one more document; equivalent to ``fit``."""
        data = bytes(data)
        for length in range(1, self.max_order + 2):
            grams = self.grams[length]
            totals = self.context_totals[length - 1]
            types = self.context_types[length - 1]
            for start in range(len(data) - length + 1):
                gram = data[start:start + length]
                if not grams[gram]:
                    types[gram[:-1]] += 1
                grams[gram] += 1
                totals[gram[:-1]] += 1
        return self

    def probability(self, context, symbol, order=None):
        """P(symbol | last ``order`` bytes of context)."""
        order = self.max_order if order is None else min(int(order), self.max_order)
        context = bytes(context)[-order:] if order else b""
        order = min(order, len(context))
        probability = 1.0 / ALPHABET
        for length in range(0, order + 1):
            history = context[len(context) - length:] if length else b""
            total = self.context_totals[length].get(history, 0)
            if not total:
                continue
            types = self.context_types[length][history]
            count = self.grams[length + 1].get(history + bytes((symbol,)), 0)
            probability = (count + types * probability) / (total + types)
        return probability

    def bits_per_byte(self, data, order=None):
        data = bytes(data)
        order = self.max_order if order is None else int(order)
        total = 0.0
        for index, symbol in enumerate(data):
            context = data[max(0, index - order):index]
            total -= math.log(self.probability(context, symbol, order), 2)
        return total / max(1, len(data))


def compressor_bits_per_byte(data, prime=b""):
    """Compressed bits/byte for gzip, bz2, xz; xz also primed on ``prime``."""
    data = bytes(data)
    size = max(1, len(data))
    result = {
        "gzip_9": 8.0 * len(gzip.compress(data, 9, mtime=0)) / size,
        "bz2_9": 8.0 * len(bz2.compress(data, 9)) / size,
        "xz_9e": 8.0 * len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)) / size,
    }
    if prime:
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME,
                    "dict_size": 1 << 26}]
        base = len(lzma.compress(prime, format=lzma.FORMAT_RAW, filters=filters))
        joint = len(lzma.compress(prime + data, format=lzma.FORMAT_RAW, filters=filters))
        result["xz_primed"] = 8.0 * max(0, joint - base) / size
    return result


def to_bits_per_bit(rates):
    return {name: rate / 8.0 for name, rate in rates.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e2-manifest.json")
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--soma-report", default="reports/r3b-e2.json")
    parser.add_argument("--out", default="reports/r3b-e2-reference-baselines.json")
    parser.add_argument("--skip-primed", action="store_true",
                        help="skip the slow xz-primed conditional cost")
    args = parser.parse_args()

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, args.manifest)
    acquisition = [data for _, data in parts["acquisition"]]
    evaluation = {
        "validation": parts["validation"][0],
        "test": parts["test"][0],
    }

    started = time.time()
    model = WittenBellByteModel(args.max_order).fit(acquisition)
    fit_seconds = time.time() - started
    prime = b"".join(acquisition) if not args.skip_primed else b""

    splits = {}
    for split, (name, data) in evaluation.items():
        ngram = {"order_%d" % order: model.bits_per_byte(data, order) / 8.0
                 for order in range(args.max_order + 1)}
        splits[split] = {
            "book": name,
            "bytes": len(data),
            "ngram_witten_bell_frozen": ngram,
            "compressors_adaptive": to_bits_per_bit(compressor_bits_per_byte(data, prime)),
        }

    soma = {}
    try:
        with open(args.soma_report) as handle:
            soma_report = json.load(handle)
        soma = {
            "validation": soma_report["curve"][-1]["validation_bits_per_bit"],
            "test": soma_report["test_bits_per_bit"],
            "max_order_bits": soma_report.get("max_order"),
            "source": args.soma_report,
        }
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        soma = {"source": args.soma_report, "unavailable": True}

    report = {
        "benchmark": "r3b-reference-baselines-v1",
        "unit": "bits/bit (byte-level bits/byte divided by 8)",
        "manifest": args.manifest,
        "acquisition_bytes": sum(len(data) for data in acquisition),
        "max_order": args.max_order,
        "fit_seconds": round(fit_seconds, 1),
        "soma": soma,
        "splits": splits,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

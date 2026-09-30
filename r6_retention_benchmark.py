#!/usr/bin/env python3
"""R6 retention and order sensitivity for CircuitMixingMemory (metrics, not gates).

Master plan §22 step 4. Freezes how forgetting and book-order sensitivity
are measured, so step 5 (consolidation) has a fixed target.

Protocol r6-retention-v1 (E2 manifest):
- The last PROBE_BYTES of each acquisition book are held out of training
  as that book's retention probe. Everything else trains, in order.
- After every book, the frozen model scores every probe plus the Jekyll
  validation book. This gives a retention matrix R[k][j] (bits/bit on
  book j's probe after learning k books).
- forgetting_j = R[final][j] - min over k >= j of R[k][j]: how much of the
  best performance on book j is lost by the end. Mean forgetting averages
  over every book except the last.
- Order sensitivity: the same run under the canonical order, the reversed
  order, and ORDER_SEEDS seeded shuffles; the spread (max - min) of final
  validation scores.
- Diagnostic: the canonical order also runs at the full budget, where
  reclamation is rare. Forgetting that remains there comes from plastic
  arbitration drift, not from the budget.

A frozen count-based n-gram has zero forgetting and zero order sensitivity
by construction (its counts commute), so these numbers are properties of
the plastic and budgeted mechanisms.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from r6_circuit_mixing_benchmark import COMPACT_BUDGET, FULL_BUDGET, build_engine, run_engine
from soma.evaluation.english import load_verified_book_corpus

PROTOCOL = "r6-retention-v1"
PROBE_BYTES = 32768
ORDER_SEEDS = (1, 2)


def split_probes(acquisition, probe_bytes=PROBE_BYTES):
    """Return [(name, train_bytes, probe_bytes)] holding out each book's tail."""
    split = []
    for name, data in acquisition:
        if len(data) < 4 * probe_bytes:
            raise ValueError("book too short for a retention probe: %s" % name)
        split.append((name, data[:-probe_bytes], data[-probe_bytes:]))
    return split


def orders_for(names, seeds=ORDER_SEEDS):
    orders = {"canonical": list(names), "reversed": list(reversed(names))}
    for seed in seeds:
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        orders["shuffle_%d" % seed] = shuffled
    return orders


def retention_metrics(order, curve):
    """Matrix, per-book forgetting, and mean forgetting from an engine curve.

    ``order`` lists book names in training order; ``curve[k]`` holds the
    scores after training on order[:k+1], keyed by probe name.
    """
    matrix = [[checkpoint["bits_per_bit"]["probe:" + name] for name in order]
              for checkpoint in curve]
    forgetting = {}
    for j, name in enumerate(order[:-1]):
        seen = [matrix[k][j] for k in range(j, len(order))]
        forgetting[name] = matrix[-1][j] - min(seen)
    mean = sum(forgetting.values()) / max(1, len(forgetting))
    return {"order": order, "matrix": matrix, "forgetting": forgetting,
            "mean_forgetting": mean,
            "validation_curve": [c["bits_per_bit"]["validation"] for c in curve]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="reports/e2-manifest.json")
    parser.add_argument("--out", default="reports/r6-retention.json")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build_engine()
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, args.manifest)
    split = split_probes(parts["acquisition"])
    names = [name for name, _, _ in split]
    started = time.time()

    with tempfile.TemporaryDirectory() as directory:
        train_paths, evals = {}, {}
        for name, train, probe in split:
            train_paths[name] = os.path.join(directory, "train-%s.bin" % name)
            evals["probe:" + name] = os.path.join(directory, "probe-%s.bin" % name)
            with open(train_paths[name], "wb") as handle:
                handle.write(train)
            with open(evals["probe:" + name], "wb") as handle:
                handle.write(probe)
        evals["validation"] = os.path.join(directory, "validation.bin")
        with open(evals["validation"], "wb") as handle:
            handle.write(parts["validation"][0][1])

        runs = [(label, order, COMPACT_BUDGET) for label, order in orders_for(names).items()]
        runs.append(("canonical_full_budget", list(names), FULL_BUDGET))
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                label: pool.submit(run_engine, {"max_circuits": budget},
                                   [train_paths[name] for name in order], evals, True)
                for label, order, budget in runs}
            raw = {label: future.result() for label, future in futures.items()}

    results = {}
    for label, order, budget in runs:
        metrics = retention_metrics(order, raw[label]["curve"])
        metrics.update({
            "budget": budget,
            "final_validation": metrics["validation_curve"][-1],
            "circuits_reclaimed": raw[label]["circuits_reclaimed"],
        })
        results[label] = metrics
    compact_labels = [label for label, _, budget in runs if budget == COMPACT_BUDGET]
    finals = [results[label]["final_validation"] for label in compact_labels]
    report = {
        "protocol": PROTOCOL,
        "manifest": args.manifest,
        "probe_bytes": PROBE_BYTES,
        "trained_bytes": sum(len(train) for _, train, _ in split),
        "runs": results,
        "summary": {
            "order_spread_final_validation": max(finals) - min(finals),
            "mean_forgetting_compact": sum(results[l]["mean_forgetting"] for l in compact_labels)
            / len(compact_labels),
            "mean_forgetting_canonical_compact": results["canonical"]["mean_forgetting"],
            "mean_forgetting_canonical_full": results["canonical_full_budget"]["mean_forgetting"],
            "final_validation_by_order": {l: results[l]["final_validation"] for l in compact_labels},
        },
        "total_seconds": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

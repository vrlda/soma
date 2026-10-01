#!/usr/bin/env python3
"""Where CircuitMixingMemory's forgetting comes from (diagnostics, not gates).

Master plan §22 step 5 evidence. Uses the r6-retention-v1 probes (canonical
book order) and reports:

- ngram_dilution: forgetting of a frozen Witten-Bell order-5 byte n-gram
  trained book by book. Counts commute, so this is the floor that any
  accumulating model reaches just by averaging books together.
- decomposition: for each budget and metaplasticity setting, each book's
  loss since it was learned (final probe score minus the score right after
  that book), split into
    weights  = final score - score with final circuits but the arbitration
               weights saved right after that book, and
    circuits = the remainder (circuit growth, count changes, reclamation).
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from r3b_reference_baselines import WittenBellByteModel
from r6_circuit_mixing_benchmark import COMPACT_BUDGET, FULL_BUDGET, build_engine, run_engine
from r6_retention_benchmark import orders_for, retention_metrics, split_probes
from soma.evaluation.english import load_verified_book_corpus
from soma.memory.mixing import LEGACY_DEFAULTS

PROTOCOL = "r6-forgetting-diagnostics-v1"
SETTINGS = (
    ("compact", COMPACT_BUDGET, 100000.0),
    ("compact_no_metaplasticity", COMPACT_BUDGET, 0.0),
    ("full", FULL_BUDGET, 100000.0),
    ("full_no_metaplasticity", FULL_BUDGET, 0.0),
)


def ngram_dilution(split, validation, order):
    train = {name: data for name, data, _ in split}
    probes = {name: probe for name, _, probe in split}
    model = WittenBellByteModel(5)
    curve = []
    for name in order:
        model.add(train[name])
        scores = {"probe:" + other: model.bits_per_byte(probes[other]) / 8.0 for other in order}
        scores["validation"] = model.bits_per_byte(validation) / 8.0
        curve.append({"bits_per_bit": scores})
    metrics = retention_metrics(order, curve)
    return {"mean_forgetting": metrics["mean_forgetting"],
            "forgetting": metrics["forgetting"],
            "final_validation": metrics["validation_curve"][-1]}


def decompose(order, result):
    rows, weight_total, circuit_total = {}, 0.0, 0.0
    for index, name in enumerate(order[:-1]):
        key = "probe:" + name
        after = result["curve"][index]["bits_per_bit"][key]
        final = result["bits_per_bit"][key]
        old_weights = result["snapshot_scores"][key]
        rows[name] = {"total": final - after, "weights": final - old_weights,
                      "circuits": old_weights - after}
        weight_total += final - old_weights
        circuit_total += old_weights - after
    count = max(1, len(order) - 1)
    return {"books": rows, "mean_total": (weight_total + circuit_total) / count,
            "mean_weights": weight_total / count, "mean_circuits": circuit_total / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="reports/e2-manifest.json")
    parser.add_argument("--out", default="reports/r6-forgetting-diagnostics.json")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build_engine()
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, args.manifest)
    split = split_probes(parts["acquisition"])
    order = orders_for([name for name, _, _ in split])["canonical"]
    validation = parts["validation"][0][1]
    started = time.time()

    with tempfile.TemporaryDirectory() as directory:
        train_paths, evals = [], {}
        for name, train, probe in split:
            train_path = os.path.join(directory, "train-%s.bin" % name)
            probe_path = os.path.join(directory, "probe-%s.bin" % name)
            with open(train_path, "wb") as handle:
                handle.write(train)
            with open(probe_path, "wb") as handle:
                handle.write(probe)
            train_paths.append(train_path)
            evals["probe:" + name] = probe_path
        snapshot = ["probe:" + name for name in order]

        def run(setting):
            label, budget, tau = setting
            job_config = dict(LEGACY_DEFAULTS, max_circuits=budget, plasticity_tau=tau)
            return label, run_engine(job_config, train_paths, evals, True,
                                     snapshot_evals=snapshot)

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            engine_runs = dict(pool.map(run, SETTINGS))

    report = {
        "protocol": PROTOCOL,
        "manifest": args.manifest,
        "order": order,
        "ngram_dilution": ngram_dilution(split, validation, order),
        "decomposition": {label: dict(decompose(order, engine_runs[label]),
                                      budget=budget, plasticity_tau=tau)
                          for label, budget, tau in SETTINGS},
        "total_seconds": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary = {label: {k: round(v, 5) for k, v in entry.items() if k.startswith("mean")}
               for label, entry in report["decomposition"].items()}
    summary["ngram_dilution"] = round(report["ngram_dilution"]["mean_forgetting"], 5)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

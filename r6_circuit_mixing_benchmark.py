#!/usr/bin/env python3
"""Frozen R6 English gates for CircuitMixingMemory (soma-mixer engine).

Same protocol as the E2 gate (``r3b_e1_benchmark.py``): verified E2
manifest, sequential acquisition over 14 books, frozen next-bit scoring on
the unseen validation book and the sealed test book. The model is
``soma/memory/mixing.py`` run through its bit-exact Rust port.

Gates:
  G1 improves_with_data: final validation beats the first checkpoint.
  G2 beats_byte_unigram: test beats the acquisition-fitted unigram bar.
  G3 beats_prior_e2: test beats the previous SOMA E2 score.
  G4 beats_ngram_order2 / G5 beats_ngram_order5: test beats the frozen
     Witten-Bell byte n-grams from reports/r3b-e2-reference-baselines.json.
  G6 bounded: live circuits never exceed the budget.
  G7 arbitration_causal: replacing plastic arbitration with longest-match
     backoff over the same circuits is worse (compact tier).
Ablations of calibration, correction, and evidence-gated growth are
reported with their measured effect; they are not gates.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from soma.evaluation.english import byte_unigram_cross_bits, load_verified_book_corpus

ROOT = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(ROOT, "engine", "soma-engine", "target", "release", "soma-mixer")
PROTOCOL = "r6-circuit-mixing-v1"
FULL_BUDGET = 1 << 24
COMPACT_BUDGET = 1 << 22
ABLATIONS = (
    ("no_arbitration", {"arbitration": False}),
    ("no_calibration", {"calibration": False}),
    ("no_correction", {"correction": False}),
    ("no_growth_gate", {"growth_threshold": 0}),
)


def run_engine(config, train_paths, eval_paths, every_file=False):
    job = {"config": config, "train": train_paths, "eval": eval_paths,
           "eval_every_file": bool(every_file)}
    completed = subprocess.run([BINARY], input=json.dumps(job), check=True,
                               capture_output=True, text=True)
    report = json.loads(completed.stdout)
    report.pop("trace", None)
    report.pop("weights", None)
    return report


def build_engine():
    subprocess.run(["cargo", "build", "--release", "--quiet", "--manifest-path",
                    os.path.join(ROOT, "engine", "soma-engine", "Cargo.toml")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e2-manifest.json")
    parser.add_argument("--baselines", default="reports/r3b-e2-reference-baselines.json")
    parser.add_argument("--prior", default="reports/r3b-e2.json")
    parser.add_argument("--out", default="reports/r6-circuit-mixing.json")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    if not args.no_build:
        build_engine()
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, args.manifest)
    acquisition = parts["acquisition"]
    validation = parts["validation"][0]
    test = parts["test"][0]
    with open(args.baselines) as handle:
        baselines = json.load(handle)
    with open(args.prior) as handle:
        prior_test = json.load(handle)["test_bits_per_bit"]

    started = time.time()
    with tempfile.TemporaryDirectory() as directory:
        def write(name, data):
            path = os.path.join(directory, name + ".bin")
            with open(path, "wb") as handle:
                handle.write(data)
            return path

        train_paths = [write("acq-%02d-%s" % (index, name), data)
                       for index, (name, data) in enumerate(acquisition)]
        eval_paths = {"validation": write("validation", validation[1]),
                      "test": write("test", test[1])}
        runs = [("full", {"max_circuits": FULL_BUDGET}, True),
                ("compact", {"max_circuits": COMPACT_BUDGET}, False)]
        if not args.skip_ablations:
            for name, change in ABLATIONS:
                config = dict(change, max_circuits=COMPACT_BUDGET)
                runs.append((name, config, False))
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {name: pool.submit(run_engine, config, train_paths, eval_paths, every)
                       for name, config, every in runs}
            results = {name: future.result() for name, future in futures.items()}

    trained = b"".join(data for _, data in acquisition)
    unigram_bar = byte_unigram_cross_bits(trained, test[1])
    ngram = baselines["splits"]["test"]["ngram_witten_bell_frozen"]
    full = results["full"]
    compact = results["compact"]
    curve = full["curve"]
    test_score = full["bits_per_bit"]["test"]

    ablations = {}
    for name, _ in ABLATIONS:
        if name not in results:
            continue
        scores = results[name]["bits_per_bit"]
        ablations[name] = {
            "validation": scores["validation"],
            "test": scores["test"],
            "validation_cost": scores["validation"] - compact["bits_per_bit"]["validation"],
            "helps": scores["validation"] > compact["bits_per_bit"]["validation"],
        }

    gates = {
        "improves_with_data": curve[-1]["bits_per_bit"]["validation"]
        < curve[0]["bits_per_bit"]["validation"],
        "beats_byte_unigram": test_score < unigram_bar,
        "beats_prior_e2": test_score < prior_test,
        "beats_ngram_order2": test_score < ngram["order_2"],
        "beats_ngram_order5": test_score < ngram["order_5"],
        "bounded": full["circuits"] <= FULL_BUDGET and compact["circuits"] <= COMPACT_BUDGET,
    }
    if "no_arbitration" in ablations:
        gates["arbitration_causal"] = ablations["no_arbitration"]["helps"]
    report = {
        "protocol": PROTOCOL,
        "manifest": args.manifest,
        "test_book": test[0],
        "validation_book": validation[0],
        "acquisition_bytes": len(trained),
        "unigram_test_bar": unigram_bar,
        "prior_e2_test": prior_test,
        "ngram_test": {"order_2": ngram["order_2"], "order_5": ngram["order_5"]},
        "full": full,
        "compact": compact,
        "ablations": ablations,
        "gates": gates,
        "all_passed": all(gates.values()),
        "total_seconds": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"gates": gates, "all_passed": report["all_passed"],
                      "full": full["bits_per_bit"], "compact": compact["bits_per_bit"],
                      "ablations": ablations}, indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

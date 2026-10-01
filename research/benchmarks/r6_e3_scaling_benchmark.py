#!/usr/bin/env python3
"""R6 E3-full scaling run for circuit mixing (master plan §22 step 9).

Pre-registered before any model saw the corpus (research/docs/r6-e3-scaling.md):

  H1 data:     at the 2**24-circuit budget, final Jekyll validation on
               E3-full beats the E3-lite result at the same budget (0.2350).
  H2 capacity: on E3-full, the 2**26-circuit budget beats the 2**24 one.

Also reported for each budget: the validation curve every EVAL_INTERVAL
books, bits/bit per acquired MB, wall time, peak RSS, and saved-state
size. Time Machine test scores are reported but are validation-grade for
this model (research/docs/r6-untouched-test.md), so they decide nothing.

The corpus is reproduced with ``research/scripts/build_e3_full_corpus.py fetch``.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
import sys
import tempfile
import time

from r6_circuit_mixing_benchmark import build_engine, run_engine
from soma.evaluation.english import load_verified_book_corpus
from soma.memory.mixing import LEGACY_DEFAULTS

PROTOCOL = "r6-e3-scaling-v1"
MANIFEST = "research/reports/e3-full-manifest.json"
E3_LITE_VALIDATION_AT_2_24 = 0.2350  # research/reports/r6-circuit-mixing-e3lite.json, full budget
BUDGETS = (("budget_2_24", 1 << 24), ("budget_2_26", 1 << 26))
EVAL_INTERVAL = 25


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--out", default="research/reports/r6-e3-scaling.json")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build_engine()
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, args.manifest)
    acquisition = parts["acquisition"]
    started = time.time()
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        train_paths = []
        for index, (name, data) in enumerate(acquisition):
            path = os.path.join(directory, "acq-%04d-%s.bin" % (index, name))
            with open(path, "wb") as handle:
                handle.write(data)
            train_paths.append(path)
        evals = {}
        for split in ("validation", "test"):
            path = os.path.join(directory, split + ".bin")
            with open(path, "wb") as handle:
                handle.write(parts[split][0][1])
            evals[split] = path
        for label, budget in BUDGETS:
            state = os.path.join(directory, label + ".somamix")
            began = time.time()
            report = run_engine(dict(LEGACY_DEFAULTS, max_circuits=budget), train_paths, evals,
                                every_file=False, extra={"eval_interval": EVAL_INTERVAL,
                                                         "save_state": state})
            results[label] = {
                "max_circuits": budget,
                "validation": report["bits_per_bit"]["validation"],
                "test_validation_grade": report["bits_per_bit"]["test"],
                "circuits": report["circuits"],
                "circuits_reclaimed": report["circuits_reclaimed"],
                "peak_rss_mb": report["peak_rss_mb"],
                "seconds": time.time() - began,
                "state_bytes": os.path.getsize(state),
                "curve": [{"cumulative_mb": point["cumulative_bytes"] / 1e6,
                           "validation": point["bits_per_bit"]["validation"],
                           "circuits": point["circuits"]} for point in report["curve"]],
            }
            os.remove(state)
    small, large = results["budget_2_24"], results["budget_2_26"]
    hypotheses = {
        "H1_data_helps": small["validation"] < E3_LITE_VALIDATION_AT_2_24,
        "H2_capacity_helps": large["validation"] < small["validation"],
    }
    report = {
        "protocol": PROTOCOL,
        "manifest": args.manifest,
        "books": len(acquisition),
        "acquisition_mb": sum(len(data) for _, data in acquisition) / 1e6,
        "e3_lite_validation_at_2_24": E3_LITE_VALIDATION_AT_2_24,
        "results": results,
        "hypotheses": hypotheses,
        "total_seconds": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"hypotheses": hypotheses,
                      "validation": {k: v["validation"] for k, v in results.items()},
                      "peak_rss_mb": {k: v["peak_rss_mb"] for k, v in results.items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

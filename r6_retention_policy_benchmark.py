#!/usr/bin/env python3
"""Retention under budget pressure (protocol r6-retention-policy-v1).

Pre-registered in docs/r6-retention-policy.md. Baseline defaults against
reclaim_fraction 0.03 + growth_pressure 8, in five (corpus, budget)
settings, each scored once on the sealed book of reports/e3u-manifest.json.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from r6_circuit_mixing_benchmark import build_engine, run_engine
from soma.evaluation.english import load_verified_book_corpus
from soma.memory.mixing import LEGACY_DEFAULTS

PROTOCOL = "r6-retention-policy-v1"
CANDIDATE = {"reclaim_fraction": 0.03, "growth_pressure": 8}
SETTINGS = (  # name, manifest, budget; longest first so the pool finishes together
    ("E_e3full_2_24", "reports/e3-full-manifest.json", 1 << 24),
    ("D_e2_2_24", "reports/e2-manifest.json", 1 << 24),
    ("C_e2_2_22", "reports/e2-manifest.json", 1 << 22),
    ("B_e2_2_20", "reports/e2-manifest.json", 1 << 20),
    ("A_e1_2_16", "reports/e1-manifest.json", 1 << 16),
)
PRESSURED = ("A_e1_2_16", "B_e2_2_20", "C_e2_2_22", "E_e3full_2_24")
LIGHT = "D_e2_2_24"
NON_INFERIORITY = 0.0005
SEALED = "reports/e3u-manifest.json"


def _load(path):
    with open(path) as handle:
        return load_verified_book_corpus(json.load(handle), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/r6-retention-policy.json")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build_engine()
    sealed = _load(SEALED)["test"][0]
    started = time.time()
    with tempfile.TemporaryDirectory() as directory:
        def write(name, data):
            path = os.path.join(directory, name)
            with open(path, "wb") as handle:
                handle.write(data)
            return path

        corpora = {}
        validation = None
        for _, manifest, _ in SETTINGS:
            if manifest in corpora:
                continue
            parts = _load(manifest)
            tag = os.path.basename(manifest).split(".")[0]
            corpora[manifest] = [write("%s-%04d" % (tag, index), data)
                                 for index, (_, data) in enumerate(parts["acquisition"])]
            validation = validation or parts["validation"][0][1]
        evals = {"sealed": write("sealed.bin", sealed[1]),
                 "validation": write("validation.bin", validation)}

        def job(spec):
            name, manifest, budget, arm = spec
            config = dict(LEGACY_DEFAULTS, max_circuits=budget)  # baseline = pre-v1 defaults
            if arm == "candidate":
                config.update(CANDIDATE)
            began = time.time()
            report = run_engine(config, corpora[manifest], evals)
            return name, arm, {
                "config": config,
                "sealed": report["bits_per_bit"]["sealed"],
                "validation_dev": report["bits_per_bit"]["validation"],
                "circuits": report["circuits"],
                "circuits_reclaimed": report["circuits_reclaimed"],
                "peak_rss_mb": report["peak_rss_mb"],
                "seconds": time.time() - began,
            }

        specs = [(name, manifest, budget, arm) for name, manifest, budget in SETTINGS
                 for arm in ("baseline", "candidate")]
        results = {}
        with ThreadPoolExecutor(args.jobs) as pool:
            for name, arm, result in pool.map(job, specs):
                results.setdefault(name, {})[arm] = result
                print(name, arm, round(result["sealed"], 5), flush=True)

    for name, pair in results.items():
        pair["difference"] = pair["candidate"]["sealed"] - pair["baseline"]["sealed"]
    hypotheses = {"H1_" + name: results[name]["difference"] < 0 for name in PRESSURED}
    hypotheses["H2_" + LIGHT] = results[LIGHT]["difference"] <= NON_INFERIORITY
    report = {
        "protocol": PROTOCOL,
        "sealed_manifest": SEALED,
        "sealed_book": sealed[0],
        "candidate": CANDIDATE,
        "non_inferiority": NON_INFERIORITY,
        "results": results,
        "hypotheses": hypotheses,
        "adopt_candidate": all(hypotheses.values()),
        "total_seconds": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"hypotheses": hypotheses, "adopt": report["adopt_candidate"],
                      "difference": {k: v["difference"] for k, v in results.items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

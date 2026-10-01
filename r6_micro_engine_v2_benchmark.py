#!/usr/bin/env python3
"""R6 Micro tier on the Rust engine, protocol r6-micro-engine-v2.

Pre-registered in docs/r6-micro-engine.md (section v2). Changes from v1,
all fixed before running: the budget is 2**17, the largest power of two
whose state bound fits 4 MiB with 28-byte circuit records; current defaults
(retention policy adopted 2026-10-01); and the resume gate compares like
with like (both runs train without interleaved evaluation). Gates and
thresholds are v1's.
"""

import argparse
import json
import os
import resource
import sys
import tempfile
import time

from r6_circuit_mixing_benchmark import build_engine, run_engine
from r6_micro_engine_benchmark import CEILINGS, REFERENCE, THRESHOLDS
from r6_tier_benchmark import _canonical_manifest_match, _slope, load_verified_manifest

PROTOCOL = "r6-micro-engine-v2"
CIRCUIT_RECORD_BYTES = 28
BUDGET = 1 << 17


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e1-manifest.json")
    parser.add_argument("--out", default="reports/r6-micro-engine-v2.json")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build_engine()
    manifest, parts = load_verified_manifest(args.manifest)
    config = {"max_circuits": BUDGET}
    with tempfile.TemporaryDirectory() as directory:
        def write(name, data):
            path = os.path.join(directory, name + ".bin")
            with open(path, "wb") as handle:
                handle.write(data)
            return path

        def state(name):
            return os.path.join(directory, name + ".somamix")

        train = [write("acq-%02d-%s" % (i, name), data)
                 for i, (name, data) in enumerate(parts["acquisition"])]
        evals = {"validation": write("validation", parts["validation"][0][1]),
                 "test": write("test", parts["test"][0][1])}
        # Timed, resource-measured run: train every book, then score once.
        began = time.time()
        model = run_engine(config, train, evals, extra={"save_state": state("full")})
        seconds = time.time() - began
        with open(state("full"), "rb") as handle:
            full_bytes = handle.read()
        # Resume: the same training split at book 2, also without evaluation.
        split = max(1, len(train) // 2)
        run_engine(config, train[:split], {}, extra={"save_state": state("half")})
        run_engine(config, train[split:], {}, extra={"load_state": state("half"),
                                                      "save_state": state("resumed")})
        with open(state("resumed"), "rb") as handle:
            resume_exact = handle.read() == full_bytes
        # Checkpoint curve: a separate run evaluating after every book.
        curve_run = run_engine(config, train, {"validation": evals["validation"]},
                               every_file=True)
        control = run_engine(dict(config, arbitration=False), train, evals)

    curve = [point["bits_per_bit"]["validation"] for point in curve_run["curve"]]
    validation = model["bits_per_bit"]["validation"]
    state_bytes = len(full_bytes)
    peak_state_bound = state_bytes + (BUDGET - model["circuits"]) * CIRCUIT_RECORD_BYTES
    control_validation = control["bits_per_bit"]["validation"]
    gates = {
        "canonical_manifest": _canonical_manifest_match(manifest),
        "state_within_ceiling": max(state_bytes, peak_state_bound) <= CEILINGS["state_bytes"],
        "rss_within_ceiling": model["peak_rss_mb"] <= CEILINGS["rss_mb"],
        "time_within_ceiling": seconds <= CEILINGS["seconds"],
        "resume_exact": resume_exact,
        "at_least_three_checkpoints": len(curve) >= 3,
        "endpoint_gain": curve[0] - curve[-1] >= THRESHOLDS["endpoint_gain"],
        "slope": _slope(curve) <= THRESHOLDS["slope"],
        "control_margin": control_validation - validation >= THRESHOLDS["control_margin"],
    }
    report = {
        "protocol": PROTOCOL,
        "manifest": args.manifest,
        "budget": BUDGET,
        "ceilings": CEILINGS,
        "thresholds": THRESHOLDS,
        "reference": REFERENCE,
        "model": {
            "validation": validation,
            "test": model["bits_per_bit"]["test"],
            "curve_validation": curve,
            "circuits": model["circuits"],
            "circuits_reclaimed": model["circuits_reclaimed"],
            "peak_rss_mb": model["peak_rss_mb"],
            "wall_seconds": seconds,
            "state_bytes": state_bytes,
            "peak_state_bound_bytes": peak_state_bound,
            "endpoint_gain": curve[0] - curve[-1],
            "slope": _slope(curve),
        },
        "control_no_arbitration": {"validation": control_validation,
                                   "test": control["bits_per_bit"]["test"],
                                   "margin": control_validation - validation},
        "resume": {"split_after_book": split, "state_identical": resume_exact},
        "harness_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gates": gates,
        "all_passed": all(gates.values()),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"gates": gates, "validation": validation, "curve": curve,
                      "test": report["model"]["test"], "control": control_validation},
                     indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

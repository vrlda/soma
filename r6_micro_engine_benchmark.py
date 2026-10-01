#!/usr/bin/env python3
"""R6 Micro tier on the Rust engine (protocol r6-micro-engine-v1).

Pre-registered in docs/r6-micro-engine.md (master plan §22 step 10, ADR 0006
item 3): circuit mixing at a 2**16-circuit budget on the canonical E1
manifest must fit the frozen Micro ceilings (4 MiB state, 512 MiB RSS,
180 s), resume byte-exactly from a mid-run file, improve over checkpoints,
and beat its capacity-matched no-arbitration control.
"""

import argparse
import json
import os
import resource
import sys
import tempfile
import time

from r6_circuit_mixing_benchmark import build_engine, run_engine
from r6_tier_benchmark import (CANONICAL_MANIFEST_SHA256, CANONICAL_MANIFEST_TOTALS,
                               _canonical_manifest_match, _slope, load_verified_manifest)

PROTOCOL = "r6-micro-engine-v1"
BUDGET = 1 << 16
CIRCUIT_RECORD_BYTES = 32
CEILINGS = {"state_bytes": 4 * 1024 * 1024, "rss_mb": 512, "seconds": 180}
THRESHOLDS = {"endpoint_gain": 0.002, "slope": -0.0001, "control_margin": 0.001}
REFERENCE = {"small_tier_test": 0.3976182720750115, "byte_unigram_test": 0.5620325638503956}
DIAGNOSTIC_BUDGET = 512
ORDERS = 11  # default orders 0-8, 10, 12
MIN_BUDGET = 256 * ORDERS  # the memory refuses smaller budgets


def _timed(config, train, evals, extra=None):
    began = time.time()
    report = run_engine(config, train, evals, every_file=True, extra=extra)
    return report, time.time() - began


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e1-manifest.json")
    parser.add_argument("--out", default="reports/r6-micro-engine.json")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build_engine()
    manifest, parts = load_verified_manifest(args.manifest)
    canonical = _canonical_manifest_match(manifest)
    config = {"max_circuits": BUDGET}
    with tempfile.TemporaryDirectory() as directory:
        def write(name, data):
            path = os.path.join(directory, name + ".bin")
            with open(path, "wb") as handle:
                handle.write(data)
            return path

        train = [write("acq-%02d-%s" % (i, name), data)
                 for i, (name, data) in enumerate(parts["acquisition"])]
        evals = {"validation": write("validation", parts["validation"][0][1]),
                 "test": write("test", parts["test"][0][1])}
        full_state = os.path.join(directory, "full.somamix")
        model, seconds = _timed(config, train, evals, {"save_state": full_state})
        with open(full_state, "rb") as handle:
            full_bytes = handle.read()

        split = max(1, len(train) // 2)
        half_state = os.path.join(directory, "half.somamix")
        resumed_state = os.path.join(directory, "resumed.somamix")
        run_engine(config, train[:split], {}, extra={"save_state": half_state})
        run_engine(config, train[split:], {}, extra={"load_state": half_state,
                                                      "save_state": resumed_state})
        with open(resumed_state, "rb") as handle:
            resume_exact = handle.read() == full_bytes

        control, _ = _timed(dict(config, arbitration=False), train, evals)
        if DIAGNOSTIC_BUDGET >= MIN_BUDGET:
            diagnostic, diagnostic_seconds = _timed({"max_circuits": DIAGNOSTIC_BUDGET},
                                                    train, evals)
            diagnostic_report = {
                "validation": diagnostic["bits_per_bit"]["validation"],
                "test": diagnostic["bits_per_bit"]["test"],
                "peak_rss_mb": diagnostic["peak_rss_mb"],
                "wall_seconds": diagnostic_seconds,
            }
        else:
            diagnostic_report = {"infeasible": "budget %d is below the memory's minimum of "
                                               "256 x %d orders = %d" % (
                                                   DIAGNOSTIC_BUDGET, ORDERS, MIN_BUDGET)}

    curve = [point["bits_per_bit"]["validation"] for point in model["curve"]]
    state_bytes = len(full_bytes)
    overhead = state_bytes - model["circuits"] * CIRCUIT_RECORD_BYTES
    peak_state_bound = overhead + BUDGET * CIRCUIT_RECORD_BYTES
    validation = model["bits_per_bit"]["validation"]
    control_validation = control["bits_per_bit"]["validation"]
    gates = {
        "canonical_manifest": canonical,
        "state_within_ceiling": max(state_bytes, peak_state_bound) <= CEILINGS["state_bytes"],
        "rss_within_ceiling": model["peak_rss_mb"] <= CEILINGS["rss_mb"],
        "time_within_ceiling": seconds <= CEILINGS["seconds"],
        "resume_exact": resume_exact,
        "at_least_three_checkpoints": len(curve) >= 3,
        "endpoint_gain": curve[0] - validation >= THRESHOLDS["endpoint_gain"],
        "slope": _slope(curve) <= THRESHOLDS["slope"],
        "control_margin": control_validation - validation >= THRESHOLDS["control_margin"],
    }
    report = {
        "protocol": PROTOCOL,
        "manifest": args.manifest,
        "manifest_canonical_sha256": CANONICAL_MANIFEST_SHA256,
        "manifest_totals": CANONICAL_MANIFEST_TOTALS,
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
            "engine_seconds": model["total_seconds"],
            "wall_seconds": seconds,
            "state_bytes": state_bytes,
            "peak_state_bound_bytes": peak_state_bound,
            "endpoint_gain": curve[0] - validation,
            "slope": _slope(curve),
        },
        "control_no_arbitration": {
            "validation": control_validation,
            "test": control["bits_per_bit"]["test"],
            "margin": control_validation - validation,
        },
        "diagnostic_512_circuits": diagnostic_report,
        "resume": {"split_after_book": split, "state_identical": resume_exact},
        "harness_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gates": gates,
        "all_passed": all(gates.values()),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"gates": gates, "validation": validation, "test": report["model"]["test"],
                      "control": control_validation, "seconds": seconds,
                      "rss_mb": model["peak_rss_mb"], "state_bytes": state_bytes},
                     indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

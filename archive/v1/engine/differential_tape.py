#!/usr/bin/env python3
"""Differential tape check: same LCG op stream in Python SparseDirectedGraph.

Compares edge-strength structure against the Rust soma-tape report.
ID schemes differ by design (Python string ids vs generational ints);
agreement is structural: edge sets and strengths must match.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MASK64 = (1 << 64) - 1


def lcg(state):
    state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
    return state, (state >> 33) & 0x7FFFFFFF


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    from soma.synapses import SparseDirectedGraph, Synapse
    graph = SparseDirectedGraph()
    state = seed
    started = time.time()
    for _ in range(count):
        state, kind = lcg(state)
        kind %= 10
        state, source = lcg(state)
        source %= 16
        state, destination = lcg(state)
        destination %= 16
        if destination == source:
            destination = (destination + 1) % 16
        if kind < 7:
            state, bits = lcg(state)
            strength = (bits / 2147483648.0) * 2.0 - 1.0
            if not graph.has("n%03d" % source, "n%03d" % destination):
                graph.add(Synapse("n%03d" % source, "n%03d" % destination, strength))
        else:
            if graph.has("n%03d" % source, "n%03d" % destination):
                graph.remove("n%03d" % source, "n%03d" % destination)
    python_us = int((time.time() - started) * 1e6)
    # NOTE: Python duplicate-add keeps the FIRST strength (matches Rust).
    edges = {}
    for synapse in graph.iter_synapses():
        src = int(synapse.source[1:])
        dst = int(synapse.destination[1:])
        edges[(src, dst)] = synapse.strength
    binary = "./engine/soma-engine/target/release/soma-tape"
    report = json.loads(subprocess.run(
        [binary, str(seed), str(count)], check=True, capture_output=True, text=True).stdout)
    rust_edges = {(s, d): strength for s, d, _, _, strength in report["edges"]}
    ok = set(edges) == set(rust_edges)
    worst = 0.0
    for key, strength in edges.items():
        worst = max(worst, abs(strength - rust_edges[key]))
    result = {
        "seed": seed,
        "ops": count,
        "edge_sets_match": ok,
        "python_edges": len(edges),
        "rust_edges": len(report["edges"]),
        "worst_abs_strength_gap": worst,
        "python_us": python_us,
        "rust_us": report["microseconds"],
        "speedup": python_us / max(1, report["microseconds"]),
    }
    result["all_passed"] = ok and worst < 1e-9
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

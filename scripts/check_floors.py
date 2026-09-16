#!/usr/bin/env python3
"""R6 quality floors: frozen release-candidate bars checked against reports.

Floors derive from measured E2/E0 runs (see docs/model-card-e2.md), not
aspirations. Fails loud on any regression.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLOORS = {
    # report path, dotted key, operator, bound
    ("reports/r3b-e2.json", "test_bits_per_bit", "<", 0.45),
    ("reports/r3b-e2.json", "final_circuits", "<", 131072),
    ("reports/r3b-e0-phase.json", "heldout.bits_per_bit", "<", 1.05),
    ("reports/r3b-sequence-memory.json", "reports.1.bits_per_bit", "<", 0.60),
    ("reports/r3c-generation.json", "suite.sampled_valid_rate", "==", 1.0),
    ("reports/r3d-dialogue.json", "all_passed", "==", True),
    ("reports/r3b-fusion.json", "all_passed", "==", True),
    ("reports/r7-instruction.json", "all_passed", "==", True),
    ("reports/r7-tool.json", "all_passed", "==", True),
    ("reports/r9-redteam.json", "all_passed", "==", True),
    ("reports/r11-glyph.json", "all_passed", "==", True),
    ("reports/r12-cart.json", "all_passed", "==", True),
}


def lookup(payload, dotted):
    for part in dotted.split("."):
        if part.isdigit():
            payload = payload[int(part)]
        else:
            payload = payload[part]
    return payload


def main():
    failures = []
    for path, dotted, operator, bound in sorted(FLOORS):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            failures.append("%s missing report %s" % (path, path))
            continue
        with open(full) as handle:
            value = lookup(json.load(handle), dotted)
        good = {"<": value < bound, "==": value == bound}[operator]
        print("%-45s %-28s = %-10s floor %s %s %s" % (
            path, dotted, round(value, 4) if isinstance(value, float) else value,
            operator, bound, "OK" if good else "FAIL"))
        if not good:
            failures.append("%s %s" % (path, dotted))
    if failures:
        print("FLOOR FAILURES: %d" % len(failures))
        return 1
    print("ALL FLOORS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

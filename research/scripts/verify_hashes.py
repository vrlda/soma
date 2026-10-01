#!/usr/bin/env python3
"""Verify every tracked artifact against research/reports/r0-baseline/SHA256SUMS.

Exit 0 when all present files match; reports missing files and mismatches.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMS = os.path.join(ROOT, "research", "reports", "r0-baseline", "SHA256SUMS")


def main():
    missing, mismatch, checked = [], [], 0
    entries = {}
    with open(SUMS) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, _, path = line.partition("  ")
            entries[path.strip()] = digest
    for path, digest in sorted(entries.items()):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            missing.append(path)
            continue
        actual = hashlib.sha256()
        with open(full, "rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                actual.update(chunk)
        checked += 1
        if actual.hexdigest() != digest:
            mismatch.append(path)
    print("checked=%d missing=%d mismatch=%d" % (checked, len(missing), len(mismatch)))
    for path in missing:
        print("missing: %s" % path)
    for path in mismatch:
        print("mismatch: %s" % path)
    return 0 if not missing and not mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rewrite research/reports/r0-baseline/SHA256SUMS from the tracked evidence set.

Policy: every git-tracked file is pinned except prose and repository
metadata (Markdown, .gitignore, .gitattributes, .github/, and the manifest
itself). Code, tests, data, configs, formats, and reports are evidence;
documentation may be edited without re-freezing anything.

Run this only after the frozen benchmarks covering every changed file have
been re-run and their results confirmed (see research/docs/reproducibility.md).
Git LFS content must be present: hashing an LFS pointer file would pin the
pointer instead of the report, so the script refuses.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMS_PATH = "research/reports/r0-baseline/SHA256SUMS"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
EXCLUDED_NAMES = {".gitignore", ".gitattributes"}
EXCLUDED_SUFFIXES = (".md", ".MD")
EXCLUDED_PREFIXES = (".github/",)


def pinned(path):
    if path == SUMS_PATH or os.path.basename(path) in EXCLUDED_NAMES:
        return False
    if path.endswith(EXCLUDED_SUFFIXES) or path.startswith(EXCLUDED_PREFIXES):
        return False
    return True


def digest(path):
    hasher = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as handle:
        head = handle.read(len(LFS_POINTER))
        if head == LFS_POINTER:
            raise SystemExit("refusing to pin LFS pointer %s: run `git lfs pull`" % path)
        hasher.update(head)
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode("utf-8").split("\0")
    paths = sorted(path for path in tracked if path and pinned(path))
    lines = ["%s  %s\n" % (digest(path), path) for path in paths]
    with open(os.path.join(ROOT, SUMS_PATH), "w") as handle:
        handle.writelines(lines)
    print("pinned=%d" % len(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

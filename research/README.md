# Research archive

Everything that produced SOMA's research evidence, moved out of the repository
root on 2026-10-02. Nothing here is needed to install or run SOMA.

| Folder | Contents |
|---|---|
| `benchmarks/` | milestone gates (`r*_benchmark.py`), pilots, studies, demos |
| `tests/` | tests of those benchmarks |
| `reports/` | frozen JSON reports and corpus manifests (large ones via Git LFS) |
| `docs/` | per-milestone records (R1–R9), with honest boundaries |
| `scripts/` | evidence tools (`update_hashes.py`, `verify_hashes.py`) and corpus builders |

Run everything from the repository root:

```bash
python3 research/benchmarks/r6_circuit_mixing_benchmark.py --help
python3 -m unittest discover -s research/tests -t .
python3 research/scripts/verify_hashes.py
```

**After the move.** Reports keep the paths and harness hashes that were
true when they were produced. Moving a script, and adding the two lines
that put the repository root on `sys.path`, changes its source hash. A
report's recorded `source_sha256` or `harness_sha256` therefore matches
the script's pre-archive version in git history, not the moved file.
Results and code behavior are unchanged; both test suites pass.

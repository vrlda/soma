# Dependencies

- Python package (`soma/`): Python 3.10 minimum (3.7-compatible syntax
  retained). Standard library only; no `requirements.txt` needed.
- Rust engine (`engine/soma-engine`, edition 2024): crates `serde`,
  `serde_json`, and `libm` (see `Cargo.toml` / `Cargo.lock`). Needed for
  `engine/tests.sh`, `r6_circuit_mixing_benchmark.py`, and the other
  differential harnesses. The Python reference paths do not require it.
- CI: Python 3.10–3.14 test matrix plus an engine job (`bash engine/tests.sh`).
- Tests: `python3 -m unittest discover -s tests -v` (about 25 minutes);
  `bash engine/tests.sh` (about 1 minute plus build).
- Large reports are stored with Git LFS; without it, two pinned reports
  fail `scripts/verify_hashes.py`.

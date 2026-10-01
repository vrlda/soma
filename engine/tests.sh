#!/bin/bash
# R5 engine gate: Rust unit tests plus every differential harness.
set -eo pipefail
cd "$(dirname "$0")/.."
cargo build --release --manifest-path engine/soma-engine/Cargo.toml
cargo test --manifest-path engine/soma-engine/Cargo.toml 2>&1 | grep -E "test result"
python3 engine/differential_tape.py 7 50000 | grep -E "all_passed"
python3 engine/differential_forward.py 0 200 | grep -E "all_passed"
python3 engine/differential_learn.py 0 | grep -E "all_passed"
python3 engine/differential_evidence.py 0 | grep -E "all_passed"
python3 engine/differential_detector.py 0 | grep -E "all_passed"
python3 engine/differential_mixer.py 0 | grep -E "all_passed"
python3 engine/differential_serve.py 0 | grep -E "\"all_passed\""
python3 -m unittest tests.test_r8_engine_brain 2>&1 | tail -1
echo "ENGINE GATES DONE"

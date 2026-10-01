#!/bin/bash
# SOMA prosumer installer (R8 rehearsal): checks, layouts, smoke test.
# Local-only unless SOMA_ENGINE_URL is set; no sudo, no secrets. Fails loud on any step.
set -euo pipefail

SOMA_ROOT="${SOMA_ROOT:-$HOME/.soma}"
PYTHON="${PYTHON:-python3}"

echo "==> checking python"
$PYTHON --version || { echo "FAIL: python3 required"; exit 1; }
$PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" \
  || { echo "FAIL: python 3.10+ required"; exit 1; }

echo "==> detecting hardware"
$PYTHON -B -c "
import platform, os
print('platform:', platform.platform())
print('cpu:', os.cpu_count())
try:
    import resource
    print('memory note: check Activity Monitor (darwin reports bytes)')
except ImportError:
    print('memory note: resource module unavailable')
"

echo "==> creating layout at $SOMA_ROOT"
mkdir -p "$SOMA_ROOT/brains" "$SOMA_ROOT/backups" "$SOMA_ROOT/bundles"

echo "==> verifying package imports"
SOMA_SRC="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$SOMA_SRC" $PYTHON -B -c "
import soma, soma.evaluation, soma.service, soma.persistence
from soma.memory import SequenceCircuitMemory
memory = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
for symbol in (0, 1, 1, 0):
    memory.observe(symbol)
memory.validate()
print('package smoke: ok')
"

echo "==> building the Rust engine (language memory for --memory mixing)"
if [ "${SOMA_SKIP_ENGINE:-0}" = "1" ]; then
  echo "    skipped (SOMA_SKIP_ENGINE=1); new brains use the original memory"
elif command -v cargo >/dev/null 2>&1; then
  cargo build --release --quiet --manifest-path "$SOMA_SRC/engine/soma-engine/Cargo.toml" \
    || { echo "FAIL: engine build failed (set SOMA_SKIP_ENGINE=1 to install without it)"; exit 1; }
  echo "    engine built: new brains default to the circuit-mixing memory"
elif [ -n "${SOMA_ENGINE_URL:-}" ]; then
  # Prebuilt, signature-verified engine (needs a trusted key; see docs/r8-signing.md).
  PYTHONPATH="$SOMA_SRC" SOMA_HOME="$SOMA_ROOT" $PYTHON -B -m soma.service.main engine-install "$SOMA_ENGINE_URL" \
    || { echo "FAIL: engine install failed (unset SOMA_ENGINE_URL to install without it)"; exit 1; }
  echo "    prebuilt engine verified and installed in $SOMA_ROOT/engine"
else
  echo "    cargo not found: engine not built; new brains use the original memory"
  echo "    (install Rust from https://rustup.rs, or set SOMA_ENGINE_URL to a signed"
  echo "    release directory ending in /, then rerun to enable --memory mixing)"
fi

echo "==> running fast unit subset"
cd "$SOMA_SRC"
PYTHONPATH="$SOMA_SRC" $PYTHON -B -m unittest \
  tests.test_r3b_sequence_memory.SequenceCircuitMemoryTests.test_state_round_trip_is_exact \
  tests.test_r3b_sequence_memory.SequenceCircuitMemoryTests.test_circuit_budget_is_hard \
  tests.test_r8_service 2>&1 | tail -3

echo "==> install complete: brains in $SOMA_ROOT/brains"
echo "    try: SOMA_BRAINS=\$SOMA_ROOT/brains python3 -m soma.service.main doctor"
echo "    (doctor reports whether the engine is available)"

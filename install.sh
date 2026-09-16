#!/bin/bash
# SOMA prosumer installer (R8 rehearsal): checks, layouts, smoke test.
# Local-only, no network, no sudo, no secrets. Fails loud on any step.
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

echo "==> running fast unit subset"
cd "$SOMA_SRC"
PYTHONPATH="$SOMA_SRC" $PYTHON -B -m unittest \
  tests.test_r3b_sequence_memory.SequenceCircuitMemoryTests.test_state_round_trip_is_exact \
  tests.test_r3b_sequence_memory.SequenceCircuitMemoryTests.test_circuit_budget_is_hard \
  tests.test_r8_service 2>&1 | tail -3

echo "==> install complete: brains in $SOMA_ROOT/brains"
echo "    try: SOMA_BRAINS=\$SOMA_ROOT/brains python3 -m soma.service.main doctor"

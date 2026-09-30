# Changelog (Keep a Changelog format)

## Unreleased

- Stop tracking build output: `engine/soma-engine/target/` (3,518 files of
  macOS binaries that could not run on Linux), `__pycache__`, `*.pyc`,
  `.DS_Store`.
- `engine/tests.sh` builds the release binaries and fails on any parity
  mismatch (`pipefail`); CI runs it and tests Python 3.10–3.14.
- `r3b_reference_baselines.py`: frozen byte n-gram and compressor
  references for the E2 English score (non-gating).
- `CircuitMixingMemory` (`soma/memory/mixing.py`) with bit-exact Rust port
  `soma-mixer`: byte-context circuits, evidence-gated growth, budgeted
  reclamation, plastic arbitration. E2 test 0.2406 bits/bit (was 0.3995);
  `r6_circuit_mixing_benchmark.py` passes 7/7 gates.
- Fix quadratic reclamation scan in `SequenceCircuitMemory` that made
  teaching a service brain take minutes (installer smoke test timed out).

## 0.3.0-alpha — language track working set

Added (all gated, all hashed in reports/r0-baseline/SHA256SUMS):

- Universal event system: envelopes v2, channels, clocks, correlation,
  transducer SDK, three byte/bit adapters plus history/phase variants.
- Calibrated evidence routing as the default router (threshold retained).
- Sequence memory: bounded suffix circuits, chunk promotion, episodic
  rules, trust arbitration, quarantine, consolidation grace, conflicts.
- English E0–E2 + E3-lite scaling (book corpora with manifests).
- Generation (constrained UTF-8), dialogue, systematicity, instruction
  skills, tools, lifelong benchmark, red-team gates, prosumer service
  (CLI/chat/HTTP/doctor), binary `.soma` format, journaling, Rust spike.

Not yet: production engine port, GPU, E3+ scale, human preference
ratings, installer/signatures/clean-machine release, multimodal input.

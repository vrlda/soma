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
- ADR 0009: the circuit memory is the organism's language substrate (master
  plan step 6). Organism plasticity rejoins the language path only via a
  pre-declared experiment. Adds a domain-neutrality test on a non-text
  Markov byte process.
- Consolidation (ADR 0008): per-weight-set metaplasticity is the default
  in circuit mixing. Forgetting −15%, order spread −25%, final validation
  better in every book order; E2 full test 0.2406. Adds
  `r6_forgetting_diagnostics.py` (weight vs circuit decomposition, n-gram
  dilution floor). Rejected candidates are recorded.
- `r6_retention_benchmark.py`: retention matrix and book-order sensitivity
  for circuit mixing. Baseline mean forgetting 0.0060 bits/bit, order spread
  0.0053. Two thirds of the forgetting persists without budget pressure.
- Mechanism audit (ADR 0007): the circuit-mixing correction stage is off by
  default (redundant with partial-byte gating; it also amplified recency
  interference). Every remaining mechanism has a positive ablation. E2
  full-budget test is now 0.2411 (was 0.2406).
- Untouched test scored once: full circuit mixing 0.2381 bits/bit on
  *The Man Who Was Thursday* vs 0.2588 for the frozen order-5 n-gram and
  0.4009 for the prior E2 memory; all pre-registered hypotheses held.
- Pre-registered untouched test for R6 circuit mixing
  (`docs/r6-untouched-test.md`, `r6_untouched_test.py`): the book, frozen
  configuration, comparisons, and hypotheses are fixed before scoring;
  single-evaluation guard; editions with ASCII quotes are refused.
- Integrity re-baseline: all frozen benchmarks covering changed code re-run
  and matched (`reports/rebaseline-2026-09-30/`); manifest policy is now
  "pin all tracked evidence, not prose" via `scripts/update_hashes.py`;
  274 files pinned, 0 mismatches.
- Docs tidy: master plan §16 carries live status for R5–R12 and §22 is the
  single ordered work list with done-criteria. README gains a "what's
  next" section and a documentation map. New `docs/README.md` index.
  Superseded content is marked historical; stale engine and dependency
  facts are corrected.
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

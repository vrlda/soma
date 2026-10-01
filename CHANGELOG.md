# Changelog (Keep a Changelog format)

## Unreleased

- Small candidate brain (`r8-small-candidate-v1`, pre-registered, sealed
  book pg78576): 0.2161 bits/bit (1.73 bits/byte) against 0.2343 for the
  same config on E2 and 0.2574 for an order-5 n-gram. All 8 gates pass,
  including 0.49 s chat at 1.5 GB RSS and byte-identical resume.
  `scripts/build_small_candidate.py` rebuilds it.
- Chat on large engine brains: a turn takes 0.38–0.5 s, down from 4–5 s.
  The running engine is reused and an unchanged memory is hard-linked,
  not rewritten.
- Signed distribution (`docs/r8-signing.md`): standard-library Ed25519
  (RFC 8032 vectors; byte-identical to OpenSSL), `key-generate`,
  `key-trust`, `sign`, `verify`, `brain-download` (verified before
  import), `engine-install` (signed prebuilt engine, safe extraction).
  `brain-import-soma` checks a `.sig` when present (`--require-signature`).
  Adds a release workflow for Linux and macOS engine packages, and an
  `install.sh` fallback via `SOMA_ENGINE_URL`. No secret key is in the
  repository.
- Micro v2 (`r6_micro_engine_v2_benchmark.py`, pre-registered): 8/9. Resume
  and effect pass; arbitration ties its no-arbitration control at this
  scale (−0.0003) once evidence-gated growth is on. Micro stays diagnostic.
- Retention under budget pressure (`r6-retention-policy-v1`, pre-registered,
  sealed untouched book pg76222): new defaults `reclaim_fraction` 0.03 and
  `growth_pressure` 8 (pressure-adaptive growth), in Python and Rust.
  Sealed-book gains are −0.0069 at 2^16, −0.0043 at 2^20, −0.0013 at 2^22,
  and −0.0021 for E3-full at 2^24; neutral at E2 2^24. Old states load
  unchanged. `LEGACY_DEFAULTS` pins earlier frozen reports. Fixed: Rust
  `loads` dropped a saved `growth_pressure`. Parity now covers continued
  learning after a load.
- E3-full scaling (step 9, pre-registered): on 108 MB, Jekyll validation
  is 0.2256 at 2^24 circuits (E3-lite 0.2350) and 0.2189 at 2^26 (6.5 GB
  RAM). More data and more capacity both help.
- Micro tier on the engine (`r6_micro_engine_benchmark.py`, pre-registered):
  35 MB RSS, 19.5 s, 1.98 MB state, all within the frozen ceilings the
  Python reference failed. v1 still fails 6/9: a 2^16 budget saturates on
  the first book, so it does not improve with more data, and the resume
  comparison mixed evaluation into one side. Recorded as failed.
- E3-full corpus (`scripts/build_e3_full_corpus.py`): 206 new public-domain
  books chosen by a fixed seeded rule, 108.4 MB acquisition, hash manifest.
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
- The installer builds the Rust engine when `cargo` is present
  (`SOMA_SKIP_ENGINE=1` opts out). `brain-create` defaults to `--memory
  auto`: circuit mixing when the engine is built, else the original memory.
  `doctor` reports engine availability with a live round trip and flags
  engine brains whose engine is missing instead of crashing.
- Chat latency: a turn takes a median of 0.70 s (was 2.3 s) through
  exact speedups to the dialogue tier: a heap-and-bucket reclamation
  index (verified against the original scan; compacted so it stays
  bounded), `SequenceCircuitMemory` state v2 with bit-string contexts (v1
  still loads), the C JSON encoder, and no duplicate validation. All eight
  dependent gate reports and the R6 reclamation report reproduce
  byte-for-byte; reclamation runs 1.8x faster.
- E3-lite scaling preview: circuit mixing reaches 0.2346 test at 18 MB
  (from 0.2406 at 11.5 MB); capacity is now binding.
- Engine serving (step 8): `soma-mixer-serve` plus `EngineMixingMemory`,
  bit-exact with the reference; `brain-create --memory mixing` brains live
  on the engine (clone, backup, `.soma` optional chunk, crash recovery).
  R3C 4/4, R3D 6/6, R7 5/5 and 4/4 on the engine-served memory via
  `--memory mixing`. Chat on engine brains uses evidence arbitration and
  deterministic decoding (opt-in `respond`/`instruct` parameters), fixing
  byte-garbage generation. `chat_turn` no longer reloads after saving.
- Circuit-mixing product parity (step 7): `distribution()` interface,
  trust-weighted observation, `Organism.enable_sequence_memory(kind="mixing")`
  with checkpoints and quarantine, and a binary `SOMAMIX1` state that Python
  and Rust write byte-identically. R4 6/6 and R9 7/7 on the new memory via
  `--memory mixing`; the default gate reports are unchanged.
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

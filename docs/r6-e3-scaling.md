# R6 E3-full scaling: pre-registration

Master plan §22 step 9. **Status: pre-registered; corpus being built; no
model has seen it.** The git history timestamps this file before the run.

## Corpus (`reports/e3-full-manifest.json`)

- **Acquisition:** the 27 E3-lite books (18 MB; see the
  [names erratum](../reports/e3-manifest-ERRATA.md)), plus about 90 MB of
  new Project Gutenberg fiction. The new books are chosen by the fixed rule
  in `scripts/build_e3_full_corpus.py`: English, LoCC PR/PS, Fiction;
  excluding every book used anywhere before (by ID and by normalized
  catalog title), collections, and texts outside 100 KB–2 MB; in a seeded
  shuffle (seed 2026) until about 90 MB. Catalog hash, eligible count, and
  every skip are recorded in the manifest.
- **Validation:** Jekyll. **Test:** Time Machine (validation-grade for this
  model, reported only).
- **Reproduction:** book texts are not committed. `python3
  scripts/build_e3_full_corpus.py fetch` re-downloads them and verifies
  every hash. Gutenberg edits files occasionally; a hash failure is
  reported, never silently accepted.

## Hypotheses (decided on Jekyll validation)

- **H1 data:** at a 2^24-circuit budget, E3-full beats E3-lite's 0.2350.
- **H2 capacity:** on E3-full, 2^26 circuits beat 2^24.

Reported for each budget: the validation curve every 25 books,
bits/bit per acquired MB, wall time, peak RSS, and saved-state size.
`r6_e3_scaling_benchmark.py` runs both budgets one after the other.

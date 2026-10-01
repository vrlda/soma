# R6 E3-full scaling: pre-registration

Master plan §22 step 9. **Status: run 2026-10-01; H1 and H2 both hold.** The git history
timestamps this pre-registration and the corpus manifest before the run.

## Corpus (`research/reports/e3-full-manifest.json`)

- **Acquisition:** the 27 E3-lite books (18 MB; see the
  [names erratum](../reports/e3-manifest-ERRATA.md)), plus about 90 MB of
  new Project Gutenberg fiction. The new books are chosen by the fixed rule
  in `research/scripts/build_e3_full_corpus.py`: English, LoCC PR/PS, Fiction;
  excluding every book used anywhere before (by ID and by normalized
  catalog title), collections, and texts outside 100 KB–2 MB; in a seeded
  shuffle (seed 2026) until about 90 MB. Catalog hash, eligible count, and
  every skip are recorded in the manifest.
- **Validation:** Jekyll. **Test:** Time Machine (validation-grade for this
  model, reported only).
- **Reproduction:** book texts are not committed. `python3
  research/scripts/build_e3_full_corpus.py fetch` re-downloads them and verifies
  every hash. Gutenberg edits files occasionally; a hash failure is
  reported, never silently accepted.

## Hypotheses (decided on Jekyll validation)

- **H1 data:** at a 2^24-circuit budget, E3-full beats E3-lite's 0.2350.
- **H2 capacity:** on E3-full, 2^26 circuits beat 2^24.

Reported for each budget: the validation curve every 25 books,
bits/bit per acquired MB, wall time, peak RSS, and saved-state size.
`r6_e3_scaling_benchmark.py` runs both budgets one after the other.

## Result (`research/reports/r6-e3-scaling.json`)

233 books, 108.4 MB of acquisition (206 new books; 13,576 eligible; 70
skipped: 69 by size, 1 failed download). Both budgets ran one after the
other on one core.

| Budget | Validation | Test (validation-grade) | Circuits | Reclaimed | Peak RSS | Time | State |
|---|---|---|---|---|---|---|---|
| 2^24 | **0.2256** | 0.2257 | 15.7M | 595.6M | 1.6 GB | 36 min | 439 MB |
| 2^26 | **0.2189** | 0.2184 | 66.2M | 302.0M | 6.5 GB | 42 min | 1.85 GB |

- **H1 holds:** 0.2256 < 0.2350 (E3-lite, same budget).
- **H2 holds:** 0.2189 < 0.2256.

Validation curve (cumulative MB → bits/bit):

| MB | 17.9 | 28.5 | 39.4 | 50.5 | 62.1 | 72.1 | 83.7 | 93.6 | 104.0 | 108.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2^24 | 0.2355 | 0.2307 | 0.2289 | 0.2291 | 0.2287 | 0.2266 | 0.2266 | 0.2268 | 0.2256 | 0.2256 |
| 2^26 | 0.2341 | 0.2284 | 0.2256 | 0.2242 | 0.2229 | 0.2209 | 0.2204 | 0.2204 | 0.2189 | 0.2189 |

**Reading.**

- Data keeps helping at both budgets, and more capacity widens the
  gain. From 17.9 MB to 108.4 MB, 2^24 gains 0.0099 and 2^26 gains 0.0152.
  At 2^24 the curve is close to flat after about 40 MB (0.2289 → 0.2256 over
  the last 69 MB), so that budget is the binding constraint. 2^26 is
  still descending, more slowly.
- 0.2189 bits/bit is 1.75 bits/byte. On this validation book the path is
  0.3979 (E2 suffix memory) → 0.2399 (E2, 11.5 MB) → 0.2350 (E3-lite) →
  0.2189 (E3-full, 2^26).
- Cost: 2^26 needs 6.5 GB of RAM and a 1.85 GB state. That is a
  Prosumer-class brain, not a Small one. 2^24 (1.6 GB RAM, 439 MB
  state) is the realistic Small budget.
- Jekyll was the tuning book for this model and Time Machine is
  validation-grade, so these are scaling measurements, not clean
  generalization claims. The untouched-book result (0.2381 on E2) is
  still the clean claim. A Small candidate trained on E3-full should be
  scored once on a fresh untouched book.
- Retention under pressure (2^24 reclaimed 596M circuits) is the
  limiting mechanism; see
  [r6-reclamation-pressure-dev.md](r6-reclamation-pressure-dev.md).

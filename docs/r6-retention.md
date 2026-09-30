# R6 retention and order sensitivity (metrics, not gates)

Script: `r6_retention_benchmark.py` (protocol `r6-retention-v1`). Report:
`reports/r6-retention.json`. Master plan §22 step 4. These are the frozen
baseline numbers that step 5 (consolidation) must improve.

## Protocol

- The last 32 KiB of each of the 14 E2 acquisition books is held out as
  that book's retention probe. The remaining 11.05 MB trains, book by book.
- After every book, the frozen model scores every probe and the Jekyll
  validation book. This gives a 14×14 retention matrix.
- **Forgetting** for a book is its final probe score minus its best score
  after it was learned. **Mean forgetting** averages all books except the
  last.
- **Order sensitivity** is the spread of final validation across four
  orders: canonical, reversed, and two seeded shuffles (compact budget).
- **Diagnostic:** the canonical order again at the full budget, where
  reclamation is rare.

A frozen count-based n-gram has zero *order* sensitivity by construction:
its counts commute. It is **not** free of forgetting, though. Trained book
by book, it loses 0.00043 per book to dilution, as later books average into
shared statistics. (An earlier version of this page wrongly claimed zero.)
That 0.00043 is the floor for any accumulating model. Everything above it
comes from the plastic and budgeted mechanisms.

## Baseline (current default configuration, 2026-09-30)

| Run | Mean forgetting | Final validation | Most forgotten books |
|---|---|---|---|
| canonical, compact | 0.0055 | 0.2467 | Moby-Dick 0.0105, Iliad 0.0094, War and Peace 0.0089 |
| reversed, compact | 0.0068 | 0.2465 | Huckleberry Finn 0.0177, War and Peace 0.0095 |
| shuffle 1, compact | 0.0065 | 0.2518 | Iliad 0.0202, Moby-Dick 0.0191 |
| shuffle 2, compact | 0.0054 | 0.2466 | Iliad 0.0148, War and Peace 0.0095 |
| canonical, **full budget** | 0.0035 | — | Moby-Dick 0.0078, War and Peace 0.0060 |

Summary: mean forgetting 0.0060 (compact, four orders); order spread 0.0053.
Units are bits per bit; 0.006 ≈ 0.05 bits per byte.

## Reading

- **Most forgetting is not caused by the budget.** At full budget, with
  about a fifth of the reclamation, forgetting keeps two thirds of its
  size (0.0035 vs 0.0055). The larger cause is drift in the plastic
  arbitration weights toward recent books, plus count dilution in shared
  contexts. This run does not separate those two.
- **Distinctive books are forgotten most:** verse translation (Iliad),
  Melville, Tolstoy's long domain, and dialect (Huckleberry Finn). Their
  specialized weighting is overwritten by later, more typical prose.
- **The last book sets the final score.** Shuffle 1 ends on Dracula and
  finishes at 0.2518, although its validation was 0.2491 two books earlier.
  The other orders finish within 0.0004 of each other.
- Final validation here is slightly worse than the gated benchmark's
  compact score (0.2467 vs 0.2461) because the probes are withheld from
  training.

## Step 5 result

Metaplasticity (ADR 0008) reduced mean forgetting from 0.0060 to 0.0051
and order spread from 0.0053 to 0.0040, and it improved final validation
in all four orders (`reports/r6-retention-metaplasticity.json`). See
[r6-consolidation.md](r6-consolidation.md). The baseline below remains the
record of the pre-consolidation model.

## Target for step 5 (as set before the work)

A consolidation mechanism should lower mean forgetting (0.0060) and order
spread (0.0053) without worsening final canonical validation (0.2467). It
must be local and bounded, with no replay buffer. A diagnostic that
freezes arbitration weights would split drift from dilution and is a
natural first experiment.

Reproduce: `python3 r6_retention_benchmark.py --jobs 3` (about 6 minutes,
about 3 GB peak).

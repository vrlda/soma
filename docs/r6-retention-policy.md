# R6 retention under budget pressure: pre-registration

Protocol `r6-retention-policy-v1` (`r6_retention_policy_benchmark.py`).
**Status: pre-registered; not yet run.** The git history timestamps this
file, the harness, and the sealed book before the run.

## Question

The development sweeps ([r6-reclamation-pressure-dev.md](r6-reclamation-pressure-dev.md))
suggest that circuit mixing loses useful circuits when its budget binds,
and that two changes reduce the loss: smaller reclamation batches and the
pressure-adaptive growth gate. Those sweeps were scored on Jekyll, the
book the model was tuned on. This protocol asks whether the change holds
on a book no one has used, before it can become the default.

## Arms

- **Baseline:** current defaults (`reclaim_fraction` 0.125, `growth_pressure` 0).
- **Candidate:** `reclaim_fraction` 0.03, `growth_pressure` 8. Fixed from
  the sweeps; nothing is tuned after this file.

| Setting | Training | Budget | Pressure |
|---|---|---|---|
| A | E1 acquisition (3.9 MB) | 2^16 | Micro: severe |
| B | E2 acquisition (11.5 MB) | 2^20 | heavy |
| C | E2 acquisition | 2^22 | moderate |
| D | E2 acquisition | 2^24 | light (the E2 default budget) |
| E | E3-full acquisition (108 MB) | 2^24 | heavy at scale |

## Sealed test book

`reports/e3u-manifest.json`: **pg76222**, *The valley of eyes unseen*
(Gilbert Collins, 478,872 bytes after boilerplate stripping). It was
chosen by `scripts/build_e3_full_corpus.py untouched`, which continues
E3-full's seeded order past every candidate that build tried and applies
the same filters. Only its metadata has been printed. Each run scores it
once, frozen, after training. Jekyll validation is reported too, for
continuity with the dev sweeps; it decides nothing.

## Hypotheses (sealed book, bits/bit)

- **H1 helps under pressure:** candidate < baseline in A, B, C, and E.
- **H2 harmless when pressure is light:** in D, candidate ≤ baseline + 0.0005.

**Decision:** if H1 and H2 all hold, the candidate becomes the default.
The scripts behind frozen reports then pin the old values, so those
reports still reproduce. If any part fails, defaults stay and the
result is recorded either way. The run takes about 45 minutes on four
cores.

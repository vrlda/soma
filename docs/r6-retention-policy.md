# R6 retention under budget pressure: pre-registration

Protocol `r6-retention-policy-v1` (`r6_retention_policy_benchmark.py`).
**Status: run 2026-10-01; all hypotheses hold; adopted as the default.**
The git history timestamps this file, the harness, and the sealed book
before the run.

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

## Result (`reports/r6-retention-policy.json`)

Sealed book pg76222, bits/bit, each scored once:

| Setting | Baseline | Candidate | Difference | Jekyll (dev) baseline → candidate |
|---|---|---|---|---|
| A E1, 2^16 | 0.3505 | 0.3437 | **−0.0069** | 0.3294 → 0.3223 |
| B E2, 2^20 | 0.2852 | 0.2809 | **−0.0043** | 0.2607 → 0.2562 |
| C E2, 2^22 | 0.2705 | 0.2693 | **−0.0013** | 0.2458 → 0.2445 |
| D E2, 2^24 | 0.2658 | 0.2656 | −0.0002 (≤ +0.0005 required) | 0.2399 → 0.2395 |
| E E3-full, 2^24 | 0.2350 | 0.2329 | **−0.0021** | 0.2256 → 0.2240 |

H1 holds in A, B, C, and E. H2 holds in D. The gains on the sealed book
match the dev sweeps closely, which suggests the Jekyll tuning did not
overfit this mechanism. The cost is up to 11% more training time under
heavy pressure (A: 22 → 31 s; E: 2611 → 2799 s). Peak RSS is unchanged.
The baseline's Jekyll numbers equal the earlier frozen reports (0.2399,
0.2458, 0.2256), which serves as a reproduction check.

**Second finding: data scaling generalizes.** At the same 2^24 budget,
training on E3-full instead of E2 improves the sealed book by 0.031
(0.2658 → 0.2350 baseline). That is the clean, never-seen-book version of
the E3 result in [r6-e3-scaling.md](r6-e3-scaling.md).

**Adoption.** The defaults are now `reclaim_fraction` 0.03 and
`growth_pressure` 8 in both `soma/memory/mixing.py` and the Rust port.
`LEGACY_DEFAULTS` holds the old values, and the scripts behind earlier
frozen reports merge it in. States saved before the change have no
`growth_pressure` field and load with it off, so existing brains behave
as before. Parity: `engine/differential_mixer.py` now also checks that
continued learning after a load is byte-identical in both languages. That
check found and fixed a Rust bug in which `loads` dropped a saved
`growth_pressure`. No result was affected, because no run had loaded such
a state.

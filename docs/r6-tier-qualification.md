# R6 Tier Qualification Harness

`r6_tier_benchmark.py` (`r6-tier-v2`) is the locked Micro/Tiny/Small scaling protocol. Its
default input is the frozen, hash-verified public-domain E1 manifest:

```bash
python3 r6_tier_benchmark.py --manifest reports/e1-manifest.json \
  --report reports/r6-tier.json
```

Protocol-v2 eligibility is locked to manifest SHA-256
`bb3eec0d2e42e762967169153bb0f007c47e102770a0c97150502df307a04d5b` and
the canonical totals acquisition `3,852,331`, validation `141,077`, and test
`181,481` bytes. A full run with another manifest is reported as
`noncanonical-full-manifest` and cannot qualify R6; subset runs are
`development-smoke`.

The tiers are fixed, not selected from the result:

| Tier | Maximum order | Circuits | State ceiling | RSS ceiling | Time ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Micro | 8 | 512 | 4 MiB | 512 MiB | 180 s |
| Tiny | 12 | 16,384 | 16 MiB | 2 GiB | 600 s |
| Small | 16 | 131,072 | 32 MiB | 8 GiB | 1,800 s |

Order, circuit, and RAM ceilings are loaded from canonical
`configs/presets.json`; the report records its SHA-256. State/time ceilings and
quality/effect thresholds are versioned in the harness.

Every tier trains on the same ordered acquisition books and evaluates the
same untouched validation/test bytes. It reports progressive validation,
elapsed time, peak RSS, serialized state bytes, resident circuits, and budget
gates. Controls are fitted only on acquisition bytes: an acquisition-fitted
byte unigram and a conventional variable-order suffix n-gram with the same
order/context budget and deterministic LRU replacement (indexed heap; the
pre-optimization scan is retained only as a locked semantic reference test).

Validation and test scoring is observational: it runs on a state clone and
cannot advance training history or event clocks. Manifest books are explicit
document boundaries for both uninterrupted and resumed acquisition. Reports
also include the harness source hash, canonical preset hash, manifest hash, and
locked Python/platform metadata.

The effect gate requires at least three acquisition checkpoints, endpoint gain
of at least 0.002 bits/bit, a negative least-squares validation slope, at least
0.001 bits/bit advantage over each matched control, at least 0.001 gain beyond
the suffix control's own endpoint gain, and the Small frozen test floor of 0.45
bits/bit. Resource budgets must also pass. Small alone is subject to the
quality floor; Micro and Tiny are diagnostic/resource-scaling tiers. Full-tier
workers checkpoint at a fixed midpoint, resume through the manifest-bound
envelope, and require an identical final state digest. Thus a last-point
improvement or a held-out-fitted baseline cannot qualify a tier.

Qualification aggregation requires execution/resource/resume gates for all
three tiers, but applies quality and control gates only to Small. Micro and
Tiny quality/control results remain diagnostic and are reported separately;
they cannot veto or establish Small quality.

These thresholds are frozen in protocol v2 before an untouched run: the
0.002 endpoint bar is the minimum effect retained over the small-run
measurement-jitter budget, 0.001 is the required practical separation from a
capacity-matched conventional control, and the slope bar requires a
multi-checkpoint trend rather than a favorable endpoint. Variance is handled
by requiring all three independent signals (endpoint, control margin, and
ordered least-squares slope); thresholds are never tuned from a result. The
report embeds this rationale and the canonical preset digest.

For a fast development smoke run, use explicit prefixes:

```bash
python3 r6_tier_benchmark.py --acquisition-bytes 1024 \
  --validation-bytes 1024 --test-bytes 1024 \
  --report /tmp/r6-tier-smoke.json
```

Such output is labeled `development-smoke`, is ineligible for R6 acceptance,
and must not be reported as a tier result. A full untouched-manifest run is the
only report with `acceptance.eligible_for_r6: true`.

## Status note (see ADR 0006)

The locked v2 full-manifest run is recorded in `reports/r6-tier.json`: Small clears
the 0.45 test floor (0.3976), resume is digest-exact, and budgets pass except Micro
reference-engine time/RSS. The static control-margin gates fail by construction
(model == control bit-identical; budgets never bind so reclamation never fires).
They are superseded by evidence per `docs/adr/0006-r6-control-margins-evidence.md`;
continual-adaptation qualification moves to `r6b-reclamation-v2`
(`reports/r6-reclamation.json`, 4/4). R6 remains OPEN with remaining items in ADR 0006.

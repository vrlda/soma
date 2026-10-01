# Circuit loss under budget pressure: development sweeps (non-gating)

2026-10-01. Development-only, Jekyll validation bits/bit, Rust engine.
Tuning happened on this book, so none of these numbers is a claim. A
default change needs a pre-registered test on a book no one has used.

Engine flags (`engine/soma-engine/src/mixer.rs`, off by default, not saved,
not in the Python reference): `reclaim_recency` and `growth_pressure`.
`reclaim_fraction` and `growth_threshold` already exist.

| Setting | E1 at 2^16 (Micro) | E2 at 2^20 | E2 at 2^22 | E2 at 2^24 |
|---|---|---|---|---|
| default (fraction 0.125, threshold 8) | 0.3294 | 0.2607 | 0.2458 | 0.2399 |
| recency-aged ranking, scale 1e5 | 0.3274 | 0.2742 | — | — |
| recency-aged ranking, scale 1e7 | 0.3294 | 0.2598 | — | — |
| growth threshold 32 (fixed) | 0.3226 | 0.2587 | 0.2465 | 0.2432 |
| growth threshold 64 (fixed) | 0.3202 | 0.2586 | — | — |
| reclaim fraction 0.03 | 0.3254 | 0.2574 | 0.2447 | 0.2399 |
| reclaim fraction 0.5 | 0.3462 | 0.2716 | — | — |
| growth pressure 16 | 0.3234 | 0.2586 | 0.2453 | — |
| **growth pressure 8 + fraction 0.03** | **0.3223** | **0.2562** | **0.2445** | — |

`growth_pressure p`: once reclamation has begun, a new circuit's parent
needs at least p × (frontier + 1) visits. The frontier is the most visits
of any circuit in the last reclaimed batch, so new circuits must out-evidence
the circuits they displace.

What it shows:

- Retention, not recency, is what matters. Ageing frequency by recency
  hurts at moderate pressure (+0.0135 at 2^20, scale 1e5).
- Reclaiming in large batches throws away useful circuits. A smaller
  batch helps under pressure and is neutral when the budget barely binds
  (2^24). The cost is more reclamation passes, each O(circuits).
- A fixed higher growth threshold helps under pressure and hurts at large
  budgets (+0.0033 at 2^24). The pressure-adaptive gate keeps most of
  the gain only where reclamation runs.
- The combination gains 0.007 at Micro, 0.0045 at 2^20, and 0.0013 at
  2^22. Gains shrink as pressure falls. Neither change makes Micro
  accumulate across books: its curve is still flat.

Next: an incremental eviction index, so a small batch costs nothing,
then a pre-registered comparison on a fresh untouched book before any
default change.

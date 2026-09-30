# R6 consolidation: metaplasticity against recency drift

Master plan §22 step 5. Baseline and protocol:
[r6-retention.md](r6-retention.md). Decision: [ADR 0008](adr/0008-metaplasticity-consolidation.md).
Evidence: `reports/r6-retention.json` (before),
`reports/r6-retention-metaplasticity.json` (after), and
`reports/r6-forgetting-diagnostics.json` (where forgetting comes from).

**Pre-set success criterion (from step 4):** lower mean forgetting and
order spread without worsening final canonical validation, using a local,
bounded mechanism with no replay buffer, and shown causal by ablation.

## Result

| Metric (r6-retention-v1) | Before | Metaplasticity (τ = 1e5) | Change |
|---|---|---|---|
| mean forgetting, compact, 4 orders | 0.00604 | 0.00510 | −15% |
| order spread, final validation | 0.00533 | 0.00401 | −25% |
| forgetting, canonical, full budget | 0.00355 | 0.00290 | −18% |
| final validation: canonical / reversed / shuffle 1 / shuffle 2 | 0.2467 / 0.2465 / 0.2518 / 0.2466 | 0.2465 / 0.2463 / 0.2503 / 0.2464 | better in all four |

The criterion is met, and the gated benchmark improves as well: full
0.2399 / 0.2406 validation / test (was 0.2402 / 0.2411), 7/7 gates. The
ablation is τ = 0, the previous default.

**Mechanism.** Each arbitration weight set counts its own updates, and its
learning rate is learning_rate × τ / (τ + updates). A heavily used set
settles; a rarely used set stays plastic. This is metaplasticity in the
sense of PROJECT.MD: plasticity that depends on the history of plasticity.
It is local (per weight set), bounded (one counter per set), and uses no
replay.

## Where forgetting comes from (`r6_forgetting_diagnostics.py`)

Each book's loss since it was learned, split by re-scoring its probe with
the final circuits but the weights saved right after that book (canonical
order):

| Setting | Total | Weight drift | Circuit changes |
|---|---|---|---|
| compact, no metaplasticity | 0.00543 | 0.00175 | 0.00368 |
| **compact, metaplasticity** | **0.00460** | **0.00091** | 0.00369 |
| full, no metaplasticity | 0.00295 | 0.00218 | 0.00077 |
| **full, metaplasticity** | **0.00219** | **0.00144** | 0.00075 |
| reference: order-5 n-gram dilution | 0.00043 | — | — |

- Metaplasticity roughly halves weight drift and leaves circuit changes
  untouched: it acts exactly on the component it targets.
- At the compact budget, most of what remains is circuit loss from
  reclamation (0.0037 vs 0.0008 at full budget). That is a capacity limit:
  4.2M circuits cannot hold fourteen books' specifics.
- Everything is well above the n-gram's dilution floor (0.00043), so more
  improvement is possible.

## Mechanisms tried (canonical order, compact budget unless noted)

Kept only if they met the criterion. All are recorded, including the
failures.

| Candidate | Mean forgetting | Final validation | Outcome |
|---|---|---|---|
| baseline | 0.0055 | 0.2467 | — |
| **metaplasticity τ = 1e5** | **0.0048** | **0.2465** | **adopted** |
| metaplasticity τ = 3e4 | 0.0040 | 0.2470 | more retention, but worse validation |
| metaplasticity τ = 1e4 / 2e3 | 0.0036 / 0.0035 | 0.2483 / 0.2502 | rejected: validation cost |
| freeze weights after 5 books (diagnostic) | 0.0043 | 0.2523 | weights matter, but freezing kills adaptation |
| rescale counts at cap (halve both) | 0.0180 | 0.2456 | rejected: recency weighting in disguise |
| count cap 65535 | 0.0099 | 0.2458 | rejected: the 1023 cap is itself a stabilizer |
| gate arbitration by circuit maturity | 0.0056 | 0.2471 | rejected: no effect |
| reclaim lowest-value circuits (advantage over parent) | 0.0158 | 0.2569 | rejected: evicts young circuits before they prove themselves |
| growth threshold 64 | 0.0044 | 0.2476 | rejected: validation cost |
| orders 0–6 / 0–4 only | 0.0065 / 0.0124 | 0.2492 / 0.2770 | long contexts *help* retention |

Two findings changed the model of the problem along the way. The count cap
acts as consolidation: saturated frequent contexts stop moving. And an
accumulating n-gram's dilution is tiny, so almost all of SOMA's forgetting
was excess, caused by its own plastic and budgeted mechanisms.

## Open

- **Circuit loss under budget pressure.** A reclamation policy that
  protects distinctive old circuits without starving new ones remains
  open. The value-based attempt failed; a grace period plus a value term
  is the obvious next variant.
- **Remaining weight drift:** 0.0009–0.0014. Stronger consolidation trades
  away validation at the rates tried.

Reproduce:

```bash
python3 r6_retention_benchmark.py --out reports/r6-retention-metaplasticity.json
python3 r6_forgetting_diagnostics.py
```

# R6 circuit mixing: growing byte-context circuits with plastic arbitration

Script: `r6_circuit_mixing_benchmark.py` (protocol `r6-circuit-mixing-v1`).
Report: `reports/r6-circuit-mixing.json`. Model: `soma/memory/mixing.py`
(`CircuitMixingMemory`), run through its Rust port
`engine/soma-engine/src/mixer.rs` (`soma-mixer`). Parity:
`engine/differential_mixer.py` reports a probability gap of exactly 0.0
across four configurations, including heavy reclamation and hashed long
orders.

## Why

The E2 gate model (`SequenceCircuitMemory`, order 16 bits) predicts from
the single longest suffix seen at least twice. It reached 0.3995 bits/bit,
between a frozen order-1 and order-2 byte n-gram and behind gzip
([reference baselines](r3b-reference-baselines.md)). Its circuits grow and
are reclaimed, but nothing learns *which* circuits to trust.

## Mechanisms

| Mechanism | What it does |
|---|---|
| Byte-aligned circuits | context = 0..12 preceding whole bytes + bits seen of the current byte (orders 0–8, 10, 12; ≤6 bytes exact, longer hashed to 48 bits) |
| Evidence-gated growth | a longer-context circuit is created only after its parent context has 8 visits |
| Hard budget + reclamation | fewest-visits-then-oldest circuits are removed in 12.5% batches |
| Plastic arbitration | every active circuit contributes log-odds evidence; delta-rule weights, one set per (circuits present × partial byte) |
| Count-state calibration | available, **off by default** (hurts: +0.0033) |
| Correction stage | available, **off by default** since 2026-09-30 (redundant with partial-byte gating; [ADR 0007](adr/0007-circuit-mixing-mechanism-audit.md)) |

Frozen scoring (`learn=False`) grows nothing and changes no plastic state
(unit-tested).

## Results (E2 manifest: 11.5 MB acquisition, Jekyll validation, Time Machine test)

| Model | Validation | Test | Circuits | Peak RSS |
|---|---|---|---|---|
| byte unigram bar | — | 0.5636 | — | — |
| SOMA E2 (`SequenceCircuitMemory`) | 0.3979 | 0.3995 | 74,784 | 1.87 GB |
| frozen Witten-Bell byte n-gram, order 5 | 0.2583 | 0.2575 | — | 0.7 GB |
| **circuit mixing, compact (4.2M budget)** | **0.2461** | **0.2474** | 3.8M | 0.40 GB |
| **circuit mixing, full (16.8M budget)** | **0.2402** | **0.2411** | 15.9M | 1.61 GB |

These are the current default (correction stage off, ADR 0007). The
configuration used for the untouched test had the correction stage on and
scored 0.2400 / 0.2406 here. 0.2411 bits/bit is 1.93 bits/byte, down from
3.20. It's 40% lower than the previous E2 score and 6.4% below the order-5
n-gram. The full run takes about 3 minutes on one core. All seven gates pass: improves with data, beats
the unigram bar, beats prior E2, beats order-2 and order-5 n-grams, bounded,
and arbitration is causal.

## Ablations (compact budget, validation bits/bit, default 0.2461)

Every variant is worse than the default ([ADR 0007](adr/0007-circuit-mixing-mechanism-audit.md)).

| Variant | Validation | Cost of the change |
|---|---|---|
| longest-match backoff instead of plastic arbitration | 0.2663 | +0.0202 (**gate**) |
| count-state calibration on | 0.2494 | +0.0033 |
| growth gate off (grow every context) | 0.2493 | +0.0032 |
| partial-byte gating off | 0.2489 | +0.0029 |
| correction stage on | 0.2461 | +0.00004 |

Mechanism matters more than budget. With the first predictor (fast
forgetting, no gating), unbounded growth reached 17.8M circuits and scored
0.2624 on test. At a fixed 4.2M budget, that predictor scored 0.2666 and
the final one scores 0.2473.

## Honest boundaries

- **Untouched test (resolved):** on a book never used anywhere,
  scored once under a pre-registered protocol, full circuit mixing scores
  0.2381 against 0.2588 for the order-5 n-gram and 0.4009 for the prior
  E2 memory; all three hypotheses hold
  ([r6-untouched-test.md](r6-untouched-test.md)). The note below explains
  why that test was needed.
- **Tuning was not blind.** About 20 configurations were compared, and
  test scores were printed next to validation during tuning. Choices were
  made on validation, and the two never disagreed on direction. Still,
  treat the Time Machine test as validation-grade for this model; a fresh
  sealed book is needed for a clean claim.
- **Recency interference.** Validation moves 0.2518 → 0.2592 after War and
  Peace (3.3 MB) and recovers after one more book. With the correction
  stage on, it was 0.2536 → 0.2647 and took about four books, because the
  correction table tracked the most recent domain. The plastic arbitration
  weights still drift toward recent books, and the final score still
  depends on book order. Consolidation is the natural SOMA next step. It
  is not solved here.
- This is still a statistical predictor. Better compression is not
  comprehension, and generation and chat have not been switched to it yet.
  The service still uses `SequenceCircuitMemory`.
- `Organism` cell/synapse plasticity is still not on this scoring path.
  The structural plasticity measured here is circuit growth, gating, and
  reclamation inside the memory.

## Next steps

These are tracked in master plan §22: a retention matrix and
consolidation (steps 4–5), and forgetting, quarantine, persistence, and
engine serving (steps 7–8). Steps 2 (untouched test) and 3 (mechanism
audit) are done.

## Reproduce

```bash
python3 r6_circuit_mixing_benchmark.py --jobs 3   # ~11 min, ~2.5 GB peak
python3 engine/differential_mixer.py 0            # parity, ~35 s
python3 -m unittest tests.test_r6_circuit_mixing
```

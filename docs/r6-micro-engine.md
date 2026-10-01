# R6 Micro tier on the engine: pre-registration

Master plan §22 step 10, [ADR 0006](adr/0006-r6-control-margins-evidence.md)
remaining item 3. **Status: pre-registered; not yet run.** The git history
timestamps this file before the run.

## Why

The locked `r6-tier-v2` Micro tier failed its resource ceilings on the
Python reference engine (278 s > 180 s, 966 MB > 512 MB; the v2 worker
timed out). The product language memory is now circuit mixing served by
the Rust engine (step 8), so the open question is whether the production
memory fits the smallest tier.

## Protocol `r6-micro-engine-v1` (`r6_micro_engine_benchmark.py`)

- **Data:** the canonical E1 manifest (`r6-tier-v2` lock: manifest digest
  `bb3eec0d…a04d5b`, acquisition 3,852,331 / validation 141,077 / test
  181,481 bytes). A different manifest is reported ineligible. Five
  acquisition books in manifest order, each one document. Validation is
  Jekyll, test Time Machine; scoring is frozen.
- **Model:** `CircuitMixingMemory` defaults on `soma-mixer`, with
  `max_circuits = 2^16`. The budget is fixed from the state format, not
  from a result. A `SOMAMIX1` circuit record is 32 bytes, so 2^16
  circuits take 2 MiB, which is under the 4 MiB ceiling with room for the
  header and weights. 2^17 would be exactly 4 MiB of records alone and
  could not fit.
- **Units:** the preset's "512 circuits" counts `SequenceCircuitMemory`
  suffix circuits, a different unit. Micro is bound here by its frozen
  resource ceilings. A 512-circuit run is reported as a diagnostic only.
- **Control:** the same engine, budget, and data with plastic arbitration
  off (longest-match backoff). This is the capacity-matched conventional
  predictor for this memory.

## Gates (all must pass)

| Gate | Threshold (frozen `r6-tier-v2` values) |
|---|---|
| canonical manifest | digest and totals match the v2 lock |
| state | final `SOMAMIX1` file ≤ 4 MiB, and the analytic peak bound (header + weights + 2^16 × 32 B) ≤ 4 MiB |
| RSS | engine peak RSS (VmHWM) ≤ 512 MiB |
| time | wall clock of the full training-plus-evaluation run ≤ 180 s |
| resume | a run that saves after book 2 and resumes from the file ends in a byte-identical state |
| checkpoints | ≥ 3 validation checkpoints (one per book) |
| endpoint gain | first-checkpoint − final validation ≥ 0.002 bits/bit |
| slope | least-squares validation slope over checkpoints ≤ −0.0001 |
| control margin | control validation − model validation ≥ 0.001 bits/bit |

Reported, not gated: test bits/bit next to the locked Small result
(0.3976) and the byte unigram (0.5620); harness Python RSS; the 512-circuit
diagnostic.

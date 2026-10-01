# R6 Micro tier on the engine: pre-registration

Master plan §22 step 10, [ADR 0006](../../docs/adr/0006-r6-control-margins-evidence.md)
remaining item 3. **Status: v1 fails (6/9); v2 fails (8/9, control margin).** Resources pass by a wide
margin. The memory saturates its budget on the first book, so it stops
improving with data. The git history timestamps the protocol before the run.

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

**Amendment (2026-10-01, before any gated number was read):** the
512-circuit diagnostic cannot run. The memory refuses budgets below
256 × orders (2,816 for the 11 default orders), and the first attempt
stopped there. The report records the diagnostic as infeasible. No
replacement budget was chosen, and the gates are unchanged.

## Result (`research/reports/r6-micro-engine.json`)

| Gate | Result | Value |
|---|---|---|
| canonical manifest | pass | |
| state ≤ 4 MiB | pass | 1.98 MB final, 2.10 MB peak bound |
| RSS ≤ 512 MiB | pass | 35 MB (the Python reference used 966 MB) |
| time ≤ 180 s | pass | 19.5 s (the Python reference took 278 s) |
| resume byte-identical | **fail** | see below |
| ≥ 3 checkpoints | pass | 5 |
| endpoint gain ≥ 0.002 | **fail** | −0.0015 |
| slope ≤ −0.0001 | **fail** | +0.00019 |
| control margin ≥ 0.001 | pass | 0.0048 (control 0.3343) |

Validation by book: 0.3279, 0.3266, 0.3311, 0.3255, 0.3294. Test is 0.3297,
against 0.3976 for the locked Small tier on the same data and 0.5620 for
the byte unigram.

**What failed and why.**

- **Effect (real).** 2^16 circuits fill on the first book, and
  27.6M circuits were reclaimed over 3.85 MB. After that, each book mostly
  replaces the last, and validation moves by recency (±0.003), not by
  accumulation. At this budget the memory is a strong small predictor
  (0.0048 better than its control and 0.068 better than Small) but not an
  accumulating one. Retaining more needs more circuits (the E3 runs) or
  a better reclamation policy. This is the open "circuit loss under
  budget pressure" item.
- **Resume (harness defect, scored as fail).** The reference run evaluated
  validation after every book; the resumed run did not. Frozen scoring
  advances `events_seen`, which is the clock that stamps `last_used`, so
  the two state files differ in clock values only. Reclamation only
  compares stamps, and frozen scoring shifts them uniformly, so
  behavior is unchanged. Checked after the run: a resumed state is
  byte-identical to an uninterrupted run without interleaved
  evaluation, and it scores exactly the same (validation 0.329428, test
  0.329696). The v1 gate still counts as failed. A later protocol must
  compare like with like.

**Decision.** The resource half of ADR 0006 item 3 is closed: the
production engine runs Micro at 7% of its RSS ceiling and 11% of its time
ceiling. The quality half is not. Micro does not keep learning from more
data at a 4 MiB state. Micro stays a diagnostic tier, as `r6-tier-v2`
already treats it. There is no re-run at another budget, because that
would be choosing the budget from the result.

**Erratum (2026-10-01, after the run).** A `SOMAMIX1` circuit record is
28 bytes (key 8, counts 4 + 4, visits 4, last use 8), not 32. The harness
used 32, which overstates the peak-state bound, so the state gate's verdict
stands. The reason given for choosing 2^16 was wrong, though: 2^17 records
take 3.67 MB, and that budget would also have fit under 4 MiB. Any later
Micro protocol should take the largest power of two whose bound fits,
computed with 28 bytes. That is 2^17.

## v2 pre-registration (`r6_micro_engine_v2_benchmark.py`)

These changes from v1 were fixed before any v2 run:

- **Budget 2^17.** This is the largest power of two whose state bound fits
  4 MiB with the correct 28-byte record (erratum above).
- **Current defaults.** This includes the retention policy adopted from
  [r6-retention-policy.md](r6-retention-policy.md).
- **Like-with-like resume.** The timed run and both resume halves train
  without interleaved evaluation. The checkpoint curve comes from a
  separate run that evaluates after every book. The timed run scores
  validation and test once at the end.
- **Same gates and thresholds as v1.** The peak-state bound is the final
  state plus 28 bytes for every circuit slot still free.

The decision rule is unchanged. If every gate passes, ADR 0006 item 3
closes. If the effect gates fail again, Micro stays a diagnostic tier and
the failure is recorded.

## v2 result (`research/reports/r6-micro-engine-v2.json`)

| Gate | Result | Value |
|---|---|---|
| canonical manifest | pass | |
| state ≤ 4 MiB | pass | 3.97 MB final, 3.99 MB peak bound (ceiling 4.19 MB) |
| RSS ≤ 512 MiB | pass | 29 MB |
| time ≤ 180 s | pass | 29.8 s |
| resume byte-identical | **pass** | like with like |
| ≥ 3 checkpoints | pass | 5 |
| endpoint gain ≥ 0.002 | **pass** | 0.0038 |
| slope ≤ −0.0001 | **pass** | −0.0012 |
| control margin ≥ 0.001 | **fail** | −0.0003 (control 0.3007, model 0.3010) |

Validation by book: 0.3047, 0.3021, 0.3056, 0.2972, 0.3010. Test is 0.3029
(v1: 0.3297; locked Small tier: 0.3976).

**Why the control caught up** (development runs after the result,
Jekyll validation, arbitration / no-arbitration control):

| Setting | Old defaults | New defaults |
|---|---|---|
| E1, 2^16 | 0.3294 / 0.3343 | 0.3223 / 0.3227 |
| E1, 2^17 | 0.3070 / 0.3121 | 0.3010 / 0.3007 |
| E2, 2^20 | 0.2607 / 0.2806 | 0.2562 / 0.2614 |

The pressure-adaptive growth gate keeps only well-evidenced long contexts,
so longest-match backoff becomes reliable. At Micro scale, that was most
of what plastic arbitration contributed. The control gains 0.0114 from
the policy and arbitration gains 0.006. With more data and budget,
arbitration still matters (0.0052 at E2 2^20). The "control" is no longer
a conventional predictor either, because it shares the evidence-gated
growth and the new reclamation. The gate is scored as written anyway.

**Decision.** v2 fails, and Micro stays a diagnostic tier. The resource
ceilings, the resume, and the effect gates all pass on the engine. What
fails is that, at 4 MiB on 3.9 MB of text, plastic arbitration adds
nothing measurable over evidence-gated structure. That is a finding about
the mechanism, not a defect to tune away: no v3 will be cut to pass this
gate.

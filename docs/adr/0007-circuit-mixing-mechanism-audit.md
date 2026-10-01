# ADR 0007: Circuit-mixing mechanism audit; correction stage off by default

- Status: accepted 2026-09-30. Master plan §22 step 3.
- Context: the step requires every shipped mechanism of
  `CircuitMixingMemory` to have a positive ablation or a recorded reason to
  stay. The only evidence used is validation (Jekyll). The untouched test
  book is spent, and the Time Machine test is validation-grade for this
  model.
- Evidence (E2 acquisition, compact budget unless noted, validation
  bits/bit, `research/reports/r6-circuit-mixing.json`):
  - Correction stage: on vs off 0.24613 vs 0.24609 at compact, and
    0.2400 vs 0.2402 at full budget. Net effect ≤ 0.0002.
  - It is redundant with partial-byte gating. With gating off, removing
    correction costs 0.0019 (0.2470 → 0.2489). With gating on, it costs
    about 0. Both condition on the partial byte.
  - Removing it also reduced recency interference: the War-and-Peace
    validation bump fell from +0.0111 to +0.0074, and recovery shortened
    from about four books to one.
- Decision: the correction stage is off by default in both the Python
  reference and the Rust port. Its code stays as an option, covered by the
  parity harness and reported as a benchmark variant. Count-state
  calibration stays off for the same reason (+0.0033 when on).
- Kept, with the ablation cost of removing each:
  - plastic arbitration: 0.0202 (a gate)
  - evidence-gated growth: 0.0032
  - partial-byte gating: 0.0029
  - the hard budget with reclamation: required for boundedness (7/7 gates)
- Cost: full-budget test moves 0.2406 → 0.2411 (Time Machine). The
  untouched-test result (0.2381) was scored with the frozen pre-registered
  configuration, correction on, and remains the record for that
  configuration.

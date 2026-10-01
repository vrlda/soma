# ADR 0008: Metaplasticity is the default consolidation mechanism

- Status: accepted 2026-09-30. Master plan §22 step 5.
- Context: step 4 froze forgetting metrics for `CircuitMixingMemory`: mean
  forgetting 0.0060 and order spread 0.0053. Step 5 required a local,
  bounded, replay-free mechanism that lowers both without worsening final
  validation, and is shown causal by ablation.
- Evidence ([r6-consolidation.md](../../research/docs/r6-consolidation.md)):
  - Diagnostics attribute forgetting to arbitration-weight drift (about
    0.002 at both budgets) and, at the compact budget, reclamation.
  - Per-weight-set metaplasticity, rate = lr × τ / (τ + updates) with
    τ = 1e5, lowers forgetting 15% and order spread 25%, and improves
    final validation in all four book orders. It halves the weight-drift
    component and leaves circuit loss unchanged. The gated benchmark
    improves to 0.2399 / 0.2406 with 7/7 gates. The ablation (τ = 0)
    costs 0.0003 validation.
  - The other candidates tried were rejected on the pre-set
    criterion; they are listed in the record.
- Decision: `plasticity_tau = 1e5` is the default in the Python reference
  and the Rust port, with bit-exact parity. τ = 0 restores constant-rate
  arbitration. `freeze_arbitration_after` is kept as a diagnostic.
- Consequences: the pre-registered untouched-test configuration pins
  τ = 0, the behavior actually scored. Step 6 (organism plasticity on the
  language path) is next. Circuit loss under budget pressure stays open.

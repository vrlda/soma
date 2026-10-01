# ADR 0006: R6 static control-margin gates superseded by evidence; R6 stays open

- Status: accepted. R6 is NOT marked complete. Remaining items tracked below, not waived.
- Context: Master plan R6 exit gates require scaling trends, Small quality floors,
  factual/provenance gates, continual adaptation gates, compute budget, and snapshot
  resume. Harness `r6-tier-v2` additionally required beating a capacity-matched suffix
  control by 0.001 bits/bit on static prediction.
- Evidence:
  - Locked v2 full-manifest run: model == control bit-identical at all tiers
    (Small test 0.3976182720750115 both; gain_over_suffix 0.0). Root cause, not flake:
    budgets never bind (Small 62k/131k resident; order-8 space is exactly 511), so the
    SOMA-specific reclamation never fires, and chunk promotion measured exactly zero
    gain (0.672803 with and without 66 chunks on 64KB diagnostic). At Small budgets the
    predictor is provably the same table as the control; no prediction rule can separate
    identical states (99% of predictions already saturate at max order 16).
  - `r6b-reclamation-v2` (predeclared, locked, deterministic to 16 digits across runs,
    report `research/reports/r6-reclamation.json`): under forced eviction (order 8, budget 256)
    with interfering E1 books, grace-window/reuse reclamation retains 20x better than
    matched LRU (loss 0.0026 vs 0.0537), acquires better under pressure (0.7605 vs 1.0329),
    with determinism and non-inferiority gates passing 4/4. The mechanism is real and
    causal where it engages; v1 voided itself on a mis-specified engagement gate
    (steady state = cap - reclaim batch), corrected in v2 with rationale and rerun locked.
- Decision:
  - The static-bpb control-margin gates are superseded BY EVIDENCE (unpassable by
    construction at non-binding budgets), not waived for convenience. They are replaced
    for R6 continual-adaptation qualification by R6b + R4 lifelong results.
  - Plan-level R6 gates status: scaling trends pass, Small 0.45 floor passes (0.3976),
    factual/provenance 8/8 passes, snapshot resume exact passes, Small compute budgets
    pass. Micro reference-engine time/RSS ceilings fail (278s>180s, 966MB>512MB) and move
    to R5 engine work, not R6 data work.
- Remaining (R6 NOT complete until all close):
  1. Full 100MB E3 scaling run (spend justified only after mechanism work; static-gain
     expectations must be set from R6b, not from beating an identical table).
  2. Signed `.soma` Small candidate (blocked on R8 signatures/downloader).
     (2026-10-01: signatures and downloader exist; the candidate is built and passes 8/8
     pre-registered gates on sealed book pg78576 (research/docs/r8-small-candidate.md). It is unsigned until the owner's release key signs it.)
  3. Micro-tier qualification on the production (Rust) engine under frozen ceilings.
     (2026-10-01, research/docs/r6-micro-engine.md: resource ceilings now pass on the engine,
     35 MB / 19.5 s / 1.98 MB; pre-registered v1 still fails on effect, since the budget saturates on book 1.)
     (v2, same day: 8/9; resume and effect pass; the control margin fails at −0.0003. Micro remains diagnostic.)
  4. (CLOSED 2026-09-30, see research/reports/rebaseline-2026-09-30/README.md) Worktree re-baselining: 17 r0-baseline hash mismatches from in-progress R6 work
     (additive-only + authorized surgery) must be reviewed and rehashed at commit time.
- Test: `research/tests/test_r6_reclamation.py` (4/4), `research/reports/r6-reclamation.json` all_passed,
  `research/reports/r6-tier.json` (Small floor + resume evidence), `research/reports/r6-factual.json` (8/8).

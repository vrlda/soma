# R1 Calibration Freeze

Locked metrics (Section 6.5): NLL, Brier, ECE, reliability diagrams, posterior coverage,
false-switch rate, missed-switch rate, oscillation count, novelty false positives,
time-to-detect, switching regret, growth regret.

## Current frozen tolerances (development seeds)

- Posterior sums to 1.0 within 1e-9; `__novelty__` mass fixed 0.05.
- LLR clip [-5, 5]; Student-t df 5; min scale 1e-3.
- Switch requires challenger n >= 16, incumbent n >= 16, margin >= 0.15 (enter).
- Return uses exit margin 0.05 (hysteresis: enter > exit).
- One outcome never switches (min-evidence floor).
- Shadow mode only: evidence leader logged per outcome, legacy threshold controls behavior.

## Rollout order (mandatory)

1. Shadow logging (done: `evidence_shadow` events, `evidence_report`, comparative
   cross-circuit scoring on every outcome).
2. Evidence-only recall, growth disabled (done: `enable_evidence_decisions`,
   `evidence_recall` first-resort branch, 6 recall tests, 2000-step smoke deterministic).
3. Bounded probing (next: novelty mass + information-gain probe budget).
4. Growth after recall + novelty pass independently (next).
5. Evidence default, threshold retained as named control (last).

## Acceptance (Section 6.6)

Calibrated posteriors, single-noise immunity, sustained-mismatch detection without fixed
warm-up, faster familiar recall, novelty separation, probe budgets respected, exact
restore, v12-v15 gates preserved without manifest changes.

## Disagreement analysis (seeds 0-5, lifetime harness replica)

Record: `research/reports/r1-disagreement-seeds0-5.txt`. Method: same organism, tapes, and
schedule; decisions off vs on. Script: `/tmp r1_disagree.py` (analysis only, frozen
benchmarks untouched).

- Seeds 0,1,3,4: identical rewards and module counts. Shadow tracked 466-495 outcomes.
- Seeds 2,5: legacy recruited a 2nd module and reactivated once; evidence mode
  deferred growth (`evidence_growth_deferred` x1) and held 1 module. Mean reward
  slightly lower on this short smoke in those seeds.
- Reading: novelty gate is strictly more conservative than legacy recruitment, as
  designed (growth needs stronger evidence than recall). Whether the deferred
  growths were false alarms or misses needs the full return-retention gates, not
  this smoke.

## Promotion status (step 17 done)

Evidence routing is the default (`evidence_router_mode=shadow`,
`evidence_decisions_enabled=True`). Legacy threshold behavior is retained as the
named control `use_threshold_router()`.

Evidence-on acceptance block (all unpatched, default-on):
- lifetime full: 24/24 acquisition, 21/24 B adaptation, 23/24 relearning, 18/24
  retention — identical to the frozen baseline audit.
- multicontext, nonlinear, adaptive, variable_order, compositional, streaming,
  general_structural, continuous, final: every primary mode passes; matched
  controls fail as designed.
- 267/267 unit tests green, including 44 R1 tests (schema, hysteresis,
  calibration metrics, shadow determinism, recall, novelty, probes, growth
  confirm/revert/expire, replay-exactness).

Design corrections made during calibration (mechanism, not gate tuning):
- Marginal outcome models cannot see context switches; scoring uses conditional
  detector predictors per Section 6.2.
- Lifetime-mean novelty dilutes; gate uses recency-weighted best fit.
- Pre-hoc growth deferral starved adaptation; growth is legacy-initiated with
  evidence confirmation (sustained recent edge) and conditional revert.
  Deadline yanks on ambiguous evidence caused measured retention collapse;
  ambiguity expires without touching control.

Probe compliance: max 16 / min 2 steps, information threshold, frozen models
during probes, early termination on posterior bounds, checkpoint-exact
(including mid-probe and pending-growth states).

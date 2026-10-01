# R3A Generic Sequence Acquisition

## Fixtures (`soma/evaluation/sequence.py`)

History buffers are disclosed transducer scaffolding (depth covers the gap plus
one; transients dropped from scoring by tail windows). The core selects lags
and conjunctions; buffers solve nothing alone (persistence control ≈ 0.2).

- lagcopy: random bits, target 5 steps back, depth 6. Fixed order-3 n-grams are
  structurally at chance: the nontrivial gate.
- periodic: 0011 cycle, next-symbol target, depth 3. Ambiguous from any single
  lag; needs the (lag0, lag1) conjunction. Beats memoryless; order-3 n-gram
  reported as upper comparator (1.0).
- bigram (diagnostic, RED, ungated): stationary stochastic stream fragments
  routing (4/4 modules on seed 0) because noise trips sustained-mismatch
  detection. Retuning frozen R1 thresholds is out of scope.

Locked: 8 seeds, 2400 steps, margin 0.15, actors lr 0.10, exploration 0.20.

## Results (8-seed means, `research/reports/r3a-gates.json`)

- lagcopy scalar: learn 0.538 vs frozen 0.200, shuffled 0.160, persistence
  0.193, ngram3 0.225. All gates pass.
- lagcopy bits: learn 0.587 vs frozen 0.199, shuffled 0.057, persistence
  0.193, ngram3 0.225. All gates pass.
- periodic scalar: learn 0.666 vs frozen 0.201, shuffled 0.119, persistence
  0.200. All gates pass.
- periodic bits: learn 0.875 vs frozen 0.200, shuffled 0.173, persistence
  0.200. All gates pass.

Causality: frozen/shuffled fail as predicted; motor-policy lesion collapses
prediction (tested); single-module lesions are routed around by recall, so the
lesion covers all motor modules. no_structure == learn: stationary fixtures
are solved by weights; explicit graph composition does not engage
(variable-order installs measured 0 on stationary streams).

## Carried (not claimed)

1. Composition-transfer through events: needs the multi-context regime that
   engages installation (v10/v11 numeric evidence stands as regression).
2. Stochastic grammar acquisition: needs router noise tolerance (R1 thresholds).
3. Multi-output discrete control: 3-output actor unproven; R3A uses scalar
   prediction heads over discrete alphabets.

# R3B English Developmental Acquisition — Staged Progress (E0 Gate Open)

## Breakthrough since the E0 stop

The stop was revisited mechanism by mechanism. Results on Alice E0:

- Head dead-gradient fix (affine motor-to-probability mapping): NLL 8.5/4.3
  down to ~1.0. Found the hidden acquisition: byte-boundary gating at 94%
  train / 89% held-out accuracy.
- Probability floor (0.02, documented): honest NLL without exact 0/1 bombs.
- Phase timing infrastructure (`text-utf8-phase-v1`, free-running mod-8
  clock declared by the adapter): intra-byte accuracy 0.50 -> 0.59 train /
  0.588 held-out; overall accuracy 0.556 -> 0.638 train / 0.627 held-out.
- Full E0 (127 KB train -> 12 KB held-out): train bpb 0.979 acc 0.632
  (msb 0.92, intra 0.59); held-out bpb 0.994 acc 0.627 (msb 0.89,
  intra 0.59). Transferred sequential structure, not memorization.

Eliminated (measured, no gain): explicit conjunction features, low
exploration, NLL rewards, full variable-order profile (worse, 1 install),
representation perturbations (exactly zero effect), hidden-afferent
variable-order profile (installs 3, worse than direct actor).

## E0 gate status: OPEN (bar 0.59 bits/bit, best held-out 0.99)

Remaining gap is byte-identity memory: integrating ~8 bits into byte
representations.

## Authorized VO surgery (branch r3b-vo-input-aff): complete, no gain

Variable-order pair features can now be owned by input-afferent motor
modules (extra-sources map, route recruits incumbent kind, gated validation,
persisted + replay-tested, 6 new tests, full 310-test suite green).
On E0-phase full acquisition (127 KB, 1M bits): 4 installs, 541 routes,
train acc 0.638 / intra 0.597, held-out acc 0.633 / intra 0.595 — identical
to no-VO within noise. Installs do not convert to accuracy on stationary
bit streams. Context-free + phase is worse (0.605); hidden-aff variants are
worse on accuracy. ~27 configurations measured; the wall is solid.

## Carried (honest)

Router noise tolerance, stochastic grammar, byte-compositional memory,
multi-output discrete control. R3B stays unmarked until the E0 gate passes.

## Shipped

- `text-utf8-v1` transducer: reversible UTF-8 bytes, MSB-first bit events plus
  generic byte framing, invalid-sequence rejection, round-trip tested.
- Recorded environment (`soma/evaluation/english.py`): byte-preserving streams,
  chapter-lineage partitioning (80/10/10), sealed test hash, license manifest.
- E0 corpus: `data/e0/alice.txt` (151 KB after boilerplate strip, public
  domain Lewis Carroll via Project Gutenberg; 13 chapters; 127/12/12 KB split).
- Baselines: byte unigram (4.74 bits/byte on validation), bit marginal (0.99),
  bit bigram (0.99), frozen core, shuffled credit.
- Lineage: `clone_brain` / `verify_clone` with identity records (tested).

## E0 result (STOP per 8.3, do not scale)

20 KB train -> 12 KB held-out validation, seed 0:

- learn: 4.45 train / 4.54 held-out bits/bit.
- frozen: 8.67 / 8.77 (saturated miscalibration).
- byte-frequency bar: 0.59 bits/bit equivalent. Gate requires beating it.

The core learns the bit marginal (accuracy 0.56) and nothing sequential.
Absolute NLL is head-limited (saturated motor outputs), but even accuracy
shows no structure: the MSB-after-byte-end regularity, a direct mapping of
the generic framing input, is not acquired with or without context modules
(4.34 vs 4.13 bits/bit).

## Diagnosis (all measured, E0 5KB slice unless noted)

0. The regularity exists and is easy: P(next=1|byte_end=1) is 0.02-0.07 vs
   0.49 otherwise. An oracle 4-cell conditional scores 0.906 bits/bit.
   The core scores 4.13 with flat quartile trend (4.11/4.49/4.33/4.43):
   confident marginal output, nothing sequential. More data will not help
   this configuration.
1. Explicit diagnostic conjunction input (bit*end, test-only adapter): no
   gain (4.07 vs 4.13). Gap is credit, not just representation.
2. Low exploration (0.02): worse (4.65/5.06). Proper-scoring NLL rewards:
   worse (4.56). Full variable-order profile: worse (5.00, 1 install).
   Context-free: marginally better (4.13) but still unacquired.
3. Routing over-segments noisy bit streams (94 warnings, 3 recruits, 2
   evidence reverts in 40k bits); reverts demonstrably yank useful recruits.
4. Saturated outputs destroy NLL calibration under every reward tried.

## Carried

E1+ ladder blocked on E0 by the runbook's own rule (data never substitutes
for a failed lower-stage mechanism). Next: nonlinear gating mechanism,
calibrated probabilistic heads, router noise tolerance. Composition,
generation, and dialogue work waits behind E0.

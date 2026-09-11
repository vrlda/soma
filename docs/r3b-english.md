# R3B English Developmental Acquisition — E0 Sequence-Memory Breakthrough

## 2026-09-11 result: the E0 byte-identity wall is crossed

The earlier conclusion that eight-bit composition was an unresolved firewall
was too broad. It was a real limitation of the scalar motor/product path, but
not of bounded local online memory in general.

Added `SequenceCircuitMemory`, a domain-neutral structural prediction memory.
Each acquired circuit represents an observed ordered suffix and retains only
local successor counts, reuse, and recency. Prediction backs off to the longest
supported circuit. Circuits grow online, obey a hard count budget, reclaim weak
old structures deterministically under pressure, and serialize exactly. It has
no byte, UTF-8, English, phase, or corpus knowledge.

On the complete existing Alice E0 acquisition/validation split with maximum
order 16 and a 131,072-circuit ceiling:

- acquisition: 127,072 bytes / 1,016,575 scored bits;
- validation: 12,122 bytes / 96,975 scored bits;
- byte-unigram bar: 0.592718 bits/bit;
- acquired sequence memory: **0.375095 bits/bit**, accuracy **0.877185**;
- intra-byte accuracy: **0.863141**;
- causal uniform-prediction lesion: 1.0 bits/bit and therefore fails the bar;
- acquired circuits: 37,134, with no reclamation required;
- measured full acquisition plus validation time: approximately 6.8 seconds
  on the development M3 Pro in the initial direct run.

Reproduce the locked mechanism run with:

```bash
python3 r3b_sequence_memory_benchmark.py
```

The complete regression suite is **316/316 green** after the addition.

### Interpretation boundary

This solves the specific E0 byte-identity bottleneck and proves that explicit
ordered event state was the missing primitive. It does not prove semantic
language acquisition, learned chunking, scalable long-term composition, or an
advantage over a matched high-order statistical context model. The mechanism
is deliberately close to a bounded variable-order context predictor. It is a
new SOMA memory substrate and a causal scaffold for the next experiment, not a
claim that lookup-style context growth is sufficient for language.

R3B remains open until this memory is integrated through the universal brain
lifecycle, exact compound checkpointing is demonstrated, repeated suffixes can
be promoted into reusable hierarchical circuits, and the next locked English
gate distinguishes compositional transfer from context-table memorization.

## Integration into the brain lifecycle (done)

- `Organism.enable_sequence_memory` owns a `SequenceCircuitMemory`: observe,
  predict, reset-history, validate, checkpoint save/load, legacy migration.
- `run_english_brain_memory` reproduces the standalone numbers exactly and
  resumes byte-identically from saved brain checkpoints (tested).
- Fixed an order-0 history-trim bug found by the ablation below.

## Order ablation: where byte structure lives (E0 slice)

Held-out bits/bit by maximum order: 0: 0.991, 1: 0.991, 2: 0.981, 4: 0.955,
8: 0.691, 16: 0.418. Orders below 8 cannot touch the bar; byte structure
lives at order 8+. Novelty split on held-out data: order>=12 predictions
0.374, backed-off predictions 0.93-1.35. Frequent patterns transfer;
unseen patterns collapse. That is table behavior with backoff, honestly
measured, and it scopes the composition problem precisely.

## Hierarchical chunk promotion (done, lossy-optional)

`promote_chunks` merges extension families into promoted chunk circuits with
aggregated counts; prediction code untouched. On full E0:

- merge all: 37,134 -> 11,950 circuits (3x), held-out 0.375 -> 0.439.
- merge one-shot children only: 37,134 -> 33,964 circuits, held-out 0.3752
  (identical). Principled forgetting: disposable memorization compresses
  losslessly; load-bearing circuits stay.

Chunks compress memory; they do not yet improve novel-context transfer.
Compositional transfer (novel arrangements of familiar parts outperforming
backoff) remains the defined next gate, alongside brain-action fusion
(memory driving organism decisions, currently harness-driven).

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

## Closing ledger (authorized push, ~30 configurations)

Leaderboard, E0 5KB slice unless noted (acc / intra / bpb):

- input-aff + phase: 0.638 / 0.593 / 0.964. Full 127KB: 0.627 held-out.
  Best accuracy. No bias variant: identical within noise.
- hidden-aff clean (no VO): 0.629 / 0.582 / 0.910. Best NLL.
- input-aff + phase + VO (branch surgery, merged): installs happen
  (nontrivial bit x phase pairs after bias removal), full-scale metrics
  identical to no-VO. Product edges wire strongly but add nothing.
- hidden-aff + VO: installs 3, worse accuracy.
- General structural: installs 1 depth-3 feature; NLL slightly better,
  accuracy worse. Machinery interferes with the direct path.
- Bit-history buffer, combo phase+history, trace features, focal rewards,
  NLL rewards, low exploration, conjunction features, representation
  perturbations (exactly zero: hidden disconnected in input-aff mode),
  context-free, byte-level regression, hidden 6/16/32 (irrelevant:
  disconnected): all flat or worse.

Key mechanism findings:

- Bias inputs poison VO pair evidence (all installs degenerate to X x bias).
  Removing bias yields correct (bit, phase) installs. Core lesson for all
  fingerprint-based selection with constant inputs.
- Installed pairs do not convert to accuracy on stationary streams; product
  edges saturate at install credit.
- Byte identity needs ~8-bit conjunctions; per-rung gains measure ~+0.005.
  Compositional sample efficiency at this noise level is the firewall.

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

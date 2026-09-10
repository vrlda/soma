# SOMA v0.2

> **Project status:** v0.2 is the validated bounded numeric synthetic learning kernel, not the completed SOMA vision or a language-capable consumer model. The sole active roadmap is [`SOMA_REAL_MODEL_MASTER_PLAN.md`](SOMA_REAL_MODEL_MASTER_PLAN.md). The first-English-brain procedure is [`SOMA_ENGLISH_DEMO_RUNBOOK.md`](SOMA_ENGLISH_DEMO_RUNBOOK.md). Completed v8–v15 work is preserved only as historical evidence in [`SOMA_SYNTHETIC_KERNEL_HISTORY.md`](SOMA_SYNTHETIC_KERNEL_HISTORY.md).

SOMA is a small, inspectable, continuously learning computational organism. Its checkpoint is the model: each interaction updates persistent neural state, local plasticity, routing evidence, learned structure, consolidation state, and resource accounting. There is no separate offline training command and no need to create a new model when the environment changes.

## Use the model

Create one persistent model:

```bash
python3 -m soma create soma-model.json --inputs 5 --reward-delay 3 --seed 7
```

Take a learning step. The reward belongs to the action whose delayed outcome has now arrived:

```bash
python3 -m soma step soma-model.json \
  --input '[0.8,-0.9,0.7,0.6,-0.8]' \
  --reward 0.20 --novelty 0.10 --exploration 0.20
```

Inspect or continue the same model from a later process:

```bash
python3 -m soma inspect soma-model.json
python3 continuous_model.py soma-model.json --steps 200
python3 continuous_model.py soma-model.json --steps 200
```

The Python lifecycle API is `SOMA.create`, `step`, `apply_outcome`, `inspect`, `save`, and `load`. Checkpoints are atomic JSON files with validated, deterministic restoration. `Organism.save_checkpoint` additionally saves a supported environment so an interrupted experiment can resume at the exact decision boundary.

To create the exact capacity envelope used by v15, pass `--inputs 6 --hidden 10 --max-cells 40 --max-synapses 320 --max-modules 6 --energy-per-step 40 --prune-reuse-ceiling 2 --reward-delay 3`.

## Demonstrated scope

The locked v14 result demonstrates one bounded organism learning continuously through noisy observations, abrupt recurring contexts, gradual drift, delayed scalar rewards, and novel compositional dynamics without resets or offline retraining. It grows and reuses graph-derived structure under fixed cell and synapse limits and beats a matched structure-disabled online control on the novel dynamics.

This is not a language model, a general-purpose pretrained model, or evidence of unrestricted intelligence. The current model consumes fixed-width numeric observations and scalar modulatory feedback; its strongest evidence is in controlled synthetic lifetime environments. Capacity is finite, adaptation is not instantaneous, aggregate benchmark success does not imply every individual trajectory succeeds, and broader real-world environments remain future work.

## Reliability envelope

The hardened test suite covers exact continuation across four arbitrary points in a complete 6,912-step lifetime, rejected checkpoint corruption, failed-write preservation, randomized long-soak graph/resource invariants, and byte-identical reproduction across independent Python processes. On the development machine, a 1,000-step release-profile fixture averages about `0.210 ms` per step; its JSON checkpoint is about `645 KB`, saves atomically in about `14.3 ms`, and restores in about `4.3 ms`. Portable regression ceilings are intentionally looser than those measurements.

All locked synthetic benchmarks v8–v14 remain green after hardening. These checks establish deterministic software behavior and bounded operation within the tested envelope; they do not establish robustness to arbitrary inputs, operating-system failures, adversarial environments, or large-scale real-world deployment.

## Final scoped evaluation

Run the locked end-to-end v15 protocol with:

```bash
python3 final_lifetime_benchmark.py
```

V15 uses one persistent six-input organism for 6,656 interactions. It acquires four direct policies, constructs recurring and novel higher-order graph circuits, recalls prior owners, reaches full six-module capacity, prunes a weak dormant circuit, reuses that slot, and returns to older tasks. The final untouched seeds 32–39 pass the frozen cohort gates with growth/reuse in `8/8`, pruning in `6/8`, and bounded resources in `8/8`.

The comparison includes structure-disabled and learning-disabled SOMA, a fixed linear online learner, a replay learner with a fixed complete degree-six basis and explicit reward decoder, and an evaluator-only oracle. SOMA beats the non-structural controls on higher-order tasks, but the heavily advantaged replay learner is much stronger (`0.666`–`0.847` mean skill versus SOMA's `0.055`–`0.291`). The project is therefore functionally complete within its declared experimental scope, not competitive with a task-informed replay system and not demonstrated beyond numeric synthetic environments.

SOMA is a deliberately small, standard-library-only implementation scaffold for a self-organizing computational organism. It has stateful cells, a sparse directed graph with stable IDs, recurrent local activity propagation, eligibility traces with three-factor local updates, threshold homeostasis, bounded structural adaptation, explicit resource accounting, and JSON checkpoints.

Run the demo with Python 3.10 (Python 3.7-compatible syntax is retained):

```bash
python3.10 demo.py
python3.10 -m unittest discover -s tests -v
```

The environment and organism are separate: `ContinuousTargetEnvironment` emits observations and rewards, while `Organism.step` consumes observations and modulators and returns actions/outputs.

Rewards are delayed by one transition: pass each returned environment reward into the next `Organism.step`; after the final transition call `organism.apply_outcome(reward)` to close the last eligibility trace without creating an extra decision.

The current version models bounded synaptic structural plasticity and hidden-module recruitment with explicit resource accounting; glia-like regulators remain reserved for later experiments.

On the included drifting-target fixture, the local learner averaged `0.8048` reward per transition across 10 seeds and 200 transitions, versus `0.7898` for the same organisms with synaptic learning disabled (`learning_rate=0`) but structural adaptation still active (deterministic exploration `0.03`). This is a narrow smoke-test signal, not evidence of general learning; the environment is tiny, the controller is rate-based, and no broad performance claim is made.

## Persistent lifetime benchmark

Run the unannounced A -> B -> A lifetime benchmark with:

```bash
python3 lifetime_benchmark.py
```

The `UnannouncedLifetimeEnvironment` changes its latent cue-to-action polarity internally. Rule A rewards action `cue`, rule B rewards `-cue`, and rule A returns. Observations contain only the cue and an independent distractor; they contain no phase, target, time, or horizon label. Phase lengths are reproducibly varied per seed using an independent schedule stream. The evaluator records phase windows separately and feeds the same deterministic exploration tape to the full learner and the matched structural-only control.

Each run uses one persistent organism and one checkpoint lineage across all phases. The report measures normalized skill (`1 - SSE / SSE_zero`) and sign accuracy, including early/tail values for initial A acquisition, B adaptation, immediate A-return retention, and A-return relearning speed. The default protocol uses 24 deterministic seeds, varied hidden phase lengths, and matched evaluator-owned exploration tapes. For this causal milestone, the evaluator installs a zero-strength direct cue-to-motor scaffold and neutralizes competing output afferents; hidden/recurrent representation learning is a later milestone. `context_actor_no_generic_rewiring` tests the causal motor learner before generic topology changes; `full_context_plus_rewiring` enables topology adaptation; `no_actor_context_plus_rewiring` disables actor learning. The gates are predeclared in `LifetimeBenchmarkConfig`; a failing result is reported as `RETENTION_OR_ADAPTATION_NOT_ESTABLISHED` rather than being hidden by a tuned seed or relaxed threshold.

The actor trace is separate from the legacy Hebbian trace. For an output mean `mu`, presynaptic activity `pre`, perturbation `xi`, and exploration scale `sigma`, it records `pre * (1 - mu**2) * xi / sigma` on output-afferent synapses and applies the next reward-prediction error exactly once. Legacy hidden/recurrent weight learning is disabled in the lifetime benchmark; use `mode="legacy_full"` only when explicitly testing the earlier rule.

The context-retention iteration adds graph-native motor engrams. Each engram is a stable motor-module cell with sensor-afferent synapses, a module-specific baseline/confidence/surprise state, and a protected dormant state. Only the module that acted receives the pending outcome and actor update. A module-specific leaky Page-Hinkley/CUSUM detector uses a persisted local outcome predictor (`[1, cue, action, cue*action, cue², action²]`) learned only during normal active operation; positive residuals reduce evidence rather than hard-resetting it. The persisted context state is `normal -> warning -> search`: warning freezes the active actor/predictor, smoothly attenuates the deterministic motor command toward a declared safe fallback, and immediately starts protected diagnostics when a dormant candidate exists; full alarm permits recruitment after the dormant queue is exhausted. Warning/search consume (but do not apply) the evaluator exploration tape, so their motor actions contain zero random noise. Dormant candidates are evaluated by direct sensor-afferent motor means while their cell state and weights remain frozen. Only informative candidate/incumbent disagreements contribute probe evidence, with bounded sequential accept/reject and a queue of all dormant modules before recruitment. Module cells, afferent synapses, pending detector/probe state, and event traces are checkpointed.

Version 6 homeostasis separates signed activity centering from magnitude regulation: signed activation EMA centers thresholds, while magnitude drives adaptation gain; motor expression uses symmetric adaptation scaling. Dormant modules freeze execution state and recover it on reactivation. The detector tolerates standardized negative residuals up to `0.35`, allowing sustained moderate degradation to accumulate without changing switch or confidence gates. Deterministic probes use the declared information threshold directly; caller exploration amplitude is not treated as probe noise.

The explicit benchmark modes are `context_actor_no_generic_rewiring`, `full_context_plus_rewiring`, and `no_actor_context_plus_rewiring`; the historical `weights_only`, `full`, and `structure_only` spellings remain aliases. The B-adaptation gate requires both improvement over B's early window and B tail skill of at least `0.10`; an improving-but-still-negative or weakly positive B phase is reported but does not pass. The lifetime gates also require positive A-initial tail and A-return tail skill. Per-seed gate outcomes, pass counts/fractions, and censored relearning counts are emitted in the JSON report. Aggregate `all_passed` is exactly the conjunction of declared aggregate gates; per-seed outcomes remain diagnostics.

The current independently verified 24-seed audit is `PASS`: immediate A-return retention is `0.279264`, B adaptation is `0.826210`, and mean relearning is `21.375` steps. Pass counts are initial acquisition `24/24`, B adaptation `21/24`, return retention `18/24`, and relearning `23/24`. The original A module is reactivated in `21/24` runs, with median observed lag `10` transitions. The direct sensor-to-motor scaffold remains part of this causal milestone; this result does not claim broad intelligence or general representation learning. Stationary-A controls remain required to detect false warnings, switching, or recruitment.

### Multi-context stress benchmark

Run the separate hidden A -> B -> C -> A -> B stress protocol with:

```bash
python3 multi_context_benchmark.py
```

This protocol uses a frozen 24-tuple manifest of independent `(organism_seed, cue_seed, schedule_seed, exploration_seed)` streams and wide randomized phase lengths in the 216–312 range. Observations are circular `(cos(theta), sin(theta))`; A, B, and C are balanced 120°-separated unit-vector policies with target `0.8 * w · observation`, so no policy has a marginal distribution advantage and zero action remains neutral. The evaluator alone records segment windows, aggregate normalized skill, sign accuracy, canonical module ownership, reactivation lag, and module/capacity events. The primary order is A→B→C→A→B; locked held-out orders A→C→B→C→A and B→A→C→B→C are also run. The matched controls are `no_actor_context` (engrams but actor learning disabled) and `no_context_single_engram` (one engram with context switching disabled).

The current strict 24-seed stress result is `PASS`. Primary canonical reuse is `22/24` for A-return and `24/24` for B-return, with median observed reactivation lag `8` transitions. Held-out orders A→C→B→C→A and B→A→C→B→C both pass with policy-aware first-occurrence identity mapping. The three-module cap is preserved. This remains a bounded continual-reuse milestone over a circular sensor task, not a broad intelligence or general representation claim; direct sensor-to-motor scaffolding and matched controls remain explicit protocol components.

### Nonlinear balanced parity benchmark (v8)

Run the locked nonlinear benchmark with:

```bash
python3.10 nonlinear_benchmark.py
```

The primary protocol uses 24 frozen seeds and one persistent organism per run. Each phase is a shuffled block stream containing every state of `{−1,+1}³` exactly once; the hidden policy sequence is A→B→C→A→B, with targets `0.8*x0*x1`, `0.8*x1*x2`, and `0.8*x2*x0`. Phase labels and targets are absent from observations, and the raw affine baseline has an exact zero-skill/zero-projection certificate.

The v8 `dendritic_representation` mode installs eight graph-native product cells. Each cell persists exactly two sensor source IDs and computes a bounded weighted product before the common cell activation; motor modules receive hidden-cell afferents only, with no direct input-to-motor edges. The pair bank covers all unordered sensor pairs before deterministic seeded repeats. This is a fixed second-order inductive bias for the benchmark, not a general learned representation mechanism.

The independently verified 24-seed primary result is `PASS`. Mean tail skills for A-initial, B-interference, C-novel, A-return, and B-return are respectively `0.862330`, `0.860557`, `0.847208`, `0.835905`, and `0.895678`; tail sign accuracies are `0.993421`, `0.996528`, `0.987209`, `0.973833`, and `0.998643`. Canonical reuse is `20/24` for A-return and `22/24` for B-return, with median first-canonical reactivation lag `11`. Stationary-A safety is `20/24`, with no extra-module recruitment. Relative A-return tail-skill margins are `+0.951854` versus the raw-linear actor and `+0.856768` versus the additive hidden no-node control.

Held-out diagnostics are reported separately from the locked primary: sensor-pair permutation reaches mean tail skill `0.630754` with `6/6` positive runs, and continuous product inputs reach `0.699795` with `6/6` positive runs. The `hidden_node_perturbation` mode remains an experimental control and does not establish a general representation learner; broad representation learning without a predeclared second-order product bias remains open.

### Adaptive dendritic fingerprint benchmark (v9)

Run the locked neutral fingerprint protocol with:

```bash
python3 adaptive_dendritic_benchmark.py
```

The v9 protocol uses 24 deterministic seeds, five phases with a 256-step base length and seeded actual lengths varying from 128 to 384, six balanced sensor coordinates, an evaluator-owned task-pair manifest, and matched raw-linear, fixed-random-dendritic, additive-hidden, no-actor, and frozen controls. The organism receives no phase labels, targets, task pairs, or schedule state. The adaptive path starts from the matched additive hidden scaffold and uses a neutral 16-outcome fingerprint: action mean `0.0`, the actual exploration tape, delayed raw reward, and all 15 unordered sensor-pair eligibilities on every outcome. Actor, baseline, confidence, and detector learning are frozen during the fingerprint. An installed pair reactivates its owner directly; an unowned pair recruits a fresh hidden-afferent motor module and installs atomically. Detector drift is `0.65`, minimum normal evidence is `64`, and each install has a `64`-step detector hold. Pair ownership, pending delayed evidence, resource state, stable IDs, and mid-fingerprint checkpoints are serialized exactly. The legacy structural-trial `min_evidence=3` field remains only for checkpoint/report compatibility and is not the v9 fingerprint window.

The independently verified default 24-seed v9 result is `ADAPTIVE_DENDRITIC_PASS`. Mean tail skills for A-initial, B-interference, C-novel, A-return, and B-return are `0.664460`, `0.669873`, `0.597893`, `0.679609`, and `0.657057`; corresponding tail sign accuracies are `1.000000`, `1.000000`, `0.947483`, `0.985677`, and `0.965104`. Canonical reuse is `21/24` for A-return and `20/24` for B-return, with median reactivation lag `11`; stationary safety is `24/24`, and mean module count is `2.875`. The A-return tail-skill margins are `+1.002709` versus raw linear modules, `+0.343468` versus fixed-random dendritic, and `+0.705433` versus the additive-hidden control. Starting cell IDs, synapse IDs, and synapse strengths are matched across the causal adaptive/additive comparison; all declared v9 gates pass.

The v9 mechanism is intentionally a second-order inductive bias, not a general representation learner. The held-out cubic diagnostic is not a primary gate (`cubic_heldout_declared_primary=false`): across seeds 0–5, tail skill is `[-0.056030, -0.031210, 0.018827, -0.031239, -0.011428, -0.026801]` (range `-0.056030..+0.018827`) and sign accuracy is `[0.484375, 0.507813, 0.554688, 0.492188, 0.507813, 0.492188]`. This negative third-order result is an explicit limitation: the benchmark demonstrates bounded pair-based continual reuse under matched controls, not higher-order or broad representation learning.

### Variable-order local representation benchmark (v10)

Run the locked v10 benchmark with:

```bash
python3 variable_order_benchmark.py
```

The default protocol uses one persistent organism through `P0 -> C0 -> P1 -> C1 -> P2 -> C2 -> P0 -> C0`, 24 frozen seeds, six independent RNG streams (`organism`, `input`, `schedule`, `exploration`, `representation`, and `task`), balanced 64-state sensor blocks, and evaluator-only task manifests. Exact parity certificates establish zero projection of pair tasks onto affine features and zero projection of cubic tasks onto all degree-`<=2` features; those certificates are protocol diagnostics, not organism inputs. Phase lengths vary by seeded schedule while preserving the declared base phase length.

The v10 learner is target-blind. It scores all 35 direct-input candidates (15 pairs plus 20 triples) using sign-symmetric local eligibility and delayed raw reward. Each context begins with a neutral 16-outcome fingerprint: mean action is zero, actual exploration tape is consumed, and actor, baseline, confidence, and detector updates are frozen. The first credible feature bootstraps onto the incumbent module; later unowned features use an atomic hidden-module recruit/install transaction. Existing owners reactivate directly. Graph cells are order-aware (`dendritic_product` for pairs and `dendritic_product_n` for triples), with one learned feature per module, bounded resources, stable IDs, and exact JSON/direct checkpoint replay. The v10 milestone checkpoint format remains compatible and migrates cleanly; the current `Organism` schema is `VERSION=11`, while v8/v9 binary behavior and checkpoint migrations remain supported.

Controls are matched additive-hidden, raw-linear-modules, no-actor, and frozen runs; the fixed random mixed-order bank remains deferred/non-gating pending a target-independent pair+triple bank control implementation. Frozen gates require first-occurrence pair tails across P0/P1/P2 and cubic tails across C0/C1/C2, returned P0/C0 performance and reuse, reactivation lag, capacity, zero direct input-to-motor edges, stationary safety, and separate pair/cubic margins over both raw and additive controls. No phase, task, target, or schedule metadata enters organism state.

The independently verified default 24-seed result is `VARIABLE_ORDER_REPRESENTATION_PASS` with all gates true. First-occurrence tail skills are P0 `.699`, C0 `.703`, P1 `.697`, C1 `.690`, P2 `.671`, and C2 `.684`; returned P0 is `0.6472287121244298` and returned C0 is `0.6906436087925579`. Canonical reuse is `24/24` for both returns, median reactivation lag is `19`, stationary safety is `24/24`, mean module count is `6.0`, and direct input-to-motor edges remain zero. Return-tail margins are pair `+0.9914474127314565` and cubic `+1.060187705933863` versus raw, and pair `+0.6754977724451956` and cubic `+0.7128091119112187` versus additive hidden.

This is a bounded direct-sensor pair/triple mechanism: maximum arity is 3, the candidate bank is finite (35 features across six first-occurrence contexts), capacity is six modules, and evaluation uses parity-style tasks with explicit balanced certificates. It does not establish arbitrary composition, language learning, or general intelligence. The v8 fixed pair representation and v9 adaptive pair fingerprint remain preserved milestones; higher-order learning beyond this declared substrate remains open.

### Graph-derived compositional continual learner (v11)

Run the locked v11 benchmark with:

```bash
python3 compositional_benchmark.py
```

The protocol uses one persistent organism across twelve phases:
`P0 -> P1 -> Q0 -> C0 -> P2 -> P3 -> Q1 -> C1 -> P0_return -> Q0_return -> C0_return -> Q1_return`.
It has 24 frozen seeds, seven independent RNG streams (`organism`, `input`, `schedule`, `exploration`, `representation`, `composition`, and `task`), and balanced 64-state blocks. The organism receives no phase, target, task-manifest, or schedule metadata.

Composition is learned from the live graph: disjoint learned direct-pair cells form graph-derived pair-of-pair candidates with four distinct leaves. Each fingerprint is exactly 16 delayed outcomes with neutral action mean; the evaluator exploration tape is consumed, delayed composition/direct evidence accumulates, and actor, baseline, confidence, and detector updates remain frozen. Direct and composition candidates use joint v10-style evidence arbitration before any mutation; stronger direct evidence blocks a noisy composition. Recall verification is target-blind and reversible: normal local evidence can reopen the same neutral fingerprint during a detector hold when an active direct route is no longer credible. Feature recruitment and ownership are atomic, resource-bounded, and checkpoint-exact, including pending owner/step/timing records and direct/JSON mid-fingerprint replay.

The causal ablation is evaluator-only and uses two clones of the same final checkpoint. The ablated clone zeros only learned composed-cell-to-owner motor edges; apart from that explicit edge-strength change on the ablated clone, evaluation performs no mutations, learning, exploration, or phase-aware routing. Matched controls include an independently trained composition-disabled v10 organism, raw linear modules, additive hidden modules, no-actor, and frozen runs.

The independently verified default 24-seed result is `COMPOSITIONAL_REPRESENTATION_PASS` with every declared gate true. Q0-return and Q1-return early means are `0.1768083074` and `0.1512030672`. Causal Q ablation is available for `24/24` seeds with mean drop `0.4320928232`; direct P/C degradation is available for `24/24` with degradation `0.0`. Q0 and Q1 task-matched install-before-tail counts are both `24/24`; four-leaf closure is `24/24`; canonical reuse is `24/24` for P0, Q0, C0, and Q1 returns; median reactivation lag is `19`; and stationary safety is `24/24`.

This is a functional bounded continuous learner over a declared graph substrate and benchmark, not a research-paper claim about general language learning, AGI, or unrestricted composition. Its finite resources, candidate construction rules, frozen manifests, balanced tasks, matched controls, and evaluator-only causal tests define the scope of the result.

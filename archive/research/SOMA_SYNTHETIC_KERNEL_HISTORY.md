# SOMA Synthetic Kernel History

> **HISTORICAL RECORD — NOT THE ACTIVE SOMA ROADMAP**
>
> This document records the design, implementation, and validation of the bounded numeric synthetic learning kernel through v15. Its completed milestones establish that scoped foundation only. They do not mean that the SOMA model, universal organism, language system, trained brain, or prosumer product is complete.
>
> All current and future project work is governed by [`SOMA_REAL_MODEL_MASTER_PLAN.md`](../SOMA_REAL_MODEL_MASTER_PLAN.md). The operational path to the first English demonstration brain is defined by [`research/SOMA_ENGLISH_DEMO_RUNBOOK.md`](SOMA_ENGLISH_DEMO_RUNBOOK.md).
>
> Preserve this file as experimental history and regression evidence. Do not continue its M0–M8 milestone sequence or use its unchecked historical questions to choose current work.

## Original scoped plan

Build SOMA into one persistent, runnable model that learns throughout its lifetime without offline retraining. The path is to stop adding isolated benchmark fixes, separate context inference from representation learning and structural growth, prove each subsystem independently, then integrate them in progressively less structured environments.

## Scope

- In: one persistent organism; online local learning; latent-context inference; circuit recall; structural growth; consolidation; pruning; bounded resources; exact checkpoint/resume; deterministic evaluation; an executable model interface; and evidence that the same checkpoint continues learning across unseen streams.
- Out: claiming AGI, matching frontier LLM language ability, biological neuron simulation, unlimited memory, zero-cost instantaneous detection of hidden task changes, offline pretraining as the primary learning mechanism, and benchmark-specific task labels or routing rules.

## Definition of done

SOMA is considered functionally complete when all of the following are true:

- A user can create one organism, stream observations and rewards into it, receive actions, save it, restore it, and continue the same lifetime through a documented API and CLI.
- The organism learns new behaviors online without resetting weights, topology, optimizer state, memory, or identity.
- It distinguishes recurring latent contexts from genuinely novel situations without receiving task IDs, phase boundaries, targets, schedules, or evaluator metadata.
- It recalls a previously useful circuit with bounded regret and measurably faster adaptation than learning a new circuit.
- It creates new computational structure only when existing circuits cannot explain the experience.
- It consolidates repeatedly useful circuits, prunes redundant or persistently useless structure, and remains within declared cell, synapse, energy, and module budgets.
- Its learned representations are not limited to a frozen list of pair, cubic, or quartic parity features; the organism can compose or construct features demanded by experience within its resource limits.
- Direct and JSON checkpoint restoration are behaviorally exact, including mid-probe, mid-update, mid-consolidation, and mid-pruning states.
- It passes the frozen regression milestones v8 through v11, the irregular streaming milestone v12, and the final unseen-environment evaluation described below.
- Its causal ablations show that learned behavior depends on the structures SOMA created, while matched fixed-topology and learning-disabled controls do not explain the result.

This definition describes a working continual-learning model. It does not imply general intelligence or unlimited capability.

## Current baseline

- v8 proves the earlier fixed nonlinear representation path.
- v9 passes adaptive pair-feature learning and bounded circuit reuse.
- v10 passes target-blind variable-order pair/triple representation learning.
- v11 passes graph-derived four-leaf composition over one persistent organism across its frozen 24-seed protocol.
- The v12 irregular-stream environment, controls, checkpointing, integrity checks, and causal evaluation are implemented and audited.
- The clean v12 four-seed baseline passes stream integrity, capacity, stationary safety, Q0/Q1 composition installation, exact four-leaf closure, causal Q ablation, direct-task preservation, and control margins.
- The remaining v12 failures are early return performance and canonical circuit reuse. Earlier reward-drop, candidate-stability, and fast-recall experiments were rejected because they traded one failure for another.
- Midpoint existing-owner resolution is retained as the locked v12 development baseline. Its enabled-versus-disabled four-seed comparison preserved every declared safety, causal, capacity, closure, acquisition, and tail gate; reduced median reactivation lag from 20 to 19 steps; improved canonical reuse for P0 from 3/4 to 4/4 and P2 from 1/4 to 3/4; and materially improved early skill for most tasks. It does not yet pass v12: every-return and canonical-reuse gates remain RED.
- The midpoint mechanism now has dedicated tests for direct and composed owner recall, unowned and cross-family-tie rejection, structural immutability, and exact direct/JSON checkpoint replay. The full suite passes 189/189 tests.

## Architectural direction

The next architecture must separate four responsibilities that are currently entangled:

1. **Change detection:** decide whether recent outcomes are inconsistent with the active circuit.
2. **Circuit inference:** estimate which existing circuit best explains current experience.
3. **Novelty decision:** decide whether no existing circuit is adequate.
4. **Structural response:** reactivate, adapt, create, consolidate, or prune structure based on that decision.

The intended router is a target-blind evidence system, not a collection of task-specific thresholds. Each circuit maintains a local predictive model of its expected outcomes. SOMA maintains evidence over existing circuits, performs bounded active probes when evidence is ambiguous, reactivates a credible owner, and permits new growth only when every existing owner remains implausible.

## Action items

- [x] **Restore and lock the baseline.** The midpoint experiment passed the full 189-test suite and the enabled-versus-disabled four-seed v12 comparison without reducing any declared acquisition, tail-skill, safety, capacity, closure, causal, or control result. It is now the v12 development baseline.

- [x] **Decompose the v12 failure with diagnostic controls.** Added evaluator-only frozen-clone runs for oracle boundary detection with learned circuit selection, learned boundary detection with oracle circuit selection, and oracle detection plus oracle selection. The controls are absent from organism state and normal acceptance output. The four-seed smoke result identifies stored-policy quality as the dominant bottleneck; routing remains secondary.

### M1 diagnostic result

The locked seeds 0–3 development smoke contains 64 return segments. Correct learned owners were graph-resolvable for 62/64 segments and learned segment owners were available for 63/64. Mean full-segment skill was:

- learned execution: 0.384
- oracle boundary with learned selection: 0.562
- learned detection with oracle selection: 0.523
- oracle boundary with oracle selection: 0.589
- perfect target policy: 1.000

The factorial effects interact: removing detection latency gains 0.178 with learned selection and 0.066 with oracle selection; correcting selection gains 0.139 with learned detection and 0.027 with oracle boundary timing. Even after both routing oracles, the remaining stored-policy gap is 0.411, larger than either routing effect. Therefore M2 must improve retained circuit policy quality as well as evidence-based routing. These are hindsight evaluator ceilings based on frozen final-state clones, not acceptance gates and not claims about an implementable oracle.

### M2 development log

- A confidence-adaptive actor-rate experiment improved mean learned return skill from 0.384 to 0.418 and the oracle-routed policy ceiling from 0.589 to 0.619, but regressed canonical reuse for C1, P0, and P2. Weaker boosts still reduced P0 reuse from 4/4 to 2/4. The experiment was fully removed and the 190-test baseline restored.
- Task-level oracle routing shows the weakest stored policies are the composed tasks Q0 (0.337) and Q1 (0.409), followed by C0 (0.490); pair-task oracle scores are 0.684–0.727. Two of eight C0 return segments have no graph-resolvable correct owner. M2 should therefore correct composed-feature expression and missing-owner inference before changing global actor plasticity.
- A provisional composed-path signal normalization now compensates for the predictable attenuation from nesting bounded product cells. At gain 1.72, mean Q oracle-routed skill rises from 0.373 to 0.566, learned return skill from 0.384 to 0.486, causal Q drop from 0.390 to 0.581, and all return-tail gates pass. Reuse reaches 4/4 for P0–P3 and Q0–Q1 and improves C0 from 1/4 to 3/4, while C1 falls from 3/4 to 2/4. The mechanism is checkpoint-exact and the full 191-test suite passes, but it remains provisional until routing removes the C1 regression.
- C1 inspection showed seeds 0 and 1 acquire the correct C1 feature only on its second occurrence after a fingerprint straddles the hidden boundary and resolves stale evidence. A configurable shorter detector-hold experiment recovered C1 reuse at 16 steps, but reduced learned return skill from 0.486 to 0.313 and regressed pair, cubic, and composed acquisition sign gates. It was fully removed. The next router must invalidate stale evidence without globally increasing switch/relearning frequency.
- A hold-local post-resolution contradiction rule was also rejected. Ten retries failed to recover C1 and reduced learned skill from 0.486 to 0.457, increased median lag from 18 to 20, and regressed five task-reuse counts. It was fully removed. Ordinary eight-outcome perturbation evidence is not reliable enough to invalidate a route; M2 needs explicit confidence/evidence state rather than another threshold over the existing aggregate.
- Added a persisted, normalized owner-versus-novelty evidence distribution in behaviorally inert shadow mode. Enabled/disabled runs have identical actions, rewards, routing, reuse, and latency; direct/JSON replay is exact. Relative to the legacy router's eventual decisions, the selected owner ranks first in all 126 existing-owner resolutions and novelty ranks first in all 31 structural installs. However, evaluator truth exposes cumulative-evidence contamination: at 16 outcomes leader accuracy is 95.2%, and wrong leaders can still receive posterior probability 0.964. Requiring the same leader at outcomes 8 and 16 yields only 95.4% accuracy. Therefore this posterior is instrumentation, not yet a calibrated routing policy; promotion requires recency-aware evidence or bounded active probes.
- An eight-outcome recency-only shadow estimator was tested and removed. It reduced completion-time evaluator-truth accuracy from 95.2% to 86.5%; discarding older evidence avoids contamination but leaves too much sampling variance. Passive cumulative and sliding-window evidence are both insufficient, so the next M2 path is bounded active owner probing with explicit uncertainty.
- Added bounded active owner probes driven by the persisted posterior. Only a graph-valid existing owner different from the incumbent can be probed; probe actions are protected from learning, rejection returns to the incumbent, and this path cannot recruit or install. Across seeds 0–3 it performs 36 probes, preserves every declared gate and every reuse count, improves learned return skill from 0.486 to 0.519, holds oracle policy quality (0.680 to 0.682), causal Q drop (0.581 to 0.582), and stationary safety (4/4), with median reactivation lag increasing from 18 to 20 under the frozen limit of 30. Mid-probe direct/JSON replay and no-growth rejection are tested; the full suite passes 194/194. The probe mechanism is accepted as the current M2 baseline.
- **M2 locked:** the frozen 24-seed run passes all M2 gates. Reuse is C0 24/24, C1 22/24, P0 24/24, P1 22/24, P2 20/24, P3 23/24, Q0 23/24, and Q1 22/24, all above the frozen 18-run floor. Median reactivation lag is 20 under the 30-step limit; stationary safety is 24/24; maximum module count is 8; Q0/Q1 installs and exact closures are 24/24; checkpoint diagnostics are exact. The complete manifest performs 233 bounded probes.
- The required 24-seed normalization opt-out comparison is non-regressive: normalization improves learned return skill from 0.453 to 0.523, oracle-routed skill from 0.612 to 0.669, causal Q drop from 0.398 to 0.577, median lag from 21 to 20, and every task reuse count; stationary safety remains 24/24. Composed signal normalization is accepted into the M2 baseline.

- [ ] **Replace threshold-driven switching with circuit evidence.** Give every motor module a persisted local outcome model and a likelihood/confidence estimate. Maintain a normalized posterior or evidence score over all existing modules using only observations, actions, rewards, and internal predictions. Add explicit uncertainty and hysteresis so one noisy reward cannot cause a switch, while sustained mismatch cannot be hidden by a long fixed detector warm-up.

- [x] **Implement bounded active identification.** Ambiguous existing-owner evidence now invokes deterministic, protected, reward-only probes using the existing bounded probe machinery. Probes stop through confidence bounds or the maximum budget, cannot learn or grow structure, and restore the incumbent after rejection. Switching is measured by return skill and reactivation latency.

- [x] **Separate recall from growth.** Existing graph-valid owners reactivate without allocation; new general features require two independent confirmations. Capacity recovery ranks only dormant, sufficiently aged, weakly reused features and reuses the reclaimed module slot atomically.

- [x] **Generalize representation construction.** General features are proposed from graph-visible paths and reusable learned subgraphs, with stable IDs, exact leaf closure and lineage, bounded depth/leaves, local fingerprint evidence, and explicit resource costs. The legacy finite pair/triple bank remains only as the lower-order substrate.

- [x] **Add consolidation and metaplasticity.** Persisted reuse, last-use age, causal ablation, module confidence/protection, synapse stability/utility, and protected identification govern retention and revision. Capacity pruning ranks weak reuse and recency while protecting active and graph-dependent circuits.

- [x] **Add adaptive pruning and capacity recovery.** Persisted reuse/recency gates reclamation; active, recent, consolidated, and dependency-required structures are protected. Atomic pruning restores nursery topology, releases exact counters, preserves reachability/ownership, reuses the module slot, and continues learning without restart.

- [x] **Pass v12 in stages.** The frozen 24-seed v12 manifest passes all acquisition, return, reuse, latency, capacity, stationary-safety, closure, causal-ablation, checkpoint, and matched-control gates simultaneously and remains green through v15.

- [x] **Move beyond balanced parity environments.** The locked v14 stream uses continuously varying noisy observations, gradual drift, abrupt recurring contexts, novel higher-order dynamics, delayed rewards, and no balanced block guarantee. One organism learns throughout without resets or offline retraining; the report covers competence, returns, structural growth/reuse, matched controls, and bounded resources.

- [x] **Add an unseen final evaluation.** Locked v15 runs one checkpoint through familiar returns, four direct contexts, three higher-order contexts, noise, delayed reward, and capacity pressure. It reports matched fixed-topology, replay-based, learning-disabled, structure-disabled, and evaluator-only oracle comparators.

- [x] **Ship the actual model interface.** The stable `SOMA`/`Organism` lifecycle API and CLI provide `create`, `step`, `apply_outcome`, `inspect`, `save`, `load`, and continuation. `continuous_model.py` continues one checkpoint across processes, and inspection exposes machine-readable event, topology, resource, and learning metrics.

- [x] **Harden reliability.** Deterministic arbitrary-boundary replay, randomized graph/resource invariants, corruption rejection, a 2,500-step soak, cross-process reproducibility, atomic-failure preservation, and explicit latency/checkpoint/restore budgets are now regression-tested.

- [x] **Declare scoped completion only after an end-to-end lifetime run.** Locked v15 starts from one small organism and demonstrates online acquisition, old-circuit recall, graph growth, consolidation, pruning, capacity recovery, bounded resources, matched-control superiority, and exact multi-checkpoint continuation without resets or offline retraining. Completion is explicitly scoped to the tested numeric synthetic domain.

## Milestone gates

### M0 — Clean experimental baseline

- Full regression suite passes.
- v8–v11 frozen outputs remain unchanged.
- v12 Stage1 protocol is reproducible and honestly reports RED.
- No rejected experimental mechanism remains in default behavior.

### M1 — Diagnosed routing bottleneck

- Oracle-control matrix separates detection, selection, and policy-quality regret.
- A written result identifies the dominant bottleneck with measurements.
- No evaluator metadata enters organism state.

### M2 — Functional latent-context router ✅ COMPLETE

- Existing contexts are selected from local evidence with bounded probes.
- Stationary false-switch rate stays below the frozen limit.
- New-context detection remains possible when all existing circuits are poor.
- Router state is checkpoint-exact.

Validated on the frozen 24-seed manifest on 2026-09-10. Remaining v12 failures belong to M3: every-return early skill and one run without a complete direct-task causal-ablation set. They do not invalidate M2's routing, novelty, safety, resource, or replay gates.

### M3 — v12 streaming pass ✅ COMPLETE

- All frozen 24-seed v12 gates pass simultaneously.
- Every task has acquisition plus two successful returns.
- Matched controls, causal ablations, capacity, and safety remain valid.
- No post-result threshold or manifest changes are permitted.

Validated on the frozen 24-seed manifest on 2026-09-10. All v12 gates pass simultaneously. Every causal Q and direct-task ablation is available (24/24); mean Q ablation drop is 0.575 and direct-task degradation is 0.000. Reuse counts range from 21/24 to 24/24, median reactivation lag is 13 steps, stationary safety is 24/24, Q0/Q1 installs and exact closures are 24/24, and the module cap remains 8. The final mechanism adds a single bounded, target-blind fingerprint refresh when novelty evidence rises sharply after the midpoint, preventing boundary-straddling evidence from consuming capacity. The complete suite passes 196/196.

### M4 — General structural learner ✅ COMPLETE

- Feature construction is graph-derived and not restricted to the v12 task family.
- Novel environments cause traceable useful growth.
- Recurring environments reuse existing structures.
- Redundant structures are recoverably pruned.

#### M4 development log

- Added an opt-in graph-derived structural substrate, kept inert for the frozen v8-v12 paths. Candidate features are now constructed from existing sensor and learned-feature nodes, with bounded depth and leaf closure rather than a fixed task manifest.
- General feature installs persist stable source-cell lineage, transitive input closure, depth, owner, nursery rollback data, and resource changes. Direct and JSON checkpoints preserve this state exactly.
- Added dependency-safe pruning and explicit reuse events. A feature used by a deeper feature cannot be pruned; pruning in dependency order restores the original additive nursery topology, synapse identities where restored, ownership maps, and exact resource counters. The first deterministic depth-2 construction/reuse/pruning fixture passes, and the full suite is 197/197.
- Autonomous delayed-reward scoring is integrated with the latent-context fingerprint path. Unowned graph-derived features require the same winner across two independent 16-outcome fingerprints before growth. A bounded target-blind structural audit prevents weak sustained mismatch from remaining invisible; separated reuse episodes update persisted consolidation counters.
- Capacity pressure automatically prunes an unreinforced, dormant, dependency-free general feature and reuses its existing motor-module slot. Active, protected, reused, and transitively required features cannot be selected. A deterministic capacity fixture proves replacement at the hard module cap without increasing module count.
- **M4 locked:** the v13 development manifest passes 8/8 for graph-derived depth-2 five-input growth, intervening direct-owner recall, high-order owner reuse, recoverable dependency-safe pruning, structure-disabled control margin, and causal feature dependence. Mean high-order tail skill is 0.730 versus 0.701 for the matched structure-disabled control; mean causal edge-ablation drop is 0.048.
- A final untouched seed block (16–23), run after the mechanism and thresholds were locked, passes every gate 8/8. Its minimum high-order tail skill is 0.705 against the frozen 0.670 floor; mean high-order skill is 0.728 versus 0.699 for the matched control, and mean causal drop is 0.048. The full suite passes 201/201 and the frozen 24-seed v12 result remains `STREAMING_COMPOSITIONAL_PASS` with unchanged 24/24 direct causal coverage, median lag 13, and minimum reuse 21/24.

### M5 — Continuous-environment pass ✅

- One organism handles abrupt switches, gradual drift, noise, and delayed rewards.
- It demonstrates retention and transfer against matched online baselines.
- Resource usage remains bounded over a long soak run.

#### M5 development log

- Added the phase-blind v14 continuous lifetime protocol: bounded continuous noisy observations, abrupt familiar and novel contexts, gradual A→B drift, a higher-noise return, three-step delayed rewards, and no balanced-state blocks. Environment checkpoints restore the stream cursor and pending reward queue exactly.
- The first eight-seed baseline was honestly RED. Resources remained bounded in 8/8 runs, but normalized tail skill was A −0.027, B 0.062, H −0.013, drift −0.178, and noisy-H −0.123; useful general growth and reuse were 0/8. Holding diagnostic perturbations over temporally persistent observations did not recover causal learning.
- Diagnosis: v12/v13 structural fingerprints and actor updates bind each arriving reward to the immediately preceding perturbation. With a multi-step reward queue, independent intervening perturbations destroy that causal pairing. M5 therefore requires persisted delayed-credit traces or explicit action/outcome correlation state inside the organism; changing detector thresholds or weakening the benchmark would not address the failure.
- Added a bounded, checkpoint-exact three-outcome credit buffer shared by actor traces and direct/general fingerprint evidence. JSON round trips with a nonempty queue resume deterministically. The continuous stream uses stochastic, unbalanced sign recurrence with independently varying magnitudes and observation noise; a 32-outcome fingerprint makes structural selection reliable without evaluator-visible phase identity.
- **M5 locked:** development passes all v14 gates. The final untouched seed block 24–31 then passes without retuning: phase-mean normalized tail skills are A `0.253`, B `0.298`, H `0.173`, drift `0.263`, and noisy-H `0.208`; graph-derived growth, reuse, and resource bounds are each `8/8`. The matched structure-disabled H and noisy-H means are `−0.122` and `−0.079`, so both causal-control margins exceed the locked `0.10` requirement. The single organism runs 6,912 interactions per seed with no reset or offline retraining.
- The full regression suite passes 205/205. Frozen v13 remains `GENERAL_STRUCTURAL_PASS`, and the frozen 24-seed v12 benchmark remains `STREAMING_COMPOSITIONAL_PASS`.

### M6 — Model release ✅

- Public lifecycle API and CLI are complete.
- Checkpoints are versioned, validated, and migration-tested.
- The final held-out lifetime run passes from a fresh organism.
- Documentation states demonstrated capabilities and limitations without broader claims.

#### M6 release log

- Added the stable `SOMA` lifecycle facade. `SOMA.create` constructs the locked continuous-learning profile; `step` learns from each interaction, `apply_outcome` closes terminal credit, `inspect` reports topology/resources/learning state, and `save`/`load` continue the same persistent organism.
- Added `python3 -m soma` commands for `create`, `step`, `outcome`, and `inspect`, plus `continuous_model.py` as a minimal process-restart example that continues one checkpoint rather than training separate models.
- Advanced organism checkpoints to version 12, retained migrations from versions 5–11, made writes atomic, and added the v14 continuous environment to compound checkpoint restoration. Direct and JSON-restored models continue deterministically with nonempty delayed-credit state.
- The README now documents the usable interface, continuous-learning semantics, demonstrated scope, and explicit limitations. The final untouched v14 lifetime run remains the fresh-organism acceptance result recorded under M5.
- **M6 locked:** the complete suite passes `209/209`, including public API, CLI, atomic save/load continuation, legacy migration, and continuous compound-checkpoint tests.

### M7 — Reliability and end-to-end hardening ✅

- A new end-to-end test runs the complete 6,912-interaction continuous lifetime twice: once uninterrupted and once with atomic save/load cycles after interactions 137, 1,023, 4,097, and 6,501. Every action and the final organism and environment states are exactly equal.
- Hardening exposed and fixed a genuine delayed-credit invariant defect: fingerprint actions whose rewards arrived after resolution retained stale pending timestamps and could make a valid live model impossible to validate or reload. The bounded deadline now includes delayed-credit depth, and post-resolution diagnostic rewards are discarded rather than contaminating normal learning.
- Checkpoint loading now rejects non-mapping model payloads, unsupported compound versions, non-finite synapses, invalid delayed queues, and persisted resource counters that disagree with actual topology. Failed atomic saves and invalid CLI calls preserve the previous checkpoint byte-for-byte and leave no temporary file behind.
- A deterministic 2,500-step randomized soak repeatedly validates reachability, stable graph ownership, exact resource counters, and the 32-cell/256-synapse limits. Identical CLI interactions in independent Python processes produce byte-identical checkpoints.
- The release performance fixture measures approximately `0.210 ms` per step, a `645,302`-byte checkpoint after 1,000 interactions, `14.3 ms` atomic save time, and `4.3 ms` restoration time on the development machine. Regression ceilings are deliberately portable: `50 ms` per step, `20 MiB` checkpoint size, and `5 s` restoration.
- Because the invariant fix changes runtime behavior, v14 was re-frozen before a new untouched acceptance block. Seeds 32–39 pass: A `0.257`, B `0.294`, H `0.220`, drift `0.228`, noisy-H `0.184`; graph growth, reuse, and resource bounds are `8/8`. Matched structure-disabled H and noisy-H are `−0.099` and `−0.098`, preserving both `+0.10` structural margins.
- **M7 locked:** the full suite passes `215/215`; v8 `PASS`, v9 `ADAPTIVE_DENDRITIC_PASS`, v10 `VARIABLE_ORDER_REPRESENTATION_PASS`, v11 `COMPOSITIONAL_REPRESENTATION_PASS`, v12 `STREAMING_COMPOSITIONAL_PASS`, and v13 `GENERAL_STRUCTURAL_PASS` all remain green.

### M8 — Scoped end-to-end completion ✅

- Added the phase-blind v15 capacity-pressure lifetime: six continuously varying noisy inputs, three-step delayed rewards, four direct contexts, recurring five-factor H, a temporary five-factor D circuit, a distinct five-factor K context at full six-module capacity, and final H/A/B returns. One organism runs the entire 6,656-interaction sequence.
- The same run must build at least two general circuits, reuse learned structure, preserve consolidated H, prune a weak dormant circuit under capacity pressure, reuse its module slot for K, and remain within 40 cells and 320 synapses. JSON save/load at four arbitrary internal boundaries is exactly identical to uninterrupted execution.
- Matched reports include SOMA, structure-disabled SOMA, learning-disabled SOMA, a fixed linear online learner, a 256-example replay learner with a fixed complete degree-six polynomial basis and explicit public-reward decoder, and an evaluator-only oracle. The strong replay comparator is intentionally advantaged and substantially outperforms SOMA; it is reported as a conventional upper comparator, not as evidence for SOMA superiority over replay.
- **Final untouched v15 acceptance (seeds 32–39):** phase-mean normalized skill is A `0.291`, B `0.217`, C `0.231`, E `0.241`, H `0.134`, temporary D `0.107`, and K `0.055`. Growth and reuse occur in `8/8`, capacity pruning in `6/8`, and resource safety in `8/8`. Structure-disabled H is `−0.180`; learning-disabled mean skills are negative throughout. Replay scores range from `0.666` to `0.847`, making the remaining performance gap explicit.
- V15 passes every frozen behavioral, structural, checkpoint, resource, and matched-control gate without post-result changes. This closes the implementation plan only for the declared small numeric synthetic lifetime domain; language, perception, embodied control, and unrestricted continual learning are not claimed.
- Final verification passes `221/221` tests. The post-v15 pruning changes leave v13 `GENERAL_STRUCTURAL_PASS` and v14 `CONTINUOUS_LIFETIME_PASS`; the earlier complete sweep remains green for v8–v12. The public API and CLI expose the exact six-module, 40-cell, 320-synapse, 40-energy, delayed-credit, consolidation/pruning profile used by v15.

## Validation policy

- Freeze manifests, controls, metrics, and thresholds before acceptance runs.
- Use small smoke manifests during mechanism development; reserve the frozen 24-seed and held-out manifests for locked candidates.
- Compare every candidate mechanism against its explicit opt-out in the same code revision.
- Reject mechanisms that improve one headline metric by degrading retained skill, safety, causal availability, capacity, or old milestones.
- Never use task identity, phase boundaries, evaluator targets, future horizon, or schedule metadata inside the organism.
- Never treat a passing benchmark as sufficient if the mechanism is task-shaped and fails the next less-structured environment.

## Principal risks

- **Identifiability:** when observations are identical across latent contexts, immediate perfect switching is impossible without informative actions and rewards. The model must minimize bounded regret rather than promise zero-latency certainty.
- **Stability versus plasticity:** rapid learning can overwrite recalled circuits; excessive consolidation can prevent adaptation.
- **Capacity:** a finite organism cannot remember unlimited unrelated behaviors. Pruning, compression, composition, and explicit resource accounting are required.
- **Noisy structural commitment:** local evidence can select plausible but incorrect features. Growth must use stronger evidence than reversible recall.
- **Benchmark overfitting:** parity tasks are useful proofs but not the final environment. Every major mechanism must survive a less structured successor benchmark.
- **Evaluation leakage:** oracle controls are diagnostic ceilings only and must never become inputs to the released organism.

## Open questions

- What final resource envelope should define the released small SOMA model: maximum cells, synapses, modules, memory, and step latency?
- Should the first released model expose only a scalar action interface, or a general vector observation/action interface from the start?
- What continuous held-out environment family should serve as the final completion test: control dynamics, navigation, or a mixed synthetic lifetime suite?

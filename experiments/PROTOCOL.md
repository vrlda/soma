# First experiments: registered before execution (2026-10-10)

## Question and scope
Can a small neural substrate learn continuously through local synaptic rules,
and retain an association across intervening experience? This is an initial
candidate, not a claim to have solved internal credit assignment or language.
Knowledge used for predictions must live in neuron/synapse state. Sensory and
outcome channels are primitive experimental interfaces, not semantic circuits.

## Candidate rules
Leaky recurrent neurons, sparse excitatory/inhibitory drive synapses, local
Oja updates, and prediction synapses with a local three-factor rule:
source trace × target neuron's own outcome error × broadcast learning gate.
No outcome error travels into hidden neurons. Trace decays are generic, not
selected for a task. Optional growth, pruning, and metaplasticity are ablated.
A temporary raw-experience replay buffer may supply sleep; it cannot answer
queries and is not counted as learned knowledge.

## Evaluation policy
All reported task scores are prequential: predict BEFORE exposing the answer,
then learn from it. No frozen inference mode for the candidate. Independent
runs initialize separate lifelong organisms; a run never resets at a task
change. Five initial seeds (0–4); confirmation seeds (10–19) are held back
from initial parameter exploration. Store settings, per-seed results, and
structure counts. Do not choose a seed or change success criteria afterward.
All ablations use the same stream and initial construction where applicable.
Learning-disabled and trivial predictors are clearly labelled controls.
A checkpoint clone may be probed, but probes must not alter the main lifetime.

## Initial suite
1. **Sequence:** four-symbol cycle, 800 experiences, then a permuted cycle
   for 800. Score the last 200 in each phase. Controls: no learning,
   last-symbol predictor, no hidden neurons, no traces, no hidden plasticity.
   Initial gate: mean accuracy >= .95 in both phases and >= .30 absolute
   advantage over no learning. This establishes elementary adaptation only.
2. **Delayed association:** independent binary cue, independently sampled
   distractors, query; target is exposed only AFTER the query prediction.
   Delays 0, 4, 16, 64; 600 trials, score last 200. Trivial chance = .5.
   Controls: no learning, no traces, no hidden neurons, no recurrence,
   no hidden plasticity. Gate: >= .90 at delays 0, 4, 16 and >= .20 advantage
   over no-learning; interpret delay 64 as exploratory. Also report log loss.
3. **Conflicting cues:** cue A, intervening experience, cue B, query: answer
   depends on the latest cue. This challenges accumulation traces; report
   independently from the basic delay gate. Target >= .90, not assumed.
4. **Continual interference:** learn mapping A, then its reversal B, then
   revisit A. Report adaptation/retention prequentially, comparing
   metaplasticity on/off. Contradictory mappings with no context cannot both
   be recalled: this is an adaptation-versus-inertia diagnostic, not proof
   of catastrophic forgetting. A later contextual task is required.
5. **Structure:** predictable vs independent random outcomes, growth/pruning
   on/off. Report births/deaths, size trajectory, accuracy and runtime.
   No fixed capacity ceiling; finite experiment duration is a safety bound.
   No claim of useful growth without an accuracy or loss benefit.
6. **Sleep:** sequential noncontradictory tasks, replay/no replay and equal
   replay during wakefulness. Match number/order of presentations and keep
   main streams paired. A benefit shared with wake replay is a rehearsal
   benefit, not evidence of sleep-specific consolidation.

## Next decisions
Do not touch real text unless the small tasks pass and the mechanisms are
causally supported. Failure on delay or interference directs the next local
rule experiment. A successful linear readout is not evidence of discovered
hierarchy, internal credit assignment, reasoning, or calibrated uncertainty.

## Follow-up registered after the initial delay failure (2026-10-10)

The v1 gate remains failed. Compare the same streams and seeds using (a) one
EMA trace per source (v1), (b) all four EMA timescales per source, and (c) all
four timescales with an event/peak trace. A peak trace is purely local:
`trace = pre` when `abs(pre) >= abs(decay * old_trace)`, otherwise the decayed
old trace. It preserves the amplitude of a brief event without a task-specific
memory variable. Compare optional normalization by the receiving neuron's
incoming trace energy; this scales learning amount, not its direction.
Keep learning rate .35; do not change the delay gates. Ablate hidden neurons,
traces, and hidden plasticity. Use seeds 0–4 to investigate; choose at most one
candidate before testing confirmation seeds 10–19. Report all attempted
configurations, including failures. Confirmation targets: >= .90 at delays
0/4/16, >= .20 over the disabled-learning control. Delay 64 remains exploratory.
A result achievable with no hidden neurons is synaptic temporal association,
not internal credit assignment or hierarchy.

## Competitive representation follow-up (registered 2026-10-10)

Initial contextual results: mixed-context rehearsal makes new-context accuracy
approximately chance, with a linear-only control behaving similarly. This
suggests missing context×cue representations, not simply insufficient replay.
Test a task-agnostic competitive dendritic rule: each hidden neuron responds
to mismatch between current sensory input and its own afferent synapses;
lateral competition selects the strongest response. A neuron's winning
activity gates local Hebbian movement of each afferent toward its source.
No task identifiers, target labels, or gradients enter the hidden rule.
Use the same contextual stream, initial 16 neurons, hidden rate .05, no
persistence or prediction traces (so current conjunctions are unambiguous).
Compare hidden learning off, no hidden neurons, and the v1 Oja network with
these same temporal settings. Test both single-context lifetime and alternating
rehearsal. Gates fixed before running: >= .90 on new context, >= .90 first-20
old-context recall with rehearsal, and >= .15 recall improvement over no-hidden.
Any success establishes small learned conjunctions only, not hierarchy or
compositional generalization. Investigate hidden-rate sensitivity with initial
seeds only; select one setting before confirmation seeds 10–19.

## Combined-rule and structural follow-up (registered 2026-10-10)

Exploration found peak traces + no hidden neurons pass gaps 0/4/16, whereas
16 leaky Oja cells degrade delayed prediction. Competitive cells with immediate
activity preserve both contextual mappings. Test the combination without
retuning: competitive hidden rule (.05), no recurrence or persistence, four
peak trace decays, unnormalized outcome learning (.35). Compare hidden cells
removed, hidden plasticity off, and traces off, first with seeds 0–4. If it
passes, select it for confirmation; otherwise confirm the simpler trace-only
candidate and report the failed combination. Delay gates are unchanged.

Prediction-surprise growth expands on irreducible noise without an accuracy
benefit. Test representation-surprise growth: a slow local/broadcast average
of the winning neuron's sensory mismatch gates exploration births; costs
increase smoothly with neuron count. Initial pool 1, growth every 20 observed
outcomes, threshold .07 and size cost .02, pruning on/off, same competitive
rule and immediate temporal settings. Compare fixed 1 and fixed 16 cells.
Contextual gate: new-context final accuracy >= .90 AND first-20 old-context
recall >= .90 without rehearsal. Random-label noise should not keep creating
neurons once sensory patterns are represented. Track all births/deaths.
This is a hypothesis about novelty, not an assumed solution to useful growth.

## Birth and consolidation follow-up (registered 2026-10-10)

Representation-based growth from one cell does not meet the no-rehearsal
retention gate (59% initial mean old recall). Test a newborn's afferents
initialized from locally present sensory activity rather than random weights,
and optional afferent metaplasticity. The latter uses only the winner's
own mismatch and recent source activity to accumulate stability evidence;
it reduces adaptation of established representations. Rate, growth thresholds,
intervals, and gates remain unchanged. Compare each mechanism alone and in
combination on seeds 0–4. Choose a setting before confirmation seeds 10–19.
A noise diagnostic must show no continual expansion after input novelty is
exhausted. Keep earlier failed growth results intact.

## Temporal conjunction grammar (registered 2026-10-10)

Confirmation shows frozen random competitive cells also pass the simple context
task: it establishes nonlinear separation, not necessary learned organization.
The next task presents two independently sampled binary symbols in separate
sensory alphabets, then a query; its legal terminal class is their parity.
A linear predictor over independent traces cannot represent this interaction.
Between episodes, eight blank experiences clear recent activity gradually;
there is never an internal reset. A configurable neutral-symbol gap separates
the two symbols. Score predictions before the terminal class is revealed.

Candidate: competitive cells (16), rate .05, immediate neuronal activity,
no recurrence, no prediction traces, local peak drive traces with decay .8.
The same local traces feed each neuron's dendritic mismatch and afferent
learning. No task-specific conjunction features or parity rule inside the
brain. Controls: no drive traces, hidden plasticity off, no hidden neurons,
no learning. Gaps 0 and 2: 1200 episodes, final 400, mean accuracy >= .90 and
>= .20 advantage over no-hidden. Gap 8 exploratory. Seeds 0–4 first, selected
setting then confirmation 10–19. Report whether hidden plasticity earns a
measurable benefit; success with frozen cells remains a random-feature result.
This finite grammar does not establish open-ended composition or hierarchy.

## Temporal representation recruitment (registered 2026-10-10)

The first grammar candidate passes gap 0 on average (94.4%) but fails gap 2
(60.1%): the gate remains failed. Test novelty-driven recruitment rather than
adding a task-specific interaction feature. A newborn copies locally active
parent afferent traces into its own traces and initial weights. When no neuron
matches within a generic mismatch tolerance, lateral competition may abstain
instead of overwriting an established prototype (vigilance). All operations
use sensory traces and neuronal mismatch, never the terminal label.

Settings: initial hidden 1; growth interval 1, representation threshold .01,
size cost .003; peak drive decay .8, hidden rate .05, vigilance .03. Compare
vigilance on/off, hidden plasticity off, fixed-one-cell, and no drive memory.
First isolate pruning off; then test pruning on. Keep the original grammar
gates (.90 at gaps 0 and 2, .20 over no-hidden), gap 8 exploratory. Seeds 0–4
first. Record neuron trajectory, so a growing lookup of numerous state
prototypes is not mistaken for compact compositional reasoning.
Vigilant competitive afferents treat correctly matched zero inputs as useful
constraints; pruning them purely for zero strength would invalidate mismatch
computation. That usefulness rule is enabled only with vigilance, preserving
legacy candidate definitions.

## Matched noise-growth control (registered 2026-10-10)

The original prediction-growth and novelty-growth runs differed in initial
size and learning rule. Add a matched control: same competitive rule, one
initial cell, rate .05, no temporal state, growth every 20 observations,
threshold .07, cost .02, pruning enabled; change only growth_rule from
representation to prediction. Predictable and random outcomes, 3000 steps,
seeds 0–4. This isolates irreducible prediction error versus input novelty.
Do not infer a rule benefit from the unmatched earlier size curves.

## Align growth with the unmet experience (registered 2026-10-10)

The query-only recruitment run was interrupted for excessive cost. A separately
reproduced 120-episode diagnostic records its partial trajectory and accuracy;
it is not a completed gate. Births were allowed only on labelled outcomes,
while persistent mismatch included every blank/prefix experience. Change the
growth clock to every experience and require CURRENT mismatch above vigilance
as well as persistent surprise. Keep .01/.003 pressure coefficients and .03
vigilance initially. Remove unused recurrent candidate edges (density zero)
for these explicitly nonrecurrent competitive runs, avoiding quadratic cost.
Use a 60-second per-run experiment budget, record incomplete runs explicitly,
and never count a partial run as a passed gate. Brain size remains uncapped.
Same grammar streams/gates, seeds 0–4, controls as before. Keep the failed
outcome-clock run and diagnostic intact.

## One lifetime across changed meanings (registered 2026-10-10)

The recruited grammar passed gaps 0/2 on ten confirmation seeds; gap 8 failed.
Pruning preserved accuracy on initial seeds but caused roughly 90–120 births
and almost as many deaths, versus about 15 births without pruning. Test a longer
single lifetime: same two-symbol grammar (gap 2), with an explicit sensory
context channel. Context 0 uses the original terminal mapping, context 1 its
reversal. Learn 1200 episodes in context 0, 1200 in context 1, then return for
400 episodes to context 0. Seven inputs, same selected neural parameters.
All context information is sensory; no internal task identifiers or resets.

Compare pruning on/off first (seeds 0–4), no replay. Record first 50 return
predictions separately from final relearning; gates: >= .90 new-context final
400 AND >= .90 first-50 old-context return. Do not call later relearning
retention. Checkpoints and structure trajectory must preserve the whole brain.
A failed return directs a consolidation/pruning experiment, not text scaling.

## Consolidation protects an old context (registered 2026-10-10)

Lifetime baseline: pruning-on mean first-50 return accuracy 86.4% (gate failed),
whereas completed pruning-off runs scored 100%. Pruning repeatedly kills
unlabelled prefix representations and prunes inactive old-context pathways.
Test four local structural safeguards: keep an active sensory representation
when its prediction pathways are unused; regrow missing prediction connections
from current local coactivity/error magnitude; accumulate afferent stability
for correctly matched absence as well as presence; protect consolidated
synapses from pruning using their own evidence. No labels enter hidden cells.

Registered settings: selected lifetime rules plus hidden/outcome metaplasticity,
match evidence, evidence decay .99999 per update, retention weight .001;
active-representation retention and prediction regrowth enabled (one candidate
per 10 observed outcomes). Compare pruning protection enabled/disabled with
other settings matched, no pruning, and all safeguards off (original baseline).
Five exploratory seeds, unchanged .90 new-context and first-50 return gates.
Confirm selected setting on seeds 10–19 if it passes. The evidence decay is
an explicit slow forgetting rate, not permanent immunity or a capacity cap.
A stability gain does not establish biological sleep; replay remains a separate
controlled scaffolding experiment.

Clarification for the safeguards-off consolidation control: other common
metaplasticity settings remain enabled; it is not numerically identical to the
original `lifetime_initial.jsonl` baseline. The protection-off control changes
only pruning protection. Configurations in each raw row are authoritative.
The selected lifetime configuration is frozen in `SELECTED_LIFETIME.json`
before confirmation on the independent lifetime streams at seeds 10–19.

## Diagnose the remaining long grammar failure (registered 2026-10-10)

Gap 8 still fails in the selected grammar while single cues survive 64 steps.
Test whether the hidden afferent traces still contain the two symbols. A clearly
labelled ORACLE baseline reads only pre-outcome trace snapshots, decodes each
sensory alphabet by its larger trace, and applies the task's terminal rule.
It never teaches, answers for, or changes SOMA. Gaps 8/16/64, five seeds, 1200
episodes: compare oracle accuracy with the brain. If the oracle succeeds while
the brain fails, representation resolution (not complete information loss) is
implicated. This is an information diagnostic, not a learnable competitor.

## Local afferent homeostasis (registered 2026-10-10)

The oracle decoded both symbols at 100% for gaps 8/16/64 on all five seeds;
SOMA stayed near chance. Information survives in traces but Euclidean matching
can ignore the weaker older signal. Test mismatch normalized by each afferent's
own incoming trace variance. Mean/variance update from its source trace only,
without target/error information. Newborns inherit locally available parent
trace statistics. Outcome learning and the task definitions are unchanged.

Settings: selected grammar rules, variance normalization rate .01, variance
floor .0001, no pruning for isolation. Gaps 0/2/8/16, initial seeds 0–4; controls
normalization off, scale statistics frozen (rate zero), and recruitment off.
Extension gate: >= .90 at gap 8 while retaining >= .90 at gaps 0/2; >= .20 over
normalization-off at gap 8. Gap 16 exploratory. Choose once before confirmation
10–19. Hidden-weight-off means afferent weights frozen, not homeostatic scale
statistics; test those separately. Preserve the original gap-8 failure.

## Combine longer temporal representation with lifetime retention (registered 2026-10-10)

All five initial homeostasis seeds pass gaps 0/2/8. Freeze the unchanged
configuration in `SELECTED_HOMEOSTASIS.json` before confirmation. Also test
compatibility with the selected consolidation/pruning rules in ONE longer
contextual lifetime at gap 8 (same 1200/1200/400 episodes, seven sensory
channels). Compare pruning protection off and normalization off. Initial seeds
0–4, then selected combination on confirmation 10–19 if it passes. Keep gates
>= .90 new-context final 400 and >= .90 first-50 old return; no resetting,
replay, or hand-designed task features. This distinguishes separately working
mechanisms from a working combination. Incomplete budget-stopped runs cannot
pass. Gap 16 is still a separate exploratory limitation.

Compute-budget note: the first combined lifetime attempt ran concurrently with
three CPU-heavy suites. Its 120-second wall budget stopped before the return
phase (partial records preserved in `lifetime_combined.jsonl`); another active
run was interrupted to avoid more uninformative partials. This is a compute
limit, not evidence of task failure. Rerun the unchanged model/streams/gates
with a 600-second budget after the other suites finish. The population remains
uncapped; do not count partial scores as successes or choose settings from them.

The extended-budget combined study completed all 15 runs: selected combination
100% new context, 99.2% old return (five seeds); no protection 80.0% old return;
normalization off did not learn the gap-8 mappings. Freeze the unchanged
combination in `SELECTED_COMBINED.json`, then confirm candidate and paired
protection-off control on streams 10–19 with the same 600-second budget.

## Confidence/familiarity diagnostic (registered 2026-10-10)

Use saved, completed combined-lifetime brains (initial seeds 0–4), clone each
checkpoint per probe condition so main lifetimes stay untouched. Probe 40
unlabelled episodes in familiar contexts 0/1 (gap 8), unseen contexts .5/-.5,
and familiar context 0 at unseen gaps 16/32/64. The brain still changes through
intrinsic rules; no outcomes are supplied. The evaluator knows labels only
for familiar integer contexts. Record accuracy where defined, max probability,
entropy, active query cell's prediction-synapse evidence, and novelty.
No thresholded abstention, word "unknown", or special uncertainty rule is put
inside SOMA. This is a qualitative diagnostic, not a self-knowledge success
claim. Confident errors or poorly calibrated novelty are recorded explicitly.

## Pass 2: local future prediction and unseen combinations (2026-10-11)

Register before running the new task. The old parity grammar is not a test of
composition: every combination occurs. Here two independent four-valued
factors emit a sum of two randomly oriented sensory directions in 24 dimensions.
Fresh independent nuisance occupies the orthogonal complement on every view.
Three noisy views are followed by a generic query marker. The binary outcome
is the sign of the sum of the two factor values (environment only). Thus each
factor contributes an additive, learnable part. The sensor directions and
factor level order are seeded; the brain receives neither factor identities
nor factor values. Query outcomes are the only labels.

Four of 16 combinations are withheld: factor indices (0,1), (1,3), (2,0),
(3,2), using values -3,-1,1,3. All factor values occur during learning; withheld
labels are balanced. No withheld combination is shown even without a label
before the test. Train 600 episodes sampled uniformly from the other twelve;
then test 40 episodes per withheld pair in a shuffled order, with ZERO outcome
feedback. Intrinsic input learning continues during testing; additionally
record the first occurrence of each withheld pair and immutable-checkpoint
single-pair probes to distinguish adaptation from initial transfer. Report
seen-combination unlabelled probes from separate clones, not further training.
A pure exact-template memorizer without a compositional rule has no learned
answer for any withheld conjunction: its unknown-template prediction is chance.
The actual selected competitive pool has sensory shortcuts/nearest matching;
run it as a stronger empirical control rather than assuming it must fail.

New module candidate: eight initially distributed predictor neurons, each
with an independent fixed random sensory observation dendrite. A predictor
receives a projected current sensory activity, but its basal synapses predicted
that activity from the PREVIOUS sensory state. Its own error updates its own
basal synapses with their local eligibility. Outcome neurons read the emitted
prediction, never raw sensory input. The observation population supplies
activity, never another neuron's error. Prediction is produced before current
sensory observation and before any outcome update. This is self-supervised
next-input prediction; there is no gradient or feedback from the outcome.

Initial settings: 24 sensory coordinates plus one marker; nuisance amplitude
2, signal amplitude .6; hidden step .2 divided by its own afferent energy,
outcome step .2 with analogous local normalization. Observation dendrites
have random weights scaled by input dimension; basal weights initialize at
random with scale .1. Initial growth pressure .5 + .2 log(1 + population),
interval 200; pruning interval 200, age 2000, usefulness threshold .00001.
Use costs/rates, not population caps. Treat these settings as a hypothesis;
record any follow-up before trying different rules or settings.

Required paired controls: original selected competitive organism
(`SELECTED_COMBINED.json` unchanged, interface dimensions adjusted), hidden
predictor updates frozen at birth, shuffled observation signal, matched-capacity
nonpredictive random projections, and chance/majority/exact-template baselines.
The three new-module controls receive exactly the candidate's birth/death
schedule and identical initial wiring, so population and edge counts match
throughout. Shuffling uses a separate RNG, preserving initialization and
structural RNG. A capacity control uses the same neurons and dendrites but
fixed random basal synapses without a prediction rule; it is deliberately
redundant with frozen learning when no structural events occur.

Exploration seeds 0–4, confirmation 10–19 only after a frozen selection.
Pass: mean withheld accuracy >= .90 AND >= .10 absolute over EVERY required
neural control, seen accuracy >= .90, and advantage on the immutable initial
transfer probes too. Individual seed ranges are reported; require at least
9/10 confirmation seeds >= .85. No claim of hidden credit assignment or
compositional generalization without all controls passing on confirmation.
A predictive loss decrease alone establishes local input prediction only.
Per-run budget 600 seconds, incomplete runs never pass. Save only diagnostic
states; preserve every result. No real text in this pass.

### Composition follow-up: reduce noisy plasticity (registered 2026-10-11)

Initial distributed predictor: withheld mean 80.875%, seen 80.833%, immutable
first-combination probes 80%; frozen/matched 53.625%, shuffled 50.5%. Gates
FAILED. The receiving error includes unpredictable observation noise; large
constant steps chase that noise. Test the same network/task/gates with hidden
step .02 (one tenth), then .05 (one quarter), keeping outcome step .2 and all
600 episodes unchanged. These are exploratory seeds 0–4 only. If neither
meets the gates, test .02 with 1800 episodes and matched streams/controls;
more exposure is a separately recorded condition, never a rescue of the
600-episode failure. The competitive control must be run at every selected
condition, not assumed weak. There is no structural/event change.

### Control for activity scale (registered 2026-10-11)

The frozen random basal weights initialize at one tenth the observation scale.
Poor frozen readout learning could therefore reflect amplitude, not meaningful
prediction. Before interpreting a hidden-learning benefit, add a stronger
control and a separately recorded candidate: local output-axon homeostasis.
Each predictor tracks ONLY its own emitted activity mean/variance (rate .01,
floor .0001); normalize its axon for outcome synapses. This never uses labels or
another neuron's error. Enable it in ALL new-module variants, preserving the
original default/runs. Test .02 and .05 hidden steps, 600 episodes, then .02 at
1800 if needed. Keep every accuracy gate and the noise/task unchanged. Report
these as separate conditions, not an edit to the failed initial experiment.
Also record genuinely frozen-hidden single-combination probes: no hidden
updates during the probe, no cumulative test adaptation. The normal test
continues intrinsic learning without outcome feedback as before.

### Stronger matched-capacity control (registered 2026-10-11)

The initial frozen and matched-capacity controls were numerically redundant.
Retain both raw records and their original definitions. Add a distinct
`hebbian_capacity` control: identical populations, synapses, birth/death
schedule, axon normalization and step, but local Oja afferent adaptation rather
than predictive error. Its update uses only its own source trace, receiving
activity, and strength. It asks whether generic unsupervised plasticity,
rather than temporal prediction specifically, explains transfer. A selected
candidate must also exceed this control by .10, with the same gate on initial
checkpoint probes. Run with the .02 homeostatic condition on seeds 0–4 before
selection. Do not claim necessity from a redundant frozen comparison.

### Representation geometry diagnostic (registered 2026-10-11)

Before confirmation, inspect learned predictor afferents without teaching or
modifying the brain. The evaluator knows the environment's two sensory
emission directions. Measure the fraction of each basal neuron's sensory
weight energy in their two-dimensional span versus the orthogonal nuisance
span, for .02/homeostatic candidate and frozen, shuffled, Oja controls.
This is a labelled diagnostic of learned denoising, not an internal decoder,
not a new accuracy gate, and not evidence that neurons discovered named factors.
Use the exact 600-episode streams, seeds 0–4; no new task or tuning.

### Linear memory baseline and the scope of a positive result (2026-10-11)

The proposed predictor and outcome readout are linear between sensory state
and logits (axon centering/scaling changes that affine map, not its depth).
Even a passed task cannot establish nonlinear hierarchy. Add an explicitly
labelled experiment-only linear neural readout over the previous sensory state,
with the same local .2 outcome step and per-afferent activity normalization.
It has no hidden prediction, no factor decoder, and no task rule. Use identical
600-episode streams, seeds 0–4, later confirmation if a candidate is selected.
This baseline is a diagnostic of whether supervised linear learning suffices;
if it matches the candidate, do not attribute task solvability to hidden credit
assignment. Preserve the registered required-control gates, additionally report
this stronger comparison and avoid claiming necessity if it is not exceeded.

### Probe precision (registered 2026-10-11)

The original independent-checkpoint metric samples just four first views, so
retain it and its gate unchanged but supplement it with all 160 held-out noisy
views, each from its own identical checkpoint clone with basal updates,
structural updates and axon-statistic updates disabled. No target is supplied.
This supplementary metric distinguishes transfer from intrinsic adaptation
and reports uncertainty from 160 probes per seed. It is not used to replace a
failed original four-probe gate. Restrict this expensive clone diagnostic to
new-module controls; the old pool retains its original four boundary probes.
Add a deterministic stream hash and assert training/test pair disjointness.

### Runtime check (2026-10-11)

The old noisy prototype control is expensive. A trial of the unchanged
standard-library reference under PyPy 3.10 preserved seed-0 scores and
structural counts but FAILED exact brain/trajectory equality; a short neural
state comparison also differed. Reject it for the paired scientific study.
Keep CPython as the study runtime; the original twelve historical CPython
receipts still match exactly. The rejected interpreter receipt is a technical
failure record, not evidence that a model/task gate passed.

## Pass 2 phase C: self-knowledge precursor, contingent diagnostic (2026-10-11)

Only run after B is confirmed or clearly blocked; this is the ONE C direction.
Clone the .02/homeostatic trained brains (initial seeds 0–4) per condition.
Continue intrinsic learning, supply NO outcome labels during 200 probe episodes
per condition. Conditions: familiar training combinations, withheld combinations,
same familiar inputs with nuisance amplitudes 4/8, reduced signal amplitude .2,
and unchanged familiar inputs whose evaluator labels are reversed. The last
condition exposes an information limit: an unseen outcome rule change cannot
be inferred from identical unlabelled inputs.

Measure pre-outcome hidden prediction residual on the three views, query
residual, outgoing probability entropy, confidence, accuracy and log loss.
All labels are evaluator-only. Measure ROC AUC for errors within each condition
(pooled per-seed ranks to avoid interpreting between-brain scale shifts), not
just across easy/hard distributions. Report error counts and undefined AUCs
when there are no correct/incorrect examples. No hard-coded abstention or
unknown policy, no threshold selection, no success claim of self-knowledge.
This qualitative study checks whether these local signals are plausible
precursors and records confident errors, including the impossible label shift.

Provenance note: initial long-running competitive trials loaded the initial
runner while reporting-only code evolved for new-module probes. Their hashes
were captured from disk at completion, so the runner hash does not guarantee
the executed snapshot. The stream generator and original organism were never
changed, and configs/seeds remain authoritative and numerically reproducible.
For confirmation, capture all source hashes at module import and freeze source
until every trial finishes. This is a provenance limitation of the initial
records; do not edit those records retroactively.

Clarification: factor level values/order are fixed (-3,-1,1,3); sensory
directions are seeded. The initial phrase "factor level order is seeded" was
inaccurate. The generator never permuted level order; brain inputs never
included factor indices. No data, task definition, or gate changes.

Confirmation uses two independent CPython worker processes (two CPU cores are
available). Each worker runs all controls for a seed, including the original
competitive pool, with the unchanged 600-second per-run budget. Neither brain
nor stream state is shared between seeds. Source is frozen from dispatch until
completion. No additional candidate tuning after confirmation starts.

Before confirmation, extend exact structural yoking through the primary
unlabelled test stream as well as training. Earlier settings had zero test
births/deaths, so actual capacities already matched; the implementation now
also enforces it if a structural event occurs. Independent checkpoint probes
freeze structure for every new-module variant. Record sparse nonzero events
plus a full schedule hash instead of thousands of identical zero-event rows.

Final audit correction: exploratory Oja seed 0 grew a ninth unit during its
unlabelled test. The earlier "zero test events" statement applied to the
original candidate/frozen/shuffled controls, not all later Oja trials.
Full-test structural yoking was implemented before confirmation and all final
new-module capacities/schedules match exactly. Earlier raw data remain intact.
Adding Oja also regrouped floating arithmetic in the new kernel; exact early
replay requires its recorded source version. Five pinned-source case checks
pass, including the source-frozen confirmation. The twelve legacy current-code
replays remain exact. No model selection or gate was changed during confirmation.


## Pass 3: nonlinear prediction and the start of unification (2026-10-11)

Registered before implementation and execution. Start from reviewed 6f92673.
The new substrate is `brain/soma.py`; both specialists remain unchanged.

### A: environment and information boundaries

Two independently chosen factors have values (-3,-1,1,3), emitted along two
seeded orthogonal directions in twelve raw coordinates with orthogonal fresh
nuisance (uniform amplitude .3). The query outcome is sign(x*y): a nonlinear
interaction, exclusively computed by the evaluator. Hold out (0,1), (0,2),
(3,1), (3,2), all unseen even without outcomes in training. Their factor values
and both signs of the interaction occur in other training combinations.
Every pair has its central inversion with the same label; both classes have
zero mean and equal mass. Consequently the population logistic optimum for
any affine raw-factor classifier is constant/chance. An affine threshold can
fit at most three of the four noiseless withheld centers, not all four.
Report an experiment-only deterministic affine-logistic symmetry check and
actual supervised linear and pass-2 predictive controls; do not assume their
finite online accuracy is exactly chance. An exact seen-template lookup has
no entry for any withheld pair (chance); no lookup is installed in the brain.

Each episode: eight blank steps, two factor views, a third factor view carrying
a query marker (thirteenth coordinate), then a future sensory view along a
third independent random direction with amplitude x*y/3. The future view is
AFTER the query and its prediction; it is never available to a query's linear
baseline or outcome neuron. Its magnitude varies continuously with the factor
product; it is sensory predictive structure, not an outcome label channel.
No future-view outcome labels are supplied. Prediction synapses learn from
this future observation on the next tick, from their own stored eligibility.
A paired predictable-nuisance condition replaces the future view with
(x*x+y*y)/6: predictable nonlinear magnitude unrelated to the sign outcome.
Query input and label distributions are identical between conditions.

Train 1200 episodes on the other twelve pairs; test 160 episodes (40 each
withheld pair), no outcome feedback. Seen probe: 120 from checkpoint clone.
Also probe every withheld episode from a single checkpoint with all plasticity
and structure frozen, sequentially with the usual blank steps. Intrinsic
learning continues in the primary test. Test and training streams hashed.
Seeds 0–4 exploration; only after a selected hypothesis is written, 10–19
confirmation. Per-run 300 second budget; partial runs never pass. At most
three preregistered exploratory hypotheses before freezing the best understood
hypothesis, even if blocked, for a diagnostic confirmation without tuning.

### A: candidate and controls

Initially four neurons, each four generic saturating dendritic branches. A
branch combines normalized local sensory event traces and a bias, tanh
nonlinearity, and a fixed excitatory/inhibitory polarity at its own soma.
The soma predicts its observation projection one step ahead. Each branch
synapse updates from its own previous presynaptic trace, its receiving soma's
own error, its fixed local polarity, and its own unsaturated activity gate
(1-abs(branch activity)). No derivatives of another cell, trainable weight
transport, outcome errors in hidden synapses, or gradient through populations.
Hidden step .05; outcome step .3. Axon/source normalization .01, event peak
decay .8, variance floor .01. Local prediction-error/branch-activity covariance
(EMA .01) gives structured-surprise pressure, unlike raw squared error.
Recruitment interval 120, pressure threshold .02 + .01 log(1+population).
Pruning interval 240, age 1200, prediction usefulness below .015, and local
evidence protection; evidence decays .999. All are rates/costs, no size cap.
These initial settings are hypotheses, not earned claims.

Required paired controls: pass-2 linear predictor selected settings (only
interface dimensions changed); supervised linear memory readout; frozen
nonlinear hidden weights with active homeostasis; shuffled hidden observations
using separate RNG; matched-capacity Oja branch adaptation without prediction
error; growth off with initial population equal to candidate final population
and pruning off; chance. Frozen/shuffled/Oja use identical wiring and exact
candidate birth/death schedules throughout train and primary test. No neuron
counters are equated with specialist counters without explaining them.

Unchanged gates: mean held-out >=.90, seen >=.90, >=.10 advantage over EACH
required neural control including fixed-final-size, >=9/10 candidate seeds
held-out >=.85. Frozen-hidden held-out >=.85 and >=.10 over frozen features.
At least one birth AND one death in >=9/10 candidate confirmation lifetimes
for the living-structure claim; report every lifetime. Mean linear-control
accuracy must be <=.60; otherwise diagnose the task leak/imbalance before any
nonlinear claim. A failed gate stays failed. Nonlinear learned-composition
claim specifically requires beating random nonlinear and linear controls.
Predictable-nuisance result is a specificity diagnostic, not a gate rescue.

### B: one substrate, one configuration

Use exactly the selected phase-A `SomaConfig` in independent lifetimes, changing
only input/output counts and seed for each interface. No specialist delegation
inside the unified brain. This substrate contains peak traces, source variance,
nonlinear local prediction, recruitment, and evidence-protected pruning.
Compare to specialists on phase-A task, original pass-2 task (600 episodes),
original grammar gap8 (1200 episodes, final400), original contextual gap8
lifetime (1200 A,1200 B,400 A return). Report every specialist/unified score,
including initial-A final400 acquisition. Acquisition >=.90 is required before
a retention conclusion; failures are reported with return scores but are not
counted as forgetting/retention successes. Seeds 0–4 for the cross-task
comparison, no task-specific retuning. Structural and temporal ablations on
this same configuration are diagnostic, not new selections.

### C: only after A and B complete

Choose the longer-time direction, NOT sleep. Same unified configuration,
fresh gap16 grammar, seeds0–4, 1200 episodes and final400. Compare gap8 and
trace-off, report failure without retuning. No endogenous replay claim.

### Provenance

Source hashes captured at import and source frozen during each dispatched
experiment batch. Freeze/commit each hypothesis before launching its runs.
Keep every result and source-specific replay receipts. No historical raw files,
original kernel behavior, selections, or vision edits. Readable stdlib brain.


### Pass 3 structural and information diagnostics (before running)

After the complete initial five-seed batch, retain the initial hypothesis if
its mechanisms are interpretable even if the all-controls gate is blocked by
a strong fixed-final-size model. Do not tune to make that control weaker.
A task-specific nonlinear success is distinct from demonstrating a benefit of
growth. Confirmation will keep the complete original gate and report its failure.

Additional structural diagnostic with the same selected configuration, seeds
0–4: 4000 unlabelled sensory ticks, uniform independent coordinates in [-1,1],
versus the same coordinate generator refreshed only every four ticks. Record
all births/deaths and trajectories. This probes whether correlation pressure
actually suppresses irreducible sensory noise; do not infer it from the rule's
name. It is an additional failure/success diagnostic, not a selection gate.

Information caveat: an unlabelled future product view nonetheless reveals the
interaction property after each query. Continued intrinsic test learning can
therefore improve on earlier test pairs. The frozen-weights/statistics probe
is essential evidence of already learned transfer; primary accuracy alone
is not proof of zero-adaptation composition. Keep both scores prominent.
An affine threshold can incidentally score 75% on four XOR centers despite
the population affine-logistic optimum being chance; observed online linear
controls, their seed ranges, and the original <=.60 mean gate remain decisive.


### Pass 3 hypothesis 2: signed underfit, before execution

The original35 runs fail the all-controls gate, despite99.125% primary task
accuracy. Independent and repeated noise both trigger all33 possible births.
Absolute residual/branch covariance counts the negative self-prediction term
as novelty. Add an ablatable `pressure_rule=underfit`: each soma maintains
EMA(error * own previous forecast), growth uses only its POSITIVE part. This
asks whether an existing forecast systematically underestimates future input,
rather than interpreting overprediction on noise as missing capacity. Keep
branch/outcome learning unchanged. Set growth threshold .001 and size cost
.002 (signed statistic has smaller scale); same120-tick birth interval.
Local predictive evidence increment .001 rather than .01 (matching .999 decay
in a bounded EMA); unchanged age/threshold/pruning240. No population cap.
Run all original controls on0–4 at1200episodes, plus a stronger temporal shuffle
that pairs each sensory episode's teacher with another randomly permuted
episode's observation (no outcome labels). The original population shuffle
and all gates remain required; do not replace their failed comparison.

Observation replacement is an experimental control interface only. Candidate
always learns its own actual observation. Temporal shuffle is generated in
experiments, with independently permuted training/testing/probe teachers,
identical actual sensory streams and candidate structural schedules. It
destroys both individual and population-mean factor/teacher alignment.
Repeat4000-tick independent/repeated noise under hypothesis2. Select the
best-understood hypothesis after both complete batches, prioritizing noise
regulation over an inflated score from continuous recruitment. If blocked,
freeze and confirm a diagnostic hypothesis without relaxing any gate.


### Pass 3 hypothesis 3: teacher covariance minus forecast covariance

Registered after the complete40-run hypothesis2 failure and before execution.
Candidate61.875%, with two extinct lifetimes; independent noise zero births
and four deaths, repeated inputs4–11 births. Signed forecast/error pressure
misses structure when the forecast has not formed. For each branch separately,
keep centered EMA(observation × previous branch activity) and EMA(previous
forecast × previous branch activity). Structural pressure is the positive
part of abs(observation covariance) minus abs(forecast covariance). Thus a
branch's sensory relation can demand capacity without a successful existing
forecast; self-generated prediction variance on independent noise does not
by itself count as unrepresented structure. All terms are local to the
receiving neuron/branch. No outcome information enters pressure.

Use original threshold .02/cost .01/120 interval, evidence increment .001
with pruning threshold .0015 (both scaled tenfold relative to hypothesis1),
same240 interval/1200 age/.999 decay. All learning/task settings unchanged.
This is the third and final exploratory hypothesis. Run the complete eight
controls on0–4 and independent/repeated noise before freezing. No further
candidate tuning. Prefer the interpretable noise-regulating candidate if
its task transfer survives; otherwise explicitly confirm a blocked hypothesis.
All original gates, including fixed-final and population shuffle, remain.

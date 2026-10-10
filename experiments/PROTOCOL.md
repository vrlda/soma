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

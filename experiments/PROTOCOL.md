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

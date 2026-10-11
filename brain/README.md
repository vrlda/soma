# The first experimental SOMA substrate

`organism.py` is a readable Python standard-library reference. It has sensory
neurons, hidden neurons, outcome neurons, and sparse drive/prediction synapses.
The small experiments fix their sensory/outcome interfaces; the internal
population has no size ceiling. Growth is limited by time, cost, and local
surprise. This is an experimental brain, not a language model or a finished
architecture.

## Experience and locality

`experience(signal, target=None)` updates activity, makes a prediction, and
only then exposes the optional outcome to its receiving neuron. Hidden cells
never receive the outcome error. `gate` is a scalar broadcast controlling
learning amount, not the desired synaptic direction. The optional observer
receives copied pre-outcome activity/trace diagnostics and cannot supply a
learning signal through the interface.

| Rule | Information it uses |
|---|---|
| Leaky neural activity | Afferent drive, previous activity, own persistence/threshold |
| Averaging or peak synaptic trace | Source activity and the synapse's previous trace |
| Local outcome update | Own source trace × receiving neuron's own outcome error × scalar gate |
| Oja afferent update | Source activity, receiving activity, own strength |
| Competitive representation | Each neuron's own sensory mismatch, generic lateral competition |
| Competitive afferent update | Source/afferent trace, receiving activity, own strength/plasticity |
| Afferent variance normalization | The synapse's own source-trace mean and variance, no outcomes |
| Recruitment | Current/persistent representation surprise, smooth size cost, elapsed interval |
| Newborn afferents | Local source activity or parent afferent traces; no task labels |
| Metaplasticity | Synapse-local matching/use evidence and an explicit decay rate |
| Pruning protection | Synapse age, use, and its own consolidation evidence |
| Connection regrowth | Current source activity and receiving outcome neuron's local error magnitude |

The candidate does not differentiate a global loss through the hidden network.
Outcome updates are local delta rules. Competitive cells are primitive sensory
pattern representations; they are not words, grammar rules, or hand-built
semantic circuits. All task rules are exclusively in `experiments/`.

`Config` keeps the initial failed rules available for reproducibility and
ablations. The selected experiment configurations are in
`experiments/SELECTED*.json`; the original defaults deliberately reproduce the
failed initial candidate. Do not assume a default `Organism(Config())` has all
of the later experimental rules enabled.

## Lifetimes and inspection

`summary()` reports population, connections, births/deaths, activity,
plasticity, and surprise. `state()` returns an independent snapshot, including
all neural/synaptic state, structural counters, and the random generator.
`save(path)` writes an atomic JSON checkpoint; `load(path)` resumes it with
exact continuation under the same implementation/configuration. The external
experiment's data generator has its own seed and is not part of the brain.

There is no inference/training switch in the candidate. Missing outcomes still
update activity, local afferent learning, and experience-clock growth. The
learning-disabled option is an explicitly labelled experimental control.

## What is still missing

The rules have not established deep credit assignment, an emergent hierarchy,
compositional generalization, conversational learning, or self-knowledge.
Some successful tasks also succeed with frozen afferent weights: those tasks
support recruitment/nonlinear activity and local outcome learning, not a claim
that each plasticity mechanism is necessary. Sleep-labelled raw-data replay is
currently experimental scaffolding matched against wake replay, not an
internal neural replay mechanism. Read the notebook before choosing the next
rule or scaling a task.

The selected competitive candidate is one representation pool with local
outcome connections. It cannot establish a hierarchy just by making the pool
larger; additional local prediction/connectivity mechanisms must earn that
claim in later experiments.

## Pass 2: distributed next-input prediction

`predictive.py` is a separate reference hypothesis, preserving all existing
`Organism` defaults and historic results. It contains two interacting hidden
populations: fixed, random sensory observation dendrites and distributed
predictor neurons. A predictor emits a forecast of its observation partner's
current activity from the previous sensory state. It then receives that
activity and updates its own basal synapses from its OWN mismatch. The partner
supplies activity, never its error. Labels reach only outcome neurons.

| New rule | Local information |
|---|---|
| Observation dendrite | Fixed random afferent strengths and current source sensory activity |
| Hidden future prediction | Own basal strengths and previous source activities |
| Predictive basal adaptation | Own source eligibility × receiving neuron's own forecast error, normalized by its own afferent energy |
| Output axon homeostasis | Own emitted activity mean and variance; no label or other neuron error |
| Local outcome learning | Source axon × receiving outcome's own error and scalar learning gate |
| Population growth | Broadcast scalar average of local surprise, elapsed interval, smooth cost |
| Connected-unit pruning | Own age/usefulness; removes associated observation/prediction cells and all their connections |
| Oja capacity control | Source trace, receiving activity and own strength; no predictive mismatch |

The readout consumes predictions, not raw sensory inputs. Axon normalization
is local homeostasis, not a directional global signal. Growth can continue
without a population cap, while its costs and cadence regulate speed. Structural
controls can replay the candidate's exact birth/death schedule; that yoking
exists solely to compare matched capacity and is not a candidate learning rule.
Saved states include both RNG streams and every synapse/neuron's history.

This architecture does not transport outcome credit into hidden neurons,
learn deep hierarchies, or discover symbolic factor identities. The question
is whether its self-supervised predictive features become useful for unseen
combinations. Read the protocol and notebook for the gates and failures.

For counting, a "unit" pairs one observation neuron with one learned predictor;
`hidden` and birth/death counters count these pairs, while `neurons` includes
both populations. `synapses` counts weighted afferents and output connections;
the fixed observation-to-partner activity coupling is implicit. These counters
must not be equated with the original organism's single-cell birth counts.

## Pass 3: one candidate substrate, nonlinear dendrites

`soma.py` is a new single substrate rather than a wrapper around either
specialist. Both earlier kernels remain intact. Each hidden neuron contains
an observation dendrite and several saturating predictive dendrites; every
forecast and outcome uses this same neuron/synapse population. All interface
sizes are sensory/outcome counts, never task identifiers.

| Rule | Local information used |
|---|---|
| Peak event trace | Source sensory activity and its own previous trace/decay |
| Source variance normalization | Source trace, its own mean and variance |
| Saturating dendrite | Its own afferents, bias and tanh activity |
| Predictive soma | Its own branches and fixed excitatory/inhibitory polarities |
| Dendritic adaptation | Own previous presynaptic eligibility, own soma prediction error, own polarity, own unsaturated-activity gate, scalar learning amount |
| Axon homeostasis | Own soma activity mean and variance |
| Outcome adaptation | Own source axon and receiving outcome neuron's own error; no feedback to hidden weights |
| Structured surprise | Own residual × own branch activity covariance; max magnitude is a broadcast structural-pressure scalar |
| Recruitment | Persistent structural pressure, elapsed interval and smooth population cost |
| Connected-neuron pruning | Own age, predictive coactivity/usefulness and decaying local predictive evidence |
| Oja capacity control | Own branch pre/post activity and strength, no predictive error |

The fixed branch polarities are part of the neuron's local wiring. No learned
outcome weights are transported back; no error from another neuron is
propagated or differentiated. The saturation gate is `1-abs(activity)`, a local
plasticity rule, not a gradient through populations. Input observations teach
the previous forecast before the current forecast/readout is formed. Only
then may an outcome label change the outcome synapses.

`SomaConfig` is shared across interfaces; task-specific selection is prohibited
in the cross-task study. `plasticity=False`, frozen hidden learning, shuffled
observations and structural yoking are explicit experimental controls, not
separate candidate inference policies. Source histories continue during frozen
probes as temporal state, while all strengths/statistics/structure are fixed.

Counts: a Soma hidden count is one neuron with all its dendrites. Births/deaths
exclude initialization and count complete connected neurons. Weighted sensory,
branch and outcome edges are counted; fixed branch-to-soma polarities are
implicit and their number is reported by `branches` in the configuration.
Removing a neuron removes all its edges. No internal population cap exists.

This starts unification of the principles; it does not claim numerical
identity with the specialists or inheritance of their measured successes.
Observation projections remain fixed in this hypothesis. Synaptic evidence
is checkpointed for inspection; structural protection currently uses each
neuron's predictive evidence, not the old organism's exact afferent safeguards.
Growth must earn a benefit against a fixed-final-size predictive control.
Check the report before interpreting task-specific hidden learning as a
universal brain, hierarchy, biological sleep, or deep outcome credit assignment.

The frozen diagnostic selection uses `pressure_rule="covariance"`: each branch
keeps centered observation/activity covariance and forecast/activity covariance.
Only observation covariance exceeding the explained forecast covariance
contributes positive structural pressure. Both originate inside the same
receiving neuron and branch. Earlier absolute-covariance and signed-underfit
rules remain ablations with preserved failures. Predictive evidence increment
and pruning threshold scale together. This greatly reduces independent-noise
recruitment over the measured4000 ticks, not a proof of asymptotic regulation.
An externally replaced observation signal is ONLY the episode-shuffled control;
it never supplies an answer to the candidate. See the failed confirmation gates.

## Pass 4: shared forecasts and persistent conjunctive branches

`SharedSoma(SharedConfig(...))` is the pass-4 hypothesis in `soma.py`.
The original Soma/SomaConfig methods above remain unchanged for exact pass-3
replay. SharedSoma instantiates one neural graph directly, without importing
or selecting a specialist. Interface counts and seeds never select a rule.

| New rule | Exactly which local information it uses |
|---|---|
| Linear future forecast | Own basal afferents' prior sensory activity and own branches' prior activity |
| Predictive update | Own observation projection minus own emitted forecast, own presynaptic eligibility, scalar gate, own incoming energy |
| Receptive-field matching | Each branch's own source trace, own afferent strength and source-local variance; raw/predictive distance mixture |
| Lateral competition/vigilance | Competing branches' mismatch scalars; a winning branch emits activity only when matched |
| Provisional persistence | One provisional dendrite's own afferents, current source traces and consecutive matching count; no example list |
| Targeted branch growth | Persistent/current sensory mismatch and cost/cadence; competition selects the soma with greatest own residual-energy scalar |
| Newborn branch | Locally available provisional afferents, initialized branch-to-own-soma and outcome strengths zero |
| Branch predictive current | Own prior compartment activity × own parent's next-input error; no error from any other parent |
| Matching adaptation/consolidation | Own source trace minus own afferent strength, own winning activity and local matching evidence |
| Outcome learning/regrowth | Source compartment activity and receiving outcome neuron's own error; labels stop here |
| Branch pruning/protection | Own activity usefulness, age and matching evidence; removes all incident branch edges |
| Outcome-edge pruning/protection | Own strength/use, age and local coactivity evidence |

Predictive neurons' observation projections are fixed; basal prediction and
branch currents learn continuously. Nonlinear receptive branches expose their
compartment activities to outcome synapses, as well as a current to their own
soma. This is an explicit compartment-output simplification, not ordinary
single-axon anatomy. It does not propagate a gradient or a label into hidden
plasticity. A provisional branch is a single plastic neural state, not a raw
experience buffer. Its replacement count is reported alongside births so
unsuccessful attempts remain observable.

Counts distinguish fixed initial predictive neurons from living branches:
`hidden` counts predictive somas, `branches` counts represented compartments,
`births/deaths` count branches, and `provisional_branches` is one scratch
compartment. Weighted edges include observation, basal, outcome, matching,
branch-to-soma and provisional afferents. No cap applies to branch population.
The initial branch/layout setting is initialization; layout replay is ONLY an
experiment's random-wiring matched-final-capacity control, never candidate
knowledge. Candidate growth must earn a benefit, and learned prediction must
beat its frozen-feature control before being called necessary.

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

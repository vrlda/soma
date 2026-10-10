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

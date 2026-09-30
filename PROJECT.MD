# SOMA — Self-Organizing Morphogenic Architecture

> **Role of this document:** the founding research vision and design
> principles. It is not a status report or a work plan. Current status and
> the ordered next work are in
> [`SOMA_REAL_MODEL_MASTER_PLAN.md`](SOMA_REAL_MODEL_MASTER_PLAN.md)
> (Sections 16 and 22). The "Instructions for Codex" section applies to any
> coding agent. The synthetic-kernel work that pursued its first
> milestones is recorded in the [history](SOMA_SYNTHETIC_KERNEL_HISTORY.md).

SOMA is an experimental artificial intelligence research project focused on building a fundamentally different kind of learning system from conventional Transformers, LLMs, and fixed-topology neural networks.

The core idea is to explore whether an artificial computational system can behave more like a biological nervous system: continuously learning during its lifetime, changing not only numerical connection strengths but also its own structure.

SOMA should not be treated as a smaller Transformer, an LLM with extra memory, a Mixture-of-Experts model, or a conventional neural network with occasional pruning. The goal is to investigate a new architectural paradigm in which the computational substrate itself develops over time.

The system should be capable of forming new connections, weakening and removing old ones, creating new computational units when additional capacity is required, deactivating or deleting redundant units, consolidating important knowledge, selectively forgetting unimportant information, adapting its own plasticity, and reorganizing itself throughout its lifetime.

The long-term ambition is to investigate artificial intelligence based on principles associated with biological development, plasticity, homeostasis, memory consolidation, structural adaptation, and lifelong learning.

This does not mean attempting to perfectly simulate a biological human brain. Neuroscience does not currently provide a sufficiently complete computational description of the brain for that to be possible. Instead, SOMA should use biological principles as inspiration for creating a computational architecture with capabilities that conventional fixed neural networks struggle with.

The central research question is:

Can a computational system autonomously develop, reorganize, grow, prune, and consolidate its own structure while continuously learning from experience without requiring conventional end-to-end retraining?

## Fundamental distinction from conventional neural networks

A conventional neural network can approximately be described by a fixed parameter vector:

$$
\theta \in \mathbb{R}^{n}
$$

Training changes the values inside that vector, but the architecture usually remains fixed.

SOMA should instead be represented as a changing computational organism:

$$
S_t = (N_t, E_t, X_t, P_t, M_t, R_t)
$$

where:

* \(N_t\) is the set of currently existing computational cells or neurons.
* \(E_t\) is the set of currently existing connections or synapses.
* \(X_t\) contains the internal dynamic states of the cells.
* \(P_t\) contains plasticity and metaplasticity states.
* \(M_t\) contains global or regional modulatory signals.
* \(R_t\) represents resources, energy budgets, or homeostatic constraints.

The model evolves through experience:

$$
S_{t+1} = F(S_t, input_t, action_t, outcome_t)
$$

This state transition is itself the learning process.

Learning therefore does not mean only changing weights.

Learning may include:

* changing synaptic strengths;
* creating synapses;
* destroying synapses;
* creating neurons or computational units;
* putting neurons into dormant states;
* removing neurons;
* changing activation thresholds;
* changing plasticity rates;
* creating specialized regions or modules;
* consolidating frequently useful circuits;
* reorganizing connectivity;
* changing transmission characteristics;
* allocating and reclaiming computational resources.

The number of neurons, synapses, and effective degrees of freedom must therefore be dynamic rather than fixed.

## Do not optimize around parameter count

SOMA should initially be very small.

There is no reason to target hundreds of millions or one billion parameters during early research.

An initial experimental organism might contain approximately:

* 10,000 to 100,000 computational neurons;
* hundreds of thousands to several million sparse synapses;
* dynamic connectivity;
* relatively small internal state vectors per neuron and synapse.

The research value is not in raw scale.

A system with 50,000 cells that genuinely develops useful structure autonomously would be far more interesting than a 900-million-parameter model that behaves like an inferior Transformer.

The primary resource should be lifetime experience and simulation rather than parameter count.

## Computational cells

SOMA should not initially assume that an artificial neuron needs to look like a standard ANN neuron.

Each computational cell may contain persistent state.

A possible conceptual structure is:

```text
Cell
    id
    type
    membrane_state
    activation_state
    threshold
    adaptation_state
    plasticity_state
    age
    utility
    energy_usage
    refractory_state
    position_or_region
```

This is only a starting abstraction and should remain extensible.

Cells should be stateful through time.

The architecture should therefore naturally represent temporal dynamics rather than treating every inference as an isolated stateless matrix operation.

An initial implementation may use simplified event-driven or rate-based neurons. Full biological spiking realism is not necessary for the first version.

The important property is persistent state and local interaction.

## Synapses

Synapses are central to SOMA.

A synapse should be more than a scalar weight.

A conceptual synapse might contain:

```text
Synapse
    source
    destination
    strength
    transmission_delay
    plasticity
    stability
    eligibility_trace
    activity_history
    age
    utility
```

Different experiments may add additional properties.

Synapses must be able to:

* strengthen;
* weaken;
* become more plastic;
* become less plastic;
* become dormant;
* reactivate;
* disappear;
* emerge between previously unconnected cells.

Connectivity must therefore be represented using a sparse dynamic graph rather than a permanently fixed dense matrix.

## Local learning

A major research objective is to avoid requiring conventional global backpropagation during lifetime learning.

The system should investigate local learning rules.

A synapse should ideally determine its update using information that could plausibly be locally or regionally available.

A generic rule may have the form:

$$
\Delta w_{ij}
=
f(
a_i,
a_j,
e_{ij},
m_t,
history_{ij}
)
$$

where:

* \(a_i\) is presynaptic activity;
* \(a_j\) is postsynaptic activity;
* \(e_{ij}\) is a local eligibility trace;
* \(m_t\) is a modulatory signal;
* \(history_{ij}\) represents recent local history.

A useful family of mechanisms to investigate is three-factor learning:

$$
\Delta w_{ij} =
\eta \cdot e_{ij} \cdot M_t
$$

The local interaction establishes an eligibility trace, while a later reward, surprise, novelty, or other modulatory signal determines whether that change should be strengthened or consolidated.

Hebbian-style learning may be one component:

$$
\Delta w_{ij} \propto a_i a_j
$$

but SOMA should not be limited to simple Hebbian learning.

The architecture should make local plasticity rules interchangeable so that different mechanisms can be experimentally compared.

## Structural plasticity

Structural plasticity is one of SOMA's defining features.

The network topology itself must evolve.

A connection between two cells should be able to appear when repeated experience indicates that communication between those cells would be useful.

Conceptually:

$$
P(create_{ij})
=
f(
coactivity,
novelty,
prediction\ error,
distance,
modulation,
resource\ availability
)
$$

The exact function should be experimentally determined.

Likewise, synapses with consistently low utility should weaken and eventually disappear.

Conceptually:

$$
P(delete_{ij})
=
f(
low\ usage,
low\ information,
redundancy,
age,
cost
)
$$

Structural changes should generally occur on a slower timescale than ordinary neural activity.

The system should avoid constantly rebuilding itself after every event.

## Neurogenesis

SOMA should eventually support the creation of new computational cells.

New cells should not be created arbitrarily.

The system should detect conditions suggesting that existing representational capacity is insufficient.

Possible signals include:

* persistent prediction error;
* persistent novelty;
* excessive local activation;
* representational saturation;
* repeated inability to discriminate important states;
* chronic competition for existing cells;
* high workload in a region;
* repeated unresolved error despite synaptic adaptation.

A conceptual growth pressure could be:

$$
G_r =
Novelty_r
\times
Load_r
\times
UnresolvedError_r
\times
AvailableResources
$$

If growth pressure remains sufficiently high, new cells may be created within a relevant region.

New cells should initially have high plasticity.

They may start with exploratory connections.

Through experience, useful connections should strengthen while irrelevant connections disappear.

Mature cells should become progressively more stable.

This creates a developmental process:

```text
new cell
    ↓
high plasticity
    ↓
exploratory connections
    ↓
experience
    ↓
specialization
    ↓
stabilization
```

## Cell death and pruning

Growth must be balanced by removal.

Otherwise the system will expand indefinitely.

Every computational element should have a cost.

A cell may have utility approximately described as:

$$
U_i =
usefulness_i
+
information_i
+
rewardContribution_i
-
computationalCost_i
$$

A cell with persistently low utility should not necessarily be deleted immediately.

Use a progression such as:

```text
active
→ low-utility
→ dormant
→ disconnected
→ removed
```

This provides an opportunity for rarely used but important knowledge to survive.

Similarly, synapses should pass through weakening and dormancy before permanent removal.

The system should investigate whether dormant structures allow faster relearning of previously forgotten behaviors.

## Homeostasis

A biological-style adaptive network needs constraints that prevent runaway activity and runaway growth.

SOMA should implement homeostatic processes.

Possible mechanisms include:

* target activation ranges;
* local excitation/inhibition balance;
* maximum regional energy budgets;
* total connection budgets;
* competition for resources;
* normalization of connection strengths;
* adaptive firing thresholds;
* penalties for redundant activity;
* limits on growth rates.

Intelligence in SOMA should emerge under limited resources.

This creates computational selection pressure.

Useful circuits survive and stabilize.

Redundant circuits disappear.

New capacity appears when persistent novelty justifies its cost.

## Metaplasticity

SOMA should distinguish between a synaptic strength and the ability of that synapse to change.

A synapse may therefore contain at least:

$$
(w_{ij}, p_{ij})
$$

where \(w\) is connection strength and \(p\) is plasticity.

Then:

$$
\Delta w_{ij}
=
p_{ij} \cdot learningSignal
$$

Plasticity itself should evolve.

New connections may initially have high plasticity.

Repeatedly useful connections may gradually become less plastic.

Highly consolidated knowledge should become difficult, but not necessarily impossible, to overwrite.

A conceptual process is:

```text
new connection
plasticity = high

repeated useful activation
plasticity decreases

long-term reliable knowledge
plasticity = very low

major contradiction or relearning pressure
plasticity may increase again
```

This mechanism should help address catastrophic forgetting.

It creates multiple memory timescales.

## Neuromodulation

SOMA should not rely entirely on one scalar global loss function.

Introduce global or regional modulatory signals.

These do not need to perfectly reproduce biological neurotransmitters.

They can serve computational roles such as:

```text
reward
novelty
surprise
uncertainty
attention
stress
exploration
salience
```

Represent them conceptually as:

$$
M_t =
[
m_{reward},
m_{novelty},
m_{uncertainty},
m_{salience},
...
]
$$

Different cell and synapse types may respond differently to the same signal.

For example, a reward signal may strengthen eligible synapses, while novelty increases plasticity or promotes exploratory structural changes.

Neuromodulation provides a mechanism for coordinating otherwise local learning rules.

## Glia-like regulatory units

A later version of SOMA should investigate non-neuronal computational units inspired by glial functions.

These units should not necessarily encode content directly.

Instead they may regulate groups of neurons or synapses.

Possible responsibilities include:

* adjusting local plasticity;
* controlling resource allocation;
* regulating growth;
* controlling pruning;
* changing excitability;
* coordinating consolidation;
* monitoring local saturation;
* regulating energy use;
* modifying learning timescales.

A glia-like unit might influence an entire local neighborhood rather than behaving as another ordinary neuron.

This should be treated as an experimental abstraction rather than an attempt at exact biological simulation.

## Multiple timescales

SOMA should operate on several distinct timescales.

### Fast timescale

Milliseconds to seconds conceptually:

* neural activity;
* sensory processing;
* actions;
* recurrent internal state.

### Medium timescale

Seconds to hours conceptually:

* synaptic adaptation;
* eligibility traces;
* changes in plasticity;
* temporary memories.

### Slow lifetime timescale

Hours to months conceptually:

* structural rewiring;
* synaptic pruning;
* neurogenesis;
* cell death;
* consolidation;
* emergence of modules;
* reorganization.

These timescales can be compressed drastically in simulation.

The conceptual separation matters more than matching biological clock time.

## Evolution and development

Eventually SOMA should distinguish between lifetime learning and evolutionary optimization.

The system should have something analogous to a genome or developmental rule set:

$$
\theta_{genome}
$$

The genome should not encode the final trained network.

It should encode rules for development and adaptation.

Conceptually:

$$
\theta_{genome}
\rightarrow
developmentalRules
\rightarrow
SOMA_t
\rightarrow
behavior
$$

Lifetime experience changes the organism.

Evolution changes the rules that determine how organisms learn and develop.

This produces three levels of adaptation:

```text
Level 1
Neural activity

Level 2
Lifetime learning and structural plasticity

Level 3
Evolution of developmental and plasticity rules
```

Evolution should not be required for SOMA v0.1.

However, the architecture should be designed so that learning rules and structural-development rules can eventually be optimized across populations or generations.

## Do not begin with language

The first SOMA implementation should not attempt to compete with an LLM.

Language introduces enormous complexity and makes it difficult to determine whether the basic architecture works.

Start with a continuously living artificial organism inside a simulated environment.

The agent should receive sensory observations and produce actions.

Conceptually:

```text
Environment
    ↓
Sensors
    ↓
SOMA
    ↓
Actions
    ↓
Environment
```

The environment should also generate biologically inspired signals such as reward, energy gain, damage, surprise, or novelty.

A simple initial world could contain:

* spatial navigation;
* food or energy resources;
* obstacles;
* harmful regions;
* objects;
* changing environmental conditions;
* eventually other agents.

The organism may have objectives such as:

* maintain energy;
* avoid damage;
* explore;
* predict future sensory states;
* collect resources;
* discover useful actions.

## First important research experiment

One of SOMA's first benchmarks should test sequential learning without task labels.

The agent should experience multiple environments or capabilities sequentially.

For example:

```text
Environment A
navigation

Environment B
object discrimination

Environment C
resource collection

Environment D
prediction of another agent

Environment A again
```

The system should never be explicitly told:

```text
Task A ended.
Task B has begun.
```

The experience should appear as one continuous lifetime.

Measure:

* retention of old capabilities;
* adaptation speed;
* catastrophic forgetting;
* number of cells;
* number of synapses;
* neuron birth rate;
* neuron death rate;
* synapse creation rate;
* pruning rate;
* energy usage;
* modularity;
* reuse of existing circuitry;
* structural growth under novelty;
* stability of consolidated circuits;
* speed of relearning after forgetting.

## Core hypotheses

SOMA should be developed scientifically rather than only as an engineering experiment.

Important hypotheses include:

$$
H_1:
Structural\ plasticity\ reduces\ catastrophic\ interference
$$

$$
H_2:
Adaptive\ neurogenesis\ improves\ learning\ under\ persistent\ novelty
$$

$$
H_3:
Activity-dependent\ pruning\ prevents\ uncontrolled\ growth
$$

$$
H_4:
Metaplasticity\ protects\ important\ previously\ acquired\ capabilities
$$

$$
H_5:
Local\ learning\ plus\ modulatory\ signals\ can\ produce\ useful\ specialization
$$

$$
H_6:
Dynamic\ topology\ enables\ better\ continual\ learning\ than\ a\ comparable\ fixed\ topology
$$

$$
H_7:
Dormant\ structural\ traces\ enable\ faster\ relearning
$$

$$
H_8:
Functional\ modules\ can\ emerge\ without\ being\ explicitly\ architected
$$

## Ablation experiments

Every major mechanism should be independently removable.

Compare the full SOMA architecture against versions with:

```text
no neurogenesis
no cell death
no synaptic pruning
no structural rewiring
no metaplasticity
no neuromodulation
no homeostatic regulation
fixed topology
global backpropagation baseline
```

This is essential.

If the full system performs better, we need to understand which mechanisms create the improvement.

## Observability

SOMA must be designed to be inspectable.

Do not create a black-box implementation where only reward is visible.

Record structural evolution throughout the organism's lifetime.

At minimum track:

```text
cell count over time
synapse count over time
average degree
degree distribution
birth rate
death rate
connection formation rate
connection removal rate
plasticity distribution
strength distribution
energy consumption
activation statistics
regional specialization
network modularity
memory retention
novelty response
task performance
```

Provide visualization tools for the evolving network.

Researchers should be able to inspect snapshots of the organism at different ages.

The system should eventually allow visual comparison such as:

```text
birth
↓
early development
↓
specialization
↓
mature organization
↓
new environment
↓
structural adaptation
```

One particularly important question is whether functional regions emerge naturally.

For example, navigation-related activity might eventually cluster into one subnetwork while memory or action-selection circuitry appears elsewhere.

Such modularization should emerge rather than being hardcoded wherever possible.

## Implementation philosophy

The initial version should prioritize correctness, interpretability, modularity, and experimentation over raw performance.

Do not prematurely optimize for GPUs.

A CPU-based sparse implementation is acceptable for early prototypes.

Use a dynamic sparse graph representation.

Different mechanisms should be implemented as interchangeable policies.

For example:

```text
PlasticityRule
GrowthRule
PruningRule
NeurogenesisRule
HomeostasisRule
ModulationRule
CellDynamics
ResourceRule
```

The architecture should allow researchers to replace any one of these without rewriting the system.

The simulation loop might conceptually look like:

```text
while organism_alive:

    observation = environment.observe()

    sensory_events = encoder(observation)

    soma.receive(sensory_events)

    soma.propagate_activity()

    action = soma.select_action()

    outcome = environment.step(action)

    soma.update_local_synapses(outcome)

    soma.update_modulatory_state(outcome)

    soma.update_homeostasis()

    if structural_update_due:
        soma.grow_connections()
        soma.prune_connections()
        soma.consider_neurogenesis()
        soma.consider_cell_dormancy_or_death()

    if consolidation_due:
        soma.consolidate()

    metrics.record()
```

Avoid making every mechanism execute every simulation tick.

Structural changes should generally run at lower frequency.

## SOMA v0.1

The first implementation should intentionally be minimal.

Suggested scope:

* stateful artificial cells;
* sparse directed synapses;
* local activation propagation;
* local synaptic plasticity;
* eligibility traces;
* one or two global modulatory signals;
* adaptive synaptic growth;
* synaptic pruning;
* simple homeostasis;
* resource budget;
* visualization and metrics;
* simple simulated environment.

Do not initially include:

* human language;
* billions of parameters;
* large-scale pretraining;
* exact biological neuron simulation;
* detailed molecular biology;
* dozens of neurotransmitter types;
* complicated glial simulation;
* evolutionary optimization;
* arbitrary self-modifying executable code.

First prove structural plasticity works.

## SOMA v0.2

Possible additions:

* neuron birth;
* neuron dormancy and death;
* metaplasticity;
* explicit consolidation;
* regional organization;
* multiple modulatory signals;
* stronger novelty detection;
* sequential-environment benchmarks.

## SOMA v0.3+

Possible later research:

* glia-like regulatory units;
* developmental stages;
* inherited developmental rule sets;
* evolution across generations;
* multiple cell types;
* inhibitory/excitatory specialization;
* learned sensory encoders;
* richer simulated environments;
* multi-agent interaction;
* hierarchical memory;
* predictive world models;
* vision;
* eventually language.

## Important design principle

Do not hardcode the final organization of the system.

Hardcode only enough primitive mechanisms for development to occur.

The desired direction is:

```text
simple developmental rules
        ↓
experience
        ↓
self-organization
        ↓
specialization
        ↓
complex behavior
```

rather than:

```text
human designer specifies complex architecture
        ↓
optimization fills in weights
```

The distinction is central to SOMA.

## Scientific success criteria

The first version does not need to achieve sophisticated intelligence.

A meaningful early success would be showing that:

1. SOMA can continuously learn several sequential behaviors.
2. It retains previous behaviors better than an appropriate fixed-network baseline.
3. Novel environments cause measurable structural adaptation.
4. Frequently useful circuits become more stable.
5. Redundant circuitry is removed.
6. Network size remains bounded through homeostasis and pruning.
7. Returning to an old environment produces immediate recall or faster relearning.
8. Distinct functional subnetworks begin to emerge without being manually specified.

Even partial evidence for these properties would justify further research.

## Long-term vision

The eventual goal is to investigate whether intelligence can arise from a continuously developing computational organism rather than a static pretrained model.

A mature SOMA system could potentially have:

* a relatively small initial developmental specification;
* continuous sensory experience;
* persistent internal state;
* lifelong learning;
* dynamically changing connectivity;
* self-directed capacity expansion;
* self-directed pruning;
* memory consolidation;
* multiple timescales of adaptation;
* emergent specialization;
* homeostatic resource constraints;
* evolutionary optimization of developmental rules.

The architecture itself becomes part of what is learned.

Instead of asking:

“What should the values of these fixed parameters be?”

SOMA asks:

“What computational structure should exist now, given everything this organism has experienced so far?”

That is the central idea of the project.

## Instructions for Codex

Treat this as a research codebase, not a production AI product.

Start by designing the smallest implementation capable of testing the central hypothesis.

Do not silently replace SOMA with a conventional PyTorch feed-forward model or Transformer.

Do not introduce backpropagation unless it is explicitly being used as a baseline or in an experiment where it is clearly isolated.

Prefer sparse, inspectable, modular systems.

Every structural change should be traceable.

Every neuron and synapse should have a stable identifier so its lifetime can be analyzed.

Use deterministic random seeds where possible.

Make experiments reproducible.

Keep environment logic separate from organism logic.

Keep biological inspiration separate from implementation assumptions so mechanisms can be tested independently.

Create automated tests for:

* neuron creation and deletion;
* synapse creation and deletion;
* activity propagation;
* plasticity updates;
* resource accounting;
* pruning;
* deterministic simulation;
* serialization/deserialization;
* preservation of graph integrity after structural modifications.

Persist full organism checkpoints so a SOMA instance can resume its lifetime rather than being “retrained.”

The first engineering milestone should be:

> Build a small SOMA organism that lives inside a simple environment, learns useful behavior through local plasticity, dynamically modifies its synaptic topology, and can be visualized throughout its lifetime.

The second milestone should be:

> Demonstrate sequential learning and compare SOMA against a fixed-topology control network under similar computational constraints.

The third milestone should be:

> Add controlled neurogenesis, metaplasticity, structural consolidation, and adaptive pruning, then perform systematic ablation experiments.

Avoid premature claims about artificial general intelligence or complete biological brain simulation.

The research should stand on measurable computational results.

The working name of the architecture and project is:

# SOMA

## Self-Organizing Morphogenic Architecture

The project should investigate whether a computational system can become more capable not merely by optimizing fixed parameters, but by continuously becoming a different computational structure throughout its lifetime.

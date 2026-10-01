"""Locked R2 fixture: latent-context polarity through two event encodings.

Same A -> B -> A cue stream reaches the unchanged core either as one scalar
channel or as three bit channels. Targets and rewards are computed outside
the bridge; the brain never sees task identity. Modes: learn (default core),
frozen (no learning), shuffled (misattributed credit), persistence
(adapter-only baseline with no core at all).
"""

import random

from ..events import Event, EventBridge
from ..events.envelope import OutcomeEvent
from ..organism import Modulators, Organism
from ..synapses import Synapse
from ..transducers import BitScalarTransducer, ScalarStreamTransducer
from ..transducers.symbol_bits import quantize

R2_ACCEPTANCE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
PHASE_LENGTHS = (150, 150, 150)
CUE_VALUES = (0.8, -0.8)
SHUFFLE_LAG = 37
MARGIN = 0.10

ADAPTERS = {
    "scalar": ScalarStreamTransducer,
    "bits": BitScalarTransducer,
}


def make_event_organism(input_size, seed, frozen=False):
    organism = Organism.create_default(
        input_size=input_size, hidden_size=6, output_size=1, seed=seed)
    for synapse in organism.graph.iter_synapses():
        if synapse.destination in organism.output_ids:
            synapse.strength = 0.0
    for input_id in organism.input_ids:
        if not organism.graph.has(input_id, organism.output_ids[0]):
            organism.graph.add(Synapse(input_id, organism.output_ids[0], 0.0, plasticity=1.0))
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    organism.enable_context_modules(max_modules=4)
    organism.actor_learning_rate = 0.0 if frozen else 0.10
    if frozen:
        organism.learning_rate = 0.0
        organism.use_threshold_router()
    return organism


def cue_stream(seed):
    rng = random.Random(seed + 999)
    cues, targets = [], []
    for length, sign in zip(PHASE_LENGTHS, (1.0, -1.0, 1.0)):
        for _ in range(length):
            cue = CUE_VALUES[0] if rng.random() < 0.5 else CUE_VALUES[1]
            cues.append(cue)
            targets.append(sign * cue)
    return cues, targets


def envelopes_for(adapter_name, value, clock, counters):
    if adapter_name == "scalar":
        event_id = counters["series"]
        counters["series"] += 1
        return [Event("series", "fixture", event_id, clock, {"value": value},
                      {"trust": "test"}).to_dict()]
    symbol = quantize(value)
    out = []
    for index, channel in enumerate(("bit0", "bit1", "bit2")):
        event_id = counters[channel]
        counters[channel] += 1
        out.append(Event(channel, "fixture", event_id, clock,
                         {"bit": float((symbol >> index) & 1)}, {"trust": "test"}).to_dict())
    return out


def run_event_context(adapter_name, seed, mode="learn"):
    """Run one locked fixture stream. Returns tail means and skill margins."""
    if adapter_name not in ADAPTERS:
        raise ValueError("unknown R2 adapter: %s" % adapter_name)
    if mode not in ("learn", "frozen", "shuffled", "persistence"):
        raise ValueError("unknown R2 mode: %s" % mode)
    adapter = ADAPTERS[adapter_name]()
    organism = make_event_organism(adapter.input_size, seed, frozen=(mode == "frozen"))
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    cues, targets = cue_stream(seed)
    steps = len(cues)
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(steps)))
    counters = {"series": 0, "bit0": 0, "bit1": 0, "bit2": 0}
    rewards = []
    lagged = []
    for step in range(steps):
        action = bridge.ingest(envelopes_for(adapter_name, cues[step], step, counters))
        while action is None:
            action = bridge.ingest(envelopes_for(adapter_name, cues[step], step, counters))
        prediction = cues[step] if mode == "persistence" else action["proposal"]["value"]
        reward = 1.0 - abs(prediction - targets[step])
        rewards.append(reward)
        if mode == "shuffled":
            lagged.append(reward)
            delivered = lagged.pop(0) if len(lagged) > SHUFFLE_LAG else 0.0
        else:
            delivered = reward
        bridge.outcome(OutcomeEvent(action["correlation_id"], step, delivered, "fixture").to_dict())
    bridge.close()
    windows = {
        "a_tail": rewards[100:150],
        "b_tail": rewards[250:300],
        "return_tail": rewards[400:450],
    }
    tails = {name: sum(values) / len(values) for name, values in windows.items()}
    return {
        "adapter": adapter_name,
        "seed": seed,
        "mode": mode,
        "tails": tails,
        "mean_tail": sum(tails.values()) / len(tails),
        "bridge_steps": bridge.steps,
        "modules": len(organism.motor_modules),
    }


def run_event_context_benchmark(seeds=R2_ACCEPTANCE_SEEDS):
    """Run learn/frozen/shuffled/persistence across adapters and locked seeds."""
    report = {"seeds": list(seeds), "adapters": {}, "gates": {}, "all_passed": False}
    for adapter_name in ADAPTERS:
        per_seed = {}
        for seed in seeds:
            per_seed[str(seed)] = {mode: run_event_context(adapter_name, seed, mode)["tails"]
                                   for mode in ("learn", "frozen", "shuffled", "persistence")}
        means = {}
        for mode in ("learn", "frozen", "shuffled", "persistence"):
            for window in ("a_tail", "b_tail", "return_tail"):
                values = [per_seed[str(seed)][mode][window] for seed in seeds]
                means["%s_%s" % (mode, window)] = sum(values) / len(values)
        gates = {}
        for window in ("a_tail", "b_tail", "return_tail"):
            for control in ("frozen", "shuffled"):
                gates["%s_beats_%s" % (window, control)] = (
                    means["learn_%s" % window] - means["%s_%s" % (control, window)] >= MARGIN)
        # Adapter-only persistence is perfect on stationary phases by
        # construction; it must fail the switched phase.
        gates["b_tail_beats_persistence"] = (
            means["learn_b_tail"] - means["persistence_b_tail"] >= MARGIN)
        adapter_passed = all(gates.values())
        report["adapters"][adapter_name] = {
            "means": means, "gates": gates, "all_passed": adapter_passed, "per_seed": per_seed,
        }
    report["gates"] = {name: info["all_passed"] for name, info in report["adapters"].items()}
    report["all_passed"] = all(report["gates"].values())
    return report

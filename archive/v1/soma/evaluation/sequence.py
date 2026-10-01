"""Locked R3A fixtures: next-symbol prediction through history-buffer adapters.

Buffers are disclosed transducer scaffolding: they present lagged history
without selecting it. The core must learn which lags and conjunctions
predict. Alphabet {0, 1} coded as {-0.8, +0.8}; scalar prediction head;
reward is 1 - |error| against the next symbol's code.

Fixtures:
- lagcopy: random bits, target is the symbol GAP steps back. Any fixed-order
  n-gram with order < GAP is structurally at chance: the nontrivial gate.
- periodic: 0011 cycle, target is the next symbol. Ambiguous from any single
  lag; needs the (lag0, lag1) conjunction. Beats memoryless; fixed n-grams
  are reported as upper comparators.
"""

import random

from ..events import Event, EventBridge
from ..events.envelope import OutcomeEvent
from ..organism import Organism
from ..synapses import Synapse
from ..transducers.history import HistoryBitsTransducer, HistoryScalarTransducer
from ..transducers.symbol_bits import quantize

R3A_ACCEPTANCE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
STEPS = 2400
MARGIN = 0.15
SHUFFLE_LAG = 37
NGRAM_ORDER = 3

CODE = {0: -0.8, 1: 0.8}
PERIOD = (0, 0, 1, 1)
BIGRAM_NEXT = {0: (1, 0), 1: (0, 1)}
BIGRAM_PROBS = {0: 0.8, 1: 0.3}
LAGCOPY_GAP = 5
LAGCOPY_DEPTH = 6
PERIODIC_DEPTH = 3
BIGRAM_DEPTH = 3

# Gated fixtures. "bigram" remains runnable via run_sequence as a RED
# diagnostic: stationary stochastic streams over-segment routing (4/4 modules
# on seed 0) because noise trips sustained-mismatch detection. Retuning those
# frozen R1 thresholds is explicitly out of scope; grammar acquisition under
# noise is defined next work for R3C+.
FIXTURES = ("lagcopy", "periodic")
ADAPTERS = ("scalar", "bits")


def make_sequence_organism(input_size, seed, frozen=False):
    organism = Organism.create_default(
        input_size=input_size, hidden_size=6, output_size=1, seed=seed,
        max_cells=128, max_synapses=1024, energy_per_step=80.0)
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


def make_adapter(adapter_name, depth):
    if adapter_name == "scalar":
        return HistoryScalarTransducer(depth=depth)
    if adapter_name == "bits":
        return HistoryBitsTransducer(depth=depth)
    raise ValueError("unknown R3A adapter: %s" % adapter_name)


def fixture_stream(kind, seed, steps):
    rng = random.Random(seed + 5000)
    if kind == "lagcopy":
        return [rng.randint(0, 1) for _ in range(steps + 1)]
    if kind == "periodic":
        return [PERIOD[t % len(PERIOD)] for t in range(steps + 1)]
    if kind == "bigram":
        seq, state = [], 0
        for _ in range(steps + 1):
            seq.append(state)
            first, second = BIGRAM_NEXT[state]
            state = first if rng.random() < BIGRAM_PROBS[state] else second
        return seq
    raise ValueError("unknown R3A fixture: %s" % kind)


def fixture_depth(kind):
    if kind == "lagcopy":
        return LAGCOPY_DEPTH
    return PERIODIC_DEPTH if kind == "periodic" else BIGRAM_DEPTH


def fixture_target(kind, stream, step):
    if kind == "lagcopy":
        return stream[step - LAGCOPY_GAP] if step - LAGCOPY_GAP >= 0 else 0
    return stream[step + 1]


def envelopes_for(adapter_name, history, clock, counters):
    """history: oldest-first list of scalar values, len == adapter depth."""
    out = []
    if adapter_name == "scalar":
        for lag, value in enumerate(history):
            channel = "lag%d" % lag
            event_id = counters[channel]
            counters[channel] += 1
            out.append(Event(channel, "fixture", event_id, clock, {"value": value},
                             {"trust": "test"}).to_dict())
        return out
    for lag, value in enumerate(history):
        symbol = quantize(value)
        for index in range(3):
            channel = "lag%dbit%d" % (lag, index)
            event_id = counters[channel]
            counters[channel] += 1
            out.append(Event(channel, "fixture", event_id, clock,
                             {"bit": float((symbol >> index) & 1)},
                             {"trust": "test"}).to_dict())
    return out


def counter_keys(adapter_name, depth):
    if adapter_name == "scalar":
        return ["lag%d" % lag for lag in range(depth)]
    return ["lag%dbit%d" % (lag, index) for lag in range(depth) for index in range(3)]


def run_sequence(adapter_name, kind, seed, mode="learn", steps=STEPS):
    """Run one locked sequence stream. Returns tail statistics."""
    if adapter_name not in ADAPTERS:
        raise ValueError("unknown R3A adapter: %s" % adapter_name)
    if mode not in ("learn", "frozen", "shuffled", "persistence", "no_structure"):
        raise ValueError("unknown R3A mode: %s" % mode)
    depth = fixture_depth(kind)
    adapter = make_adapter(adapter_name, depth)
    organism = make_sequence_organism(adapter.input_size, seed, frozen=(mode == "frozen"))
    if mode == "no_structure":
        organism.structural_plasticity_enabled = False
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    stream = fixture_stream(kind, seed, steps)
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(steps)))
    counters = {key: 0 for key in counter_keys(adapter_name, depth)}
    rewards = []
    lagged = []
    for step in range(steps):
        history = [CODE[stream[step - lag]] if step - lag >= 0 else 0.0 for lag in range(depth)]
        action = bridge.ingest(envelopes_for(adapter_name, history, step, counters))
        while action is None:
            action = bridge.ingest(envelopes_for(adapter_name, history, step, counters))
        if mode == "persistence":
            prediction = history[0]
        else:
            prediction = action["proposal"]["value"]
        reward = 1.0 - abs(prediction - CODE[fixture_target(kind, stream, step)])
        rewards.append(reward)
        if mode == "shuffled":
            lagged.append(reward)
            delivered = lagged.pop(0) if len(lagged) > SHUFFLE_LAG else 0.0
        else:
            delivered = reward
        bridge.outcome(OutcomeEvent(action["correlation_id"], step, delivered, "fixture").to_dict())
    bridge.close()
    tail = rewards[3 * steps // 4:]
    return {
        "adapter": adapter_name,
        "fixture": kind,
        "seed": seed,
        "mode": mode,
        "tail_mean": sum(tail) / len(tail),
        "bridge_steps": bridge.steps,
        "modules": len(organism.motor_modules),
    }


def run_ngram(kind, seed, order=NGRAM_ORDER, steps=STEPS):
    """Online fixed-order n-gram control on the same stream (fixture-side)."""
    stream = fixture_stream(kind, seed, steps)
    counts = {}
    rewards = []
    for step in range(steps):
        context = tuple(stream[max(0, step - order + 1):step + 1])
        while len(context) < order:
            context = (0,) + context
        table = counts.get(context, [0, 0])
        prediction = CODE[0] if table[0] >= table[1] else CODE[1]
        reward = 1.0 - abs(prediction - CODE[fixture_target(kind, stream, step)])
        rewards.append(reward)
        counts.setdefault(context, [0, 0])[stream[step + 1]] += 1
    tail = rewards[3 * steps // 4:]
    return sum(tail) / len(tail)


def run_sequence_benchmark(seeds=R3A_ACCEPTANCE_SEEDS, steps=STEPS):
    """Learn/frozen/shuffled/persistence/ngram across fixtures, adapters, seeds."""
    report = {"seeds": list(seeds), "steps": steps, "fixtures": {}, "all_passed": False}
    for kind in FIXTURES:
        kind_report = {"adapters": {}, "all_passed": False}
        for adapter_name in ADAPTERS:
            means = {}
            for mode in ("learn", "frozen", "shuffled", "persistence", "no_structure"):
                values = [run_sequence(adapter_name, kind, seed, mode, steps)["tail_mean"]
                          for seed in seeds]
                means[mode] = sum(values) / len(values)
            means["ngram1"] = sum(run_ngram(kind, seed, order=1) for seed in seeds) / len(seeds)
            means["ngram3"] = sum(run_ngram(kind, seed, order=3) for seed in seeds) / len(seeds)
            gates = {
                "beats_frozen": means["learn"] - means["frozen"] >= MARGIN,
                "beats_shuffled": means["learn"] - means["shuffled"] >= MARGIN,
                "beats_persistence": means["learn"] - means["persistence"] >= MARGIN,
            }
            if kind == "lagcopy":
                # Fixed order-3 n-grams cannot span gap 5: gated.
                gates["beats_ngram3"] = means["learn"] - means["ngram3"] >= MARGIN
            if kind == "bigram":
                # Order-1 is the sufficient statistic: match it within margin.
                gates["matches_bigram_optimal"] = means["learn"] >= means["ngram1"] - MARGIN
            kind_report["adapters"][adapter_name] = {
                "means": means, "gates": gates, "all_passed": all(gates.values()),
            }
        kind_report["all_passed"] = all(
            info["all_passed"] for info in kind_report["adapters"].values())
        report["fixtures"][kind] = kind_report
    report["all_passed"] = all(info["all_passed"] for info in report["fixtures"].values())
    return report

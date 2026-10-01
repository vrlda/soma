"""Locked R11-first-step fixture: jittered glyphs through raw pixel events.

Two classes (vertical/horizontal center bars, +-1px jitter, one noise
pixel) under an A -> B -> A mapping switch (B inverts the classes).
Long phases: the high-dimensional context detector needs longer evidence
than scalar streams. Gates use tail means, matching house practice.
"""

import random

from ..events import Event, EventBridge
from ..events.envelope import OutcomeEvent
from ..organism import Organism
from ..synapses import Synapse
from ..transducers.vision import GlyphTransducer, render_horizontal, render_vertical

R11_ACCEPTANCE_SEEDS = (0, 1, 2, 3, 4, 5)
STEPS = 1800
MARGIN = 0.10
SIZE = 6


def make_vision_organism(seed, frozen=False):
    adapter = GlyphTransducer()
    organism = Organism.create_default(
        input_size=adapter.input_size, hidden_size=6, output_size=1, seed=seed,
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


def glyph_stream(seed, steps):
    """Precompute (class, grid) pairs. Class 0 = vertical, 1 = horizontal."""
    rng = random.Random(seed + 31337)
    stream = []
    for _ in range(steps):
        cls = rng.randint(0, 1)
        shift = (rng.randint(-1, 1), 0) if cls == 0 else (0, rng.randint(-1, 1))
        noise = [(rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1))]
        grid = render_vertical(shift=shift, noise=noise) if cls == 0 else render_horizontal(
            shift=shift, noise=noise)
        stream.append((cls, grid))
    return stream


def run_glyphs(seed, mode="learn", steps=STEPS):
    """Run one locked vision stream. Returns phase tail means."""
    if mode not in ("learn", "frozen", "shuffled", "persistence"):
        raise ValueError("unknown R11 mode: %s" % mode)
    adapter = GlyphTransducer()
    organism = make_vision_organism(seed, frozen=(mode == "frozen"))
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    stream = glyph_stream(seed, steps)
    third = steps // 3
    order = [0] * third + [1] * third + [0] * (steps - 2 * third)
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(steps)))
    counters = {"px%d" % index: 0 for index in range(SIZE * SIZE)}
    rewards = []
    lagged = []
    for step in range(steps):
        cls, grid = stream[step]
        envelopes = []
        for index in range(SIZE * SIZE):
            channel = "px%d" % index
            envelopes.append(Event(channel, "camera", counters[channel], step,
                                   {"on": float(grid[index // SIZE][index % SIZE])},
                                   {"trust": "test"}).to_dict())
            counters[channel] += 1
        action = bridge.ingest(envelopes)
        target = (0.8 if cls == 0 else -0.8) * (1.0 if order[step] == 0 else -1.0)
        if mode == "persistence":
            prediction = 0.8 if cls == 0 else -0.8
        else:
            prediction = action["proposal"]["value"]
        reward = 1.0 - abs(prediction - target)
        rewards.append(reward)
        if mode == "shuffled":
            lagged.append(reward)
            delivered = lagged.pop(0) if len(lagged) > 37 else 0.0
        else:
            delivered = reward
        bridge.outcome(OutcomeEvent(action["correlation_id"], step, delivered, "camera").to_dict())
    bridge.close()
    tails = {}
    for name, (start, end) in (("a_tail", (3 * third // 4, third)),
                               ("b_tail", (third + 3 * third // 4, 2 * third)),
                               ("return_tail", (2 * third + 3 * (steps - 2 * third) // 4, steps))):
        window = rewards[start:end]
        tails[name] = sum(window) / max(1, len(window))
    tails["modules"] = len(organism.motor_modules)
    return tails


def run_glyph_benchmark(seeds=R11_ACCEPTANCE_SEEDS, steps=STEPS):
    """Learn/frozen/shuffled/persistence across locked seeds."""
    report = {"seeds": list(seeds), "steps": steps, "modes": {}, "all_passed": False}
    for mode in ("learn", "frozen", "shuffled", "persistence"):
        tails = [run_glyphs(seed, mode, steps) for seed in seeds]
        report["modes"][mode] = {
            window: sum(entry[window] for entry in tails) / len(tails)
            for window in ("a_tail", "b_tail", "return_tail")
        }
    learn = report["modes"]["learn"]
    gates = {}
    for window in ("a_tail", "b_tail", "return_tail"):
        for control in ("frozen", "shuffled"):
            gates["%s_beats_%s" % (window, control)] = (
                learn[window] - report["modes"][control][window] >= MARGIN)
    # Adapter-only persistence nails stationary phases and fails switched B.
    gates["b_tail_beats_persistence"] = learn["b_tail"] - report["modes"]["persistence"]["b_tail"] >= MARGIN
    report["gates"] = gates
    report["all_passed"] = all(gates.values())
    return report

import json
import os
import tempfile
import unittest

from soma.evaluation.sequence import (
    ADAPTERS,
    envelopes_for,
    counter_keys,
    fixture_depth,
    fixture_stream,
    fixture_target,
    make_sequence_organism,
    run_ngram,
    run_sequence,
)
from soma.events import EventBridge
from soma.events.envelope import OutcomeEvent
from soma.organism import Organism


def _run_short(adapter_name, kind, seed, steps, organism=None, bridge=None,
               bridge_payload=None, start=0, total=None):
    import random
    from soma.evaluation.sequence import CODE, make_adapter
    depth = fixture_depth(kind)
    adapter = make_adapter(adapter_name, depth)
    fresh = organism is None
    organism = organism or make_sequence_organism(adapter.input_size, seed)
    if fresh:
        tape = random.Random((seed + 1) * 1000033 + 3571)
        organism.set_exploration_tape(
            tuple(tape.uniform(-1.0, 1.0) for _ in range(total or (start + steps))))
    bridge = bridge or EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    if bridge_payload is not None:
        bridge.load_state_dict(json.loads(json.dumps(bridge_payload)))
    stream = fixture_stream(kind, seed, start + steps)
    counters = {key: start for key in counter_keys(adapter_name, depth)}
    rewards = []
    for step in range(start, start + steps):
        history = [CODE[stream[step - lag]] if step - lag >= 0 else 0.0 for lag in range(depth)]
        action = bridge.ingest(envelopes_for(adapter_name, history, step, counters))
        while action is None:
            action = bridge.ingest(envelopes_for(adapter_name, history, step, counters))
        reward = 1.0 - abs(action["proposal"]["value"] - CODE[fixture_target(kind, stream, step)])
        rewards.append(reward)
        bridge.outcome(OutcomeEvent(action["correlation_id"], step, reward, "fixture").to_dict())
    return organism, bridge, rewards


class R3ASequenceTests(unittest.TestCase):
    def test_streams_deterministic(self):
        self.assertEqual(fixture_stream("lagcopy", 2, 50), fixture_stream("lagcopy", 2, 50))
        self.assertEqual(fixture_stream("periodic", 2, 50), fixture_stream("periodic", 2, 50))
        self.assertEqual(fixture_target("lagcopy", fixture_stream("lagcopy", 2, 50), 10),
                         fixture_stream("lagcopy", 2, 50)[5])

    def test_learn_beats_frozen_spot(self):
        for adapter_name in ADAPTERS:
            for kind in ("lagcopy", "periodic"):
                learn = run_sequence(adapter_name, kind, 1, "learn", steps=1200)["tail_mean"]
                frozen = run_sequence(adapter_name, kind, 1, "frozen", steps=1200)["tail_mean"]
                self.assertGreater(learn - frozen, 0.10,
                                   "%s/%s learn=%.3f frozen=%.3f" % (adapter_name, kind, learn, frozen))

    def test_lesion_collapses_prediction(self):
        organism, bridge, rewards = _run_short("scalar", "lagcopy", 2, 600, total=700)
        tail_intact = sum(rewards[-100:]) / 100
        # Lesion every motor module: single-module lesions are routed around
        # by recall, which would mask the causal contribution.
        module_cells = {module.cell_id for module in organism.motor_modules.values()}
        for synapse in organism.graph.iter_synapses():
            if synapse.destination in module_cells:
                synapse.strength = 0.0
        # Freeze further learning: the probe window measures the ablated
        # circuit only, with no recovery.
        organism.actor_learning_rate = 0.0
        organism.learning_rate = 0.0
        _, _, lesioned = _run_short("scalar", "lagcopy", 2, 100, organism=organism,
                                    bridge=bridge, start=600)
        tail_lesioned = sum(lesioned) / len(lesioned)
        self.assertGreater(tail_intact - tail_lesioned, 0.10)

    def test_ngram_control_at_chance_on_lagcopy(self):
        value = run_ngram("lagcopy", 0, order=3, steps=600)
        self.assertGreaterEqual(value, -0.2)
        self.assertLessEqual(value, 0.5)

    def test_resume_exact(self):
        for adapter_name in ADAPTERS:
            full_organism, _, full_rewards = _run_short(adapter_name, "periodic", 3, 300)
            half_organism, half_bridge, first = _run_short(
                adapter_name, "periodic", 3, 137, total=300)
            payload = json.loads(json.dumps(half_bridge.state_dict()))
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "organism.json")
                half_organism.save(path)
                restored = Organism.load(path)
                resumed, _, rest = _run_short(
                    adapter_name, "periodic", 3, 163, organism=restored,
                    bridge_payload=payload, start=137)
                resumed.save(path)
                self.assertEqual(Organism.load(path).state_dict(), resumed.state_dict())
            self.assertEqual(first + rest, full_rewards)
            self.assertEqual(resumed.state_dict(), full_organism.state_dict())

    def test_task_identity_absent(self):
        organism, _, _ = _run_short("bits", "lagcopy", 4, 100)
        blob = json.dumps(organism.state_dict(), sort_keys=True, default=str).lower()
        for token in ('"copy"', '"target"', '"phase"', '"lagcopy"', '"fixture_task"', '"task_id"'):
            self.assertNotIn(token, blob)


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest

from soma.evaluation.temporal import (
    ADAPTERS,
    cue_stream,
    envelopes_for,
    make_event_organism,
)
from soma.events import EventBridge
from soma.events.envelope import OutcomeEvent
from soma.organism import Organism


def _run(adapter_name, seed, steps, organism=None, bridge_payload=None, start=0):
    adapter = ADAPTERS[adapter_name]()
    organism = organism or make_event_organism(adapter.input_size, seed)
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    if bridge_payload is not None:
        bridge.load_state_dict(json.loads(json.dumps(bridge_payload)))
    cues, targets = cue_stream(seed)
    counters = {"series": start, "bit0": start, "bit1": start, "bit2": start}
    rewards = []
    for step in range(start, start + steps):
        action = bridge.ingest(envelopes_for(adapter_name, cues[step], step, counters))
        while action is None:
            action = bridge.ingest(envelopes_for(adapter_name, cues[step], step, counters))
        reward = 1.0 - abs(action["proposal"]["value"] - targets[step])
        rewards.append(reward)
        bridge.outcome(OutcomeEvent(action["correlation_id"], step, reward, "fixture").to_dict())
    return organism, bridge, rewards


class R2BridgeTests(unittest.TestCase):
    def test_resume_exact_at_arbitrary_boundary(self):
        for adapter_name in ("scalar", "bits"):
            full_organism, full_bridge, full_rewards = _run(adapter_name, 3, 300)
            half_organism, half_bridge, first_rewards = _run(adapter_name, 3, 137)
            bridge_payload = json.loads(json.dumps(half_bridge.state_dict()))
            with tempfile.TemporaryDirectory() as directory:
                organism_path = os.path.join(directory, "organism.json")
                half_organism.save(organism_path)
                restored_organism = Organism.load(organism_path)
                self.assertEqual(restored_organism.state_dict(), half_organism.state_dict())
                resumed_organism, resumed_bridge, rest_rewards = _run(
                    adapter_name, 3, 163, organism=restored_organism,
                    bridge_payload=bridge_payload, start=137)
                resumed_organism.save(organism_path)
                reloaded = Organism.load(organism_path)
                self.assertEqual(reloaded.state_dict(), resumed_organism.state_dict())
            self.assertEqual(first_rewards + rest_rewards, full_rewards)
            self.assertEqual(resumed_organism.state_dict(), full_organism.state_dict())

    def test_bridge_state_round_trip(self):
        organism, bridge, _ = _run("scalar", 4, 50)
        payload = bridge.state_dict()
        clone = EventBridge(make_event_organism(2, 4), __import__(
            "soma.transducers", fromlist=["ScalarStreamTransducer"]).ScalarStreamTransducer(),
            novelty=0.10, exploration=0.20)
        clone.load_state_dict(json.loads(json.dumps(payload)))
        self.assertEqual(clone.state_dict(), bridge.state_dict())

    def test_task_identity_absent_from_brain(self):
        for adapter_name in ("scalar", "bits"):
            organism, _, _ = _run(adapter_name, 5, 100)
            blob = json.dumps(organism.state_dict(), sort_keys=True, default=str).lower()
            # Quoted keys: avoids collisions with legitimate fields like
            # target_activity while catching any task-shaped state.
            for token in ('"copy"', '"target"', '"phase"', '"polarity"', '"cue_stream"',
                          '"fixture_task"', '"task_id"', '"schedule"'):
                self.assertNotIn(token, blob)

    def test_correlation_mismatch_rejected(self):
        adapter = ADAPTERS["scalar"]()
        bridge = EventBridge(make_event_organism(2, 6), adapter)
        action = bridge.ingest(envelopes_for("scalar", 0.5, 0, {"series": 0, "bit0": 0, "bit1": 0, "bit2": 0}))
        with self.assertRaises(ValueError):
            bridge.outcome(OutcomeEvent(action["correlation_id"] + 1, 0, 0.5, "fixture").to_dict())
        with self.assertRaises(ValueError):
            bridge.ingest(envelopes_for("scalar", 0.5, 1, {"series": 1, "bit0": 0, "bit1": 0, "bit2": 0}))

    def test_no_credit_drains_without_numeric_zero_learning(self):
        adapter = ADAPTERS["scalar"]()
        bridge = EventBridge(make_event_organism(adapter.input_size, 8), adapter,
                             novelty=0.10, exploration=0.20)
        bridge.ingest(envelopes_for("scalar", 0.5, 0,
                                   {"series": 0, "bit0": 0, "bit1": 0, "bit2": 0}))
        baseline = bridge.organism.reward_baseline
        bridge.no_credit()
        checkpoint = json.loads(json.dumps(bridge.state_dict()))
        clone = EventBridge(make_event_organism(adapter.input_size, 9), adapter,
                            novelty=0.10, exploration=0.20)
        clone.load_state_dict(checkpoint)
        self.assertEqual(clone.state_dict(), bridge.state_dict())
        bridge.ingest(envelopes_for("scalar", 0.6, 1,
                                    {"series": 1, "bit0": 1, "bit1": 1, "bit2": 1}))
        self.assertEqual(bridge.organism.reward_baseline, baseline)
        self.assertTrue(bridge.organism._pending_outcome)
        self.assertTrue(any(event.get("kind") == "outcome_discarded_no_credit"
                            for event in bridge.organism.events))

    def test_no_credit_rejects_delayed_credit_before_mutation(self):
        adapter = ADAPTERS["scalar"]()
        organism = make_event_organism(adapter.input_size, 10)
        organism.enable_delayed_credit(1)
        before = organism.state_dict()
        energy_before = organism.resources.energy_used
        with self.assertRaises(ValueError):
            organism.step((0.5,), __import__("soma").Modulators(credit=False))
        self.assertEqual(organism.step_count, 0)
        self.assertEqual(organism.resources.energy_used, energy_before)
        self.assertEqual(organism.state_dict(), before)

    def test_no_credit_rejects_outstanding_adaptive_proposal_before_mutation(self):
        adapter = ADAPTERS["scalar"]()
        organism = make_event_organism(adapter.input_size, 11)
        organism.adaptive_dendritic_proposal = {"owner_module": organism.active_motor_module}
        before = organism.state_dict()
        with self.assertRaises(ValueError):
            organism.step((0.5,), __import__("soma").Modulators(credit=False))
        self.assertEqual(organism.step_count, 0)
        self.assertEqual(organism.state_dict(), before)

    def test_no_credit_does_not_start_new_adaptive_proposal(self):
        adapter = ADAPTERS["scalar"]()
        organism = make_event_organism(adapter.input_size, 12)
        organism.enable_adaptive_dendritic_learning(proposal_interval=1)
        organism.step((0.5, 1.0), __import__("soma").Modulators(credit=False))
        self.assertIsNone(organism.adaptive_dendritic_proposal)

    def test_closed_bridge_rejects_use(self):
        adapter = ADAPTERS["scalar"]()
        bridge = EventBridge(make_event_organism(2, 7), adapter)
        bridge.close()
        with self.assertRaises(ValueError):
            bridge.ingest(envelopes_for("scalar", 0.5, 0, {"series": 0, "bit0": 0, "bit1": 0, "bit2": 0}))
        with self.assertRaises(ValueError):
            bridge.close()

    def test_buffer_bound_enforced(self):
        from soma.events import ChannelBuffer, Event
        buffer = ChannelBuffer(["bit0", "bit1", "bit2"], max_buffered=4)
        for index in range(4):
            buffer.push(Event("bit0", "s", index, index, {"bit": 1.0}, {}))
        with self.assertRaises(ValueError):
            buffer.push(Event("bit0", "s", 4, 4, {"bit": 1.0}, {}))


if __name__ == "__main__":
    unittest.main()

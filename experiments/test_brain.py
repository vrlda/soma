"""Scientific validity: causality, locality, checkpoint continuity, structure."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from brain import Config, Organism


class BrainTests(unittest.TestCase):
    def test_prediction_precedes_outcome_and_hidden_has_no_target_access(self):
        a = Organism(Config(inputs=2, outputs=2))
        b = Organism.from_state(a.state())
        self.assertEqual(a.experience([1, 0], 0), b.experience([1, 0], 1))
        for nid in a.hidden:
            self.assertEqual(a.neurons[nid], b.neurons[nid])
        self.assertEqual([s.strength for s in a.synapses if s.kind == "drive"],
                         [s.strength for s in b.synapses if s.kind == "drive"])
        self.assertNotEqual([s.strength for s in a.synapses if s.kind == "prediction"],
                            [s.strength for s in b.synapses if s.kind == "prediction"])

    def test_checkpoint_resumes_exact_lifetime_and_rng_including_growth(self):
        a = Organism(Config(inputs=2, outputs=2, growth=True, growth_interval=1,
                            growth_threshold=-1, growth_cost=0, pruning=True))
        for i in range(10):
            a.experience([i % 2, (i + 1) % 2], i % 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            a.save(path)
            b = Organism.load(path)
        for i in range(30):
            x, y = [i % 2, (i + 1) % 2], (i + 1) % 2
            self.assertEqual(a.experience(x, y), b.experience(x, y))
        self.assertEqual(json.dumps(a.state(), sort_keys=True), json.dumps(b.state(), sort_keys=True))
        self.assertGreater(a.births, 0)

    def test_no_learning_control_preserves_all_strengths(self):
        a = Organism(Config(inputs=2, outputs=2, learning=False, growth=True, pruning=True))
        weights = [s.strength for s in a.synapses]
        for i in range(40):
            a.experience([1, 0], i % 2)
        self.assertEqual(weights, [s.strength for s in a.synapses])
        self.assertEqual(a.births, 0)

    def test_pruning_never_leaves_dangling_synapses(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=3, pruning=True,
                            prune_interval=1, prune_age=1, prune_threshold=1e9))
        for i in range(5):
            p = a.experience([1, 0], 0)
            self.assertAlmostEqual(sum(p), 1)
        self.assertEqual(len(a.hidden), 0)
        self.assertGreater(a.synapse_deaths, 0)
        for s in a.synapses:
            self.assertIn(s.source, a.neurons)
            self.assertIn(s.target, a.neurons)

    def test_ablation_removes_all_temporal_paths(self):
        cfg = Config(inputs=2, outputs=2, traces=False, persistence=False,
                     recurrence=False, hidden_plasticity=False)
        a, b = Organism(cfg), Organism(cfg)
        a.experience([1, 0])
        b.experience([0, 1])
        self.assertEqual(a.experience([0, 0]), b.experience([0, 0]))

    def test_invalid_experience_does_not_change_state(self):
        a = Organism(Config())
        before = a.state()
        for x, y in (([0], 0), ([math.nan] * 4, 0), ([0] * 4, 8)):
            with self.assertRaises(ValueError):
                a.experience(x, y)
        self.assertEqual(before, a.state())

    def test_state_and_clone_do_not_share_mutable_neuron_lists(self):
        a = Organism(Config(inputs=2, outputs=2, growth=True, growth_interval=1,
                            growth_threshold=-1, growth_cost=0))
        snapshot = a.state()
        b = Organism.from_state(snapshot)
        original_hidden = a.hidden[:]
        b.experience([1, 0], 0)
        self.assertEqual(a.hidden, original_hidden)
        self.assertEqual(snapshot["hidden"], original_hidden)
        self.assertEqual(a.tick, 0)

    def test_peak_trace_retains_brief_event_amplitude(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=0,
                            trace_bank=True, trace_rule="peak"))
        a.experience([1, 0])
        a.experience([0, 0])
        traces = [s for s in a.synapses if s.source == a.sensory[0] and s.kind == "prediction"]
        for synapse in traces:
            self.assertAlmostEqual(synapse.trace, synapse.decay)

    def test_competitive_afferent_learning_is_local(self):
        a = Organism(Config(inputs=2, outputs=2, hidden_rule="competitive",
                            persistence=False, recurrence=False, traces=False))
        before = {id(s): s.strength for s in a.synapses}
        a.experience([1, 0], 1)
        winner = max(a.hidden, key=lambda nid: a.neurons[nid].activity)
        for s in a.synapses:
            if s.kind == "drive" and s.target != winner:
                self.assertEqual(s.strength, before[id(s)])

    def test_pruned_competitive_neuron_cannot_win_by_missing_afferents(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=2,
                            hidden_rule="competitive", persistence=False))
        missing = a.hidden[0]
        a.synapses = [s for s in a.synapses if not (s.kind == "drive" and s.target == missing)]
        a.experience([1, 0])
        self.assertEqual(a.neurons[missing].activity, 0)


if __name__ == "__main__":
    unittest.main()

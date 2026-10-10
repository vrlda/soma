"""Scientific validity: causality, locality, checkpoint continuity, structure."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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

    def test_unlabelled_experience_can_recruit_structure(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=1,
                            hidden_rule="competitive", growth=True,
                            growth_rule="representation", growth_clock="experience",
                            growth_interval=1, growth_threshold=-1, growth_cost=0,
                            vigilance=.001, growth_requires_current_novelty=True,
                            seed_birth_from_activity=True, persistence=False))
        a.experience([1, 0])
        self.assertEqual(a.births, 1)
        self.assertEqual(a.observations, 0)

    def test_representation_growth_does_not_duplicate_a_current_match(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=1,
                            hidden_rule="competitive", growth=True,
                            growth_rule="representation", growth_clock="experience",
                            growth_interval=1, growth_threshold=-1, growth_cost=0,
                            vigilance=.001, growth_requires_current_novelty=True,
                            seed_birth_from_activity=True, persistence=False))
        a.experience([1, 0])
        births = a.births
        a.experience([1, 0])
        self.assertEqual(a.births, births)

    def test_matched_zero_afferent_is_a_useful_constraint(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=1,
                            hidden_rule="competitive", vigilance=.01,
                            persistence=False, traces=False, pruning=True, prune_age=10,
                            prune_interval=1, prune_threshold=.001))
        for s in a.synapses:
            if s.kind == "drive":
                s.strength = 1.0 if s.source == a.sensory[0] else 0.0
        for _ in range(25):
            a.experience([1, 0], 0)
        constraints = [s for s in a.synapses if s.kind == "drive" and s.source == a.sensory[1]]
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].strength, 0)
        self.assertGreater(constraints[0].usefulness, .001)

    def test_temporal_competitive_checkpoint_continuation(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=1,
                            hidden_rule="competitive", growth=True, pruning=True,
                            growth_rule="representation", growth_clock="experience",
                            growth_interval=1, growth_threshold=.001, growth_cost=.001,
                            vigilance=.01, growth_requires_current_novelty=True,
                            seed_birth_from_activity=True, persistence=False,
                            drive_trace_rule="peak", drive_density=0,
                            prune_age=10, prune_interval=5, afferent_normalization=True))
        for i in range(20):
            a.experience([i % 2, (i + 1) % 2], i % 2)
        b = Organism.from_state(json.loads(json.dumps(a.state())))
        for i in range(50):
            self.assertEqual(a.experience([i % 2, 0], i % 2), b.experience([i % 2, 0], i % 2))
        self.assertEqual(a.state(), b.state())

    def test_failed_checkpoint_replacement_preserves_previous_lifetime(self):
        a = Organism(Config(inputs=2, outputs=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.json"
            a.save(path)
            previous = path.read_bytes()
            a.experience([1, 0], 0)
            with patch("brain.organism.os.replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    a.save(path)
            self.assertEqual(previous, path.read_bytes())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_consolidated_local_path_survives_inactivity(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=0,
                            trace_bank=True, trace_rule="peak", metaplasticity=True,
                            pruning=True, prune_age=100, prune_interval=10,
                            retention_protection=True))
        b = Organism(Config(inputs=2, outputs=2, initial_hidden=0,
                            trace_bank=True, trace_rule="peak", metaplasticity=True,
                            pruning=True, prune_age=100, prune_interval=10))
        for i in range(400):
            cue = i % 2
            for organism in (a, b):
                organism.experience([float(cue == 0), float(cue == 1)], cue)
        for _ in range(2000):
            a.experience([0, 0])
            b.experience([0, 0])
        self.assertGreater(a.experience([1, 0])[0], .8)
        self.assertFalse(any(s.source == b.sensory[0] and s.kind == "prediction" for s in b.synapses))
        self.assertTrue(any(s.source == a.sensory[0] and s.kind == "prediction" for s in a.synapses))

    def test_missing_prediction_path_can_regrow_from_local_coactivity(self):
        a = Organism(Config(inputs=2, outputs=2, initial_hidden=1,
                            hidden_rule="competitive", persistence=False,
                            traces=False, regrow_predictions=True,
                            synapse_growth_interval=1, synapse_growth_threshold=0))
        nid = a.hidden[0]
        for s in a.synapses:
            if s.kind == "drive":
                s.strength = 1.0 if s.source == a.sensory[0] else 0.0
        a.synapses = [s for s in a.synapses if not (s.source == nid and s.kind == "prediction")]
        a.experience([1, 0], 0)
        self.assertTrue(any(s.source == nid and s.kind == "prediction" for s in a.synapses))
        for _ in range(40):
            p = a.experience([1, 0], 0)
        self.assertGreater(p[0], .9)

    def test_diagnostic_snapshot_has_no_new_outcome_or_mutable_state_alias(self):
        a = Organism(Config(inputs=2, outputs=2))
        b = Organism.from_state(a.state())
        receipts_a, receipts_b = [], []
        a.experience([1, 0], 0, observer=receipts_a.append)
        b.experience([1, 0], 1, observer=receipts_b.append)
        self.assertEqual(receipts_a, receipts_b)
        self.assertIsInstance(receipts_a[0]["traces"], tuple)
        receipts_a[0]["sensory"] = ()
        self.assertEqual(len(a.sensory), 2)

    def test_afferent_homeostasis_uses_only_source_history(self):
        a = Organism(Config(inputs=2, outputs=2, hidden_rule="competitive",
                            afferent_normalization=True))
        b = Organism.from_state(a.state())
        for i in range(20):
            x = [i % 2, (i + 1) % 2]
            a.experience(x, 0)
            b.experience(x, 1)
        self.assertEqual([(s.pre_mean, s.pre_variance) for s in a.synapses if s.kind == "drive"],
                         [(s.pre_mean, s.pre_variance) for s in b.synapses if s.kind == "drive"])

    def test_no_learning_freezes_homeostatic_statistics(self):
        a = Organism(Config(inputs=2, outputs=2, afferent_normalization=True, learning=False))
        for _ in range(10):
            a.experience([1, 0], 0)
        for s in a.synapses:
            self.assertEqual((s.pre_mean, s.pre_variance), (0, 1))


if __name__ == "__main__":
    unittest.main()

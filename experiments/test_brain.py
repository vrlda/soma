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



class PredictiveBrainTests(unittest.TestCase):
    def make(self, **kwargs):
        from brain.predictive import PredictiveBrain, PredictiveConfig
        return PredictiveBrain(PredictiveConfig(inputs=3, initial_hidden=3, **kwargs))

    def test_predictive_hidden_updates_ignore_outcome_label(self):
        from brain.predictive import PredictiveBrain
        a=self.make();a.experience([1.,-.5,.2])
        b=PredictiveBrain.from_state(a.state())
        self.assertEqual(a.experience([.2,1.,-.5],0), b.experience([.2,1.,-.5],1))
        for x,y in zip(a.units,b.units):
            self.assertEqual(x.basal,y.basal)
            self.assertEqual(x.observation,y.observation)
            self.assertEqual(x.error,y.error)
        self.assertNotEqual(a.bias,b.bias)

    def test_predictive_update_ignores_distant_error(self):
        from brain.predictive import PredictiveBrain
        a=self.make();a.experience([1.,-.5,.2]);a.experience([.2,1.,-.5])
        b=PredictiveBrain.from_state(a.state())
        b.units[1].error=1000000.
        b.units[1].activity=-1000000.
        a._local_prediction_update(a.units[0], .3, 1.)
        b._local_prediction_update(b.units[0], .3, 1.)
        self.assertEqual(a.units[0],b.units[0])

    def test_predictive_local_rule_uses_previous_source_trace(self):
        a=self.make(hidden_rate=.2)
        a.experience([1.,0.,0.])
        before=a.units[0].basal[1].strength
        a.experience([0.,1.,0.])
        self.assertEqual(a.units[0].basal[1].strength,before)
        self.assertEqual(a.units[0].basal[0].trace,1.)

    def test_predictive_frozen_control_preserves_hidden_weights(self):
        a=self.make(hidden_learning=False,growth=False,pruning=False)
        old=[[s.strength for s in u.basal] for u in a.units]
        for i in range(20):a.experience([i%2,.5,-.2],i%2)
        self.assertEqual(old,[[s.strength for s in u.basal] for u in a.units])

    def test_predictive_save_resume_exact_with_shuffle_and_growth(self):
        from brain.predictive import PredictiveBrain
        a=self.make(shuffled=True,normalize_activity=True,growth_interval=1,growth_threshold=-1.,growth_cost=0.)
        for i in range(3):a.experience([1.,-.5,.2],i%2)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';a.save(path);b=PredictiveBrain.load(path)
        for i in range(5):
            self.assertEqual(a.experience([.1,float(i%2),-.3],i%2),
                             b.experience([.1,float(i%2),-.3],i%2))
        self.assertEqual(a.state(),b.state())
        self.assertGreater(a.births,0)

    def test_predictive_pruning_removes_whole_connected_unit(self):
        a=self.make(growth=False,prune_interval=1,prune_age=1,prune_threshold=1e9)
        a.experience([1.,0.,0.]);p=a.experience([1.,0.,0.])
        self.assertEqual(len(a.units),0)
        self.assertEqual(a.deaths,3)
        self.assertEqual(a.summary()['synapses'],0)
        self.assertAlmostEqual(sum(p),1.)
        self.assertEqual(a.experience([0.,0.,0.]),[.5,.5])

    def test_predictive_structural_yoking_matches_capacity(self):
        a=self.make(growth_interval=1,growth_threshold=-1.,growth_cost=0.)
        b=self.make(hidden_learning=False,growth=False)
        for i in range(3):
            a.experience([1.,.2,-.3]);b.experience([1.,.2,-.3],structure_override=a.last_event)
            self.assertEqual(a.summary()['synapses'],b.summary()['synapses'])
            self.assertEqual([u.id for u in a.units],[u.id for u in b.units])
        self.assertEqual(a.rng.getstate(),b.rng.getstate())

    def test_predictive_axon_homeostasis_is_label_independent(self):
        from brain.predictive import PredictiveBrain
        a=self.make(normalize_activity=True)
        a.experience([1.,0.,0.]);b=PredictiveBrain.from_state(a.state())
        a.experience([0.,1.,0.],0);b.experience([0.,1.,0.],1)
        for x,y in zip(a.units,b.units):
            self.assertEqual(x.activity_mean,y.activity_mean)
            self.assertEqual(x.activity_variance,y.activity_variance)
            self.assertEqual(x.axon,y.axon)
        self.assertNotEqual(a.units[0].activity_variance,1.)

    def test_predictive_hebbian_control_ignores_observation_error(self):
        from brain.predictive import PredictiveBrain
        a=self.make(hidden_rule='oja');a.experience([1.,0.,0.]);a.experience([0.,1.,0.])
        b=PredictiveBrain.from_state(a.state())
        a._local_prediction_update(a.units[0], -100., 1.)
        b._local_prediction_update(b.units[0], 100., 1.)
        self.assertEqual(a.units[0].basal,b.units[0].basal)

    def test_predictive_hidden_lifetime_is_independent_of_labels(self):
        a=self.make(normalize_activity=True)
        b=self.make(normalize_activity=True)
        for i in range(40):
            x=[float(i%3==0),float(i%3==1),float(i%3==2)]
            a.experience(x,i%2);b.experience(x)
        for x,y in zip(a.units,b.units):
            self.assertEqual(x.basal,y.basal)
            self.assertEqual(x.observation,y.observation)
            self.assertEqual(x.activity_mean,y.activity_mean)
            self.assertEqual(x.activity_variance,y.activity_variance)
        self.assertNotEqual(a.bias,b.bias)

    def test_uncertainty_auc_counts_ties_and_missing_classes(self):
        from experiments.predictive_uncertainty import auc
        self.assertEqual(auc([(0.,0),(1.,1)]),1.)
        self.assertEqual(auc([(1.,0),(0.,1)]),0.)
        self.assertEqual(auc([(1.,0),(1.,1)]),.5)
        self.assertIsNone(auc([(1.,0)]))

    def test_predictive_snapshot_and_invalid_signal_are_safe(self):
        a=self.make();state=a.state()
        with self.assertRaises(ValueError):a.experience([1.,float('nan'),0.])
        self.assertEqual(state,a.state())
        state['units'][0]['basal'][0]['strength']=10000.
        self.assertNotEqual(a.units[0].basal[0].strength,10000.)



class SomaTests(unittest.TestCase):
    def test_hidden_lifetime_ignores_outcome_labels(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, growth_interval=3, growth_threshold=-1, pruning=True))
        b = Soma.from_state(a.state())
        for i in range(40):
            a.experience([(-1.) ** i, .2], i % 2)
            b.experience([(-1.) ** i, .2], (i + 1) % 2)
        sa, sb = a.state(), b.state()
        for state in (sa, sb):
            state.pop('bias')
            for neuron in state['neurons']:
                neuron.pop('outcomes')
        self.assertEqual(sa, sb)

    def test_branch_update_ignores_distant_error_and_current_sources(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, initial_hidden=2))
        a.experience([.4, -.2])
        b = Soma.from_state(a.state())
        b.neurons[1].error = 10000
        b.neurons[1].activity = -10000
        a._learn_prediction(a.neurons[0], .8, 1.)
        b._learn_prediction(b.neurons[0], .8, 1.)
        self.assertEqual(a.neurons[0], b.neurons[0])

    def test_save_resume_exact_with_both_structural_events(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, initial_hidden=2, shuffled=True, growth_interval=2,
                            growth_threshold=-1, prune_interval=4, prune_age=3, prune_threshold=1e6))
        for i in range(7):
            a.experience([i % 2, .5], i % 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'brain.json'
            a.save(path)
            b = Soma.load(path)
        for i in range(20):
            self.assertEqual(a.experience([i % 2, -.5], i % 2), b.experience([i % 2, -.5], i % 2))
        self.assertEqual(a.state(), b.state())
        self.assertGreater(a.births, 0)
        self.assertGreater(a.deaths, 0)
        for neuron in a.neurons:
            self.assertTrue(all(edge.source == neuron.id for edge in neuron.outcomes))
            self.assertTrue(all(0 <= edge.source <= a.config.inputs
                                for branch in neuron.dendrites for edge in branch.synapses))
        self.assertEqual(len(a.neurons), a.config.initial_hidden + a.births - a.deaths)

    def test_invalid_input_and_snapshot_are_isolated(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2))
        before = a.state()
        for x, y in (([0], None), ([0, math.nan], None), ([0, 0], 3)):
            with self.assertRaises(ValueError):
                a.experience(x, y)
        self.assertEqual(before, a.state())
        b = Soma.from_state(before)
        b.neurons[0].dendrites[0].synapses[0].strength = 1e6
        self.assertEqual(before, a.state())

    def test_frozen_hidden_and_fully_frozen_controls(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, hidden_learning=False, growth=False, pruning=False))
        strengths = [[edge.strength for branch in neuron.dendrites for edge in branch.synapses] for neuron in a.neurons]
        for i in range(20):
            a.experience([i % 2, .5], i % 2)
        self.assertEqual(strengths, [[edge.strength for branch in neuron.dendrites for edge in branch.synapses] for neuron in a.neurons])
        b = Soma(SomaConfig(inputs=2, plasticity=False))
        before = b.state()
        for i in range(10):
            b.experience([i % 2, .5])
        self.assertEqual(before['means'], b.means)
        self.assertEqual(before['variances'], b.variances)
        self.assertEqual(b.births, 0)
        self.assertEqual(b.deaths, 0)

    def test_structural_yoking_is_exact_and_label_independent(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, growth_interval=2, growth_threshold=-1,
                            prune_interval=4, prune_age=3, prune_threshold=1e6))
        b = Soma(SomaConfig(inputs=2, hidden_learning=False))
        for i in range(20):
            a.experience([.5, -.4])
            b.experience([.5, -.4], structure_override=a.last_event)
            self.assertEqual(a.summary()['synapses'], b.summary()['synapses'])
            self.assertEqual([n.id for n in a.neurons], [n.id for n in b.neurons])
            self.assertEqual(a.last_event, b.last_event)

    def test_nonlinear_task_has_no_future_view_before_query(self):
        from experiments.nonlinear import stream, symmetry_check, label, HELD
        train, test, _ = stream(0, episodes=120)
        self.assertFalse({p for p, _ in train} & set(HELD))
        self.assertEqual(sum(label(p) for p, _ in test), 80)
        other, _, _ = stream(0, episodes=120, condition='predictable_nuisance')
        for (_, frames), (_, altered) in zip(train, other):
            self.assertEqual(frames[:11], altered[:11])
        self.assertEqual(symmetry_check()['held']['accuracy'], .5)

    def test_event_trace_preserves_a_brief_source_and_homeostasis_is_local(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, growth=False, pruning=False))
        a.experience([1., 0.])
        a.experience([0., 0.])
        self.assertAlmostEqual(a.traces[0], .8)
        self.assertEqual(a.traces[1], 0.)
        self.assertGreater(a.variances[0], 0.)
        self.assertEqual(a.means[1], 0.)

    def test_local_predictive_evidence_protects_a_connected_neuron(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, initial_hidden=2, growth=False, prune_interval=1,
                            prune_age=1, prune_threshold=.2))
        for neuron in a.neurons:
            neuron.age = 2
            neuron.usefulness = 0.
        a.neurons[0].evidence = 1.
        a.experience([0., 0.])
        self.assertEqual([neuron.id for neuron in a.neurons], [0])
        self.assertEqual(a.deaths, 1)

    def test_signed_underfit_does_not_recruit_on_systematic_overprediction(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, initial_hidden=1, pressure_rule='underfit',
                            growth_interval=1, growth_threshold=.0001, growth_cost=0, pruning=False))
        neuron = a.neurons[0]
        neuron.activity = 2.
        a._learn_prediction(neuron, 0., 1.)
        self.assertLess(neuron.underfit, 0.)
        a.experience([0., 0.])
        self.assertEqual(a.births, 0)

    def test_temporally_replaced_teacher_only_changes_hidden_local_learning(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, growth=False, pruning=False))
        a.experience([.5, .2])
        b = Soma.from_state(a.state())
        a.experience([.5, .2], observation_signal=[1., 1.])
        b.experience([.5, .2], observation_signal=[-1., -1.])
        self.assertEqual(a.traces, b.traces)
        self.assertEqual(a.means, b.means)
        self.assertNotEqual(a.neurons[0].dendrites, b.neurons[0].dendrites)

    def test_covariance_pressure_requires_unexplained_sensory_relation(self):
        from brain.soma import Soma, SomaConfig
        a = Soma(SomaConfig(inputs=2, initial_hidden=1, pressure_rule='covariance',
                            plasticity=False, growth=False, pruning=False))
        branch = a.neurons[0].dendrites[0]
        branch.observed_covariance = .1
        branch.forecast_covariance = .3
        a.experience([0., 0.])
        self.assertEqual(a.pressure, 0.)
        branch.observed_covariance = -.4
        a.experience([0., 0.])
        self.assertAlmostEqual(a.pressure, .1)


if __name__ == "__main__":
    unittest.main()

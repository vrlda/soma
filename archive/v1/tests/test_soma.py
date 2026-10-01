import os
import tempfile
import unittest
import math
import json

from soma.cells import Cell
from soma import ContinuousTargetEnvironment, Modulators, Organism
from soma.resources import ResourceBudget
from soma.synapses import SparseDirectedGraph, Synapse, synapse_to_dict


class SomaTests(unittest.TestCase):
    def test_signed_homeostasis_does_not_turn_zero_mean_activity_into_bias(self):
        cell = Cell("signed")
        for _ in range(500):
            cell.observe(0.8 if _ % 2 == 0 else -0.8)
            cell.update_homeostasis(0.04)
        self.assertLess(abs(cell.signed_activity_ema), 0.05)
        self.assertLess(abs(cell.threshold), 0.10)
        self.assertGreater(cell.activity_ema, 0.50)

    def test_stationary_circular_stream_keeps_thresholds_far_from_saturation(self):
        organism = Organism.create_default(seed=103)
        for step in range(3000):
            organism.step((math.cos(0.17 * step), math.sin(0.17 * step)), Modulators())
        self.assertLess(max(abs(cell.threshold) for cell in organism.cells.values()), 0.25)

    def test_adaptation_is_symmetric_gain_and_dormant_state_is_frozen(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=101)
        organism.enable_context_modules(max_modules=2)
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        cell = organism.cells[candidate.cell_id]
        cell.activation = 0.37
        cell.adaptation = 0.40
        cell.threshold = 0.10
        cell.signed_activity_ema = 0.12
        for synapse in organism._module_synapses(candidate):
            synapse.strength = 0.8
        frozen = (cell.activation, cell.adaptation, cell.threshold, cell.signed_activity_ema)
        organism._propagate((0.5,))
        self.assertEqual(frozen, (cell.activation, cell.adaptation, cell.threshold, cell.signed_activity_ema))
        candidate.dormant = False
        organism._propagate((0.5,))
        self.assertGreater(cell.activation, 0.0)
        self.assertAlmostEqual(cell.activation, organism._module_mean_from_current_inputs(candidate), places=2)

    def test_legacy_v5_checkpoint_migrates_signed_homeostasis_state(self):
        organism = Organism.create_default(seed=102)
        legacy = organism.state_dict()
        legacy["version"] = 5
        for payload in legacy["cells"].values():
            payload.pop("signed_activity_ema", None)
        restored = Organism.from_state_dict(legacy)
        self.assertEqual(restored.VERSION, 12)
        self.assertTrue(all(cell.signed_activity_ema == 0.0 for cell in restored.cells.values()))

    def test_graph_integrity_and_stable_ids(self):
        organism = Organism.create_default(seed=3)
        organism.validate()
        ids = set(organism.cells)
        self.assertEqual(len(ids), len(organism.cells))
        synapse_ids = [s.id for s in organism.graph.iter_synapses()]
        self.assertEqual(len(synapse_ids), len(set(synapse_ids)))
        self.assertTrue(all(identifier.startswith("synapse-") for identifier in synapse_ids))
        for synapse in list(organism.graph.iter_synapses()):
            self.assertIn(synapse.destination, organism.graph.outgoing[synapse.source])

    def test_recurrent_propagation_is_stateful(self):
        organism = Organism.create_default(input_size=1, hidden_size=2, output_size=1, seed=8)
        organism.graph.add(Synapse("output-000", "hidden-000", 0.4))
        first = organism.step((1.0,)).outputs
        second = organism.step((0.0,)).outputs
        self.assertNotEqual(first, second)

    def test_three_factor_plasticity_uses_reward(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=2)
        synapse = Synapse("input-000", "hidden-000", 0.2, plasticity=1.0)
        organism.graph = type(organism.graph)()
        organism.graph.add(synapse)
        before = synapse.strength
        organism.step((1.0,), Modulators())
        organism.step((1.0,), Modulators(reward=1.0))
        self.assertGreater(synapse.eligibility_trace, 0.0)
        self.assertNotEqual(before, synapse.strength)

    def test_deleted_and_recreated_synapse_gets_new_lifetime_id(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=4)
        edge = next(iter(organism.graph.iter_synapses()))
        old_id = edge.id
        organism.graph.remove(edge.source, edge.destination)
        recreated = Synapse(edge.source, edge.destination, edge.strength)
        organism.graph.add(recreated)
        self.assertNotEqual(old_id, recreated.id)
        self.assertIn(old_id, organism.graph.retired_ids)
        organism.validate()

    def test_growth_and_pruning_are_bounded_and_traceable(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=1, max_synapses=30)
        initial = len(organism.graph.synapses)
        for index in range(12):
            organism.step((1.0, 1.0), Modulators(reward=0.0 if index == 0 else 0.2, novelty=1.0))
        self.assertLessEqual(len(organism.graph.synapses), organism.resources.max_synapses)
        self.assertGreaterEqual(len(organism.events), 1)
        self.assertGreaterEqual(len(organism.graph.synapses), 1)
        organism.validate()
        self.assertEqual(organism.resources.counters["cells"], len(organism.cells))
        self.assertEqual(organism.resources.counters["synapses"], len(organism.graph.synapses))
        self.assertAlmostEqual(organism.resources.total_energy, sum(metric["energy_used"] for metric in organism.metrics_history))
        self.assertTrue(initial >= 1)
        self.assertTrue(all(metric["energy_used"] <= organism.resources.energy_per_step for metric in organism.metrics_history))

    def test_energy_parameters_and_baseline_cap_are_enforced(self):
        with self.assertRaises(ValueError):
            ResourceBudget(energy_per_step=-1.0)
        base = Organism.create_default(seed=6)
        with self.assertRaises(ValueError):
            Organism(base.cells, base.graph, base.input_ids, base.output_ids, resources=ResourceBudget(energy_per_step=1.0))

    def test_seeded_determinism(self):
        a = Organism.create_default(seed=99)
        b = Organism.create_default(seed=99)
        for i in range(8):
            modulators = Modulators(reward=0.2 if i % 2 else 0.0, novelty=0.3)
            self.assertEqual(a.step((0.4, -0.2), modulators).outputs, b.step((0.4, -0.2), modulators).outputs)
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_resource_budget_and_serialization_resume(self):
        organism = Organism.create_default(seed=5, max_cells=20, max_synapses=15)
        for index in range(5):
            organism.step((0.8, 0.1), Modulators(reward=0.0 if index == 0 else 0.3, novelty=0.8))
        organism.validate()
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            organism.save(path)
            restored = Organism.load(path)
            self.assertEqual(organism.state_dict(), restored.state_dict())
            self.assertEqual(organism.step((0.1, 0.2), Modulators(reward=0.1)).outputs, restored.step((0.1, 0.2), Modulators(reward=0.1)).outputs)
        finally:
            os.unlink(path)

    def test_environment_checkpoint_matches_uninterrupted_continuation(self):
        environment = ContinuousTargetEnvironment(seed=13, horizon=80)
        organism = Organism.create_default(seed=13)
        observation = environment.reset()
        for _ in range(25):
            result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.02))
            observation = environment.step(result.outputs[0])
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            organism.save_checkpoint(path, environment)
            expected = []
            for _ in range(20):
                result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.02))
                observation = environment.step(result.outputs[0])
                expected.append((result.outputs, environment.target, observation.reward))
            resumed, resumed_environment = Organism.load_checkpoint(path)
            resumed_observation = ObservationProxy(resumed_environment)
            actual = []
            for _ in range(20):
                result = resumed.step(resumed_observation.values, Modulators(reward=resumed_observation.reward, exploration=0.02))
                resumed_observation = ObservationProxy(resumed_environment, result.outputs[0])
                actual.append((result.outputs, resumed_observation.target, resumed_observation.reward))
            self.assertEqual(expected, actual)
        finally:
            os.unlink(path)

    def test_learning_beats_zero_weight_control_with_structure_active(self):
        def evaluate(rate):
            scores = []
            for seed in range(10):
                environment = ContinuousTargetEnvironment(seed=seed, horizon=200)
                organism = Organism.create_default(seed=seed)
                organism.learning_rate = rate
                observation = environment.reset()
                score = 0.0
                for _ in range(200):
                    result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.03))
                    observation = environment.step(result.outputs[0])
                    score += observation.reward
                organism.apply_outcome(observation.reward)
                scores.append(score / 200.0)
            return sum(scores) / len(scores)
        frozen = evaluate(0.0)
        learning = evaluate(0.08)
        self.assertGreater(learning, frozen + 0.005)

    def test_terminal_outcome_does_not_create_an_extra_decision(self):
        environment = ContinuousTargetEnvironment(seed=21, horizon=3)
        organism = Organism.create_default(seed=21)
        observation = environment.reset()
        for _ in range(3):
            result = organism.step(observation.values, Modulators(reward=observation.reward))
            observation = environment.step(result.outputs[0])
        self.assertEqual(organism.step_count, 3)
        organism.apply_outcome(observation.reward)
        self.assertEqual(organism.step_count, 3)
        self.assertFalse(organism.state_dict()["pending_outcome"])
        with self.assertRaises(ValueError):
            organism.apply_outcome(observation.reward)
        self.assertEqual(organism.metrics_history[-1]["outcome_reward"], observation.reward)

    def test_hidden_motor_modules_use_only_hidden_afferents(self):
        organism = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=60)
        organism.enable_context_modules(max_modules=2, afferent_kind="hidden")
        for module in organism.motor_modules.values():
            self.assertEqual(module.afferent_kind, "hidden")
            afferents = organism._module_synapses(module)
            self.assertTrue(afferents)
            self.assertTrue(all(organism.cells[synapse.source].kind == "hidden" for synapse in afferents))
            self.assertFalse(any(synapse.source in organism.input_ids for synapse in afferents))

    def test_hidden_actor_trace_uses_causal_penultimate_afferent(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=68)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_representation_learning(rate=0.02, noise=0.2)
        organism.set_representation_tape((1.0,))
        organism.set_exploration_tape((0.5,))
        incoming = organism.graph.get("input-000", "hidden-000")
        self.assertIsNotNone(incoming)
        incoming.strength = 0.0
        module = organism.motor_modules[organism.active_motor_module]
        afferent = organism._module_synapses(module)[0]
        afferent.strength = 1.0

        organism.step((0.0,), Modulators(exploration=0.2))

        causal_pre = organism._motor_causal_presynaptic[module.id][afferent.source]
        module_mean = organism.cells[module.cell_id].activation
        expected = causal_pre * (1.0 - module_mean * module_mean) * 0.5 / 0.2
        self.assertAlmostEqual(afferent.actor_eligibility_trace, expected, places=14)
        self.assertNotAlmostEqual(causal_pre, organism.cells[afferent.source].activation, places=6)
        self.assertNotAlmostEqual(
            afferent.actor_eligibility_trace,
            organism.cells[afferent.source].activation * (1.0 - module_mean * module_mean) * 0.5 / 0.2,
            places=6,
        )

    def test_representation_bias_is_not_centered_by_hidden_homeostasis(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=69)
        organism.enable_representation_learning(rate=0.02, noise=0.2)
        hidden = organism.cells["hidden-000"]
        hidden.representation_bias = 0.37
        for _ in range(500):
            hidden.observe(0.8)
            organism._homeostasis()
        self.assertEqual(hidden.threshold, 0.0)
        self.assertEqual(hidden.representation_bias, 0.37)

    def test_node_perturbation_is_local_and_uses_base_formula(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=61)
        graph = SparseDirectedGraph()
        graph.add(Synapse("input-000", "hidden-000", 0.4))
        graph.add(Synapse("hidden-000", "output-000", 0.0))
        organism.graph = graph
        organism.resources.counters["synapses"] = 2
        organism.enable_representation_learning(rate=0.1, noise=0.2)
        organism.set_representation_tape((0.5,))
        organism._propagate((0.4,), propagation_steps=2)
        base = math.tanh(0.4 * 0.4)
        expected = 0.4 * (1.0 - base * base) * 0.5 / 0.2
        incoming = organism.graph.get("input-000", "hidden-000")
        self.assertAlmostEqual(incoming.node_eligibility_trace, expected, places=14)
        self.assertAlmostEqual(organism.cells["hidden-000"].representation_bias_eligibility, (1.0 - base * base) * 0.5 / 0.2, places=14)
        self.assertEqual(organism.graph.get("hidden-000", "output-000").node_eligibility_trace, 0.0)

    def test_node_reward_is_delayed_once_and_clears_trace(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=62)
        organism.enable_representation_learning(rate=0.1, noise=0.2)
        organism.set_representation_tape((0.5,))
        organism._propagate((0.4,), propagation_steps=2)
        incoming = organism.graph.get("input-000", "hidden-000")
        before = incoming.strength
        trace = incoming.node_eligibility_trace
        organism._pending_outcome = True
        organism._consume_outcome(0.5)
        self.assertAlmostEqual(incoming.strength, before + 0.1 * trace * 0.5, places=14)
        self.assertEqual(incoming.node_eligibility_trace, 0.0)
        unchanged = incoming.strength
        organism._consume_outcome(0.0)
        self.assertEqual(incoming.strength, unchanged)

    def test_representation_bias_receives_local_reward_credit(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=63)
        organism.enable_representation_learning(rate=0.02, noise=0.2)
        organism.set_representation_tape((1.0,))
        organism._propagate((0.0,), propagation_steps=2)
        cell = organism.cells["hidden-000"]
        self.assertNotEqual(cell.representation_bias_eligibility, 0.0)
        organism._pending_outcome = True
        organism._consume_outcome(0.1)
        self.assertNotEqual(cell.representation_bias, 0.0)
        self.assertEqual(cell.representation_bias_eligibility, 0.0)

    def test_hidden_noise_is_suppressed_but_tape_is_consumed_in_warning_probe(self):
        noisy = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=64)
        quiet = Organism.from_state_dict(noisy.state_dict())
        noisy.enable_context_modules(max_modules=1, afferent_kind="hidden")
        quiet.enable_context_modules(max_modules=1, afferent_kind="hidden")
        noisy.enable_representation_learning(rate=0.02, noise=0.2)
        noisy.set_representation_tape((0.75,) * 3)
        noisy.context_state = "warning"
        noisy._propagate((0.8, -0.3), propagation_steps=2)
        quiet._propagate((0.8, -0.3), propagation_steps=2)
        self.assertEqual(noisy.representation_cursor, 3)
        self.assertTrue(all(synapse.node_eligibility_trace == 0.0 for synapse in noisy.graph.iter_synapses()))
        self.assertTrue(all(cell.representation_bias_eligibility == 0.0 for cell in noisy.cells.values()))
        for identifier in ("hidden-000", "hidden-001", "hidden-002"):
            self.assertEqual(noisy.cells[identifier].activation, quiet.cells[identifier].activation)
        noisy.context_state = "search"
        noisy.context_probe_module = noisy.active_motor_module
        noisy.set_representation_tape((0.25,) * 3)
        noisy._propagate((-0.2, 0.4), propagation_steps=2)
        self.assertEqual(noisy.representation_cursor, 3)
        self.assertTrue(all(synapse.node_eligibility_trace == 0.0 for synapse in noisy.graph.iter_synapses()))

    def test_v6_checkpoint_migrates_representation_defaults(self):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=65)
        organism.enable_context_modules(max_modules=1)
        legacy = organism.state_dict()
        legacy["version"] = 6
        for name in ("representation_learning_enabled", "representation_learning_rate", "representation_noise", "representation_rng_state", "representation_tape", "representation_cursor", "context_afferent_kind"):
            legacy.pop(name, None)
        for payload in legacy["cells"].values():
            payload.pop("representation_bias", None)
            payload.pop("representation_bias_eligibility", None)
        for payload in legacy["synapses"]:
            payload.pop("node_eligibility_trace", None)
        for payload in legacy["motor_modules"].values():
            payload.pop("afferent_kind", None)
        restored = Organism.from_state_dict(legacy)
        self.assertEqual(restored.VERSION, 12)
        self.assertFalse(restored.representation_learning_enabled)
        self.assertEqual(restored.context_afferent_kind, "input")
        self.assertTrue(all(module.afferent_kind == "input" for module in restored.motor_modules.values()))

    def test_v7_checkpoint_preserves_pending_representation_action_exactly(self):
        organism = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=66)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_representation_learning(rate=0.02, noise=0.2)
        organism.set_representation_tape((0.25,) * 6)
        organism.step((0.8, -0.3), Modulators(exploration=0.0))
        restored = Organism.from_state_dict(organism.state_dict())
        expected = organism.step((0.2, 0.7), Modulators(reward=0.1, exploration=0.0)).outputs
        actual = restored.step((0.2, 0.7), Modulators(reward=0.1, exploration=0.0)).outputs
        self.assertEqual(expected, actual)
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_hidden_module_recruitment_failure_is_atomic(self):
        organism = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=67)
        organism.resources.max_synapses = len(organism.graph.synapses) + 2
        cells = tuple(sorted(organism.cells))
        synapses = tuple(organism.graph.synapses)
        with self.assertRaises(RuntimeError):
            organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        self.assertEqual(tuple(sorted(organism.cells)), cells)
        self.assertEqual(tuple(organism.graph.synapses), synapses)
        self.assertEqual(organism.motor_modules, {})

    def test_dendritic_product_cell_uses_two_weighted_afferents(self):
        organism = Organism.create_dendritic_pair_bank(input_size=3, hidden_size=3, output_size=1, seed=70)
        self.assertEqual(organism.cells["hidden-000"].activation_type, "dendritic_product")
        self.assertEqual(organism.cells["hidden-000"].dendritic_sources, ("input-000", "input-001"))
        organism._propagate((1.0, -1.0, 1.0), propagation_steps=1)
        self.assertAlmostEqual(organism.cells["hidden-000"].activation, math.tanh(-1.0), places=14)
        organism.validate()

    def test_dendritic_pair_bank_covers_pairs_without_motor_sensor_edges(self):
        first = Organism.create_dendritic_pair_bank(input_size=3, hidden_size=8, output_size=1, seed=71)
        second = Organism.create_dendritic_pair_bank(input_size=3, hidden_size=8, output_size=1, seed=72)
        pairs = [cell.dendritic_sources for identifier, cell in sorted(first.cells.items()) if cell.kind == "hidden"]
        self.assertEqual(tuple(pairs[:3]), (("input-000", "input-001"), ("input-000", "input-002"), ("input-001", "input-002")))
        self.assertEqual(pairs[:3], [cell.dendritic_sources for identifier, cell in sorted(second.cells.items()) if cell.kind == "hidden"][:3])
        first.enable_context_modules(max_modules=1, afferent_kind="hidden")
        self.assertFalse(any(s.source in first.input_ids and first.cells[s.destination].kind == "motor_module" for s in first.graph.iter_synapses()))
        self.assertEqual(first.resources.counters["synapses"], len(first.graph.synapses))

    def test_v7_migration_defaults_to_additive_and_v8_pair_resume_is_exact(self):
        additive = Organism.create_default(seed=73).state_dict()
        additive["version"] = 7
        for payload in additive["cells"].values():
            payload.pop("activation_type", None)
            payload.pop("dendritic_sources", None)
        restored_additive = Organism.from_state_dict(additive)
        self.assertTrue(all(cell.activation_type == "additive" and not cell.dendritic_sources for cell in restored_additive.cells.values()))

        organism = Organism.create_dendritic_pair_bank(input_size=3, hidden_size=3, output_size=1, seed=74)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.set_exploration_tape((0.25, -0.5))
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        restored = Organism.from_state_dict(organism.state_dict())
        expected = organism.step((-1.0, 1.0, 1.0), Modulators(reward=0.1, exploration=0.2)).outputs
        actual = restored.step((-1.0, 1.0, 1.0), Modulators(reward=0.1, exploration=0.2)).outputs
        self.assertEqual(expected, actual)
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def _adaptive_dendritic_fixture(self, seed=75):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=seed)
        organism.enable_motor_bootstrap(scale=0.10)
        organism.enable_adaptive_dendritic_learning(proposal_interval=1)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.structural_plasticity_enabled = False
        organism.legacy_learning_enabled = False
        organism.set_exploration_tape((0.4,) * 16)
        return organism

    def _adaptive_dendritic_shadow_fixture(self, seed=90):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=seed)
        organism.enable_adaptive_dendritic_learning(proposal_interval=1)
        organism.enable_adaptive_dendritic_shadow_learning()
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.structural_plasticity_enabled = False
        organism.legacy_learning_enabled = False
        organism.actor_learning_rate = 0.1
        organism.set_exploration_tape((0.4,) * 64)
        return organism

    def _fingerprint_fixture(self, seed=120, max_modules=1):
        organism = self._adaptive_dendritic_shadow_fixture(seed=seed)
        organism.context_max_modules = max_modules
        return organism

    def _set_fingerprint_evidence(self, organism, pair=("input-000", "input-001")):
        owner = organism.motor_modules[organism.active_motor_module]
        key = organism._adaptive_dendritic_pair_key(pair)
        organism.adaptive_dendritic_fingerprint_active = True
        organism.adaptive_dendritic_fingerprint_owner = owner.id
        organism.adaptive_dendritic_fingerprint_incumbent = owner.id
        organism.context_state = "fingerprint"
        organism.adaptive_dendritic_fingerprint_count = 16
        organism.adaptive_dendritic_fingerprint_evidence = {
            key: {"sum": 1.0, "sumsq": 1.0, "count": 16.0}
        }
        return owner, pair

    def test_adaptive_dendritic_proposal_is_atomic_and_has_stable_cell_id(self):
        organism = self._adaptive_dendritic_fixture()
        before = organism.state_dict()
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["cell_id"], proposal["old_cell"]["id"])
        self.assertEqual(organism.resources.counters["synapses"], len(organism.graph.synapses))
        self.assertEqual(len(proposal["new_sources"]), 2)
        self.assertGreaterEqual(len(proposal["old_synapses"]), 1)
        self.assertEqual({payload["destination"] for payload in proposal["new_synapses"]}, {proposal["cell_id"]})
        self.assertNotEqual(before["cells"][proposal["cell_id"]]["activation_type"], "dendritic_product")
        self.assertTrue(any(event["kind"] == "adaptive_dendritic_proposed" for event in organism.events))

    def test_adaptive_dendritic_counterfactual_uses_same_input_and_aligns_credit(self):
        organism = self._adaptive_dendritic_fixture(seed=78)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        self.assertIsNotNone(proposal)
        self.assertNotEqual(proposal["counterfactual_activation"], proposal["old_activation"])
        delta = proposal["new_activation"] - proposal["counterfactual_activation"]
        expected = 0.0
        for module in organism.motor_modules.values():
            for synapse in organism._module_synapses(module):
                if synapse.source != proposal["cell_id"]:
                    continue
                pre = organism._motor_causal_presynaptic[module.id][proposal["cell_id"]]
                expected += synapse.strength * delta * synapse.actor_eligibility_trace / pre
        self.assertAlmostEqual(proposal["local_actor_eligibility"], expected, places=14)
        self.assertEqual(set(proposal["counterfactual_sources"]), {payload["source"] for payload in proposal["old_synapses"]})

    def test_adaptive_dendritic_accepts_once_from_local_reward_credit(self):
        organism = self._adaptive_dendritic_fixture()
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        proposal["local_eligibility"] = 1.0
        organism.adaptive_dendritic_min_evidence = 1
        proposal["pending_local_eligibility"] = 1.0
        cell_id = proposal["cell_id"]
        organism.adaptive_dendritic_proposal_interval = 100
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=1.0, exploration=0.2))
        self.assertEqual(organism.adaptive_dendritic_accept_count, 1)
        self.assertEqual(organism.adaptive_dendritic_reject_count, 0)
        self.assertTrue(any(event.get("kind") == "adaptive_dendritic_accepted" and event.get("cell_id") == cell_id for event in organism.events))
        organism._apply_adaptive_dendritic_outcome(1.0)
        self.assertEqual(organism.adaptive_dendritic_accept_count, 1)

    def test_adaptive_dendritic_revert_restores_graph_and_resources(self):
        organism = self._adaptive_dendritic_fixture()
        original_ids = tuple(synapse.id for synapse in organism.graph.iter_synapses())
        original_cells = {identifier: dict(cell.__dict__) for identifier, cell in organism.cells.items()}
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        proposal["local_eligibility"] = -1.0
        organism.adaptive_dendritic_min_evidence = 1
        proposal["pending_local_eligibility"] = -1.0
        organism._apply_context_reward(1.0)
        organism._apply_adaptive_dendritic_outcome(1.0)
        self.assertIsNone(organism.adaptive_dendritic_proposal)
        self.assertEqual(organism.adaptive_dendritic_reject_count, 1)
        self.assertEqual(tuple(synapse.id for synapse in organism.graph.iter_synapses()), original_ids)
        self.assertEqual(organism.resources.counters["synapses"], len(organism.graph.synapses))
        self.assertEqual(organism.cells[proposal["cell_id"]].__dict__, original_cells[proposal["cell_id"]])
        self.assertTrue(set(payload["synapse_id"] for payload in proposal["new_synapses"]) <= organism.graph.retired_ids)
        self.assertGreaterEqual(organism.graph.next_synapse_index, proposal["old_next_synapse_index"] + len(proposal["new_synapses"]))
        self.assertGreater(organism.resources.structural_events, 0)
        expected_motor = {
            (payload["source"], payload["destination"]): payload
            for payload in proposal["motor_afferents"]
        }
        actual_motor = {
            (synapse.source, synapse.destination): dict(synapse.__dict__)
            for synapse in organism.graph.iter_synapses()
            if organism.cells[synapse.destination].kind == "motor_module"
        }
        self.assertEqual(actual_motor, expected_motor)

    def test_adaptive_dendritic_accepted_product_is_not_overwritten(self):
        organism = self._adaptive_dendritic_fixture(seed=79)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        first = organism.adaptive_dendritic_proposal
        first["local_eligibility"] = 1.0
        organism.adaptive_dendritic_min_evidence = 1
        first["pending_local_eligibility"] = 1.0
        cell_id = first["cell_id"]
        organism.adaptive_dendritic_proposal_interval = 100
        organism._apply_adaptive_dendritic_outcome(1.0)
        accepted_sources = organism.cells[cell_id].dendritic_sources
        organism.adaptive_dendritic_proposal_interval = 1
        for index in range(1, 12):
            organism.step(tuple(1.0 if (index >> bit) & 1 else -1.0 for bit in range(3)), Modulators(reward=0.0, exploration=0.2))
        self.assertEqual(organism.cells[cell_id].activation_type, "dendritic_product")
        self.assertEqual(organism.cells[cell_id].dendritic_sources, accepted_sources)

    def test_adaptive_dendritic_pending_proposal_checkpoint_replays_exactly(self):
        organism = self._adaptive_dendritic_fixture(seed=76)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        self.assertIsNotNone(organism.adaptive_dendritic_proposal)
        restored = Organism.from_state_dict(organism.state_dict())
        self.assertEqual(organism.state_dict(), restored.state_dict())
        expected = organism.step((-1.0, 1.0, -1.0), Modulators(reward=1.0, exploration=0.2))
        actual = restored.step((-1.0, 1.0, -1.0), Modulators(reward=1.0, exploration=0.2))
        self.assertEqual(expected, actual)
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_adaptive_dendritic_resource_failure_leaves_graph_unchanged(self):
        cells = {
            "input-000": Cell("input-000", "input"),
            "input-001": Cell("input-001", "input"),
            "hidden-000": Cell("hidden-000", "hidden"),
            "output-000": Cell("output-000", "output"),
        }
        graph = SparseDirectedGraph()
        graph.add(Synapse("input-000", "hidden-000", 0.2))
        graph.add(Synapse("input-000", "output-000", 0.0))
        graph.add(Synapse("input-001", "output-000", 0.0))
        graph.add(Synapse("hidden-000", "output-000", 0.2))
        organism = Organism(cells, graph, ("input-000", "input-001"), ("output-000",), seed=77, resources=ResourceBudget(max_cells=8, max_synapses=4))
        organism.enable_adaptive_dendritic_learning(proposal_interval=1)
        organism.context_enabled = True
        before = organism.state_dict()
        organism._adaptive_dendritic_maybe_propose()
        self.assertIsNone(organism.adaptive_dendritic_proposal)
        self.assertEqual(organism.state_dict(), before)

    def test_adaptive_dendritic_trial_requires_sequential_confident_evidence(self):
        organism = self._adaptive_dendritic_fixture(seed=82)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        self.assertEqual(organism.adaptive_dendritic_min_evidence, 3)
        for count in (1, 2):
            proposal["pending_local_eligibility"] = 1.0
            organism.step_count += 1
            organism._apply_adaptive_dendritic_outcome(1.0)
            self.assertIs(organism.adaptive_dendritic_proposal, proposal)
            self.assertEqual(proposal["evidence_count"], count)
        proposal["pending_local_eligibility"] = 1.0
        organism.step_count += 1
        organism._apply_adaptive_dendritic_outcome(1.0)
        self.assertEqual(organism.adaptive_dendritic_accept_count, 1)
        self.assertIsNone(organism.adaptive_dendritic_proposal)

    def test_adaptive_dendritic_confidence_bound_rejects_negative_trial(self):
        organism = self._adaptive_dendritic_fixture(seed=83)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        for _ in range(3):
            proposal["pending_local_eligibility"] = -1.0
            organism.step_count += 1
            organism._apply_adaptive_dendritic_outcome(1.0)
            if organism.adaptive_dendritic_proposal is None:
                break
        self.assertIsNone(organism.adaptive_dendritic_proposal)
        self.assertEqual(organism.adaptive_dendritic_reject_count, 1)

    def test_adaptive_dendritic_delayed_outcome_is_counted_once_per_step(self):
        organism = self._adaptive_dendritic_fixture(seed=84)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        proposal["pending_local_eligibility"] = 1.0
        organism.step_count += 1
        organism._apply_adaptive_dendritic_outcome(1.0)
        organism._apply_adaptive_dendritic_outcome(1.0)
        self.assertEqual(proposal["evidence_count"], 1)
        self.assertEqual(len(proposal["trial_evidence"]), 1)

    def test_adaptive_dendritic_ownership_change_cancels_before_reward(self):
        organism = self._adaptive_dendritic_fixture(seed=85)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        organism.context_state = "warning"
        organism.motor_modules[organism.active_motor_module].detector_cusum = organism.context_warning_threshold + 1.0
        organism.adaptive_dendritic_proposal_interval = 100
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=0.5, exploration=0.2))
        self.assertIsNone(organism.adaptive_dendritic_proposal)
        self.assertTrue(any(event.get("kind") == "adaptive_dendritic_cancelled" for event in organism.events))
        self.assertEqual(organism.cells[proposal["cell_id"]].activation_type, "additive")

    def test_adaptive_dendritic_trial_does_not_update_detector_or_module_baseline(self):
        organism = self._adaptive_dendritic_fixture(seed=86)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        module = organism.motor_modules[proposal["owner_module"]]
        before = (module.baseline, module.confidence, module.detector_count, module.detector_cusum, module.total_reward)
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=0.5, exploration=0.2))
        after = (module.baseline, module.confidence, module.detector_count, module.detector_cusum, module.total_reward)
        self.assertEqual(after, before)
        self.assertEqual(proposal["evidence_count"], 1)

    def test_adaptive_dendritic_candidate_insulates_dormant_module_weights(self):
        organism = self._adaptive_dendritic_fixture(seed=87)
        organism.context_max_modules = 2
        dormant = organism._recruit_motor_module()
        dormant.dormant = True
        owner = organism.motor_modules[organism.active_motor_module]
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        proposal = organism.adaptive_dendritic_proposal
        candidate = proposal["cell_id"]
        owner_weight = organism.graph.get(candidate, owner.cell_id).strength
        dormant_weight = organism.graph.get(candidate, dormant.cell_id).strength
        self.assertIn(owner_weight, (-organism.motor_bootstrap_scale, organism.motor_bootstrap_scale))
        self.assertEqual(dormant_weight, 0.0)

    def test_adaptive_dendritic_module_search_state_is_checkpointed(self):
        organism = self._adaptive_dendritic_fixture(seed=88)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        module_id = organism.active_motor_module
        self.assertEqual(len(organism.adaptive_dendritic_pairs_tried[module_id]), 1)
        restored = Organism.from_state_dict(organism.state_dict())
        self.assertEqual(restored.adaptive_dendritic_pairs_tried, organism.adaptive_dendritic_pairs_tried)
        self.assertEqual(restored.adaptive_dendritic_module_accepted, organism.adaptive_dendritic_module_accepted)

    def test_adaptive_dendritic_checkpoint_replays_mid_trial_evidence(self):
        organism = self._adaptive_dendritic_fixture(seed=89)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=0.0, exploration=0.2))
        self.assertIsNotNone(organism.adaptive_dendritic_proposal)
        self.assertEqual(organism.adaptive_dendritic_proposal["evidence_count"], 1)
        restored = Organism.from_state_dict(organism.state_dict())
        expected = organism.step((1.0, 1.0, -1.0), Modulators(reward=0.1, exploration=0.2))
        actual = restored.step((1.0, 1.0, -1.0), Modulators(reward=0.1, exploration=0.2))
        self.assertEqual(expected, actual)
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_adaptive_shadow_delayed_alignment_and_once_only(self):
        organism = self._adaptive_dendritic_shadow_fixture()
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        pending = organism.adaptive_dendritic_shadow_pending
        self.assertEqual(pending["owner_module"], organism.active_motor_module)
        self.assertEqual(pending["step"], organism.step_count)
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=0.25, exploration=0.2))
        evidence = organism.adaptive_dendritic_shadow_evidence[organism.active_motor_module]
        self.assertTrue(evidence)
        count = sum(int(value["count"]) for value in evidence.values())
        organism._adaptive_dendritic_shadow_accumulate(1.0)
        organism._adaptive_dendritic_shadow_accumulate(1.0)
        self.assertEqual(sum(int(value["count"]) for value in evidence.values()), count + 3)

    def test_adaptive_shadow_warning_does_not_update_evidence(self):
        organism = self._adaptive_dendritic_shadow_fixture(seed=91)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        organism.context_state = "warning"
        organism.motor_modules[organism.active_motor_module].detector_cusum = organism.context_warning_threshold + 1.0
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=0.25, exploration=0.2))
        self.assertIsNone(organism.adaptive_dendritic_shadow_pending)
        self.assertFalse(any(value["count"] for entries in organism.adaptive_dendritic_shadow_evidence.values() for value in entries.values()))

    def test_adaptive_shadow_sign_symmetry_preserves_signed_direction(self):
        positive = self._adaptive_dendritic_shadow_fixture(seed=92)
        negative = self._adaptive_dendritic_shadow_fixture(seed=92)
        for organism, reward in ((positive, 1.0), (negative, -1.0)):
            organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
            organism._adaptive_dendritic_shadow_accumulate(reward)
        key = "input-000|input-001"
        self.assertAlmostEqual(positive.adaptive_dendritic_shadow_evidence["motor-module-000"][key]["sum"], -negative.adaptive_dendritic_shadow_evidence["motor-module-000"][key]["sum"])

    def test_adaptive_shadow_install_is_atomic_and_sets_owner_gradient(self):
        organism = self._adaptive_dendritic_shadow_fixture(seed=93)
        organism.context_max_modules = 2
        dormant = organism._recruit_motor_module()
        dormant.dormant = True
        owner = organism.motor_modules[organism.active_motor_module]
        old_ids = tuple(synapse.id for synapse in organism.graph.iter_synapses())
        candidate_id = min(
            cell.id for cell in organism.cells.values()
            if cell.kind == "hidden" and cell.activation_type == "additive"
        )
        unrelated_before = {
            (synapse.source, synapse.destination): dict(synapse.__dict__)
            for synapse in organism.graph.iter_synapses()
            if organism.cells[synapse.destination].kind == "motor_module"
            and synapse.source != candidate_id
        }
        self.assertTrue(organism._adaptive_dendritic_shadow_install(owner, ("input-000", "input-001"), 2.0, 1.0, 16))
        product = next(cell for cell in organism.cells.values() if cell.activation_type == "dendritic_product")
        self.assertEqual(product.id, candidate_id)
        self.assertEqual(product.dendritic_sources, ("input-000", "input-001"))
        self.assertAlmostEqual(organism.graph.get(product.id, owner.cell_id).strength, 0.2)
        self.assertEqual(organism.graph.get(product.id, dormant.cell_id).strength, 0.0)
        unrelated_after = {
            (synapse.source, synapse.destination): dict(synapse.__dict__)
            for synapse in organism.graph.iter_synapses()
            if organism.cells[synapse.destination].kind == "motor_module"
            and synapse.source != product.id
        }
        self.assertEqual(unrelated_after, unrelated_before)
        self.assertTrue(set(old_ids).issubset({s.id for s in organism.graph.iter_synapses()} | organism.graph.retired_ids))

    def test_adaptive_shadow_checkpoint_replays_mid_evidence(self):
        organism = self._adaptive_dendritic_shadow_fixture(seed=94)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        organism.step((-1.0, 1.0, -1.0), Modulators(reward=0.0, exploration=0.2))
        self.assertEqual(sum(int(value["count"]) for entries in organism.adaptive_dendritic_shadow_evidence.values() for value in entries.values()), 3)
        restored = Organism.from_state_dict(organism.state_dict())
        expected = organism.step((1.0, 1.0, -1.0), Modulators(reward=0.1, exploration=0.2))
        actual = restored.step((1.0, 1.0, -1.0), Modulators(reward=0.1, exploration=0.2))
        self.assertEqual(expected, actual)
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_neutral_fingerprint_uses_zero_mean_actual_tape_and_freezes_context_state(self):
        organism = self._fingerprint_fixture()
        organism.context_safe_action = 0.8
        owner = organism.motor_modules[organism.active_motor_module]
        owner.baseline = 0.31
        owner.confidence = 0.73
        owner.detector_count = 19
        owner.detector_cusum = 1.4
        owner.detector_frozen = True
        predictor = tuple(owner.detector_predictor)
        before = (owner.baseline, owner.confidence, owner.detector_count, owner.detector_cusum, owner.detector_frozen)
        organism.context_state = "warning"
        organism._adaptive_dendritic_begin_fingerprint()
        organism._propagate((1.0, -1.0, 1.0))
        action = organism._context_output(Modulators(exploration=0.2))[0]
        organism._adaptive_dendritic_fingerprint_record_pending()
        self.assertAlmostEqual(action, 0.08)
        self.assertEqual(organism.exploration_cursor, 1)
        self.assertEqual(organism.context_pending_mode, "fingerprint")
        organism._pending_outcome = True
        organism._consume_outcome(0.5)
        after = (owner.baseline, owner.confidence, owner.detector_count, owner.detector_cusum, owner.detector_frozen)
        self.assertEqual(after, before)
        self.assertEqual(tuple(owner.detector_predictor), predictor)

    def test_fingerprint_resolves_on_terminal_sixteenth_delayed_outcome(self):
        organism = self._fingerprint_fixture()
        organism.context_state = "warning"
        organism._adaptive_dendritic_begin_fingerprint()
        for _ in range(16):
            organism.step((1.0, -1.0, 1.0), Modulators(reward=0.0, exploration=0.2))
        self.assertEqual(organism.adaptive_dendritic_fingerprint_count, 15)
        self.assertTrue(organism.adaptive_dendritic_fingerprint_active)
        organism.apply_outcome(0.0)
        self.assertEqual(organism.adaptive_dendritic_fingerprint_count, 16)
        self.assertFalse(organism.adaptive_dendritic_fingerprint_active)
        self.assertEqual(organism.context_state, "normal")

    def test_zero_exploration_fingerprint_finishes_neutral_without_install(self):
        organism = self._fingerprint_fixture()
        organism.context_state = "warning"
        organism._adaptive_dendritic_begin_fingerprint()
        for _ in range(16):
            result = organism.step((1.0, -1.0, 1.0), Modulators(reward=0.0, exploration=0.0))
            self.assertEqual(result.outputs, (0.0,))
        self.assertEqual(organism.adaptive_dendritic_fingerprint_count, 15)
        organism.apply_outcome(0.0)
        self.assertFalse(organism.adaptive_dendritic_fingerprint_active)
        self.assertEqual(organism.context_state, "normal")
        self.assertEqual(len(organism.motor_modules), 1)
        self.assertFalse(any(cell.activation_type == "dendritic_product" for cell in organism.cells.values()))

    def test_fingerprint_direct_incumbent_route_preserves_confidence(self):
        organism = self._fingerprint_fixture()
        owner = organism.motor_modules[organism.active_motor_module]
        pair = ("input-000", "input-001")
        self.assertTrue(organism._adaptive_dendritic_shadow_install(owner, pair, 2.0, 1.0, 16))
        owner.confidence = 0.67
        self._set_fingerprint_evidence(organism, pair)
        organism._adaptive_dendritic_fingerprint_route()
        self.assertEqual(organism.active_motor_module, owner.id)
        self.assertAlmostEqual(owner.confidence, 0.67)
        self.assertFalse(organism.adaptive_dendritic_fingerprint_active)

    def test_fingerprint_dormant_owner_route_preserves_confidence(self):
        organism = self._fingerprint_fixture(max_modules=2)
        incumbent = organism.motor_modules[organism.active_motor_module]
        dormant = organism._recruit_motor_module()
        dormant.dormant = True
        pair = ("input-000", "input-001")
        self.assertTrue(organism._adaptive_dendritic_shadow_install(dormant, pair, 2.0, 1.0, 16))
        dormant.confidence = 0.61
        self._set_fingerprint_evidence(organism, pair)
        organism.adaptive_dendritic_fingerprint_incumbent = incumbent.id
        organism._adaptive_dendritic_fingerprint_route()
        self.assertEqual(organism.active_motor_module, dormant.id)
        self.assertFalse(dormant.dormant)
        self.assertAlmostEqual(dormant.confidence, 0.61)

    def test_fingerprint_unowned_route_recruits_and_installs(self):
        organism = self._fingerprint_fixture(max_modules=2)
        _, pair = self._set_fingerprint_evidence(organism)
        organism._adaptive_dendritic_fingerprint_route()
        self.assertEqual(len(organism.motor_modules), 2)
        owner_id = organism.adaptive_dendritic_pair_owners[organism._adaptive_dendritic_pair_key(pair)]
        self.assertEqual(owner_id, "motor-module-001")
        self.assertEqual(organism.active_motor_module, owner_id)
        self.assertTrue(any(cell.activation_type == "dendritic_product" and cell.dendritic_sources == pair for cell in organism.cells.values()))

    def test_fingerprint_unowned_recruit_install_failure_is_atomic(self):
        organism = self._fingerprint_fixture(max_modules=2)
        hidden_count = len([cell for cell in organism.cells.values() if cell.kind == "hidden"])
        organism.resources.max_synapses = len(organism.graph.synapses) + hidden_count
        before_cells = tuple(sorted(organism.cells))
        before_synapses = tuple(synapse.id for synapse in organism.graph.iter_synapses())
        before_counters = dict(organism.resources.counters)
        self._set_fingerprint_evidence(organism)
        organism._adaptive_dendritic_fingerprint_route()
        self.assertEqual(tuple(sorted(organism.cells)), before_cells)
        self.assertEqual(tuple(synapse.id for synapse in organism.graph.iter_synapses()), before_synapses)
        self.assertEqual(dict(organism.resources.counters), before_counters)
        self.assertNotIn("motor-module-001", organism.motor_modules)

    def test_fingerprint_mid_episode_replays_direct_and_json_checkpoints(self):
        organism = self._fingerprint_fixture()
        organism.context_state = "warning"
        organism._adaptive_dendritic_begin_fingerprint()
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = json.loads(json.dumps(organism.state_dict()))
        restored = Organism.from_state_dict(encoded)
        for index in range(20):
            values = (1.0, -1.0, 1.0) if index % 2 == 0 else (-1.0, 1.0, -1.0)
            expected = organism.step(values, Modulators(reward=0.1, exploration=0.2))
            self.assertEqual(expected, direct.step(values, Modulators(reward=0.1, exploration=0.2)))
            self.assertEqual(expected, restored.step(values, Modulators(reward=0.1, exploration=0.2)))
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_validate_rejects_pair_owner_without_product_or_with_unknown_owner(self):
        organism = self._fingerprint_fixture()
        state = organism.state_dict()
        state["adaptive_dendritic_pair_owners"] = {"input-000|input-001": "motor-module-000"}
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(state)
        state["adaptive_dendritic_pair_owners"] = {"input-000|input-001": "motor-module-999"}
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(state)

    def test_validate_rejects_duplicate_pair_owner_assignments(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=121)
        organism.configure_dendritic_pair_bank()
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        state = organism.state_dict()
        state["adaptive_dendritic_pair_owners"] = {
            "input-000|input-001": "motor-module-000",
            "input-000|input-002": "motor-module-000",
        }
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(state)


class VariableOrderSubstrateTests(unittest.TestCase):
    def _fixture(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=141)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_variable_order(3)
        return organism

    def test_general_graph_constructor_builds_depth_reuses_and_prunes_recoverably(self):
        organism = Organism.create_default(input_size=4, hidden_size=5, output_size=1, seed=240)
        organism.enable_context_modules(max_modules=3, afferent_kind="hidden")
        first_owner = organism._recruit_motor_module(afferent_kind="hidden")
        second_owner = organism._recruit_motor_module(afferent_kind="hidden")
        first_owner.dormant = True
        second_owner.dormant = True
        organism.enable_general_structural_learning(max_depth=3, max_leaves=6)
        initial_synapses = len(organism.graph.synapses)

        first_cell = organism.install_general_feature(("input-000", "input-001"), first_owner.id, 0.5)
        first_key = organism.general_module_features[first_owner.id]
        depth_two = next(
            item for item in organism.general_feature_candidates()
            if first_cell in item["sources"] and "input-002" in item["sources"]
        )
        second_cell = organism.install_general_feature(depth_two["sources"], second_owner.id, -0.4)
        second_key = organism.general_module_features[second_owner.id]
        organism.mark_general_feature_reused(second_key)
        self.assertEqual(organism.general_feature_lineage[second_key]["leaf_closure"], ["input-000", "input-001", "input-002"])
        self.assertEqual(organism.general_feature_lineage[second_key]["depth"], 2)
        self.assertFalse(organism.prune_general_feature(first_key))

        restored = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(restored.state_dict(), organism.state_dict())
        self.assertTrue(restored.prune_general_feature(second_key))
        self.assertTrue(restored.prune_general_feature(first_key))
        self.assertEqual(restored.cells[first_cell].activation_type, "additive")
        self.assertEqual(restored.cells[second_cell].activation_type, "additive")
        self.assertEqual(len(restored.graph.synapses), initial_synapses)
        self.assertEqual(restored.resources.counters["synapses"], initial_synapses)
        self.assertEqual(restored.general_feature_prune_count, 2)
        self.assertTrue(any(event.get("kind") == "general_feature_reused" for event in restored.events))
        self.assertEqual(sum(event.get("kind") == "general_feature_pruned" for event in restored.events), 2)
        restored.validate()

    def test_general_route_requires_repeat_confirmation_then_reuses_owner(self):
        organism = Organism.create_default(input_size=4, hidden_size=5, output_size=1, seed=241)
        organism.enable_context_modules(max_modules=3, afferent_kind="hidden")
        organism.enable_variable_order_learning(9001, max_order=3)
        organism._variable_order_internal_transition = True
        direct_cell = organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-000", 0.5)
        organism._variable_order_internal_transition = False
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.variable_order_install_count = 1
        organism.enable_general_structural_learning(max_depth=3, max_leaves=6)
        candidate = next(
            item for item in organism.general_feature_candidates()
            if direct_cell in item["sources"] and "input-002" in item["sources"]
        )

        def evidence():
            organism.general_fingerprint_evidence = {
                candidate["key"]: {"sum": 8.0, "sumsq": 4.0, "count": 16.0}
            }
            organism.variable_order_fingerprint_active = True
            organism.variable_order_fingerprint_owner = organism.active_motor_module
            organism.variable_order_fingerprint_incumbent = organism.active_motor_module

        evidence()
        self.assertTrue(organism._general_structural_route())
        self.assertNotIn(candidate["key"], organism.general_feature_owners)
        self.assertEqual(organism.general_novelty_confirmations, 1)
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(encoded.state_dict(), organism.state_dict())
        evidence()
        self.assertTrue(organism._general_structural_route())
        owner_id = organism.general_feature_owners[candidate["key"]]
        self.assertNotEqual(owner_id, "motor-module-000")
        self.assertEqual(organism.general_novelty_confirmations, 0)

        organism.active_motor_module = "motor-module-000"
        organism.motor_modules["motor-module-000"].dormant = False
        organism.motor_modules[owner_id].dormant = True
        organism.step_count += 128
        evidence()
        self.assertTrue(organism._general_structural_route())
        self.assertEqual(organism.active_motor_module, owner_id)
        self.assertEqual(organism.general_feature_reuse[candidate["key"]], 1)
        self.assertEqual(sum(event.get("kind") == "general_feature_installed" for event in organism.events), 1)

    def test_general_route_prunes_unreused_leaf_and_reuses_module_at_capacity(self):
        organism = Organism.create_default(input_size=6, hidden_size=6, output_size=1, seed=242)
        organism.enable_context_modules(max_modules=4, afferent_kind="hidden")
        modules = [organism.motor_modules["motor-module-000"]]
        modules.extend(organism._recruit_motor_module(afferent_kind="hidden") for _ in range(3))
        organism.enable_variable_order_learning(9002, max_order=3)

        def install_direct(feature, owner):
            key = organism._dendritic_feature_key(feature)
            organism._variable_order_internal_transition = True
            try:
                cell_id = organism.install_dendritic_feature(feature, owner.id, 0.5)
            finally:
                organism._variable_order_internal_transition = False
            organism.variable_order_module_features[owner.id] = key
            organism.variable_order_install_count += 1
            return cell_id

        pair_cell = install_direct(("input-000", "input-001"), modules[0])
        old_triple = install_direct(("input-002", "input-003", "input-004"), modules[1])
        new_triple = install_direct(("input-002", "input-003", "input-005"), modules[3])
        organism.enable_general_structural_learning(3, 6)
        old_candidate = next(item for item in organism.general_feature_candidates() if set(item["sources"]) == {pair_cell, old_triple})
        organism.install_general_feature(old_candidate["sources"], modules[2].id, 0.5)
        old_key = organism.general_module_features[modules[2].id]
        new_candidate = next(item for item in organism.general_feature_candidates() if set(item["sources"]) == {pair_cell, new_triple})
        organism.active_motor_module = modules[3].id
        for module in modules:
            module.dormant = module.id != modules[3].id
        organism.general_novelty_candidate = new_candidate["key"]
        organism.general_novelty_confirmations = 1
        organism.step_count = organism.general_audit_interval
        organism.general_fingerprint_evidence = {new_candidate["key"]: {"sum": 8.0, "sumsq": 4.0, "count": 16.0}}
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = modules[3].id
        organism.variable_order_fingerprint_incumbent = modules[3].id
        self.assertTrue(organism._general_structural_route())
        self.assertEqual(len(organism.motor_modules), 4)
        self.assertNotIn(old_key, organism.general_feature_owners)
        self.assertEqual(organism.general_feature_owners[new_candidate["key"]], modules[2].id)
        self.assertEqual(organism.general_feature_prune_count, 1)
        organism.validate()

    def test_cubic_cell_evaluates_balanced_signs(self):
        organism = self._fixture()
        cell_id = organism.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", 0.6)
        cell = organism.cells[cell_id]
        for values in ((1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)):
            organism._propagate(values)
            expected_sign = values[0] * values[1] * values[2]
            self.assertEqual(cell.activation > 0.0, expected_sign > 0.0)
            self.assertGreater(abs(cell.activation), 0.5)

    def test_binary_product_regression_is_unchanged(self):
        first = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=142)
        first.configure_dendritic_pair_bank()
        second = Organism.from_state_dict(first.state_dict())
        second.enable_variable_order(3)
        for values in ((1.0, -1.0), (-1.0, -1.0), (1.0, 1.0)):
            self.assertEqual(first.step(values, Modulators(exploration=0.0)), second.step(values, Modulators(exploration=0.0)))

    def test_variable_order_enable_validation_and_persistence(self):
        organism = Organism.create_default(input_size=3, hidden_size=2, output_size=1, seed=143)
        with self.assertRaises(ValueError):
            organism.enable_variable_order(2)
        with self.assertRaises(ValueError):
            organism.enable_variable_order(4)
        organism.enable_variable_order(3)
        self.assertTrue(organism.variable_order_enabled)
        self.assertEqual(organism.max_dendritic_order, 3)

    def test_cubic_checkpoint_replays_direct_and_json(self):
        organism = self._fixture()
        organism.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", -0.4)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.0))
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        for values in ((-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)):
            expected = organism.step(values, Modulators(exploration=0.0))
            self.assertEqual(expected, direct.step(values, Modulators(exploration=0.0)))
            self.assertEqual(expected, encoded.step(values, Modulators(exploration=0.0)))
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())

    def test_v8_migration_disables_variable_order_and_preserves_binary_cells(self):
        organism = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=144)
        organism.configure_dendritic_pair_bank()
        legacy = organism.state_dict()
        legacy["version"] = 8
        legacy.pop("variable_order_enabled")
        legacy.pop("max_dendritic_order")
        legacy.pop("variable_order_feature_owners")
        restored = Organism.from_state_dict(legacy)
        self.assertFalse(restored.variable_order_enabled)
        self.assertEqual(restored.max_dendritic_order, 2)
        self.assertEqual(restored.variable_order_feature_owners, {})
        self.assertTrue(all(cell.activation_type == "dendritic_product" for cell in restored.cells.values() if cell.kind == "hidden"))

    def test_malformed_arity_source_key_owner_and_duplicate_rejected(self):
        organism = self._fixture()
        organism.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", 0.2)
        state = organism.state_dict()
        malformed = dict(state)
        malformed["cells"] = {key: dict(value) for key, value in state["cells"].items()}
        malformed["cells"]["hidden-000"]["dendritic_sources"] = ["input-000", "input-001"]
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_feature_owners"] = {"k3:input-000|input-001|input-999": "motor-module-000"}
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_feature_owners"] = {"input-000|input-001|input-002": "motor-module-000"}
        with self.assertRaises((AssertionError, ValueError)):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_feature_owners"] = {"k3:input-000|input-001|input-002": "motor-module-999"}
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        duplicate = self._fixture()
        duplicate.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", 0.2)
        second = duplicate.cells["hidden-001"]
        for source in list(duplicate.graph.incoming.get(second.id, [])):
            duplicate.graph.remove(source, second.id)
        second.activation_type = "dendritic_product_n"
        second.dendritic_sources = ("input-000", "input-001", "input-002")
        for source in second.dendritic_sources:
            duplicate.graph.add(Synapse(source, second.id, 1.0, plasticity=0.0))
        duplicate.resources.counters["synapses"] = len(duplicate.graph.synapses)
        with self.assertRaises(AssertionError):
            duplicate.validate()

    def test_atomic_install_preserves_unrelated_afferents_and_failure_state(self):
        organism = self._fixture()
        organism.context_max_modules = 2
        other = organism._recruit_motor_module()
        candidate_id = "hidden-000"
        owner = organism.motor_modules["motor-module-000"]
        # Give the candidate and unrelated afferents distinctive metadata so
        # the install test covers both isolation and byte-for-byte retention.
        for synapse in organism.graph.iter_synapses():
            if synapse.destination == owner.cell_id and synapse.source != candidate_id:
                synapse.strength = 0.17
                synapse.plasticity = 0.23
                synapse.utility = 0.31
            elif synapse.source == candidate_id and organism.cells[synapse.destination].kind == "motor_module":
                synapse.strength = 0.41 if synapse.destination == owner.cell_id else -0.37
                synapse.plasticity = 0.47
                synapse.utility = 0.59
                synapse.actor_eligibility_trace = 0.9
        unrelated = {
            synapse.id: synapse_to_dict(synapse)
            for synapse in organism.graph.iter_synapses()
            if organism.cells[synapse.destination].kind == "motor_module" and synapse.source != candidate_id
        }
        candidate_edges = {
            synapse.destination: synapse_to_dict(synapse)
            for synapse in organism.graph.iter_synapses()
            if synapse.source == candidate_id and organism.cells[synapse.destination].kind == "motor_module"
        }
        before = organism.state_dict()
        organism.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", 0.35)
        after_unrelated = {
            synapse.id: synapse_to_dict(synapse)
            for synapse in organism.graph.iter_synapses()
            if organism.cells[synapse.destination].kind == "motor_module" and synapse.source != candidate_id
        }
        self.assertEqual(unrelated, after_unrelated)
        self.assertEqual(organism.graph.get(candidate_id, owner.cell_id).strength, 0.35)
        self.assertEqual(organism.graph.get(candidate_id, owner.cell_id).plasticity, 0.0)
        self.assertEqual(organism.graph.get(candidate_id, other.cell_id).strength, 0.0)
        self.assertEqual(organism.graph.get(candidate_id, other.cell_id).plasticity, 0.0)
        self.assertEqual(organism.graph.get(candidate_id, owner.cell_id).actor_eligibility_trace, 0.0)
        self.assertEqual(organism.graph.get(candidate_id, other.cell_id).actor_eligibility_trace, 0.0)
        self.assertNotEqual(candidate_edges[owner.cell_id]["strength"], organism.graph.get(candidate_id, owner.cell_id).strength)
        installed_strength = organism.graph.get(candidate_id, owner.cell_id).strength
        for synapse in organism._module_synapses(owner):
            synapse.actor_eligibility_trace = 0.0
        organism.pending_motor_module = owner.id
        organism.context_pending_mode = "normal"
        organism._apply_context_reward(1.0)
        self.assertEqual(installed_strength, organism.graph.get(candidate_id, owner.cell_id).strength)
        failed = self._fixture()
        failed.resources.energy_per_step = 0.01
        failed_before = failed.state_dict()
        with self.assertRaises(RuntimeError):
            failed.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", 0.35)
        self.assertEqual(failed_before, failed.state_dict())

    def test_atomic_install_validation_failure_restores_candidate_edges(self):
        organism = self._fixture()
        organism.context_max_modules = 2
        other = organism._recruit_motor_module()
        candidate_id = "hidden-000"
        for synapse in organism.graph.iter_synapses():
            if synapse.source == candidate_id and organism.cells[synapse.destination].kind == "motor_module":
                synapse.strength = 0.44
                synapse.plasticity = 0.66
                synapse.actor_eligibility_trace = 0.9
        before = organism.state_dict()
        original_validate = organism.validate
        calls = [0]

        def fail_once():
            calls[0] += 1
            if calls[0] == 1:
                raise AssertionError("injected install validation failure")
            return original_validate()

        organism.validate = fail_once
        with self.assertRaises(AssertionError):
            organism.install_dendritic_feature(("input-000", "input-001", "input-002"), "motor-module-000", 0.35)
        organism.validate = original_validate
        self.assertEqual(before, organism.state_dict())
        self.assertEqual(calls[0], 1)

    def test_existing_pair_owner_map_remains_valid(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=145)
        organism.configure_dendritic_pair_bank()
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.adaptive_dendritic_pair_owners["input-000|input-001"] = "motor-module-000"
        organism.enable_variable_order(3)
        organism.validate()

    def test_variable_order_pair_install_uses_binary_tag_and_k2_owner(self):
        organism = self._fixture()
        cell_id = organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-000", 0.25)
        self.assertEqual(organism.cells[cell_id].activation_type, "dendritic_product")
        self.assertEqual(organism.variable_order_feature_owners["k2:input-000|input-001"], "motor-module-000")
        organism.validate()

    def test_v10_feature_bank_checkpoint_and_delayed_once(self):
        organism = Organism.create_default(input_size=6, hidden_size=8, output_size=1, seed=146)
        organism.enable_context_modules(max_modules=6, afferent_kind="hidden")
        organism.enable_variable_order_learning(13007)
        self.assertEqual(len(organism.variable_order_feature_order), 35)
        self.assertEqual(len(set(organism.variable_order_feature_order)), 35)
        organism.set_exploration_tape((0.25,) * 64)
        values = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        organism.step(values, Modulators(exploration=0.2))
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        for _ in range(17):
            expected = organism.step(values, Modulators(reward=0.0, exploration=0.2))
            self.assertEqual(expected, direct.step(values, Modulators(reward=0.0, exploration=0.2)))
            self.assertEqual(expected, encoded.step(values, Modulators(reward=0.0, exploration=0.2)))
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())

    def test_v10_zero_exploration_fingerprint_terminates_without_install(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=147)
        organism.enable_context_modules(max_modules=6, afferent_kind="hidden")
        organism.enable_variable_order_learning(15013)
        organism.set_exploration_tape((0.5,) * 32)
        for _ in range(18):
            organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.0))
        self.assertFalse(organism.variable_order_fingerprint_active)
        self.assertEqual(organism.variable_order_install_count, 0)

    def test_v10_fingerprint_raw_credit_is_consumed_once(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=148)
        organism.enable_context_modules(max_modules=6, afferent_kind="hidden")
        organism.enable_variable_order_learning(15013)
        key = "k2:input-000|input-001"
        organism.variable_order_fingerprint_evidence = {}
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_pending = {
            "owner_module": "motor-module-000", "step": 1,
            "eligibilities": [{"key": key, "eligibility": 1.0}],
        }
        organism._variable_order_accumulate_fingerprint(0.5)
        first = dict(organism.variable_order_fingerprint_evidence[key])
        organism._variable_order_accumulate_fingerprint(0.5)
        self.assertEqual(first, organism.variable_order_fingerprint_evidence[key])

    def test_v10_normal_window_can_trigger_target_blind_fingerprint(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=149)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_variable_order_learning(17001)
        organism.variable_order_fingerprint_active = False
        organism.context_state = "normal"
        organism.variable_order_module_features = {}
        organism.pending_motor_module = "motor-module-000"
        for index in range(organism.variable_order_normal_window):
            organism.variable_order_normal_pending = {
                "owner_module": "motor-module-000", "step": organism.step_count,
                "eligibilities": [
                    {"key": "k2:input-000|input-001", "eligibility": 1.0},
                    {"key": "k2:input-000|input-002", "eligibility": 1.0 if index % 2 == 0 else -1.0},
                ],
            }
            organism._variable_order_accumulate_normal(1.0)
        self.assertTrue(organism.variable_order_fingerprint_active)
        self.assertEqual(organism.variable_order_normal_window, 8)
        organism.validate()

    def test_v10_sign_flip_selects_same_feature_and_opposite_strength(self):
        def route(sign, seed):
            organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=seed)
            organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
            organism.enable_variable_order_learning(17002)
            organism.variable_order_fingerprint_active = True
            organism.variable_order_fingerprint_owner = "motor-module-000"
            organism.variable_order_fingerprint_incumbent = "motor-module-000"
            organism.variable_order_fingerprint_evidence = {
                "k2:input-000|input-001": {"sum": 8.0 * sign, "sumsq": 8.0, "count": 8.0},
                "k2:input-000|input-002": {"sum": 1.0, "sumsq": 1.0, "count": 1.0},
            }
            organism._variable_order_route()
            key = organism.variable_order_module_features["motor-module-000"]
            feature_cell = next(cell for cell in organism.cells.values() if cell.activation_type in ("dendritic_product", "dendritic_product_n") and organism._dendritic_feature_key(cell.dendritic_sources) == key)
            edge = organism.graph.get(feature_cell.id, "motor-module-000")
            return key, edge.strength

        positive_key, positive_strength = route(1.0, 150)
        negative_key, negative_strength = route(-1.0, 151)
        self.assertEqual(positive_key, negative_key)
        self.assertGreater(positive_strength, 0.0)
        self.assertLess(negative_strength, 0.0)

    def test_v10_representation_seed_orders_ties_but_ties_do_not_install(self):
        first = Organism.create_default(input_size=6, hidden_size=8, output_size=1, seed=152)
        second = Organism.create_default(input_size=6, hidden_size=8, output_size=1, seed=152)
        other = Organism.create_default(input_size=6, hidden_size=8, output_size=1, seed=153)
        for organism in (first, second, other):
            organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        first.enable_variable_order_learning(17003)
        second.enable_variable_order_learning(17003)
        other.enable_variable_order_learning(17004)
        self.assertEqual(first.variable_order_feature_order, second.variable_order_feature_order)
        self.assertNotEqual(first.variable_order_feature_order, other.variable_order_feature_order)
        tied = {key: first._variable_order_feature_rank(first._variable_order_key(feature)) for key, feature in (("a", first.variable_order_feature_order[0]), ("b", first.variable_order_feature_order[1]))}
        self.assertNotEqual(tied["a"], tied["b"])
        for organism in (first, other):
            organism.variable_order_fingerprint_evidence = {
                "k2:input-000|input-001": {"sum": 8.0, "sumsq": 8.0, "count": 8.0},
                "k2:input-000|input-002": {"sum": 8.0, "sumsq": 8.0, "count": 8.0},
            }
            organism._variable_order_route()
            self.assertEqual(organism.variable_order_module_features, {})

    def test_v10_legacy_pair_owner_imports_without_duplicate_install(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=154)
        organism.configure_dendritic_pair_bank()
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        feature = ("input-000", "input-001")
        product = next(cell for cell in organism.cells.values() if cell.activation_type == "dendritic_product" and tuple(cell.dendritic_sources) == feature)
        existing_edge = organism.graph.get(product.id, "motor-module-000")
        if existing_edge is None:
            organism.graph.add(Synapse(product.id, "motor-module-000", 0.27, plasticity=0.0))
        else:
            existing_edge.strength = 0.27
            existing_edge.plasticity = 0.0
        organism.adaptive_dendritic_pair_owners["input-000|input-001"] = "motor-module-000"
        organism.enable_variable_order_learning(17005)
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = "motor-module-000"
        organism.variable_order_fingerprint_incumbent = "motor-module-000"
        before_synapses = organism.state_dict()["synapses"]
        organism.variable_order_fingerprint_evidence = {
            "k2:input-000|input-001": {"sum": 8.0, "sumsq": 8.0, "count": 8.0},
            "k2:input-000|input-002": {"sum": 1.0, "sumsq": 1.0, "count": 1.0},
        }
        organism._variable_order_route()
        self.assertEqual(organism.variable_order_feature_owners["k2:input-000|input-001"], "motor-module-000")
        self.assertEqual(before_synapses, organism.state_dict()["synapses"])

    def test_v10_pending_records_validate_strictly(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=155)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_variable_order_learning(17006)
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        organism.variable_order_fingerprint_pending = {
            "owner_module": "motor-module-000", "step": organism.step_count,
            "eligibilities": [{"key": "k2:input-000|input-001", "eligibility": 0.2}],
        }
        organism.validate()
        malformed = organism.state_dict()
        malformed["variable_order_fingerprint_pending"]["eligibilities"].append(dict(malformed["variable_order_fingerprint_pending"]["eligibilities"][0]))
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_fingerprint_pending"]["owner_module"] = "unknown"
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_fingerprint_pending"]["eligibilities"][0]["eligibility"] = float("nan")
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["pending_outcome"] = False
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_fingerprint_pending"] = None
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_fingerprint_pending"]["step"] = organism.step_count - 1
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)
        malformed = organism.state_dict()
        malformed["variable_order_fingerprint_pending"]["step"] = organism.step_count + 1
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)

    def test_v10_normal_pending_replays_direct_and_json(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=156)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_variable_order_learning(17007)
        organism.variable_order_fingerprint_active = False
        organism.context_state = "normal"
        organism.step((1.0, -1.0, 1.0), Modulators(exploration=0.2))
        organism.variable_order_normal_pending = {
            "owner_module": "motor-module-000", "step": organism.step_count,
            "eligibilities": [{"key": "k2:input-000|input-001", "eligibility": 0.2}],
        }
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        organism.apply_outcome(0.3)
        direct.apply_outcome(0.3)
        encoded.apply_outcome(0.3)
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())

    def test_v10_contradictory_pair_owner_maps_rejected(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=157)
        organism.configure_dendritic_pair_bank()
        organism.enable_context_modules(max_modules=2, afferent_kind="hidden")
        pair = ("input-000", "input-001")
        product = next(cell for cell in organism.cells.values() if cell.activation_type == "dendritic_product" and tuple(cell.dendritic_sources) == pair)
        for owner_id in ("motor-module-000", "motor-module-001"):
            edge = organism.graph.get(product.id, owner_id)
            if edge is None:
                organism.graph.add(Synapse(product.id, owner_id, 0.2, plasticity=0.0))
        organism.adaptive_dendritic_pair_owners["input-000|input-001"] = "motor-module-000"
        organism.variable_order_feature_owners["k2:input-000|input-001"] = "motor-module-001"
        with self.assertRaises(AssertionError):
            organism.validate()

    def test_midpoint_resolves_only_credible_existing_direct_owner_without_mutation(self):
        organism = Organism.create_default(input_size=3, hidden_size=4, output_size=1, seed=158)
        organism.enable_context_modules(max_modules=2, afferent_kind="hidden")
        organism.enable_variable_order(3)
        key = "k2:input-000|input-001"
        organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-000", 0.4, cell_id="hidden-000")
        incumbent = organism._recruit_motor_module()
        organism.variable_order_module_features["motor-module-000"] = key
        organism.enable_variable_order_learning(17008)
        organism.enable_variable_order_midpoint_existing_owner_resolution()
        organism.active_motor_module = incumbent.id
        organism.motor_modules["motor-module-000"].dormant = True
        incumbent.dormant = False
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = incumbent.id
        organism.variable_order_fingerprint_incumbent = incumbent.id
        organism.variable_order_fingerprint_count = 8
        organism.variable_order_fingerprint_evidence = {
            key: {"sum": 8.0, "sumsq": 8.0, "count": 8.0},
            "k2:input-000|input-002": {"sum": 1.0, "sumsq": 8.0, "count": 8.0},
        }
        structural_before = (
            tuple(sorted(organism.cells)),
            tuple(synapse_to_dict(s) for s in organism.graph.iter_synapses()),
            dict(organism.resources.counters),
        )
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))

        self.assertTrue(organism._variable_order_midpoint_resolve())
        self.assertTrue(direct._variable_order_midpoint_resolve())
        self.assertTrue(encoded._variable_order_midpoint_resolve())
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())
        self.assertEqual(organism.active_motor_module, "motor-module-000")
        self.assertFalse(organism.variable_order_fingerprint_active)
        self.assertEqual(organism.events[-2]["kind"], "variable_order_midpoint_existing_owner_resolved")
        self.assertEqual(organism.events[-2]["family"], "direct")
        structural_after = (
            tuple(sorted(organism.cells)),
            tuple(synapse_to_dict(s) for s in organism.graph.iter_synapses()),
            dict(organism.resources.counters),
        )
        self.assertEqual(structural_before, structural_after)

    def test_midpoint_rejects_unowned_winner_and_cross_family_tie(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=159)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_variable_order_learning(17009)
        organism.enable_variable_order_midpoint_existing_owner_resolution()
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = "motor-module-000"
        organism.variable_order_fingerprint_incumbent = "motor-module-000"
        organism.variable_order_fingerprint_count = 8
        organism.variable_order_fingerprint_evidence = {
            "k2:input-000|input-001": {"sum": 8.0, "sumsq": 8.0, "count": 8.0},
        }
        self.assertFalse(organism._variable_order_midpoint_resolve())
        organism.variable_order_feature_owners["k2:input-000|input-001"] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.composition_fingerprint_evidence = {
            "c4:hidden-000|hidden-001::input-000|input-001|input-002|input-003": {
                "sum": 8.0, "sumsq": 8.0, "count": 8.0,
            },
        }
        self.assertFalse(organism._variable_order_midpoint_resolve())
        self.assertTrue(organism.variable_order_fingerprint_active)

    def test_owner_evidence_shadow_normalizes_owner_and_novelty_and_replays(self):
        organism = self._fixture()
        organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-000", 0.3, cell_id="hidden-000")
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.enable_variable_order_learning(17010)
        organism.enable_variable_order_owner_evidence()
        organism.variable_order_fingerprint_count = 7
        organism.variable_order_fingerprint_evidence = {
            "k2:input-000|input-001": {"sum": 7.0, "sumsq": 7.0, "count": 7.0},
            "k2:input-000|input-002": {"sum": 1.0, "sumsq": 7.0, "count": 7.0},
        }
        organism._variable_order_refresh_owner_evidence()
        posterior = organism.variable_order_owner_posterior
        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertGreater(posterior["motor-module-000"], posterior["__novelty__"])
        restored = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(restored.state_dict(), organism.state_dict())
        organism.validate()

    def test_owner_probe_rejection_returns_to_incumbent_without_growth_and_replays(self):
        organism = self._fixture()
        organism.context_max_modules = 2
        candidate = organism._recruit_motor_module()
        organism.enable_variable_order_learning(17011)
        organism.enable_variable_order_owner_evidence()
        organism.enable_variable_order_owner_probes()
        organism.variable_order_fingerprint_incumbent = "motor-module-000"
        before_modules = tuple(organism.motor_modules)
        self.assertTrue(organism._variable_order_begin_owner_probe(candidate.id))
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(direct.state_dict(), encoded.state_dict())
        for clone in (organism, direct, encoded):
            clone.step_count = 1
            clone.context_probe_remaining = 0
            clone.context_probe_count = 1
            clone.context_probe_sum = -1.0
            clone.context_probe_sumsq = 1.0
            clone._context_maybe_switch()
            self.assertEqual(clone.active_motor_module, "motor-module-000")
            self.assertEqual(tuple(clone.motor_modules), before_modules)
            self.assertEqual(clone.context_state, "normal")
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())

    def test_owner_probe_acceptance_reactivates_without_learning_or_growth(self):
        organism = self._fixture()
        organism.context_max_modules = 2
        candidate = organism._recruit_motor_module()
        organism.enable_variable_order_learning(17012)
        organism.enable_variable_order_owner_evidence()
        organism.enable_variable_order_owner_probes()
        organism.variable_order_fingerprint_incumbent = "motor-module-000"
        before_structure = (tuple(organism.cells), tuple(synapse_to_dict(edge) for edge in organism.graph.iter_synapses()))
        self.assertTrue(organism._variable_order_begin_owner_probe(candidate.id))
        organism.step_count = 1
        organism.context_probe_remaining = 0
        organism.context_probe_count = 2
        organism.context_probe_sum = 1.0
        organism.context_probe_sumsq = 0.5
        organism._context_maybe_switch()
        self.assertEqual(organism.active_motor_module, candidate.id)
        self.assertEqual(organism.context_state, "normal")
        self.assertIsNone(organism.context_probe_module)
        after_structure = (tuple(organism.cells), tuple(synapse_to_dict(edge) for edge in organism.graph.iter_synapses()))
        self.assertEqual(before_structure, after_structure)
        organism.validate()

    def test_v9_stage2_checkpoint_migrates_to_v11_neutral_learner(self):
        organism = self._fixture()
        legacy = organism.state_dict()
        legacy["version"] = 9
        for key in (
            "variable_order_learning_enabled", "variable_order_representation_seed", "variable_order_rng_state",
            "variable_order_feature_order", "variable_order_module_features", "variable_order_fingerprint_window",
            "variable_order_fingerprint_active", "variable_order_fingerprint_owner", "variable_order_fingerprint_incumbent",
            "variable_order_fingerprint_start_step", "variable_order_fingerprint_deadline", "variable_order_fingerprint_count",
            "variable_order_fingerprint_evidence", "variable_order_fingerprint_pending", "variable_order_normal_window",
            "variable_order_normal_evidence", "variable_order_normal_pending", "variable_order_normal_count",
            "variable_order_detector_hold_until", "variable_order_install_count", "variable_order_route_count",
        ):
            legacy.pop(key, None)
        restored = Organism.from_state_dict(legacy)
        self.assertEqual(restored.VERSION, 12)
        self.assertFalse(restored.variable_order_learning_enabled)
        self.assertEqual(restored.variable_order_normal_window, 8)

    def test_variable_order_learner_rejects_multiple_direct_features_per_module(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=183)
        organism.enable_context_modules(max_modules=1, afferent_kind="hidden")
        organism.enable_variable_order(3)
        organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-000", 0.2, cell_id="hidden-000")
        organism.install_dendritic_feature(("input-000", "input-002"), "motor-module-000", 0.2, cell_id="hidden-001")
        with self.assertRaises(AssertionError):
            organism.enable_variable_order_learning(18301)


class CompositionalSubstrateTests(unittest.TestCase):
    def _fixture(self):
        organism = Organism.create_default(input_size=4, hidden_size=6, output_size=1, seed=181, max_cells=32, max_synapses=128)
        organism.enable_context_modules(max_modules=3, afferent_kind="hidden")
        organism.enable_variable_order(3)
        left = organism.install_dendritic_feature(("input-000", "input-001"), "motor-module-000", 0.5, cell_id="hidden-000")
        second = organism._recruit_motor_module()
        right = organism.install_dendritic_feature(("input-002", "input-003"), second.id, 0.5, cell_id="hidden-001")
        owner = organism._recruit_motor_module()
        organism.enable_compositional_substrate(19001)
        return organism, left, right, owner.id

    def test_composed_cell_sign_amplitude_and_three_pass_current_input(self):
        organism, left, right, owner_id = self._fixture()
        composed = organism.install_composed_feature((left, right), owner_id, 0.7, cell_id="hidden-002")
        self.assertEqual(organism._propagation_steps(), 3)
        organism._propagate((1.0, 1.0, 1.0, 1.0))
        positive = organism.cells[composed].activation
        organism._propagate((-1.0, 1.0, 1.0, 1.0))
        negative = organism.cells[composed].activation
        self.assertGreater(positive, 0.0)
        self.assertLess(negative, 0.0)
        self.assertGreater(abs(positive), 0.1)
        self.assertGreater(abs(negative), 0.1)
        self.assertEqual(organism.graph.get(left, composed).transmission_delay, 0)
        self.assertEqual(organism.graph.get(right, composed).transmission_delay, 0)

    def test_composed_install_rejects_overlap_duplicate_and_nested_sources(self):
        organism, left, right, owner_id = self._fixture()
        with self.assertRaises(ValueError):
            organism.install_composed_feature((left, left), owner_id, 0.2, cell_id="hidden-002")
        composed = organism.install_composed_feature((left, right), owner_id, 0.2, cell_id="hidden-002")
        with self.assertRaises(ValueError):
            organism.install_composed_feature((left, right), owner_id, 0.2, cell_id="hidden-003")
        with self.assertRaises(AssertionError):
            organism.cells[composed].dendritic_sources = (composed, right)
            organism.validate()

    def test_composed_install_event_is_post_commit_and_exact(self):
        organism, left, right, owner_id = self._fixture()
        before_events = len(organism.events)
        composed = organism.install_composed_feature((left, right), owner_id, 0.35, cell_id="hidden-002")
        self.assertEqual(len(organism.events), before_events + 1)
        event = organism.events[-1]
        self.assertEqual(event["kind"], "composed_feature_installed")
        self.assertEqual(event["cell_id"], composed)
        self.assertEqual(tuple(event["source_cells"]), tuple(sorted((left, right))))
        self.assertEqual(event["owner_module"], owner_id)
        self.assertAlmostEqual(event["strength"], 0.35)
        self.assertEqual(event["feature_key"], organism.composition_module_features[owner_id])
        self.assertEqual(len(event["leaf_closure"]), 2)

    def test_midpoint_resolves_existing_composed_owner_without_direct_verification(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.35, cell_id="hidden-002")
        organism.variable_order_module_features = {
            module_id: key for key, module_id in organism.variable_order_feature_owners.items()
        }
        organism.enable_variable_order_learning(19002)
        organism.enable_variable_order_midpoint_existing_owner_resolution()
        key = organism.composition_module_features[owner_id]
        organism.active_motor_module = "motor-module-000"
        organism.motor_modules["motor-module-000"].dormant = False
        organism.motor_modules[owner_id].dormant = True
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = "motor-module-000"
        organism.variable_order_fingerprint_incumbent = "motor-module-000"
        organism.variable_order_fingerprint_count = 8
        organism.composition_fingerprint_evidence = {
            key: {"sum": -8.0, "sumsq": 8.0, "count": 8.0},
        }
        self.assertTrue(organism._variable_order_midpoint_resolve())
        self.assertEqual(organism.active_motor_module, owner_id)
        self.assertFalse(organism.composition_direct_route_pending)
        self.assertEqual(organism.events[-2]["family"], "composition")
        organism.validate()

    def test_composition_reverse_maps_reject_unknown_or_aliased_modules(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.35, cell_id="hidden-002")
        state = organism.state_dict()
        key = next(iter(state["composition_feature_owners"]))
        state["composition_module_features"]["motor-module-999"] = key
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(state)
        malformed = organism.state_dict()
        malformed["composition_module_features"][owner_id] = "c4:alias::input-000|input-001|input-002|input-003"
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)

    def test_composition_normalizer_seed_and_checkpoint_are_exact(self):
        organism, left, right, owner_id = self._fixture()
        before_v10_rng = organism.variable_order_rng.getstate()
        composed = organism.install_composed_feature((left, right), owner_id, -0.4, cell_id="hidden-002")
        organism.cells[composed].dendritic_normalizer = 0.5
        organism.composition_rng.random()
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())
        self.assertEqual(before_v10_rng, organism.variable_order_rng.getstate())
        organism.cells[composed].dendritic_normalizer = 2.1
        with self.assertRaises(AssertionError):
            organism.validate()

    def test_composition_signal_normalization_is_fixed_before_install_and_persisted(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_signal_normalization(1.72)
        composed = organism.install_composed_feature((left, right), owner_id, 0.7, cell_id="hidden-002")
        self.assertEqual(organism.cells[composed].dendritic_normalizer, 1.72)
        restored = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(restored.state_dict(), organism.state_dict())
        with self.assertRaises(ValueError):
            organism.enable_composition_signal_normalization(1.5)

    def test_composition_install_is_atomic_and_isolates_candidate_traces(self):
        organism, left, right, owner_id = self._fixture()
        candidate = "hidden-002"
        original_max_synapses = organism.resources.max_synapses
        original_energy_used = organism.resources.energy_used
        organism.resources.energy_used = organism.resources.energy_per_step
        organism.resources.max_synapses = len(organism.graph.synapses) + 1
        before = organism.state_dict()
        with self.assertRaises(RuntimeError):
            organism.install_composed_feature((left, right), owner_id, 0.4, cell_id=candidate)
        self.assertEqual(before, organism.state_dict())
        organism.resources.max_synapses = original_max_synapses
        organism.resources.energy_used = original_energy_used
        unrelated = {}
        for synapse in organism.graph.iter_synapses():
            if organism.cells[synapse.destination].kind == "motor_module" and synapse.source != candidate:
                synapse.eligibility_trace = 0.4
                synapse.actor_eligibility_trace = 0.6
                unrelated[synapse.id] = synapse_to_dict(synapse)
            if synapse.source == candidate and organism.cells[synapse.destination].kind == "motor_module":
                synapse.actor_eligibility_trace = 0.9
        organism.install_composed_feature((left, right), owner_id, 0.4, cell_id=candidate)
        for synapse in organism.graph.iter_synapses():
            if synapse.source == candidate and organism.cells[synapse.destination].kind == "motor_module":
                self.assertEqual(synapse.actor_eligibility_trace, 0.0)
        self.assertEqual(
            unrelated,
            {synapse.id: synapse_to_dict(synapse) for synapse in organism.graph.iter_synapses()
             if organism.cells[synapse.destination].kind == "motor_module" and synapse.source != candidate},
        )

    def test_composition_dependency_cells_are_protected_from_structural_adaptation(self):
        organism, left, right, owner_id = self._fixture()
        composed = organism.install_composed_feature((left, right), owner_id, 0.4, cell_id="hidden-002")
        before = {synapse.id: synapse_to_dict(synapse) for synapse in organism.graph.iter_synapses()
                  if synapse.source in (left, right, composed) or synapse.destination == composed}
        organism.structural_interval = 1
        organism._structural_adaptation(Modulators(novelty=1.0))
        after = {synapse.id: synapse_to_dict(synapse) for synapse in organism.graph.iter_synapses()
                 if synapse.source in (left, right, composed) or synapse.destination == composed}
        self.assertEqual(before, after)
        self.assertEqual(organism.cells[composed].activation_type, "dendritic_product_composed")
        self.assertFalse(any(s.source in organism.input_ids and organism.cells[s.destination].kind == "motor_module" for s in organism.graph.iter_synapses()))

    def test_v10_checkpoint_migrates_with_composition_disabled(self):
        organism = Organism.create_default(input_size=3, hidden_size=3, output_size=1, seed=182)
        state = organism.state_dict()
        state["version"] = 10
        state.pop("compositional_substrate_enabled", None)
        state.pop("max_composed_leaves", None)
        state.pop("composition_representation_seed", None)
        state.pop("composition_rng_state", None)
        state.pop("composition_feature_owners", None)
        state.pop("composition_module_features", None)
        state.pop("composition_feature_lineage", None)
        state.pop("composition_install_count", None)
        for payload in state["cells"].values():
            payload.pop("dendritic_normalizer", None)
        restored = Organism.from_state_dict(state)
        self.assertEqual(restored.VERSION, 12)
        self.assertFalse(restored.compositional_substrate_enabled)
        self.assertEqual(restored.composition_feature_owners, {})
        self.assertTrue(all(cell.dendritic_normalizer == 1.0 for cell in restored.cells.values()))

    def test_stage3_candidate_bank_is_graph_derived_and_task_blind(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        bank = organism._composition_feature_bank()
        self.assertEqual(len(bank), 1)
        self.assertEqual(set(bank[0][1:]), {left, right})
        state = organism.state_dict()
        self.assertNotIn("task_features", state)
        self.assertNotIn("target", state)

    def test_stage3_fingerprint_pending_has_owner_step_and_exact_sixteenth_boundary(self):
        organism, left, right, owner_id = self._fixture()
        organism.variable_order_feature_owners["k2:input-000|input-001"] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = owner_id
        organism.variable_order_fingerprint_incumbent = owner_id
        organism.variable_order_fingerprint_start_step = 0
        organism.variable_order_fingerprint_deadline = 16
        organism.context_state = "fingerprint"
        organism.context_pending_mode = "fingerprint"
        organism.context_pending_exploration_value = 0.5
        organism.context_pending_exploration_sigma = 0.2
        organism._propagate((1.0, 1.0, 1.0, 1.0))
        organism._composition_record_fingerprint_pending()
        pending = organism.composition_fingerprint_pending
        self.assertEqual(pending["owner_module"], owner_id)
        self.assertEqual(pending["step"], organism.step_count)
        self.assertEqual(len(pending["eligibilities"]), 1)
        direct = Organism.from_state_dict(organism.state_dict())
        encoded = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
        self.assertEqual(organism.state_dict(), direct.state_dict())
        self.assertEqual(organism.state_dict(), encoded.state_dict())
        malformed = organism.state_dict()
        malformed["composition_fingerprint_pending"]["step"] = organism.step_count - 1
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(malformed)

    def test_stage3_evidence_must_match_current_graph_candidates(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        state = organism.state_dict()
        state["composition_fingerprint_evidence"] = {
            "c4:hidden-100|hidden-101::input-000|input-001|input-002|input-003": {"sum": 1.0, "sumsq": 1.0, "count": 1.0}
        }
        with self.assertRaises(AssertionError):
            Organism.from_state_dict(state)

    def test_stage3_composition_learning_consumes_each_delayed_outcome_and_resolves_on_16th(self):
        organism, left, right, owner_id = self._fixture()
        organism.variable_order_feature_owners["k2:input-000|input-001"] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        self.assertEqual(organism.variable_order_fingerprint_window, 16)
        key = organism._composition_feature_bank()[0][0]
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = owner_id
        organism.variable_order_fingerprint_incumbent = owner_id
        organism.variable_order_fingerprint_start_step = 0
        organism.variable_order_fingerprint_deadline = 16
        organism.context_state = "fingerprint"
        calls = []
        organism._variable_order_route = lambda: calls.append(organism.variable_order_fingerprint_count)
        for step in range(1, 17):
            organism.step_count = step
            organism.composition_fingerprint_pending = {
                "owner_module": owner_id, "step": step, "exploration_value": 0.5,
                "exploration_sigma": 0.2, "eligibilities": [{"key": key, "eligibility": 1.0}],
            }
            organism._composition_accumulate_fingerprint(1.0)
            self.assertEqual(organism.composition_fingerprint_pending, None)
            self.assertEqual(organism.composition_fingerprint_evidence[key]["count"], float(step))
            organism.variable_order_fingerprint_pending = {
                "owner_module": owner_id, "step": step, "eligibilities": [],
            }
            organism._variable_order_accumulate_fingerprint(0.0)
        self.assertEqual(calls, [16])

    def test_stage3_composition_route_recruit_failure_rolls_back_and_existing_owner_reuses(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        key = organism._composition_feature_bank()[0][0]
        evidence = {"sum": 10.0, "sumsq": 10.0, "count": 16.0}
        organism.composition_fingerprint_evidence = {key: dict(evidence)}
        # Existing direct owners make the route take the recruit/install path.
        organism.variable_order_module_features["motor-module-000"] = "placeholder"
        organism.context_max_modules = 4
        organism.resources.max_synapses = len(organism.graph.synapses) + 1
        before = organism.state_dict()
        self.assertIsNone(organism._composition_route(owner_id))
        self.assertEqual(organism.state_dict(), before)
        organism.resources.max_synapses = 128
        organism.variable_order_module_features.clear()
        result = organism._composition_route(owner_id)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], owner_id)
        self.assertEqual(len(organism.motor_modules), 3)
        self.assertEqual(organism._composition_route(owner_id)[0], owner_id)

    def test_stage3_unavailable_causal_metrics_cannot_pass(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        result = {"final_state": organism.state_dict()}
        config = type("Config", (), {"tail_fraction": 0.25})()
        from soma.lifetime import _evaluate_composition_causal_ablation
        diagnostic = _evaluate_composition_causal_ablation(result, config)
        self.assertFalse(diagnostic["available"])
        self.assertFalse(diagnostic["direct_available"])

    def test_stage3_joint_arbitration_prefers_narrowly_stronger_composition(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        key = organism._composition_feature_bank()[0][0]
        organism.composition_fingerprint_evidence = {key: {"sum": 2.279, "sumsq": 1.0, "count": 8.0}}
        result = organism._composition_route(owner_id, (2.205, "k3:direct"))
        self.assertIsNotNone(result)
        self.assertEqual(result[0], owner_id)

    def test_stage3_joint_arbitration_stronger_direct_blocks_noisy_composition(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        key = organism._composition_feature_bank()[0][0]
        organism.composition_fingerprint_evidence = {key: {"sum": 2.279, "sumsq": 1.0, "count": 8.0}}
        before = organism.state_dict()
        self.assertIsNone(organism._composition_route(owner_id, (3.0, "k3:direct")))
        self.assertEqual(organism.state_dict(), before)

    def test_stage3_joint_arbitration_ties_use_deterministic_key_order(self):
        organism, left, right, owner_id = self._fixture()
        organism.enable_composition_learning()
        key = organism._composition_feature_bank()[0][0]
        organism.composition_fingerprint_evidence = {key: {"sum": 2.279, "sumsq": 1.0, "count": 8.0}}
        self.assertIsNone(organism._composition_route(owner_id, (2.279, "a-direct")))
        other, _, _, other_owner = self._fixture()
        other.enable_composition_learning()
        other_key = other._composition_feature_bank()[0][0]
        other.composition_fingerprint_evidence = {other_key: {"sum": 2.279, "sumsq": 1.0, "count": 8.0}}
        self.assertIsNone(other._composition_route(other_owner, (2.279, "z-direct")))

    def test_stage3_composition_retry_can_override_detector_hold_after_negative_streak(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.4, cell_id="hidden-002")
        organism.variable_order_feature_owners["k2:input-000|input-001"] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        organism.variable_order_fingerprint_active = False
        organism.variable_order_fingerprint_owner = None
        organism.variable_order_fingerprint_pending = None
        organism.active_motor_module = owner_id
        organism.context_state = "normal"
        organism.variable_order_detector_hold_until = organism.step_count + 64
        organism.motor_modules[owner_id].negative_streak = organism.context_negative_streak
        organism._context_maybe_switch()
        self.assertTrue(organism.variable_order_fingerprint_active)
        self.assertTrue(any(event["kind"] == "composition_fast_recall_retry_started" for event in organism.events))

    def test_stage3_composition_retry_does_not_override_hold_without_negative_streak(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.4, cell_id="hidden-002")
        organism.variable_order_feature_owners["k2:input-000|input-001"] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = "k2:input-000|input-001"
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        organism.variable_order_fingerprint_active = False
        organism.variable_order_fingerprint_owner = None
        organism.variable_order_fingerprint_pending = None
        organism.active_motor_module = owner_id
        organism.context_state = "normal"
        organism.variable_order_detector_hold_until = organism.step_count + 64
        organism.motor_modules[owner_id].negative_streak = 0
        organism._context_maybe_switch()
        self.assertFalse(organism.variable_order_fingerprint_active)
        self.assertFalse(any(event["kind"] == "composition_fast_recall_retry_started" for event in organism.events))

    def test_stage3_normal_verification_retries_weak_direct_route_during_hold(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.4, cell_id="hidden-002")
        direct_key = "k2:input-000|input-001"
        organism.variable_order_feature_owners[direct_key] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = direct_key
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        organism.variable_order_fingerprint_active = False
        organism.variable_order_fingerprint_owner = None
        organism.variable_order_fingerprint_pending = None
        organism.composition_direct_route_pending = True
        organism.active_motor_module = "motor-module-000"
        organism.context_state = "normal"
        organism.variable_order_detector_hold_until = organism.step_count + 64
        for index in range(8):
            organism.variable_order_normal_pending = {"owner_module": "motor-module-000", "step": index + 1, "eligibilities": [{"key": direct_key, "eligibility": 1.0}]}
            organism.step_count = index + 1
            organism._variable_order_accumulate_normal(1.0 if index < 6 else -1.0)
        self.assertTrue(organism.variable_order_fingerprint_active)
        self.assertTrue(any(event["kind"] == "composition_normal_verification_retry_started" for event in organism.events))

    def test_stage3_normal_verification_keeps_credible_direct_route(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.4, cell_id="hidden-002")
        direct_key = "k2:input-000|input-001"
        organism.variable_order_feature_owners[direct_key] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = direct_key
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        organism.variable_order_fingerprint_active = False
        organism.variable_order_fingerprint_owner = None
        organism.variable_order_fingerprint_pending = None
        organism.active_motor_module = "motor-module-000"
        organism.context_state = "normal"
        organism.variable_order_detector_hold_until = organism.step_count + 64
        for index in range(8):
            organism.variable_order_normal_pending = {"owner_module": "motor-module-000", "step": index + 1, "eligibilities": [{"key": direct_key, "eligibility": 1.0}]}
            organism.step_count = index + 1
            organism._variable_order_accumulate_normal(1.0)
        self.assertFalse(organism.variable_order_fingerprint_active)
        self.assertFalse(any(event["kind"] == "composition_normal_verification_retry_started" for event in organism.events))

    def test_stage3_normal_verification_does_not_retry_composed_owner_without_direct_map(self):
        organism, left, right, owner_id = self._fixture()
        organism.install_composed_feature((left, right), owner_id, 0.4, cell_id="hidden-002")
        direct_key = "k2:input-000|input-001"
        organism.variable_order_feature_owners[direct_key] = "motor-module-000"
        organism.variable_order_module_features["motor-module-000"] = direct_key
        organism.variable_order_feature_owners["k2:input-002|input-003"] = "motor-module-001"
        organism.variable_order_module_features["motor-module-001"] = "k2:input-002|input-003"
        organism.enable_variable_order_learning(19002, max_order=3)
        organism.enable_composition_learning()
        organism.variable_order_fingerprint_active = False
        organism.variable_order_fingerprint_owner = None
        organism.variable_order_fingerprint_pending = None
        organism.active_motor_module = owner_id
        organism.context_state = "normal"
        organism.variable_order_detector_hold_until = organism.step_count + 64
        organism.composition_direct_route_pending = True
        for index in range(8):
            organism.variable_order_normal_pending = {"owner_module": owner_id, "step": index + 1, "eligibilities": [{"key": direct_key, "eligibility": 1.0}]}
            organism.step_count = index + 1
            organism._variable_order_accumulate_normal(-1.0)
        self.assertFalse(organism.variable_order_fingerprint_active)
        self.assertFalse(any(event["kind"] == "composition_normal_verification_retry_started" for event in organism.events))

class ObservationProxy:
    """Small test helper that derives the next observation from a checkpointed env."""

    def __init__(self, environment, action=None):
        if action is None:
            self.values = (environment.target, float(environment.t) / environment.horizon)
            self.reward = environment.last_reward
            self.target = environment.target
        else:
            result = environment.step(action)
            self.values = result.values
            self.reward = result.reward
            self.target = environment.target


if __name__ == "__main__":
    unittest.main()

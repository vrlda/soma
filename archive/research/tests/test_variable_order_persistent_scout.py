import copy
import unittest

import r6_temporal_composition_pilot as temporal_pilot
from soma.organism import Organism


def _organism():
    organism = Organism.create_default(
        input_size=3,
        hidden_size=2,
        output_size=1,
        seed=41,
        max_synapses=256,
        energy_per_step=100.0,
    )
    organism.enable_context_modules(max_modules=2)
    organism.enable_variable_order_learning(991, max_order=3)
    return organism


class PersistentScoutTests(unittest.TestCase):
    def test_default_is_inert_and_v12_migration_is_supported(self):
        organism = _organism()
        state = organism.state_dict()
        for key in (
            "variable_order_persistent_scout_enabled",
            "variable_order_persistent_evidence_budget",
            "variable_order_persistent_min_count",
            "variable_order_persistent_min_z",
            "variable_order_persistent_min_margin",
            "variable_order_persistent_evidence",
            "variable_order_persistent_evidence_count",
            "variable_order_persistent_episode_count",
            "variable_order_persistent_last_telemetry",
        ):
            state.pop(key, None)
        restored = Organism.from_state_dict(state)
        self.assertFalse(restored.variable_order_persistent_scout_enabled)
        self.assertEqual(restored.variable_order_persistent_evidence, {})
        self.assertEqual(restored.variable_order_persistent_evidence_count, 0)

    def test_opt_in_uses_prediction_error_and_keeps_all_feature_families(self):
        organism = _organism()
        organism.enable_variable_order_persistent_scout(evidence_budget=8, min_count=2)
        self.assertIn(("input-000", "input-001", "input-002"), organism.variable_order_feature_order)
        key = organism._dendritic_feature_key(("input-000", "input-001", "input-002"))
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = organism.active_motor_module
        organism.variable_order_fingerprint_pending = {
            "owner_module": organism.active_motor_module,
            "step": organism.step_count,
            "eligibilities": [{"key": key, "eligibility": 1.0}],
            "general_eligibilities": [],
        }
        organism._variable_order_accumulate_fingerprint(1.0, prediction_error=0.25)
        self.assertAlmostEqual(organism.variable_order_fingerprint_evidence[key]["sum"], 0.25)
        self.assertAlmostEqual(organism.variable_order_persistent_evidence[key]["sum"], 0.25)
        self.assertEqual(organism.variable_order_persistent_evidence[key]["count"], 1.0)

    def test_explicit_candidate_subset_excludes_fixed_bias_without_id_heuristic(self):
        organism = Organism.create_default(input_size=4, hidden_size=2, output_size=1,
                                           seed=42, max_synapses=256, energy_per_step=100.0)
        organism.enable_context_modules(max_modules=2)
        organism.enable_variable_order_learning(
            992, max_order=3,
            candidate_input_ids=("input-000", "input-001", "input-002"))
        organism.enable_variable_order_persistent_scout(evidence_budget=8, min_count=2)
        self.assertEqual(organism.variable_order_candidate_input_ids,
                         ("input-000", "input-001", "input-002"))
        self.assertNotIn(("input-000", "input-001", "input-003"), organism.variable_order_feature_order)
        self.assertTrue(all("input-003" not in feature for feature in organism.variable_order_feature_order))
        restored = Organism.from_state_dict(copy.deepcopy(organism.state_dict()))
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_legacy_none_candidate_subset_preserves_full_family(self):
        organism = _organism()
        self.assertIsNone(organism.variable_order_candidate_input_ids)
        self.assertIn(("input-000", "input-001", "input-002"), organism.variable_order_feature_order)

    def test_feature_bank_fallback_respects_explicit_candidate_subset(self):
        organism = Organism.create_default(input_size=4, hidden_size=2, output_size=1,
                                           seed=43, max_synapses=256, energy_per_step=100.0)
        organism.enable_variable_order_learning(
            993, max_order=3,
            candidate_input_ids=("input-000", "input-001", "input-002"))
        organism.variable_order_feature_order = ()
        self.assertTrue(organism._variable_order_feature_bank())
        self.assertTrue(all("input-003" not in feature
                            for feature in organism._variable_order_feature_bank()))

    def test_evidence_and_telemetry_roundtrip_exactly(self):
        organism = _organism()
        organism.enable_variable_order_persistent_scout(evidence_budget=8, min_count=2)
        key = organism._dendritic_feature_key(("input-000", "input-001", "input-002"))
        organism.variable_order_persistent_evidence = {key: {"sum": 1.25, "sumsq": 2.0, "count": 3.0}}
        organism.variable_order_persistent_evidence_count = 3
        organism.variable_order_persistent_episode_count = 1
        organism.variable_order_persistent_last_telemetry = {
            "candidate_count": 1,
            "winner_rank": 1,
            "top_two_scores": [0.88],
            "evidence_count": 3,
            "install_reason": "awaiting_evidence_budget",
        }
        state = organism.state_dict()
        restored = Organism.from_state_dict(copy.deepcopy(state))
        self.assertEqual(restored.state_dict(), state)

    def test_evidence_accumulates_across_episodes_and_resume(self):
        organism = _organism()
        organism.enable_variable_order_persistent_scout(evidence_budget=8, min_count=2)
        organism.variable_order_fingerprint_window = 100
        key = organism._dendritic_feature_key(("input-000", "input-001", "input-002"))
        for index, signal in enumerate((0.25, -0.5)):
            organism.variable_order_fingerprint_active = True
            organism.variable_order_fingerprint_owner = organism.active_motor_module
            organism.variable_order_fingerprint_count = 0
            organism.variable_order_fingerprint_evidence = {}
            organism.variable_order_fingerprint_pending = {
                "owner_module": organism.active_motor_module,
                "step": organism.step_count,
                "eligibilities": [{"key": key, "eligibility": 1.0}],
                "general_eligibilities": [],
            }
            organism._variable_order_accumulate_fingerprint(signal, prediction_error=signal)
            if index == 0:
                # Resume exactly at the episode boundary; cumulative evidence
                # must survive without restarting the scout budget.
                organism = Organism.from_state_dict(copy.deepcopy(organism.state_dict()))
        self.assertEqual(organism.variable_order_persistent_evidence[key]["count"], 2.0)
        self.assertAlmostEqual(organism.variable_order_persistent_evidence[key]["sum"], -0.25)
        resumed = Organism.from_state_dict(copy.deepcopy(organism.state_dict()))
        self.assertEqual(resumed.variable_order_persistent_evidence, organism.variable_order_persistent_evidence)
        self.assertEqual(resumed.variable_order_persistent_evidence_count, 2)

    def test_underpowered_or_tied_winner_never_installs(self):
        organism = _organism()
        organism.enable_variable_order_persistent_scout(evidence_budget=4, min_count=2)
        first = organism._dendritic_feature_key(("input-000", "input-001", "input-002"))
        second = organism._dendritic_feature_key(("input-000", "input-001"))
        organism.variable_order_persistent_evidence = {
            first: {"sum": 4.0, "sumsq": 4.0, "count": 4.0},
            second: {"sum": -4.0, "sumsq": 4.0, "count": 4.0},
        }
        organism.variable_order_persistent_evidence_count = 4
        organism.variable_order_fingerprint_active = True
        organism.variable_order_fingerprint_owner = organism.active_motor_module
        organism._variable_order_persistent_route()
        self.assertEqual(organism.variable_order_feature_owners, {})
        self.assertEqual(organism.variable_order_persistent_last_telemetry["install_reason"], "winner_not_multiplicity_safe")
        self.assertEqual(organism.variable_order_persistent_last_telemetry["candidate_count"], 2)

    def test_budget_runs_consecutive_episodes_and_terminal_decision_does_not_repeat(self):
        examples = temporal_pilot._examples(0, 1, 512)
        organism = temporal_pilot._new_organism(0, "persistent_variable_order")
        state, resources = temporal_pilot._train(organism, examples, 0)
        self.assertGreaterEqual(resources["persistent_scout"]["evidence_count"], 128)
        self.assertGreaterEqual(resources["persistent_scout"]["episode_count"], 8)
        telemetry = resources["persistent_scout"]["telemetry"]
        self.assertIn(telemetry["terminal_state"], ("accepted", "rejected", "inconclusive"))
        self.assertEqual(telemetry["family_size"], 2600)
        self.assertIn("winner_feature_rank", telemetry)
        self.assertIn("winner_z", telemetry)
        self.assertIn("runner_score", telemetry)
        self.assertIn("winner_runner_margin", telemetry)
        resumed = Organism.from_state_dict(copy.deepcopy(state))
        event_count = len(resumed.events)
        route_count = resumed.variable_order_route_count
        resumed._variable_order_persistent_route()
        self.assertEqual(len(resumed.events), event_count)
        self.assertEqual(resumed.variable_order_route_count, route_count)


if __name__ == "__main__":
    unittest.main()

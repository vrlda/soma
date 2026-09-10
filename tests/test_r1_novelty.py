import unittest

from soma import SOMA
from soma.routing.evidence import CircuitEvidence


def _scored(circuit_id, log_evidence, outcomes=20, scored=None):
    item = CircuitEvidence(circuit_id)
    item.log_evidence = float(log_evidence)
    item.normal_outcomes = outcomes
    item.scored_outcomes = outcomes if scored is None else scored
    return item


class R1NoveltyTests(unittest.TestCase):
    def test_scored_outcomes_round_trip_and_legacy_compat(self):
        item = _scored("m1", -3.0)
        self.assertEqual(CircuitEvidence.from_dict(item.to_dict()).to_dict(), item.to_dict())
        legacy = item.to_dict()
        del legacy["scored_outcomes"]
        restored = CircuitEvidence.from_dict(legacy)
        self.assertEqual(restored.scored_outcomes, 0)

    def test_novelty_unjustified_when_fit_good(self):
        model = SOMA.create(input_size=2, seed=31)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        organism.circuit_evidence["motor-module-000"] = _scored("motor-module-000", -5.0).to_dict()
        organism.evidence_recent_best_ll = -0.5
        organism.evidence_recent_count = 20
        # Recent best fit above threshold: circuits explain outcomes, no growth.
        self.assertFalse(organism.evidence_novelty_justified())

    def test_novelty_justified_when_recent_fit_poor(self):
        model = SOMA.create(input_size=2, seed=32)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        organism.circuit_evidence["motor-module-000"] = _scored("motor-module-000", -5.0).to_dict()
        organism.evidence_recent_best_ll = -4.0
        organism.evidence_recent_count = 20
        # Sustained poor recent fit despite long-sampled circuits: grow.
        self.assertTrue(organism.evidence_novelty_justified())

    def test_novelty_requires_min_evidence(self):
        model = SOMA.create(input_size=2, seed=33)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        organism.circuit_evidence["motor-module-000"] = _scored("motor-module-000", -60.0, outcomes=2).to_dict()
        organism.evidence_recent_best_ll = -4.0
        organism.evidence_recent_count = 2
        self.assertFalse(organism.evidence_novelty_justified())

    def test_recent_fit_tracks_outcomes_and_ignores_single_noise(self):
        model = SOMA.create(input_size=2, seed=30)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        organism.step((0.5, -0.25), __import__("soma").Modulators(novelty=0.05, exploration=0.0))
        first_ll = organism.evidence_recent_best_ll
        first_count = organism.evidence_recent_count
        organism.step((-0.5, 0.25), __import__("soma").Modulators(reward=50.0, novelty=0.05, exploration=0.0))
        # One huge surprise moves the average but cannot alone justify growth.
        self.assertGreater(organism.evidence_recent_count, first_count)
        self.assertNotEqual(organism.evidence_recent_best_ll, first_ll)
        self.assertFalse(organism.evidence_novelty_justified())

    def test_novelty_disabled_by_threshold_control(self):
        model = SOMA.create(input_size=2, seed=34)
        organism = model.organism
        organism.use_threshold_router()
        organism.circuit_evidence["motor-module-000"] = _scored("motor-module-000", -60.0).to_dict()
        self.assertFalse(organism.evidence_novelty_justified())
        self.assertTrue(organism._evidence_growth_allowed())

    def test_search_recruit_starts_pending_watch(self):
        model = SOMA.create(input_size=2, seed=35)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        organism.circuit_evidence["motor-module-000"] = _scored("motor-module-000", -5.0).to_dict()
        organism.context_state = "search"
        organism.step_count = 100
        organism.variable_order_learning_enabled = False
        organism.variable_order_fingerprint_enabled = False
        organism.adaptive_dendritic_fingerprint_enabled = False
        organism.adaptive_dendritic_fingerprint_active = False
        before = len(organism.motor_modules)
        organism._context_maybe_switch()
        # Legacy initiation restored: recruits, then watches for confirmation.
        self.assertEqual(len(organism.motor_modules), before + 1)
        self.assertIsNotNone(organism.evidence_pending_growth)
        kinds = [e.get("kind") for e in organism.events]
        self.assertIn("evidence_growth_pending", kinds)
        restored = type(organism).from_state_dict(organism.state_dict())
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_growth_confirms_on_sustained_lead(self):
        model = SOMA.create(input_size=2, seed=37)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        recruited = organism._recruit_motor_module()
        incumbent_id = organism.active_motor_module
        organism.circuit_evidence[recruited.id] = _scored(recruited.id, 0.0).to_dict()
        organism.circuit_evidence[incumbent_id] = _scored(incumbent_id, -10.0).to_dict()
        from soma.routing.evidence import normalize_posterior
        organism.evidence_posterior = normalize_posterior(
            {recruited.id: 0.0, incumbent_id: -10.0}, novelty_mass=0.05)
        organism._evidence_begin_growth_watch(recruited.id, incumbent_id=incumbent_id)
        organism.evidence_pending_growth["edge_ema"] = 1.0
        organism.evidence_pending_growth["edge_n"] = 20
        organism._evidence_check_growth_watch()
        self.assertIsNone(organism.evidence_pending_growth)
        kinds = [e.get("kind") for e in organism.events]
        self.assertIn("evidence_growth_confirmed", kinds)

    def test_growth_reverts_at_deadline_when_incumbent_leads(self):
        model = SOMA.create(input_size=2, seed=38)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        recruited = organism._recruit_motor_module()
        incumbent_id = organism.active_motor_module
        organism.active_motor_module = recruited.id
        organism.circuit_evidence[recruited.id] = _scored(recruited.id, -10.0).to_dict()
        organism.circuit_evidence[incumbent_id] = _scored(incumbent_id, 0.0).to_dict()
        from soma.routing.evidence import normalize_posterior
        organism.evidence_posterior = normalize_posterior(
            {recruited.id: -10.0, incumbent_id: 0.0}, novelty_mass=0.05)
        organism._evidence_begin_growth_watch(recruited.id, incumbent_id=incumbent_id)
        organism.evidence_pending_growth["edge_ema"] = -1.0
        organism.evidence_pending_growth["edge_n"] = 20
        organism.step_count = organism.evidence_pending_growth["deadline_step"] + 1
        organism._evidence_check_growth_watch()
        self.assertIsNone(organism.evidence_pending_growth)
        self.assertEqual(organism.active_motor_module, incumbent_id)
        self.assertTrue(organism.motor_modules[recruited.id].dormant)
        kinds = [e.get("kind") for e in organism.events]
        self.assertIn("evidence_growth_reverted", kinds)
        restored = type(organism).from_state_dict(organism.state_dict())
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_growth_expires_without_yank_on_ambiguity(self):
        model = SOMA.create(input_size=2, seed=39)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        recruited = organism._recruit_motor_module()
        incumbent_id = organism.active_motor_module
        organism.active_motor_module = recruited.id
        organism.circuit_evidence[recruited.id] = _scored(recruited.id, -1.0).to_dict()
        organism.circuit_evidence[incumbent_id] = _scored(incumbent_id, -1.1).to_dict()
        from soma.routing.evidence import normalize_posterior
        organism.evidence_posterior = normalize_posterior(
            {recruited.id: -1.0, incumbent_id: -1.1}, novelty_mass=0.05)
        organism._evidence_begin_growth_watch(recruited.id, incumbent_id=incumbent_id)
        organism.evidence_pending_growth["edge_ema"] = 0.0
        organism.evidence_pending_growth["edge_n"] = 20
        organism.step_count = organism.evidence_pending_growth["deadline_step"] + 1
        organism._evidence_check_growth_watch()
        self.assertIsNone(organism.evidence_pending_growth)
        # Ambiguous evidence: control stays where legacy left it.
        self.assertEqual(organism.active_motor_module, recruited.id)
        kinds = [e.get("kind") for e in organism.events]
        self.assertIn("evidence_growth_expired", kinds)

    def test_probe_outcomes_do_not_move_detector(self):
        model = SOMA.create(input_size=2, seed=36)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        # Drive a few normal steps to settle a detector predictor.
        organism.step((0.5, -0.25), __import__("soma").Modulators(novelty=0.05, exploration=0.0))
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        frozen_predictor = list(candidate.detector_predictor)
        frozen_count = candidate.detector_count
        organism.context_probe_module = candidate.id
        organism.context_probe_queue = []
        organism.context_probe_remaining = 4
        organism.context_probe_incumbent_module = organism.active_motor_module
        organism.context_probe_origin = "alarm"
        organism.pending_motor_module = candidate.id
        organism.context_pending_mode = "search"
        organism._pending_outcome = True
        organism._apply_context_reward(0.3)
        self.assertEqual(list(candidate.detector_predictor), frozen_predictor)
        self.assertEqual(candidate.detector_count, frozen_count)


if __name__ == "__main__":
    unittest.main()

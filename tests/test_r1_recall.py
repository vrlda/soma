import unittest

from soma import SOMA
from soma.routing.evidence import CircuitEvidence, update_evidence


def _evidence_with_outcomes(circuit_id, outcomes):
    item = CircuitEvidence(circuit_id)
    for outcome in outcomes:
        update_evidence(item, outcome)
    return item


class R1RecallTests(unittest.TestCase):
    def test_decisions_require_shadow_first(self):
        model = SOMA.create(input_size=2, seed=21)
        model.organism.use_threshold_router()
        with self.assertRaises(ValueError):
            model.organism.enable_evidence_decisions(True)

    def test_recall_candidate_none_without_dormant_owner(self):
        model = SOMA.create(input_size=2, seed=22)
        model.organism.enable_evidence_router("shadow")
        model.organism.enable_evidence_decisions(True)
        self.assertIsNone(model.organism._evidence_recall_candidate())

    def test_recall_selects_evidenced_dormant_owner(self):
        model = SOMA.create(input_size=2, seed=23)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        recruited = organism._recruit_motor_module()
        recruited.dormant = True
        incumbent_id = organism.active_motor_module
        # Comparative evidence: dormant candidate explains recent outcomes far
        # better than the incumbent; both meet the minimum-evidence floor.
        leader = _evidence_with_outcomes(recruited.id, [0.8] * 20)
        leader.log_evidence = 0.0
        lagging = _evidence_with_outcomes(incumbent_id, [-0.8] * 20)
        lagging.log_evidence = -10.0
        organism.circuit_evidence[recruited.id] = leader.to_dict()
        organism.circuit_evidence[incumbent_id] = lagging.to_dict()
        from soma.routing.evidence import normalize_posterior
        scores = {key: float(value["log_evidence"]) for key, value in organism.circuit_evidence.items()}
        organism.evidence_posterior = normalize_posterior(scores, novelty_mass=0.05)
        self.assertEqual(organism._evidence_recall_candidate(), recruited.id)

    def test_recall_single_outcome_insufficient(self):
        model = SOMA.create(input_size=2, seed=24)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        recruited = organism._recruit_motor_module()
        recruited.dormant = True
        incumbent_id = organism.active_motor_module
        leader = _evidence_with_outcomes(recruited.id, [0.8])
        leader.log_evidence = 0.0
        lagging = _evidence_with_outcomes(incumbent_id, [-0.8])
        lagging.log_evidence = -10.0
        organism.circuit_evidence[recruited.id] = leader.to_dict()
        organism.circuit_evidence[incumbent_id] = lagging.to_dict()
        from soma.routing.evidence import normalize_posterior
        scores = {key: float(value["log_evidence"]) for key, value in organism.circuit_evidence.items()}
        organism.evidence_posterior = normalize_posterior(scores, novelty_mass=0.05)
        self.assertIsNone(organism._evidence_recall_candidate())

    def test_search_state_recall_reactivates_and_replays(self):
        model = SOMA.create(input_size=2, seed=25)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        organism.enable_evidence_decisions(True)
        recruited = organism._recruit_motor_module()
        recruited.dormant = True
        incumbent_id = organism.active_motor_module
        leader = _evidence_with_outcomes(recruited.id, [0.8] * 20)
        leader.log_evidence = 0.0
        lagging = _evidence_with_outcomes(incumbent_id, [-0.8] * 20)
        lagging.log_evidence = -10.0
        organism.circuit_evidence[recruited.id] = leader.to_dict()
        organism.circuit_evidence[incumbent_id] = lagging.to_dict()
        from soma.routing.evidence import normalize_posterior
        scores = {key: float(value["log_evidence"]) for key, value in organism.circuit_evidence.items()}
        organism.evidence_posterior = normalize_posterior(scores, novelty_mass=0.05)
        organism.context_state = "search"
        organism.step_count = 100
        organism.variable_order_detector_hold_until = 0
        organism.adaptive_dendritic_detector_hold_until = 0
        organism.variable_order_fingerprint_active = False
        organism.adaptive_dendritic_fingerprint_active = False
        organism._context_maybe_switch()
        self.assertEqual(organism.active_motor_module, recruited.id)
        self.assertEqual(organism.context_state, "normal")
        self.assertFalse(organism.motor_modules[recruited.id].dormant)
        kinds = [e.get("kind") for e in organism.events]
        self.assertIn("evidence_recall", kinds)
        restored = type(organism).from_state_dict(organism.state_dict())
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_decisions_enabled_persists(self):
        model = SOMA.create(input_size=2, seed=26)
        model.organism.enable_evidence_router("shadow")
        model.organism.enable_evidence_decisions(True)
        restored = type(model.organism).from_state_dict(model.organism.state_dict())
        self.assertTrue(restored.evidence_decisions_enabled)


if __name__ == "__main__":
    unittest.main()

import unittest

from soma import SOMA
from soma.routing.evidence import (
    CircuitEvidence,
    normalize_posterior,
    probe_decision,
    update_evidence,
)


def _seeded(circuit_id, outcomes, log_evidence=None):
    item = CircuitEvidence(circuit_id)
    for outcome in outcomes:
        update_evidence(item, outcome)
    if log_evidence is not None:
        item.log_evidence = float(log_evidence)
    return item


def _enable(organism):
    organism.enable_evidence_router("shadow")
    organism.enable_evidence_decisions(True)
    return organism


def _posterior(organism):
    scores = {key: float(value["log_evidence"]) for key, value in organism.circuit_evidence.items()}
    organism.evidence_posterior = normalize_posterior(scores, novelty_mass=0.05)


class R1ProbeTests(unittest.TestCase):
    def test_probe_decision_boundaries(self):
        self.assertEqual(probe_decision(0.0, 1, 0.0, 1), "continue")
        self.assertEqual(probe_decision(0.0, 8, -10.0, 8), "accept")
        self.assertEqual(probe_decision(-10.0, 8, 0.0, 8), "reject")
        # Strong candidate lead but below floor: keep gathering.
        self.assertEqual(probe_decision(0.0, 1, -10.0, 1), "continue")

    def test_probe_initiates_on_ambiguity(self):
        model = SOMA.create(input_size=2, seed=41)
        organism = _enable(model.organism)
        challenger = organism._recruit_motor_module()
        challenger.dormant = True
        incumbent_id = organism.active_motor_module
        organism.circuit_evidence[challenger.id] = _seeded(challenger.id, [0.5] * 20, log_evidence=-1.0).to_dict()
        organism.circuit_evidence[incumbent_id] = _seeded(incumbent_id, [0.5] * 20, log_evidence=-1.2).to_dict()
        _posterior(organism)
        organism.context_state = "warning"
        organism.step_count = 100
        organism.variable_order_fingerprint_active = False
        organism.adaptive_dendritic_fingerprint_active = False
        self.assertTrue(organism._evidence_maybe_begin_probe())
        self.assertEqual(organism.context_probe_module, challenger.id)
        self.assertEqual(organism.context_probe_origin, "evidence_probe")
        self.assertEqual(organism.context_state, "search")

    def test_no_probe_when_incumbent_decisive(self):
        model = SOMA.create(input_size=2, seed=42)
        organism = _enable(model.organism)
        challenger = organism._recruit_motor_module()
        challenger.dormant = True
        incumbent_id = organism.active_motor_module
        organism.circuit_evidence[challenger.id] = _seeded(challenger.id, [0.5] * 20, log_evidence=-10.0).to_dict()
        organism.circuit_evidence[incumbent_id] = _seeded(incumbent_id, [0.5] * 20, log_evidence=0.0).to_dict()
        _posterior(organism)
        organism.context_state = "warning"
        self.assertFalse(organism._evidence_maybe_begin_probe())
        self.assertIsNone(organism.context_probe_module)

    def test_probe_freezes_models_and_resolves_accept(self):
        model = SOMA.create(input_size=2, seed=43)
        organism = _enable(model.organism)
        challenger = organism._recruit_motor_module()
        challenger.dormant = True
        incumbent_id = organism.active_motor_module
        organism.circuit_evidence[challenger.id] = _seeded(challenger.id, [0.5] * 20, log_evidence=-1.0).to_dict()
        organism.circuit_evidence[incumbent_id] = _seeded(incumbent_id, [0.5] * 20, log_evidence=-1.2).to_dict()
        _posterior(organism)
        organism.context_state = "warning"
        organism.step_count = 100
        organism.variable_order_fingerprint_active = False
        organism.adaptive_dendritic_fingerprint_active = False
        self.assertTrue(organism._evidence_maybe_begin_probe())
        frozen_mean = CircuitEvidence.from_dict(organism.circuit_evidence[challenger.id]).outcome_mean
        # Feed clearly challenger-favoring probe outcomes through the live path.
        for _ in range(organism.context_probe_max_steps + 2):
            if organism.context_probe_module is None:
                break
            pending = organism.pending_motor_module
            organism.pending_motor_module = organism.context_probe_module
            organism.context_pending_mode = "search"
            organism._pending_outcome = True
            organism._apply_context_reward(0.0)
            organism.pending_motor_module = pending
            organism._context_maybe_switch()
        live = CircuitEvidence.from_dict(organism.circuit_evidence[challenger.id])
        self.assertEqual(live.outcome_mean, frozen_mean)
        self.assertGreater(live.probe_scored, 0)
        kinds = [e.get("kind") for e in organism.events]
        self.assertTrue("evidence_probe_accepted" in kinds or "evidence_probe_rejected" in kinds)
        restored = type(organism).from_state_dict(organism.state_dict())
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_probe_instance_replays_exactly(self):
        left = SOMA.create(input_size=2, seed=44)
        _enable(left.organism)
        challenger = left.organism._recruit_motor_module()
        challenger.dormant = True
        incumbent_id = left.organism.active_motor_module
        left.organism.circuit_evidence[challenger.id] = _seeded(challenger.id, [0.5] * 20, log_evidence=-1.0).to_dict()
        left.organism.circuit_evidence[incumbent_id] = _seeded(incumbent_id, [0.5] * 20, log_evidence=-1.2).to_dict()
        _posterior(left.organism)
        left.organism.context_state = "warning"
        left.organism.step_count = 100
        left.organism.variable_order_fingerprint_active = False
        left.organism.adaptive_dendritic_fingerprint_active = False
        left.organism._evidence_maybe_begin_probe()
        right = SOMA(organism=type(left.organism).from_state_dict(left.organism.state_dict()))
        self.assertEqual(right.organism.state_dict(), left.organism.state_dict())
        self.assertEqual(right.organism.context_probe_module, challenger.id)


if __name__ == "__main__":
    unittest.main()

import copy
import unittest

from soma import SOMA


class R1ShadowTests(unittest.TestCase):
    def test_default_mode_is_evidence_shadow(self):
        model = SOMA.create(input_size=2, seed=7)
        self.assertEqual(model.organism.evidence_router_mode, "shadow")
        self.assertTrue(model.organism.evidence_decisions_enabled)
        before = model.organism.state_dict()
        model.step((0.5, -0.25), exploration=0.0)
        model.step((-0.5, 0.25), reward=0.1, exploration=0.0)
        kinds = [e.get("kind") for e in model.organism.events]
        self.assertIn("evidence_shadow", kinds)
        self.assertGreater(model.organism.evidence_shadow_steps, 0)
        self.assertEqual(before["step_count"] + 2, model.organism.state_dict()["step_count"])

    def test_threshold_router_named_control_is_silent(self):
        model = SOMA.create(input_size=2, seed=7)
        model.organism.use_threshold_router()
        self.assertEqual(model.organism.evidence_router_mode, "threshold")
        self.assertFalse(model.organism.evidence_decisions_enabled)
        model.step((0.5, -0.25), exploration=0.0)
        model.step((-0.5, 0.25), reward=0.1, exploration=0.0)
        kinds = [e.get("kind") for e in model.organism.events]
        self.assertNotIn("evidence_shadow", kinds)
        self.assertEqual(model.organism.circuit_evidence, {})
        self.assertEqual(model.organism.evidence_shadow_steps, 0)

    def test_shadow_accumulates_and_replays_exactly(self):
        left = SOMA.create(input_size=2, seed=11)
        left.organism.enable_evidence_router("shadow")
        inputs = [(0.5, -0.25), (-0.5, 0.25), (0.9, 0.1), (-0.2, -0.8)]
        left.step(inputs[0], exploration=0.0)
        for idx, values in enumerate(inputs[1:], start=1):
            left.step(values, reward=0.05 * idx, exploration=0.0)
        report = left.organism.evidence_report()
        self.assertEqual(report["mode"], "shadow")
        self.assertGreater(report["shadow_steps"], 0)
        self.assertAlmostEqual(sum(report["posterior"].values()), 1.0)
        # Direct state round-trip.
        restored = SOMA(organism=type(left.organism).from_state_dict(left.organism.state_dict()))
        self.assertEqual(restored.organism.state_dict(), left.organism.state_dict())
        # JSON save/load round-trip.
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "shadow.json")
            left.save(path)
            loaded = SOMA.load(path)
            self.assertEqual(loaded.organism.state_dict(), left.organism.state_dict())
        # Continued steps stay deterministic.
        right = SOMA(organism=type(left.organism).from_state_dict(copy.deepcopy(left.organism.state_dict())))
        left_out = left.step((0.1, 0.2), reward=0.0, exploration=0.0)
        right_out = right.step((0.1, 0.2), reward=0.0, exploration=0.0)
        self.assertEqual(left_out, right_out)
        self.assertEqual(left.organism.state_dict(), right.organism.state_dict())

    def test_conditional_scoring_separates_circuits(self):
        model = SOMA.create(input_size=2, seed=19)
        organism = model.organism
        organism.enable_evidence_router("shadow")
        challenger = organism._recruit_motor_module()
        incumbent_id = organism.active_motor_module
        features = organism._context_feature_values(0.5)
        dim = len(features)
        # Incumbent predicts +1, challenger predicts -1 on shared features.
        incumbent = organism.motor_modules[incumbent_id]
        incumbent.detector_predictor = [0.0] * dim
        incumbent.detector_predictor[0] = 1.0
        incumbent.detector_variance = 0.01
        challenger.detector_predictor = [0.0] * dim
        challenger.detector_predictor[0] = -1.0
        challenger.detector_variance = 0.01
        organism.context_pending_features = tuple(features)
        organism.context_pending_mode = "normal"
        organism.context_state = "normal"
        organism.pending_motor_module = incumbent_id
        organism._evidence_shadow_update(incumbent_id, -1.0)
        challenger_score = organism.circuit_evidence[challenger.id]["log_evidence"]
        incumbent_score = organism.circuit_evidence[incumbent_id]["log_evidence"]
        self.assertGreater(challenger_score, incumbent_score)
        self.assertGreater(
            organism.evidence_posterior[challenger.id],
            organism.evidence_posterior[incumbent_id],
        )

    def test_legacy_checkpoint_migrates(self):
        model = SOMA.create(input_size=2, seed=13)
        legacy = model.organism.state_dict()
        for key in ("evidence_router_mode", "circuit_evidence", "evidence_posterior", "evidence_shadow_steps"):
            legacy.pop(key, None)
        restored = type(model.organism).from_state_dict(legacy)
        self.assertEqual(restored.evidence_router_mode, "shadow")
        self.assertTrue(restored.evidence_decisions_enabled)
        self.assertEqual(restored.circuit_evidence, {})

    def test_shadow_rejects_bad_mode(self):
        model = SOMA.create(input_size=2, seed=17)
        with self.assertRaises(ValueError):
            model.organism.enable_evidence_router("evidence")


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from soma.routing.evidence import (
    CircuitEvidence,
    calibration_metrics,
    normalize_posterior,
    should_return_to_incumbent,
    should_switch,
    student_t_logpdf,
    update_evidence,
)


class R1EvidenceTests(unittest.TestCase):
    def test_schema_round_trip(self):
        item = CircuitEvidence("m1", outcome_mean=0.5, outcome_scale=0.2, eff_n=10.0)
        item.posterior = 0.7
        restored = CircuitEvidence.from_dict(item.to_dict())
        self.assertEqual(restored.to_dict(), item.to_dict())

    def test_schema_rejects_bad_version(self):
        payload = CircuitEvidence("m1").to_dict()
        payload["version"] = 999
        with self.assertRaises(ValueError):
            CircuitEvidence.from_dict(payload)

    def test_schema_rejects_nonfinite(self):
        payload = CircuitEvidence("m1").to_dict()
        payload["outcome_mean"] = float("nan")
        with self.assertRaises(ValueError):
            CircuitEvidence.from_dict(payload)

    def test_student_t_heavy_tail(self):
        near = student_t_logpdf(0.1, 0.0, 1.0)
        far = student_t_logpdf(10.0, 0.0, 1.0)
        self.assertTrue(math.isfinite(near) and math.isfinite(far))
        self.assertGreater(near, far)

    def test_update_bounded_and_deterministic(self):
        first = CircuitEvidence("a")
        second = CircuitEvidence("a")
        ll1 = update_evidence(first, 100.0, step=3)
        ll2 = update_evidence(second, 100.0, step=3)
        self.assertEqual(ll1, ll2)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreaterEqual(ll1, -5.0)
        self.assertLessEqual(ll1, 5.0)
        self.assertEqual(first.normal_outcomes, 1)

    def test_update_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            update_evidence(CircuitEvidence("a"), float("inf"))

    def test_posterior_reserves_novelty(self):
        posterior = normalize_posterior({"m1": 0.0, "m2": -1.0}, novelty_mass=0.1)
        self.assertAlmostEqual(posterior["__novelty__"], 0.1)
        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertGreater(posterior["m1"], posterior["m2"])

    def test_single_noisy_reward_cannot_switch(self):
        self.assertFalse(should_switch(0.9, 0.1, 1, 100))

    def test_sustained_evidence_switches(self):
        self.assertTrue(should_switch(0.8, 0.4, 32, 64))
        self.assertFalse(should_return_to_incumbent(0.8, 0.4))
        self.assertTrue(should_return_to_incumbent(0.4, 0.8))

    def test_calibration_metrics(self):
        metrics = calibration_metrics([0.9, 0.1, 0.8], [1, 0, 1])
        self.assertEqual(metrics["n"], 3)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)
        empty = calibration_metrics([], [])
        self.assertEqual(empty["n"], 0)


if __name__ == "__main__":
    unittest.main()

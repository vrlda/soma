import unittest

from soma.evaluation.embodied import (
    CartSim,
    SafetyController,
    run_cart,
    run_replay,
)


class R12EmbodiedTests(unittest.TestCase):
    def test_sim_deterministic(self):
        first = run_replay(0, [0.5, -0.3, 0.0] * 20)
        second = run_replay(0, [0.5, -0.3, 0.0] * 20)
        self.assertEqual(first, second)

    def test_sim_state_round_trip(self):
        sim = CartSim(3, goal=-0.6)
        sim.reset()
        sim.step(0.5)
        restored = CartSim.from_state_dict(sim.state_dict())
        self.assertEqual(restored.state_dict(), sim.state_dict())
        with self.assertRaises(ValueError):
            CartSim.from_state_dict({"version": 999})

    def test_safety_bounds_max_force(self):
        report = run_replay(1, [1.0] * 300)
        self.assertEqual(report["stops"], 0)
        self.assertGreater(report["engagements"], 0)

    def test_force_clamp_counted(self):
        sim = CartSim(0)
        sim.reset()
        _, _ = sim.step(5.0)
        self.assertEqual(sim.clamps, 1)

    def test_learn_runs_short(self):
        tails = run_cart(0, "learn", steps=300)
        self.assertIn("tail_mean", tails)
        self.assertEqual(tails["stops"], 0)


if __name__ == "__main__":
    unittest.main()

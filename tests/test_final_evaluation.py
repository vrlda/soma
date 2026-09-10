import json
import unittest

from soma import Organism
from soma.lifetime import (
    FINAL_LIFETIME_SEQUENCE,
    FinalLifetimeConfig,
    FinalLifetimeEnvironment,
    run_final_lifetime,
    run_final_lifetime_benchmark,
)


class FinalEvaluationTests(unittest.TestCase):
    def test_environment_is_phase_blind_and_checkpoint_exact(self):
        config = FinalLifetimeConfig(seeds=(0,))
        environment = FinalLifetimeEnvironment(17, 19, config)
        observation = environment.reset()
        self.assertFalse(hasattr(observation, "phase"))
        for _ in range(37):
            observation = environment.step(0.1)
        restored = FinalLifetimeEnvironment.from_state_dict(json.loads(json.dumps(environment.state_dict())))
        self.assertEqual(restored.state_dict(), environment.state_dict())
        self.assertEqual(restored.step(-0.2), environment.step(-0.2))

    def test_one_lifetime_grows_reuses_prunes_and_stays_bounded(self):
        result = run_final_lifetime(0, FinalLifetimeConfig(seeds=(0,)))
        self.assertGreaterEqual(result["general_growth"], 2)
        self.assertGreater(result["general_reuse"], 0)
        self.assertGreater(result["general_prunes"], 0)
        self.assertTrue(result["resource_bounded"])
        self.assertEqual(len(result["phases"]), len(FINAL_LIFETIME_SEQUENCE))

    def test_arbitrary_json_checkpoint_boundaries_match_uninterrupted_lifetime(self):
        config = FinalLifetimeConfig(seeds=(0,))
        direct = run_final_lifetime(0, config)
        resumed = run_final_lifetime(0, config, checkpoint_steps=(137, 1023, 3073, 5501))
        self.assertEqual(resumed, direct)

    def test_control_matrix_is_explicit_and_evaluator_oracle_is_separate(self):
        report = run_final_lifetime_benchmark(FinalLifetimeConfig(seeds=(0,)))
        self.assertEqual(set(report["runs"]), {"soma", "structure_disabled", "learning_disabled", "fixed_topology", "replay"})
        self.assertTrue(report["protocol"]["oracle_is_evaluator_only"])
        self.assertEqual(set(report["protocol"]["controls"]), set(report["runs"]) | {"oracle"})
        self.assertTrue(all(value == 1.0 for value in report["final_lifetime"]["oracle_phase_tail_skill"].values()))
        self.assertIn("all_passed", report["final_lifetime"])

    def test_final_environment_compound_checkpoint_loads(self):
        config = FinalLifetimeConfig(seeds=(0,))
        environment = FinalLifetimeEnvironment(23, 29, config)
        organism = Organism.create_default(input_size=6, hidden_size=6, seed=31)
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "final.json")
            organism.save_checkpoint(path, environment)
            restored, restored_environment = Organism.load_checkpoint(path)
        self.assertEqual(restored.state_dict(), organism.state_dict())
        self.assertEqual(restored_environment.state_dict(), environment.state_dict())


if __name__ == "__main__":
    unittest.main()

import json
import os
import subprocess
import sys
import tempfile
import unittest

from soma import SOMA
from soma.lifetime import ContinuousLifetimeBenchmarkConfig, ContinuousLifetimeEnvironment
from soma.organism import Organism


class PublicModelTests(unittest.TestCase):
    def test_public_model_save_load_and_continue_exactly(self):
        model = SOMA.create(input_size=5, seed=41, reward_delay=3)
        model.step((1.0, -1.0, 0.9, -0.8, 0.7), exploration=0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.soma.json")
            model.save(path)
            restored = SOMA.load(path)
            self.assertEqual(restored.organism.state_dict(), model.organism.state_dict())
            left = model.step((-1.0, 1.0, -0.9, 0.8, -0.7), reward=0.0, exploration=0.2)
            right = restored.step((-1.0, 1.0, -0.9, 0.8, -0.7), reward=0.0, exploration=0.2)
            self.assertEqual(left, right)
            self.assertEqual(restored.organism.state_dict(), model.organism.state_dict())

    def test_public_constructor_exposes_final_capacity_profile(self):
        model = SOMA.create(
            input_size=6, hidden_size=10, seed=43, reward_delay=3,
            max_cells=40, max_synapses=320, max_modules=6,
            energy_per_step=40.0, prune_reuse_ceiling=2,
        )
        self.assertEqual(model.organism.context_max_modules, 6)
        self.assertEqual(model.organism.resources.energy_per_step, 40.0)
        self.assertEqual(model.organism.general_prune_reuse_ceiling, 2)

    def test_version_11_checkpoint_migrates_to_current_version(self):
        model = SOMA.create(input_size=2, seed=7)
        legacy = model.organism.state_dict()
        legacy["version"] = 11
        restored = Organism.from_state_dict(legacy)
        self.assertEqual(restored.state_dict()["version"], Organism.VERSION)

    def test_continuous_environment_checkpoint_loads(self):
        config = ContinuousLifetimeBenchmarkConfig(seeds=(0,))
        environment = ContinuousLifetimeEnvironment(17, 19, config)
        organism = SOMA.create(input_size=5, seed=9, reward_delay=3).organism
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "lifetime.json")
            organism.save_checkpoint(path, environment)
            restored, restored_environment = Organism.load_checkpoint(path)
            self.assertEqual(restored.state_dict(), organism.state_dict())
            self.assertEqual(restored_environment.state_dict(), environment.state_dict())

    def test_cli_create_step_and_inspect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")
            create = subprocess.run(
                [sys.executable, "-m", "soma", "create", path, "--inputs", "2", "--seed", "5"],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(create.stdout)["steps"], 0)
            step = subprocess.run(
                [sys.executable, "-m", "soma", "step", path, "--input", "[0.5,-0.25]", "--exploration", "0"],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(len(json.loads(step.stdout)["outputs"]), 1)
            inspect = subprocess.run(
                [sys.executable, "-m", "soma", "inspect", path, "--events", "0"],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(inspect.stdout)["steps"], 1)


if __name__ == "__main__":
    unittest.main()

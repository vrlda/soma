import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest

from soma import Modulators, Observation, Organism, SOMA
from soma.lifetime import CONTINUOUS_LIFETIME_MANIFEST, ContinuousLifetimeBenchmarkConfig, ContinuousLifetimeEnvironment


class ReliabilityTests(unittest.TestCase):
    def test_corrupt_model_and_compound_checkpoints_are_rejected(self):
        state = SOMA.create(input_size=5, seed=3, reward_delay=3).organism.state_dict()
        corruptions = []
        unsupported = copy.deepcopy(state)
        unsupported["version"] = 999
        corruptions.append(unsupported)
        bad_resources = copy.deepcopy(state)
        bad_resources["resources"]["counters"]["cells"] += 1
        corruptions.append(bad_resources)
        bad_strength = copy.deepcopy(state)
        bad_strength["synapses"][0]["strength"] = float("nan")
        corruptions.append(bad_strength)
        bad_delay = copy.deepcopy(state)
        bad_delay["delayed_credit_queue"] = [{"pending_motor_module": "missing"}] * 4
        corruptions.append(bad_delay)
        for corrupted in corruptions:
            with self.assertRaises((ValueError, AssertionError)):
                Organism.from_state_dict(corrupted)
        with self.assertRaises(ValueError):
            Organism.from_state_dict([])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.json")
            with open(path, "w") as handle:
                json.dump({"checkpoint_version": 99, "organism": state, "environment": {}}, handle)
            with self.assertRaises(ValueError):
                Organism.load_checkpoint(path)

    def test_randomized_long_run_preserves_graph_and_resource_invariants(self):
        model = SOMA.create(input_size=5, seed=101, reward_delay=3, max_cells=32, max_synapses=256)
        rng = random.Random(202)
        rewards = [0.0, 0.0, 0.0]
        for step in range(2500):
            values = tuple(rng.uniform(-1.0, 1.0) for _ in range(5))
            result = model.step(values, reward=rewards.pop(0), novelty=0.1, exploration=0.2)
            target = math.prod(1.0 if value >= 0.0 else -1.0 for value in values)
            rewards.append(max(-1.0, min(1.0, (1.0 - (result.outputs[0] - target) ** 2) / 4.0)))
            if step % 100 == 0:
                model.organism.validate()
                self.assertEqual(model.organism.resources.counters["cells"], len(model.organism.cells))
                self.assertEqual(model.organism.resources.counters["synapses"], len(model.organism.graph.synapses))
        model.organism.validate()
        self.assertLessEqual(len(model.organism.cells), 32)
        self.assertLessEqual(len(model.organism.graph.synapses), 256)

    def test_full_lifetime_replays_exactly_across_arbitrary_checkpoints(self):
        config = ContinuousLifetimeBenchmarkConfig(seeds=(24,))
        organism_seed, input_seed, noise_seed, exploration_seed = CONTINUOUS_LIFETIME_MANIFEST[24]

        def execute(checkpoint_steps):
            environment = ContinuousLifetimeEnvironment(input_seed, noise_seed, config)
            model = SOMA.create(input_size=5, seed=organism_seed, reward_delay=config.reward_delay)
            tape_rng = random.Random(exploration_seed)
            model.organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
            observation = environment.reset()
            actions = []
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "exact.json")
                for step in range(environment.horizon):
                    result = model.step(observation.values, reward=observation.reward, novelty=config.novelty, exploration=config.exploration)
                    actions.append(result.outputs)
                    observation = environment.step(result.outputs[0])
                    if step + 1 in checkpoint_steps:
                        model.organism.save_checkpoint(path, environment)
                        organism, environment = Organism.load_checkpoint(path)
                        model = SOMA(organism)
                        values = environment.inputs[environment.t] if environment.t < environment.horizon else environment.inputs[-1]
                        observation = Observation(values, environment.last_reward, environment.t >= environment.horizon)
                model.apply_outcome(observation.reward)
                return actions, model.organism.state_dict(), environment.state_dict()

        uninterrupted = execute(set())
        resumed = execute({137, 1023, 4097, 6501})
        self.assertEqual(resumed, uninterrupted)

    def test_cross_process_cli_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, name) for name in ("a.json", "b.json")]
            for path in paths:
                subprocess.run([sys.executable, "-m", "soma", "create", path, "--inputs", "2", "--seed", "29"], check=True, capture_output=True)
                for values, reward in (("[0.4,-0.2]", "0"), ("[-0.3,0.8]", "0.15"), ("[0.9,0.1]", "-0.05")):
                    subprocess.run([sys.executable, "-m", "soma", "step", path, "--input", values, "--reward", reward], check=True, capture_output=True)
            with open(paths[0], "rb") as left, open(paths[1], "rb") as right:
                self.assertEqual(left.read(), right.read())

    def test_failed_atomic_save_and_invalid_cli_leave_checkpoint_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")
            model = SOMA.create(input_size=2, seed=31)
            model.save(path)
            with open(path, "rb") as handle:
                original = handle.read()
            model.organism.events.append({"not_json": object()})
            with self.assertRaises(TypeError):
                model.save(path)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)
            self.assertFalse(any(name.startswith(".soma-") for name in os.listdir(directory)))
            failed = subprocess.run(
                [sys.executable, "-m", "soma", "step", path, "--input", "not-json"],
                capture_output=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)

    def test_release_performance_envelope(self):
        model = SOMA.create(input_size=5, seed=53, reward_delay=3)
        started = time.perf_counter()
        for step in range(1000):
            value = 1.0 if step % 2 else -1.0
            model.step((value, -value, value, -value, value), reward=0.1 if step >= 3 else 0.0, exploration=0.2)
        elapsed = time.perf_counter() - started
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")
            model.save(path)
            size = os.path.getsize(path)
            restore_started = time.perf_counter()
            restored = SOMA.load(path)
            restore_elapsed = time.perf_counter() - restore_started
        restored.organism.validate()
        self.assertLess(elapsed / 1000.0, 0.05)
        self.assertLess(size, 20 * 1024 * 1024)
        self.assertLess(restore_elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()

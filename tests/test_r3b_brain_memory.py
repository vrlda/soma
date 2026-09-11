import os
import tempfile
import unittest

from soma.organism import Organism


class BrainMemoryTests(unittest.TestCase):
    def test_enable_observe_predict(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=0)
        organism.enable_sequence_memory((0, 1), max_order=4, max_circuits=128)
        for symbol in [0, 0, 1, 1] * 25:
            organism.observe_sequence_event(symbol)
        distribution, order = organism.sequence_distribution()
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreaterEqual(order, 0)
        organism.validate()

    def test_disabled_by_default_and_guarded(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=1)
        self.assertIsNone(organism.sequence_memory)
        with self.assertRaises(ValueError):
            organism.observe_sequence_event(0)
        with self.assertRaises(ValueError):
            organism.sequence_distribution()
        with self.assertRaises(ValueError):
            organism.reset_sequence_history()
        organism.validate()

    def test_double_enable_rejected(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=2)
        organism.enable_sequence_memory((0, 1))
        with self.assertRaises(ValueError):
            organism.enable_sequence_memory((0, 1))

    def test_checkpoint_round_trip_exact(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=3)
        organism.enable_sequence_memory((0, 1), max_order=6, max_circuits=256)
        for symbol in [0, 1, 1, 0, 0, 1] * 30:
            organism.observe_sequence_event(symbol)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.json")
            organism.save(path)
            restored = Organism.load(path)
            self.assertEqual(restored.state_dict(), organism.state_dict())
            self.assertEqual(restored.sequence_distribution(), organism.sequence_distribution())
        organism.observe_sequence_event(1)
        restored.observe_sequence_event(1)
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_legacy_checkpoint_migrates_empty(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=4)
        payload = organism.state_dict()
        del payload["sequence_memory"]
        del payload["sequence_memory_symbols"]
        restored = Organism.from_state_dict(payload)
        self.assertIsNone(restored.sequence_memory)
        restored.validate()

    def test_fuse_guards(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=6)
        with self.assertRaises(ValueError):
            organism.fuse_with_memory(0.5, 1)
        organism.enable_sequence_memory((0, 1))
        with self.assertRaises(ValueError):
            organism.fuse_with_memory(2.0, 1)
        with self.assertRaises(ValueError):
            organism.fuse_with_memory(0.5, 7)

    def test_fuse_trust_follows_evidence(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=7)
        organism.enable_sequence_memory((0, 1), max_order=4, max_circuits=128)
        for symbol in [1, 1, 1, 1] * 20:
            organism.observe_sequence_event(symbol)
        distribution, _ = organism.sequence_distribution()
        memory_probability = distribution[1]
        fused = organism.fuse_with_memory(0.0, 1)
        # Blend stays between its inputs at cold-start trust.
        self.assertGreaterEqual(fused, 0.0)
        self.assertLessEqual(fused, memory_probability)
        before = organism.fusion_ll_memory - organism.fusion_ll_motor
        organism.observe_sequence_event(1)
        after = organism.fusion_ll_memory - organism.fusion_ll_motor
        self.assertGreater(after, before)
        # Earned trust pulls later fusions toward memory.
        later = organism.fuse_with_memory(0.0, 1)
        self.assertGreater(later, fused)

    def test_fuse_checkpoint_resume_exact(self):
        import tempfile
        import os
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=8)
        organism.enable_sequence_memory((0, 1), max_order=4, max_circuits=128)
        stream = [0, 1, 1, 0, 1, 0, 0, 1] * 10
        for symbol in stream[:40]:
            organism.observe_sequence_event(symbol)
            organism.fuse_with_memory(0.4, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.json")
            organism.save(path)
            restored = Organism.load(path)
            self.assertEqual(restored.state_dict(), organism.state_dict())
        for symbol in stream[40:]:
            organism.observe_sequence_event(symbol)
            left = organism.fuse_with_memory(0.4, 1)
            restored.observe_sequence_event(symbol)
            right = restored.fuse_with_memory(0.4, 1)
            self.assertEqual(left, right)
        self.assertEqual(restored.state_dict(), organism.state_dict())

    def test_reset_keeps_circuits(self):
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=5)
        organism.enable_sequence_memory((0, 1), max_order=4, max_circuits=128)
        for symbol in [0, 0, 1, 1] * 10:
            organism.observe_sequence_event(symbol)
        created = organism.sequence_memory.circuits_created
        organism.reset_sequence_history()
        self.assertEqual(organism.sequence_memory.circuits_created, created)
        self.assertGreater(len(organism.sequence_memory.circuits), 1)


if __name__ == "__main__":
    unittest.main()

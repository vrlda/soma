import os
import tempfile
import unittest

from soma.organism import Organism


def _brain(seed=0):
    organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=seed)
    organism.enable_sequence_memory((0, 1), max_order=6, max_circuits=512)
    for symbol in ([0, 0, 1, 1, 0, 1] * 30):
        organism.observe_sequence_event(symbol)
    return organism


class QuarantineTests(unittest.TestCase):
    def test_staging_learns_nothing(self):
        organism = _brain()
        before = organism.state_dict()
        before.pop("quarantine_staged")
        before.pop("quarantine_receipts")
        receipt = organism.quarantine_events([1, 1, 1, 0, 1, 0] * 20)
        self.assertEqual(receipt, 1)
        after = organism.state_dict()
        after.pop("quarantine_staged")
        after.pop("quarantine_receipts")
        self.assertEqual(before, after)
        self.assertEqual(organism.sequence_distribution(), organism.sequence_distribution())

    def test_approve_learns_and_clears(self):
        organism = _brain()
        organism.quarantine_events([1] * 40)
        count = organism.approve_quarantine()
        self.assertEqual(count, 40)
        self.assertEqual(organism.quarantine_staged, [])
        distribution, _ = organism.sequence_distribution()
        self.assertGreater(distribution[1], 0.5)

    def test_discard_drops(self):
        organism = _brain()
        before = organism.sequence_memory.state_dict()
        organism.quarantine_events([1, 0] * 30)
        self.assertEqual(organism.discard_quarantine(), 60)
        self.assertEqual(organism.sequence_memory.state_dict(), before)

    def test_bounds_and_guards(self):
        organism = _brain()
        organism.quarantine_max_staged = 10
        with self.assertRaises(ValueError):
            organism.quarantine_events([0] * 11)
        with self.assertRaises(ValueError):
            organism.quarantine_events([7])
        with self.assertRaises(ValueError):
            organism.approve_quarantine(weight=0)
        empty = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=9)
        with self.assertRaises(ValueError):
            empty.quarantine_events([0])
        with self.assertRaises(ValueError):
            empty.approve_quarantine()

    def test_round_trip_with_staged(self):
        organism = _brain()
        organism.quarantine_events([0, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.json")
            organism.save(path)
            restored = Organism.load(path)
            self.assertEqual(restored.state_dict(), organism.state_dict())
            self.assertEqual(restored.quarantine_staged, [0, 1, 1])

    def test_poison_slice_leaves_predictions_exact(self):
        from soma.evaluation.english import bit_stream, load_corpus, split_chapters, partition_documents
        data = load_corpus("data/e0/alice.txt")
        acquisition, _, _, _ = partition_documents(split_chapters(data), 0)
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=11)
        organism.enable_sequence_memory((0, 1), max_order=8, max_circuits=4096)
        bits, _ = bit_stream(acquisition[:6000])
        for symbol in bits[:-1]:
            organism.observe_sequence_event(symbol)
        organism.reset_sequence_history()
        probe = bits[1000:1200]
        for symbol in probe:
            organism.observe_sequence_event(symbol, learn=False)
        before = organism.sequence_distribution()
        poison = [1 - int(bit) for bit in probe]
        organism.reset_sequence_history()
        organism.quarantine_events(poison)
        for symbol in probe:
            organism.observe_sequence_event(symbol, learn=False)
        after = organism.sequence_distribution()
        self.assertEqual(before, after)
        organism.approve_quarantine()
        organism.reset_sequence_history()
        for symbol in probe:
            organism.observe_sequence_event(symbol, learn=False)
        moved = organism.sequence_distribution()
        self.assertNotEqual(before, moved)


if __name__ == "__main__":
    unittest.main()

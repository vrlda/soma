import unittest

from soma.evaluation.english import (
    byte_unigram_bits,
    load_corpus,
    partition_documents,
    run_english_sequence_memory,
    split_chapters,
)
from soma.memory import SequenceCircuitMemory


class SequenceCircuitMemoryTests(unittest.TestCase):
    def test_learns_ordered_events_not_only_marginal(self):
        memory = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=128)
        stream = [0, 0, 1, 1] * 100
        correct = 0
        scored = 0
        for index, symbol in enumerate(stream[:-1]):
            memory.observe(symbol)
            prediction = max(memory.distribution()[0], key=memory.distribution()[0].get)
            if index >= 32:
                correct += int(prediction == stream[index + 1])
                scored += 1
        self.assertGreater(correct / scored, 0.95)

    def test_state_round_trip_is_exact(self):
        memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=512)
        for symbol in [0, 1, 1, 0, 1] * 20:
            memory.observe(symbol)
        restored = SequenceCircuitMemory.from_state_dict(memory.state_dict())
        self.assertEqual(restored.state_dict(), memory.state_dict())
        self.assertEqual(restored.distribution(), memory.distribution())

    def test_midstream_restore_continues_exactly(self):
        prefix = [0, 1, 1, 0, 0, 1] * 20
        suffix = [1, 0, 1, 0, 0, 0, 1] * 20
        uninterrupted = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=512)
        for symbol in prefix:
            uninterrupted.observe(symbol)
        restored = SequenceCircuitMemory.from_state_dict(uninterrupted.state_dict())
        left_predictions = []
        right_predictions = []
        for symbol in suffix:
            left_predictions.append(uninterrupted.distribution())
            right_predictions.append(restored.distribution())
            uninterrupted.observe(symbol)
            restored.observe(symbol)
        self.assertEqual(left_predictions, right_predictions)
        self.assertEqual(uninterrupted.state_dict(), restored.state_dict())

    def test_consolidation_window_protects_young_circuits(self):
        young = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=32)
        pattern = [0, 0, 0, 1, 1, 1, 0, 1] * 2
        for symbol in pattern:
            young.observe(symbol)
        taught = tuple(pattern[:8])
        self.assertIn(taught, young.circuits)
        state = 7
        for _ in range(10):
            state = (1103515245 * state + 12345) & 0x7fffffff
            young.observe((state >> 8) % 2)
        # Inside the grace window (2 * max_order events) the just-taught
        # episode survives budget pressure...
        self.assertIn(taught, young.circuits)
        for _ in range(400):
            state = (1103515245 * state + 12345) & 0x7fffffff
            young.observe((state >> 8) % 2)
        # ...while sustained pressure still reclaims: bounded growth holds
        # and stale weak circuits lose fair competition after grace expiry.
        self.assertGreater(young.circuits_reclaimed, 0)
        self.assertLessEqual(len(young.circuits), 32)
        young.validate()

    def test_rehearsed_table_recall_without_episodic(self):
        from soma.evaluation.dialogue import (
            fresh_dialogue, respond, teach_fact, text_to_bits,
        )
        from soma.memory import EpisodicBuffer
        background = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=512)
        for symbol in [0, 0, 1, 1, 0, 1] * 20:
            background.observe(symbol)
        dialogue = fresh_dialogue()
        episodic = EpisodicBuffer()
        teach_fact(dialogue, episodic, "USER AA BB AGENT ok ", "USER AA ", "BB")
        teach_fact(dialogue, episodic, "USER AA BB AGENT ok ", "USER AA ", "BB")
        bare = EpisodicBuffer()
        response = respond(background, dialogue, bare, "USER AA ", 4,
                           seed=1, deterministic=True)
        self.assertTrue(response.startswith(b"BB"))

    def test_circuit_budget_is_hard(self):
        memory = SequenceCircuitMemory(tuple(range(4)), max_order=5, max_circuits=32)
        state = 7
        for index in range(1000):
            state = (1103515245 * state + 12345) & 0x7fffffff
            memory.observe((state >> 8) % 4)
            self.assertLessEqual(len(memory.circuits), 32)
        self.assertGreater(memory.circuits_reclaimed, 0)
        memory.validate()

    def test_chunks_merge_and_round_trip(self):
        memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=4096)
        stream = [0, 0, 1, 1, 0, 1, 0, 0] * 40
        for symbol in stream:
            memory.observe(symbol)
        before = len(memory.circuits)
        promoted = memory.promote_chunks(min_order=4, min_reuse=5, min_concentration=0.6)
        self.assertGreater(promoted, 0)
        self.assertLess(len(memory.circuits), before)
        memory.validate()
        restored = SequenceCircuitMemory.from_state_dict(memory.state_dict())
        self.assertEqual(restored.state_dict(), memory.state_dict())
        self.assertEqual(restored.distribution(), memory.distribution())

    def test_chunks_reject_bad_bounds(self):
        memory = SequenceCircuitMemory((0, 1))
        with self.assertRaises(ValueError):
            memory.promote_chunks(min_order=0)
        with self.assertRaises(ValueError):
            memory.promote_chunks(min_concentration=1.5)

    def test_selective_merge_keeps_proven_circuits(self):
        memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=4096)
        for symbol in [0, 0, 1, 1] * 60 + [0, 1] * 4:
            memory.observe(symbol)
        full = SequenceCircuitMemory.from_state_dict(memory.state_dict())
        full.promote_chunks(min_order=4, min_reuse=5, min_concentration=0.6)
        selective = SequenceCircuitMemory.from_state_dict(memory.state_dict())
        selective.promote_chunks(min_order=4, min_reuse=5, min_concentration=0.6,
                                 max_merged_reuse=2)
        self.assertGreaterEqual(len(selective.circuits), len(full.circuits))
        selective.validate()

    def test_rejects_corrupt_state(self):
        memory = SequenceCircuitMemory((0, 1), max_order=3, max_circuits=32)
        for symbol in (0, 1, 0, 1):
            memory.observe(symbol)
        payload = memory.state_dict()
        payload["circuits"][0]["counts"][0] = -1
        with self.assertRaises(ValueError):
            SequenceCircuitMemory.from_state_dict(payload)

    def test_alice_e0_crosses_byte_unigram_bar_and_lesion_does_not(self):
        data = load_corpus("data/e0/alice.txt")
        acquisition, validation, _, _ = partition_documents(split_chapters(data), 0)
        reports, memory = run_english_sequence_memory(
            (acquisition[:20000], validation), max_order=16,
            max_circuits=131072)
        lesion_reports, _ = run_english_sequence_memory(
            (acquisition[:20000], validation), max_order=16,
            max_circuits=131072, lesion=True)
        bar = byte_unigram_bits(validation) / 8.0
        self.assertLess(reports[1]["bits_per_bit"], bar)
        self.assertGreater(lesion_reports[1]["bits_per_bit"], bar)
        self.assertGreater(memory.circuits_created, 1)


if __name__ == "__main__":
    unittest.main()

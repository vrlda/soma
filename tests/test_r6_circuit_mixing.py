import copy
import os
import unittest

from soma.memory import CircuitMixingMemory
from soma.memory.mixing import ORDER_SHIFT, circuit_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alice(start, length):
    with open(os.path.join(ROOT, "data", "e0", "alice.txt"), "rb") as handle:
        return handle.read()[start:start + length]


class CircuitKeyTests(unittest.TestCase):
    def test_exact_orders_pack_bytes_and_partial(self):
        self.assertEqual(circuit_key(2, b"xab", 5), (2 << ORDER_SHIFT) | (0x6162 << 8) | 5)
        self.assertEqual(circuit_key(0, b"xab", 1), 1)

    def test_long_orders_are_hashed_per_order(self):
        history = bytes(range(40))
        keys = {circuit_key(order, history, 1) for order in (7, 8, 12, 32)}
        self.assertEqual(len(keys), 4)
        for order in (7, 12):
            self.assertEqual(circuit_key(order, history, 1) >> ORDER_SHIFT, order)


class CircuitMixingMemoryTests(unittest.TestCase):
    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            CircuitMixingMemory(orders=(2, 1))
        with self.assertRaises(ValueError):
            CircuitMixingMemory(orders=(0, 33))
        with self.assertRaises(ValueError):
            CircuitMixingMemory(orders=(0, 1), max_circuits=100)
        with self.assertRaises(ValueError):
            CircuitMixingMemory().observe(2)

    def test_learning_lowers_cross_entropy(self):
        text = alice(2000, 3000)
        memory = CircuitMixingMemory(max_circuits=1 << 16)
        untrained = copy.deepcopy(memory).score_bytes(text)
        memory.observe_bytes(text * 2)
        memory.reset_history()
        self.assertLess(memory.score_bytes(text), 0.5 * untrained)

    def test_frozen_scoring_changes_no_learned_state(self):
        memory = CircuitMixingMemory(max_circuits=1 << 14)
        memory.observe_bytes(alice(0, 3000))
        before = (dict((k, list(v)) for k, v in memory.circuits.items()),
                  copy.deepcopy(memory.weights), copy.deepcopy(memory.calibration),
                  copy.deepcopy(memory.correction), memory.circuits_created)
        memory.reset_history()
        memory.score_bytes(alice(5000, 1000), learn=False)
        after = (dict((k, list(v)) for k, v in memory.circuits.items()),
                 memory.weights, memory.calibration, memory.correction,
                 memory.circuits_created)
        self.assertEqual(before, after)

    def test_budget_is_hard_and_reclaims(self):
        memory = CircuitMixingMemory(orders=(0, 1, 2, 3), max_circuits=2048)
        memory.observe_bytes(alice(0, 4000))
        self.assertLessEqual(len(memory.circuits), 2048)
        self.assertGreater(memory.circuits_reclaimed, 0)
        self.assertEqual(memory.circuits_created - memory.circuits_reclaimed,
                         len(memory.circuits))

    def test_growth_gate_defers_long_contexts(self):
        text = alice(0, 3000)
        gated = CircuitMixingMemory(max_circuits=1 << 18, growth_threshold=8)
        eager = CircuitMixingMemory(max_circuits=1 << 18, growth_threshold=0)
        gated.observe_bytes(text)
        eager.observe_bytes(text)
        self.assertLess(gated.circuits_created, eager.circuits_created // 2)

    def test_plastic_arbitration_beats_longest_match(self):
        train, held_out = alice(0, 12000), alice(12000, 2000)
        scores = {}
        for arbitration in (True, False):
            memory = CircuitMixingMemory(max_circuits=1 << 18, arbitration=arbitration,
                                         learning_rate=0.015)
            memory.observe_bytes(train)
            memory.reset_history()
            scores[arbitration] = memory.score_bytes(held_out)
        self.assertLess(scores[True], scores[False])

    def test_probabilities_stay_inside_floor(self):
        memory = CircuitMixingMemory(max_circuits=1 << 14)
        for byte in b"\x00\xff" * 200:
            for shift in range(7, -1, -1):
                probability = memory.predict()
                self.assertTrue(0.0 < probability < 1.0)
                memory.observe((byte >> shift) & 1)


if __name__ == "__main__":
    unittest.main()

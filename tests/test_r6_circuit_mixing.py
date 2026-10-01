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

    def test_growth_pressure_engages_only_under_reclamation(self):
        text = alice(0, 12000)
        roomy = [CircuitMixingMemory(max_circuits=1 << 18, growth_pressure=pressure)
                 for pressure in (0, 4)]
        tight = [CircuitMixingMemory(max_circuits=4096, growth_pressure=pressure)
                 for pressure in (0, 4)]
        for memory in roomy + tight:
            memory.observe_bytes(text)
        self.assertEqual(roomy[1].circuits_reclaimed, 0)
        self.assertEqual(roomy[0].circuits, roomy[1].circuits)
        self.assertEqual(roomy[0].weights, roomy[1].weights)
        self.assertGreater(tight[1].reclaim_frontier, 0)
        self.assertLess(tight[1].circuits_created, tight[0].circuits_created)

    def test_growth_pressure_state_round_trips_and_is_absent_when_off(self):
        memory = CircuitMixingMemory(max_circuits=4096, growth_pressure=3)
        memory.observe_bytes(alice(0, 8000))
        state = memory.dumps()
        restored = CircuitMixingMemory.loads(state)
        self.assertEqual((restored.growth_pressure, restored.reclaim_frontier),
                         (3, memory.reclaim_frontier))
        self.assertEqual(restored.dumps(), state)
        self.assertNotIn(b"growth_pressure", CircuitMixingMemory(max_circuits=4096).dumps())
        with self.assertRaises(ValueError):
            CircuitMixingMemory(growth_pressure=-1)

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

    def test_metaplasticity_slows_well_used_weight_sets(self):
        text = alice(0, 2000)
        fast = CircuitMixingMemory(max_circuits=1 << 16, plasticity_tau=0.0)
        slow = CircuitMixingMemory(max_circuits=1 << 16, plasticity_tau=50.0)
        fast.observe_bytes(text)
        slow.observe_bytes(text)
        self.assertEqual(sum(fast.weight_updates), sum(slow.weight_updates))
        drift = lambda memory: sum(abs(w - memory.initial_weight)
                                   for row in memory.weights for w in row)
        self.assertLess(drift(slow), drift(fast))
        with self.assertRaises(ValueError):
            CircuitMixingMemory(plasticity_tau=-1.0)

    def test_frozen_arbitration_stops_weight_learning(self):
        memory = CircuitMixingMemory(max_circuits=1 << 16, freeze_arbitration_after=800)
        memory.observe_bytes(alice(0, 100))
        before = [list(row) for row in memory.weights]
        memory.observe_bytes(alice(100, 400))
        self.assertEqual(before, memory.weights)
        self.assertGreater(len(memory.circuits), 0)

    def test_learns_a_non_text_byte_process(self):
        # Domain neutrality (ADR 0009): a seeded order-2 Markov chain over
        # 16 arbitrary byte values, with no text structure at all.
        import math
        import random
        rng = random.Random(9)
        symbols = rng.sample(range(256), 16)
        table = {}
        history, stream = [symbols[0], symbols[1]], []
        for _ in range(12000):
            key = tuple(history[-2:])
            if key not in table:
                table[key] = rng.sample(symbols, 3)
            nxt = rng.choice(table[key])
            stream.append(nxt)
            history.append(nxt)
        data = bytes(stream)
        train, held_out = data[:10000], data[10000:]
        counts = {b: train.count(b) for b in set(train)}
        unigram = -sum(math.log2(counts.get(b, 0.5) / len(train)) for b in held_out) / (8 * len(held_out))
        memory = CircuitMixingMemory(max_circuits=1 << 18)
        memory.observe_bytes(train)
        memory.reset_history()
        self.assertLess(memory.score_bytes(held_out), 0.6 * unigram)

    def test_distribution_matches_sequence_memory_contract(self):
        memory = CircuitMixingMemory(max_circuits=1 << 16)
        memory.observe_bytes(alice(0, 2000))
        distribution, order = memory.distribution()
        self.assertEqual(set(distribution), {0, 1})
        self.assertAlmostEqual(distribution[0] + distribution[1], 1.0)
        self.assertEqual(distribution[1], memory.probability(1))
        self.assertGreaterEqual(order, 8)
        self.assertEqual(order % 8, memory.partial.bit_length() - 1)
        fresh, fresh_order = CircuitMixingMemory().distribution()
        self.assertEqual(fresh_order, 0)
        with self.assertRaises(ValueError):
            memory.probability(2)

    def test_weight_scales_counts_up_to_the_cap(self):
        memory = CircuitMixingMemory(orders=(0,), max_circuits=4096, count_limit=10)
        memory.observe(1, weight=4)
        self.assertEqual(memory.circuits[1][:2], [0, 4])
        memory.observe(1, weight=50)
        self.assertEqual(memory.circuits[3][:2], [0, 10])
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                memory.observe(0, weight=bad)

    def test_binary_state_round_trips_exactly(self):
        memory = CircuitMixingMemory(max_circuits=4096, calibration=True, correction=True,
                                     count_limit=255)
        memory.observe_bytes(alice(0, 3000))
        blob = memory.dumps()
        restored = CircuitMixingMemory.loads(blob)
        self.assertEqual(restored.dumps(), blob)
        self.assertEqual(restored.distribution(), memory.distribution())
        corrupted = bytearray(blob)
        corrupted[len(blob) // 2] ^= 1
        with self.assertRaises(ValueError):
            CircuitMixingMemory.loads(bytes(corrupted))
        with self.assertRaises(ValueError):
            CircuitMixingMemory.loads(b"NOTSOMA!" + blob[8:])

    def test_organism_owns_mixing_memory_with_quarantine(self):
        from soma.organism import Organism
        organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=0)
        organism.enable_sequence_memory((0, 1), kind="mixing",
                                        mixing_config={"max_circuits": 8192})
        for bit in [0, 0, 1, 1] * 40:
            organism.observe_sequence_event(bit)
        before = organism.sequence_distribution()
        organism.quarantine_events([1, 0] * 20)
        self.assertEqual(organism.sequence_distribution(), before)
        restored = Organism.from_state_dict(organism.state_dict())
        self.assertEqual(restored.sequence_distribution(), before)
        organism.approve_quarantine(weight=2)
        self.assertNotEqual(organism.sequence_distribution(), before)
        with self.assertRaises(ValueError):
            Organism.create_default(input_size=2, hidden_size=2, output_size=1,
                                    seed=0).enable_sequence_memory((0, 1, 2), kind="mixing")

    def test_probabilities_stay_inside_floor(self):
        memory = CircuitMixingMemory(max_circuits=1 << 14)
        for byte in b"\x00\xff" * 200:
            for shift in range(7, -1, -1):
                probability = memory.predict()
                self.assertTrue(0.0 < probability < 1.0)
                memory.observe((byte >> shift) & 1)


if __name__ == "__main__":
    unittest.main()

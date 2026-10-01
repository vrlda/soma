"""Speed changes to SequenceCircuitMemory must not change behavior."""

import random
import unittest

from soma.memory import SequenceCircuitMemory


class ReferenceReclaim(SequenceCircuitMemory):
    """The pre-optimization _reclaim, verbatim except for the class."""

    def _reclaim(self):
        if len(self.circuits) < self.max_circuits:
            return
        if getattr(self, "_reclaim_exhausted_at", None) == self.events_seen:
            return
        protected = set(self._contexts())
        grace = 2 * max(1, self.max_order)
        candidates = []
        for context, circuit in self.circuits.items():
            if not context or context in protected:
                continue
            if self.events_seen - circuit["last_used"] < grace:
                continue
            support = sum(circuit["counts"])
            candidates.append((support, circuit["reuse"], circuit["last_used"],
                               -len(context), context))
        if not candidates:
            self._reclaim_exhausted_at = self.events_seen
            return
        candidates.sort()
        batch = max(1, min(1024, self.max_circuits // 100))
        for item in candidates[:batch]:
            del self.circuits[item[-1]]
            self.circuits_reclaimed += 1


class SequenceMemorySpeedTests(unittest.TestCase):
    def test_reclaim_matches_reference_exactly(self):
        for seed, (order, budget) in enumerate(((16, 300), (64, 2048), (8, 128))):
            rng = random.Random(seed)
            fast = SequenceCircuitMemory((0, 1), max_order=order, max_circuits=budget)
            reference = ReferenceReclaim((0, 1), max_order=order, max_circuits=budget)
            for step in range(6000):
                bit = rng.randrange(2) if rng.random() < 0.3 else step % 3 % 2
                weight = rng.choice((1, 1, 2))
                learn = rng.random() < 0.9
                fast.observe(bit, learn=learn, weight=weight)
                reference.observe(bit, learn=learn, weight=weight)
                if step % 997 == 0:
                    fast.reset_history()
                    reference.reset_history()
            self.assertGreater(reference.circuits_reclaimed, 0)
            self.assertEqual(fast.circuits, reference.circuits)
            self.assertEqual(fast.circuits_reclaimed, reference.circuits_reclaimed)
            self.assertEqual(fast.distribution(), reference.distribution())

    def test_index_survives_unobserve_chunks_persistence_and_direct_edits(self):
        rng = random.Random(11)
        fast = SequenceCircuitMemory((0, 1), max_order=12, max_circuits=200)
        reference = ReferenceReclaim((0, 1), max_order=12, max_circuits=200)
        for step in range(4000):
            bit = rng.randrange(2)
            fast.observe(bit)
            reference.observe(bit)
            if step == 1500:
                taught = [1, 0, 1, 1, 0, 0, 1]
                for memory in (fast, reference):
                    before = list(memory.history)
                    for symbol in taught:
                        memory.observe(symbol, weight=3)
                    memory.unobserve(taught, before, weight=3)
            if step == 2200:
                for memory in (fast, reference):
                    memory.promote_chunks(min_order=4, min_reuse=3, min_concentration=0.6)
            if step == 2800:
                fast = SequenceCircuitMemory.from_state_dict(fast.state_dict())
                reference = ReferenceReclaim.from_state_dict(reference.state_dict())
            if step == 3300:
                for memory in (fast, reference):
                    for context, circuit in sorted(memory.circuits.items())[:40]:
                        if context:
                            circuit["counts"] = [0] * len(circuit["counts"])
        self.assertGreater(reference.circuits_reclaimed, 0)
        self.assertEqual(fast.circuits, reference.circuits)
        self.assertEqual(fast.circuits_reclaimed, reference.circuits_reclaimed)

    def test_reclaim_index_stays_bounded(self):
        # A tiny budget reclaims 2 circuits per call while every event
        # touches up to 9; stale heap entries must not accumulate.
        memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=256)
        rng = random.Random(5)
        largest = [0, 0]
        for step in range(60000):
            memory.observe(rng.randrange(2))
            heap = getattr(memory, "_reclaim_heap", None) or []
            half = 0 if step < 30000 else 1
            largest[half] = max(largest[half], len(heap))
        self.assertGreater(memory.circuits_reclaimed, 1000)
        # Compaction threshold plus at most one grace window of flushed entries.
        window = 2 * memory.max_order * (memory.max_order + 1)
        self.assertLessEqual(max(largest), 2 * memory.max_circuits + 64 + window)
        self.assertLessEqual(largest[1], largest[0] + window)
        pending = sum(len(v) for v in (memory._reclaim_buckets or {}).values())
        self.assertLess(pending, 2 * memory.max_order * (memory.max_order + 1) + 64)

    def test_v2_state_round_trips_and_v1_still_loads(self):
        memory = SequenceCircuitMemory((0, 1), max_order=24, max_circuits=4096)
        for step in range(3000):
            memory.observe((step * 7 // 3) % 2)
        state = memory.state_dict()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["context_encoding"], "bits")
        restored = SequenceCircuitMemory.from_state_dict(state)
        self.assertEqual(restored.circuits, memory.circuits)
        self.assertEqual(restored.state_dict(), state)
        legacy = dict(state, version=1)
        legacy.pop("context_encoding")
        legacy["circuits"] = [dict(item, context=[int(c) for c in item["context"]])
                              for item in state["circuits"]]
        legacy["chunks"] = [dict(item, context=[int(c) for c in item["context"]])
                            for item in state["chunks"]]
        self.assertEqual(SequenceCircuitMemory.from_state_dict(legacy).circuits, memory.circuits)

    def test_non_binary_symbols_keep_list_contexts(self):
        memory = SequenceCircuitMemory(("a", "b", "c"), max_order=3, max_circuits=256)
        for symbol in "abcabcacb" * 5:
            memory.observe(symbol)
        state = memory.state_dict()
        self.assertEqual(state["context_encoding"], "list")
        self.assertEqual(SequenceCircuitMemory.from_state_dict(state).circuits, memory.circuits)


if __name__ == "__main__":
    unittest.main()

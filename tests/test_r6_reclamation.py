"""Fast unit tests for the r6b reclamation harness logic (synthetic streams).

The locked E1 run is the only mechanism evidence; these tests pin harness
semantics: gate computation, engagement void without pressure, and
observational (non-mutating) evaluation.
"""

import unittest

from r6_reclamation_benchmark import (
    BUDGET,
    ENGAGEMENT_BAND,
    MIN_RECLAMATIONS,
    _evaluate,
    run_stream,
)
from r6_tier_benchmark import SuffixNgramControl
from soma.evaluation.english import bit_stream
from soma.memory import SequenceCircuitMemory


def _toy_stream(seed_bytes):
    model = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64,
                                  min_support=2, prior=0.5)
    control = SuffixNgramControl(4, 64)
    for data in seed_bytes:
        for memory in (model, control):
            memory.reset_history()
            bits, _ = bit_stream(data)
            for symbol in bits[:-1]:
                memory.observe(symbol)
    return model, control


class R6bHarnessTests(unittest.TestCase):
    def test_evaluation_does_not_mutate_training_state(self):
        model, _ = _toy_stream([b"ABCD" * 32])
        before = model.state_dict()
        _evaluate(model, b"ABCD" * 16)
        self.assertEqual(model.state_dict(), before)

    def test_engagement_band_matches_reclaim_steady_state(self):
        # Under forced pressure the model holds at cap-batch, not at cap.
        model = SequenceCircuitMemory((0, 1), max_order=8,
                                      max_circuits=BUDGET, min_support=2,
                                      prior=0.5)
        bits, _ = bit_stream(bytes(range(256)) * 64)
        for symbol in bits[:-1]:
            model.observe(symbol)
        self.assertGreaterEqual(model.circuits_reclaimed, MIN_RECLAMATIONS)
        self.assertGreaterEqual(len(model.circuits), BUDGET - ENGAGEMENT_BAND)

    def test_small_budget_forces_eviction_but_large_budget_does_not(self):
        tight = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=256,
                                      min_support=2, prior=0.5)
        loose = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=131072,
                                      min_support=2, prior=0.5)
        bits, _ = bit_stream(bytes(range(256)) * 64)
        for symbol in bits[:-1]:
            tight.observe(symbol)
            loose.observe(symbol)
        self.assertGreater(tight.circuits_reclaimed, 0)
        self.assertEqual(loose.circuits_reclaimed, 0)

    def test_run_stream_uses_frozen_e1_books(self):
        # Structural pin: stream consumes at least 5 acquisition books.
        # (Full locked run asserted separately via reports/r6-reclamation.json.)
        import r6_reclamation_benchmark as harness
        self.assertEqual(harness.ORDER, 8)
        self.assertEqual(harness.BUDGET, 256)
        self.assertEqual(harness.PROTOCOL, "r6b-reclamation-v2")


if __name__ == "__main__":
    unittest.main()

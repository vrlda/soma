import unittest

from soma.evaluation.generate import (
    _bits_form_valid_prefix,
    _is_valid_utf8_prefix,
    continuation_nll,
    generate,
    generate_constrained,
    temper,
)
from soma.memory import SequenceCircuitMemory


def _tiny_memory():
    memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=512)
    for symbol in ([0, 0, 1, 1, 0, 1] * 30):
        memory.observe(symbol)
    return memory


class R3CGenerationTests(unittest.TestCase):
    def test_temper_bounds(self):
        self.assertEqual(temper(0.7, 0.0), 1.0)
        self.assertEqual(temper(0.3, 0.0), 0.0)
        middle = temper(0.7, 1.0)
        self.assertAlmostEqual(middle, 0.7)
        flat = temper(0.9, 1000.0)
        self.assertAlmostEqual(flat, 0.5, delta=0.05)

    def test_constrained_always_valid(self):
        for prefix in (b"Hi ", b"\xc3\xa9 ", b"A" * 10):
            output, report = generate_constrained(_tiny_memory(), prefix, 12,
                                                  deterministic=False, seed=5)
            self.assertTrue(report["valid_utf8"], repr(output))

    def test_constrained_rejects_bad_prefix(self):
        with self.assertRaises(ValueError):
            generate_constrained(_tiny_memory(), b"\xff\xfe", 10)
        with self.assertRaises(ValueError):
            generate_constrained(_tiny_memory(), "text", 10)
        with self.assertRaises(ValueError):
            generate_constrained(_tiny_memory(), b"ok ", 0)

    def test_generation_never_trains(self):
        memory = _tiny_memory()
        before = [sum(circuit["counts"]) for circuit in memory.circuits.values()]
        generate_constrained(memory, b"test ", 16, deterministic=False, seed=9)
        after = [sum(circuit["counts"]) for circuit in memory.circuits.values()]
        self.assertEqual(before, after)
        generate(memory, b"test ", 16, deterministic=True)
        newest = [sum(circuit["counts"]) for circuit in memory.circuits.values()]
        self.assertEqual(before, newest)

    def test_history_changes_likelihood(self):
        intact = continuation_nll(_tiny_memory(), b"\x00\x00", b"\x01")
        lesioned = _tiny_memory()
        for circuit in lesioned.circuits.values():
            circuit["counts"] = [0] * len(circuit["counts"])
        degraded = continuation_nll(lesioned, b"\x00\x00", b"\x01")
        # Uniform predictions score exactly chance...
        self.assertAlmostEqual(degraded, 1.0)
        # ...while learned history says something specific (here: confidently
        # wrong pattern overgeneralization, which the natural-data benchmark
        # shows is the exception, not the rule).
        self.assertNotAlmostEqual(intact, degraded)

    def test_horizon_invariance_beyond_memory(self):
        memory = _tiny_memory()
        first = continuation_nll(memory, b"A" * 30 + b"\x00", b"\x01")
        memory = _tiny_memory()
        second = continuation_nll(memory, b"B" * 30 + b"\x00", b"\x01")
        self.assertAlmostEqual(first, second)


if __name__ == "__main__":
    unittest.main()

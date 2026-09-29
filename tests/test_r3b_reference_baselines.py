import math
import unittest

from r3b_reference_baselines import (
    WittenBellByteModel, compressor_bits_per_byte, to_bits_per_bit,
)


class WittenBellByteModelTests(unittest.TestCase):
    def setUp(self):
        self.model = WittenBellByteModel(3).fit([b"the cat sat on the mat. " * 20])

    def test_distribution_normalizes(self):
        for context in (b"", b"t", b"th", b"the", b"zzz"):
            total = sum(self.model.probability(context, symbol) for symbol in range(256))
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_unseen_symbol_keeps_mass(self):
        self.assertGreater(self.model.probability(b"the", ord("Q")), 0.0)

    def test_order_zero_empty_model_is_uniform(self):
        empty = WittenBellByteModel(2).fit([])
        self.assertAlmostEqual(empty.bits_per_byte(b"abc"), 8.0)

    def test_context_beats_unigram_on_repetitive_text(self):
        data = b"the cat sat on the mat. " * 4
        self.assertLess(self.model.bits_per_byte(data, 3),
                        self.model.bits_per_byte(data, 0))
        self.assertTrue(math.isfinite(self.model.bits_per_byte(b"\xff\x00", 3)))


class CompressorTests(unittest.TestCase):
    def test_primed_xz_benefits_from_seen_text(self):
        prime = b"a line of public text that repeats. " * 200
        rates = compressor_bits_per_byte(prime[:2000], prime)
        self.assertEqual(set(rates), {"gzip_9", "bz2_9", "xz_9e", "xz_primed"})
        self.assertLess(rates["xz_primed"], rates["xz_9e"])
        self.assertAlmostEqual(to_bits_per_bit({"x": 8.0})["x"], 1.0)


if __name__ == "__main__":
    unittest.main()

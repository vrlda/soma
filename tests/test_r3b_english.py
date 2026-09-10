import math
import os
import tempfile
import unittest

from soma.evaluation.english import (
    ADAPTERS,
    bit_bigram_bits,
    bit_marginal_bits,
    bit_stream,
    bits_per_bit,
    byte_unigram_bits,
    run_english,
)
from soma.evaluation.lineage import clone_brain, read_lineage, verify_clone
from soma.transducers.text_bytes import decode_bits, encode_bytes


class R3BEnglishTests(unittest.TestCase):
    def test_bits_per_bit_math(self):
        self.assertAlmostEqual(bits_per_bit([0.5, 0.5], [0, 1]), 1.0)
        self.assertLess(bits_per_bit([0.9, 0.9], [1, 1]), 0.2)
        self.assertGreater(bits_per_bit([0.99, 0.99], [0, 0]), 5.0)

    def test_byte_unigram_math(self):
        data = b"aaab"
        expected = -(3 * math.log(0.75, 2) + math.log(0.25, 2)) / 4
        self.assertAlmostEqual(byte_unigram_bits(data), expected)

    def test_bit_baselines_near_chance_on_uniform(self):
        data = bytes([0xAA, 0x55] * 32)
        marginal = bit_marginal_bits(data, data)
        bigram = bit_bigram_bits(data, data)
        self.assertLess(marginal, 1.1)
        self.assertLess(bigram, 1.1)

    def test_bit_stream_boundaries(self):
        bits, ends = bit_stream(b"A")
        self.assertEqual(len(bits), 8)
        self.assertEqual(ends, [0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(decode_bits([bits]).hex(), "41")

    def test_lineage_clone_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "genesis.json")
            with open(source, "w") as handle:
                handle.write("{}")
            destination = os.path.join(directory, "branch.json")
            record = clone_brain(source, destination, "e0-pilot")
            self.assertEqual(record["reason"], "e0-pilot")
            self.assertEqual(read_lineage(destination)["identity"], record["identity"])
            verify_clone(source, destination)
            with open(destination, "w") as handle:
                handle.write('{"tampered": true}')
            with self.assertRaises(ValueError):
                verify_clone(source, destination)
            with self.assertRaises(ValueError):
                clone_brain(source, destination, "again")
            with self.assertRaises(ValueError):
                clone_brain(source, os.path.join(directory, "x.json"), "")

    def test_encode_precedes_stream(self):
        data = "Hi!".encode("utf-8")
        rows = encode_bytes(data)
        bits, _ = bit_stream(data)
        flat = [int(bit) for row in rows for bit in row]
        self.assertEqual(bits, flat)

    def test_phase_adapter_declared(self):
        adapter = ADAPTERS["phase"]()
        self.assertEqual(adapter.spec.name, "text-utf8-phase-v1")
        self.assertEqual(adapter.input_size, 6)
        frame = {"bit": {"payload": {"value": 1.0}},
                 "boundary": {"payload": {"byte_end": 0.0}},
                 "phase0": {"payload": {"on": 1.0}},
                 "phase1": {"payload": {"on": 0.0}},
                 "phase2": {"payload": {"on": 0.0}}}
        self.assertEqual(adapter.pack_frame(frame), [1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            adapter.pack_frame({"bit": {"payload": {"value": 1.0}},
                                "boundary": {"payload": {"byte_end": 0.0}}})

    def test_msb_rule_learned_on_tiny_slice(self):
        data = bytes([0x41, 0x42, 0x43, 0x44] * 64)
        report = run_english(data, 11, "learn")
        self.assertIn("accuracy_msb", report)
        self.assertIn("accuracy_intra", report)
        self.assertGreaterEqual(report["accuracy_msb"], 0.5)


if __name__ == "__main__":
    unittest.main()

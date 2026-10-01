import unittest

from soma.transducers import (
    TextBytesTransducer,
    decode_bits,
    encode_bytes,
    validate_utf8,
)


class R3BTransducerTests(unittest.TestCase):
    def test_round_trip_arbitrary_bytes(self):
        for data in (b"Hello, world!", "caf\u00e9 \u4e2d\u2615".encode("utf-8"),
                     bytes(range(256)), b"", b"\x00\xff\x7f"):
            rows = encode_bytes(data)
            self.assertEqual(len(rows), len(data))
            self.assertTrue(all(len(row) == 8 for row in rows))
            self.assertEqual(decode_bits(rows), data)

    def test_msb_first(self):
        self.assertEqual(encode_bytes(b"A"), [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def test_rejects_non_bytes(self):
        with self.assertRaises(ValueError):
            encode_bytes("text")
        with self.assertRaises(ValueError):
            encode_bytes([65])

    def test_rejects_short_rows_and_bad_bits(self):
        with self.assertRaises(ValueError):
            decode_bits([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            decode_bits([[0.0] * 7 + [2.0]])

    def test_validate_utf8(self):
        self.assertTrue(validate_utf8("plain ascii".encode("utf-8")))
        self.assertTrue(validate_utf8("caf\u00e9".encode("utf-8")))
        with self.assertRaises(ValueError):
            validate_utf8(b"\xff\xfe\x00bad")
        with self.assertRaises(ValueError):
            validate_utf8(b"\x80abc")

    def test_pack_unpack_shapes(self):
        adapter = TextBytesTransducer()
        self.assertEqual(adapter.input_size, 3)
        frame = {"bit": {"payload": {"value": 1.0}},
                 "boundary": {"payload": {"byte_end": 0.0}}}
        self.assertEqual(adapter.pack_frame(frame), [1.0, 0.0, 1.0])
        proposal = adapter.unpack_action([0.7])
        self.assertAlmostEqual(proposal["value"], 0.85)
        adapter.spec.action_schema.check_payload(proposal)
        with self.assertRaises(ValueError):
            adapter.pack_frame({"bit": {"payload": {"value": 2.0}},
                                "boundary": {"payload": {"byte_end": 0.0}}})

    def test_state_round_trip(self):
        adapter = TextBytesTransducer()
        state = adapter.state_dict()
        adapter.load_state_dict(state)
        with self.assertRaises(ValueError):
            adapter.load_state_dict({"name": "other"})

    def test_probability_floor(self):
        adapter = TextBytesTransducer()
        self.assertEqual(adapter.unpack_action([-1.0])["value"], 0.02)
        self.assertEqual(adapter.unpack_action([1.0])["value"], 0.98)
        self.assertAlmostEqual(adapter.unpack_action([0.0])["value"], 0.5)


if __name__ == "__main__":
    unittest.main()

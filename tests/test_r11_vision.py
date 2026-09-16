import unittest

from soma.evaluation.vision import glyph_stream, make_vision_organism, run_glyphs
from soma.transducers.vision import (
    GlyphTransducer,
    render_horizontal,
    render_vertical,
)


class R11VisionTests(unittest.TestCase):
    def test_renders(self):
        vertical = render_vertical()
        horizontal = render_horizontal()
        self.assertEqual(len(vertical), 6)
        self.assertTrue(all(cell in (0, 1) for row in vertical for cell in row))
        self.assertEqual(sum(sum(row) for row in vertical), 6)
        self.assertEqual(sum(sum(row) for row in horizontal), 6)
        self.assertNotEqual(vertical, horizontal)
        shifted = render_vertical(shift=(1, 0))
        self.assertNotEqual(shifted, vertical)
        noisy = render_vertical(noise=[(0, 0)])
        self.assertEqual(sum(sum(row) for row in noisy), 7)

    def test_adapter_packs(self):
        adapter = GlyphTransducer()
        self.assertEqual(adapter.input_size, 37)
        frame = {"px%d" % index: {"payload": {"on": 1.0 if index == 0 else 0.0}}
                 for index in range(36)}
        packed = adapter.pack_frame(frame)
        self.assertEqual(len(packed), 37)
        self.assertEqual(packed[0], 1.0)
        self.assertEqual(adapter.unpack_action([0.5]), {"value": 0.5})
        with self.assertRaises(ValueError):
            adapter.pack_frame({"px0": {"payload": {"on": 2.0}},
                                **{"px%d" % index: {"payload": {"on": 0.0}}
                                   for index in range(1, 36)}})

    def test_stream_deterministic(self):
        self.assertEqual(glyph_stream(3, 50), glyph_stream(3, 50))
        self.assertEqual(len(glyph_stream(3, 50)), 50)

    def test_learn_runs_and_beats_frozen_short(self):
        learn = run_glyphs(0, "learn", steps=300)
        frozen = run_glyphs(0, "frozen", steps=300)
        self.assertGreater(learn["a_tail"], frozen["a_tail"])
        self.assertIn("modules", learn)


if __name__ == "__main__":
    unittest.main()

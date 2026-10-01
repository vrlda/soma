import copy
import json
import unittest

from soma.transducers import BoundedLagWorkspaceTransducer, TextBytesTransducer
from soma.transducers.history import HistoryScalarTransducer


def _text_frame(bit, end):
    return {"bit": {"payload": {"value": bit}}, "boundary": {"payload": {"byte_end": end}}}


class BoundedLagWorkspaceTests(unittest.TestCase):
    def test_binary_sign_centering_bias_and_transient_padding(self):
        adapter = BoundedLagWorkspaceTransducer(TextBytesTransducer(), depth=2)
        self.assertEqual(adapter.pack_frame(_text_frame(1, 0)), [1.0, -1.0, 0.0, 0.0, 1.0])
        self.assertEqual(adapter.pack_frame(_text_frame(0, 1)), [-1.0, 1.0, 1.0, -1.0, 1.0])
        self.assertEqual(adapter.input_size, 5)
        self.assertEqual(adapter.spec.channels[0].name, "bit")

    def test_generic_nonbinary_schema_preserves_bounded_signals(self):
        adapter = BoundedLagWorkspaceTransducer(HistoryScalarTransducer(depth=3), depth=2)
        frame = {
            "lag0": {"payload": {"value": 0.75}},
            "lag1": {"payload": {"value": -0.5}},
            "lag2": {"payload": {"value": 2.0}},
        }
        values = adapter.pack_frame(frame)
        self.assertEqual(values, [0.75, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in values))

    def test_state_roundtrip_and_split_resume_are_exact(self):
        first = BoundedLagWorkspaceTransducer(TextBytesTransducer(), depth=8)
        first.pack_frame(_text_frame(1, 0))
        first.pack_frame(_text_frame(0, 0))
        checkpoint = json.loads(json.dumps(first.state_dict()))
        resumed = BoundedLagWorkspaceTransducer(TextBytesTransducer(), depth=8)
        resumed.load_state_dict(copy.deepcopy(checkpoint))
        self.assertEqual(resumed.state_dict(), checkpoint)
        next_frame = _text_frame(1, 1)
        self.assertEqual(resumed.pack_frame(next_frame), first.pack_frame(next_frame))
        self.assertEqual(resumed.state_dict(), first.state_dict())

    def test_configuration_and_history_validation(self):
        with self.assertRaises(ValueError):
            BoundedLagWorkspaceTransducer(TextBytesTransducer(), depth=0)
        adapter = BoundedLagWorkspaceTransducer(TextBytesTransducer(), depth=2)
        state = adapter.state_dict()
        state["history"] = [[0.0] * 3]
        with self.assertRaises(ValueError):
            adapter.load_state_dict(state)
        state = adapter.state_dict()
        state["depth"] = 3
        with self.assertRaises(ValueError):
            adapter.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()

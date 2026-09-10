import unittest

from soma.events import (
    ActionEvent,
    ChannelBuffer,
    ChannelRegistry,
    ChannelSchema,
    DiscreteHead,
    Event,
    LogicalClock,
    OutcomeEvent,
    ScalarHead,
)
from soma.transducers import ScalarStreamTransducer, SymbolBitsTransducer
from soma.transducers.symbol_bits import dequantize, quantize


def _event(channel="series", source="fixture", event_id=0, clock=0, value=0.5):
    return Event(channel, source, event_id, clock, {"value": value}, {"trust": "test"})


class R2EventTests(unittest.TestCase):
    def test_envelope_round_trip(self):
        event = _event()
        self.assertEqual(Event.from_dict(event.to_dict()).to_dict(), event.to_dict())
        action = ActionEvent("prediction", 3, 3, {"value": 0.1}, "scalar-v1")
        self.assertEqual(ActionEvent.from_dict(action.to_dict()).to_dict(), action.to_dict())
        outcome = OutcomeEvent(3, 5, 0.25, "fixture")
        self.assertEqual(OutcomeEvent.from_dict(outcome.to_dict()).to_dict(), outcome.to_dict())

    def test_forbidden_task_keys_rejected(self):
        with self.assertRaises(ValueError):
            Event("series", "fixture", 0, 0, {"task_id": "A"}, {})
        with self.assertRaises(ValueError):
            Event("series", "fixture", 0, 0, {"value": 0.5}, {}).to_dict().update({"nope": 1}) or Event.from_dict(
                {"channel": "c", "source": "s", "event_id": 0, "clock": 0,
                 "payload": {"value": 0.5}, "provenance": {}, "nope": 1})

    def test_nonfinite_and_bad_clock_rejected(self):
        with self.assertRaises(ValueError):
            _event(value=float("nan"))
        with self.assertRaises(ValueError):
            _event(clock=-1)

    def test_clock_orders_sources(self):
        clock = LogicalClock()
        clock.observe(_event(event_id=0, clock=0))
        clock.observe(_event(event_id=1, clock=1))
        with self.assertRaises(ValueError):
            clock.observe(_event(event_id=1, clock=2))
        with self.assertRaises(ValueError):
            clock.observe(_event(event_id=2, clock=0))
        restored = LogicalClock.from_dict(clock.to_dict())
        self.assertEqual(restored.to_dict(), clock.to_dict())

    def test_registry_match_and_bounds(self):
        registry = ChannelRegistry()
        registry.register(ChannelSchema("series", ["value"], low=-1.0, high=1.0))
        registry.check_event(_event(value=0.5))
        with self.assertRaises(ValueError):
            registry.check_event(_event(value=2.0))
        with self.assertRaises(ValueError):
            registry.check_event(Event("other", "s", 0, 0, {"value": 0.0}, {}))
        self.assertEqual(ChannelRegistry.from_dict(registry.to_dict()).to_dict(), registry.to_dict())

    def test_buffer_join_and_bound(self):
        buffer = ChannelBuffer(["bit0", "bit1"], max_buffered=2)
        self.assertIsNone(buffer.pop_frame())
        buffer.push(Event("bit0", "s", 0, 0, {"bit": 1.0}, {}))
        self.assertIsNone(buffer.pop_frame())
        buffer.push(Event("bit1", "s", 0, 0, {"bit": 0.0}, {}))
        frame = buffer.pop_frame()
        self.assertEqual(sorted(frame.keys()), ["bit0", "bit1"])
        buffer.push(Event("bit0", "s", 1, 1, {"bit": 1.0}, {}))
        buffer.push(Event("bit0", "s", 2, 2, {"bit": 1.0}, {}))
        with self.assertRaises(ValueError):
            buffer.push(Event("bit0", "s", 3, 3, {"bit": 1.0}, {}))

    def test_heads(self):
        head = ScalarHead()
        self.assertEqual(head.predict([2.0])["value"], 1.0)
        symbols = ["a", "b", "c"]
        discrete = DiscreteHead(symbols)
        distribution = discrete.distribution([3.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertEqual(discrete.sample([3.0, 1.0, 0.0]), "a")
        self.assertEqual(discrete.sample([3.0, 1.0, 0.0]), "a")

    def test_scalar_adapter(self):
        adapter = ScalarStreamTransducer()
        frame = {"series": {"payload": {"value": 0.25}}}
        self.assertEqual(adapter.pack_frame(frame), [0.25, 1.0])
        self.assertEqual(adapter.unpack_action([0.7]), {"value": 0.7})
        adapter.scan_boundary({"series": [0.1, 0.2]})
        with self.assertRaises(ValueError):
            adapter.scan_boundary({"task_id": "A"})

    def test_symbol_adapter(self):
        adapter = SymbolBitsTransducer()
        frame = {"bit0": {"payload": {"bit": 1.0}}, "bit1": {"payload": {"bit": 0.0}},
                 "bit2": {"payload": {"bit": 1.0}}}
        self.assertEqual(adapter.pack_frame(frame), [1.0, 0.0, 1.0, 1.0])
        proposal = adapter.unpack_action([0.9, 0.1, 0.8])
        self.assertEqual(proposal, {"bit0": 1.0, "bit1": 0.0, "bit2": 1.0})
        self.assertEqual(quantize(1.0), 7)
        self.assertEqual(quantize(-1.0), 0)
        self.assertAlmostEqual(dequantize(quantize(0.3)), 0.3, delta=0.15)
        state = adapter.state_dict()
        adapter.load_state_dict(state)
        with self.assertRaises(ValueError):
            adapter.pack_frame({"bit0": {"payload": {"bit": 2.0}}, "bit1": {"payload": {"bit": 0.0}},
                                "bit2": {"payload": {"bit": 0.0}}})


if __name__ == "__main__":
    unittest.main()

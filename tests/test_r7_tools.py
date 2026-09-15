import json
import unittest

from soma.evaluation.tools import (
    ToolRegistry,
    ToolSpec,
    _bits_of,
    _prefix_legal,
    propose_call,
)
from soma.memory import SequenceCircuitMemory


def _weather():
    return ToolSpec("get_weather", {"city": ["paris", "london"]},
                    ["weather"], {'{"city": "paris"}': "sunny"})


class R7ToolTests(unittest.TestCase):
    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            ToolSpec("", {"a": ["b"]}, ["t"], {})
        with self.assertRaises(ValueError):
            ToolSpec("x", {}, ["t"], {})
        with self.assertRaises(ValueError):
            ToolSpec("x", {"a": ["b"]}, [], {})

    def test_instances_bounded(self):
        spec = _weather()
        instances = spec.instances()
        self.assertEqual(len(instances), 2)
        for instance in instances:
            payload = json.loads(instance.decode("utf-8"))
            self.assertEqual(payload["tool"], "get_weather")
            self.assertIn(payload["arguments"]["city"], ("paris", "london"))

    def test_registry_triggers(self):
        registry = ToolRegistry()
        registry.register(_weather())
        with self.assertRaises(ValueError):
            registry.register(_weather())
        self.assertEqual([t.name for t in registry.triggered("WEATHER today")], ["get_weather"])
        self.assertEqual(registry.triggered("hello"), [])

    def test_prefix_legality(self):
        spec = _weather()
        instances = spec.instances()
        full = instances[0]
        self.assertTrue(_prefix_legal(_bits_of(full), instances))
        self.assertFalse(_prefix_legal(_bits_of(b'{"nope": 1}'), instances))

    def test_propose_call_valid(self):
        spec = _weather()
        memory = SequenceCircuitMemory((0, 1), max_order=8, max_circuits=512)
        for bit in _bits_of(b'{"tool": "get_weather", "arguments": {"city": "paris"}}'):
            memory.observe(bit)
        memory.reset_history()
        for bit in _bits_of(b'{"tool": "get_we'):
            memory.observe(bit, learn=False)
        call, report = propose_call(memory, spec, max_bytes=96, seed=0)
        payload = json.loads(call.decode("utf-8"))
        self.assertEqual(payload["tool"], "get_weather")
        self.assertIn(payload["arguments"]["city"], ("paris", "london"))
        self.assertTrue(report["complete"])

    def test_execute_lookup(self):
        spec = _weather()
        self.assertEqual(spec.execute({"city": "paris"}), "sunny")
        self.assertEqual(spec.execute({"city": "nowhere"}), "")


if __name__ == "__main__":
    unittest.main()

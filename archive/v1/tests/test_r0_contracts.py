import os
import unittest

from soma.r0.event_schema import (
    adapter_payload_allowed,
    validate_action,
    validate_event,
    validate_outcome,
)


class R0ContractTests(unittest.TestCase):
    def test_valid_event_passes(self):
        envelope = {
            "channel": "text-bytes",
            "source": "transducer",
            "event_id": 1,
            "clock": 0,
            "payload": {"bytes": [104, 105]},
            "provenance": {"trust": "test"},
        }
        self.assertTrue(validate_event(envelope))

    def test_event_missing_clock_fails(self):
        with self.assertRaises(ValueError):
            validate_event({
                "channel": "t",
                "source": "s",
                "event_id": 1,
                "payload": {},
                "provenance": {},
            })

    def test_transducer_boundary_rejects_task_identity(self):
        leaks = adapter_payload_allowed({"task_id": "A", "bytes": [1]})
        self.assertIn("task_id", leaks)
        clean = adapter_payload_allowed({"bytes": [1]})
        self.assertEqual(clean, [])

    def test_action_and_outcome_correlate(self):
        action = {"channel": "text", "event_id": 7, "clock": 3, "proposal": {}, "effector_schema": "bytes"}
        outcome = {"correlation_id": 7, "clock": 5, "outcome": 0.5, "source": "env"}
        self.assertTrue(validate_action(action))
        self.assertTrue(validate_outcome(outcome))

    def test_workspace_bounds_documented(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "workspace-contract.md")
        with open(path) as handle:
            text = handle.read()
        self.assertIn("retrieval bandwidth", text)


if __name__ == "__main__":
    unittest.main()

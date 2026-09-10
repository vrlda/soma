"""Stdlib validator for soma-event v1. No organism imports."""

REQUIRED_EVENT = ("channel", "source", "event_id", "clock", "payload", "provenance")
REQUIRED_ACTION = ("channel", "event_id", "clock", "proposal", "effector_schema")
REQUIRED_OUTCOME = ("correlation_id", "clock", "outcome", "source")

FORBIDDEN_ADAPTER_KEYS = ("task_id", "phase", "target", "schedule", "answer", "label")


def validate_event(envelope):
    missing = [k for k in REQUIRED_EVENT if k not in envelope]
    if missing:
        raise ValueError("event missing: %s" % ",".join(missing))
    if not isinstance(envelope["clock"], (int, float)):
        raise ValueError("clock must be numeric")
    return True


def validate_action(envelope):
    missing = [k for k in REQUIRED_ACTION if k not in envelope]
    if missing:
        raise ValueError("action missing: %s" % ",".join(missing))
    return True


def validate_outcome(envelope):
    missing = [k for k in REQUIRED_OUTCOME if k not in envelope]
    if missing:
        raise ValueError("outcome missing: %s" % ",".join(missing))
    return True


def adapter_payload_allowed(payload):
    return [k for k in FORBIDDEN_ADAPTER_KEYS if k in payload]

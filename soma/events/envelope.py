"""Versioned universal event envelopes (protocol v2).

Superset of the R0 v1 envelope: adds shape/duration/boundary/scope fields.
Validation is pure stdlib; task-identity keys are rejected at the boundary.
"""

import math

PROTOCOL_VERSION = 2

REQUIRED_EVENT = ("channel", "source", "event_id", "clock", "payload", "provenance")
OPTIONAL_EVENT = ("shape", "duration", "boundary", "uncertainty", "correlation_id",
                  "learning_permission", "trust", "access_scope")
REQUIRED_ACTION = ("channel", "event_id", "clock", "proposal", "effector_schema")
OPTIONAL_ACTION = ("correlation_id", "budget", "stop")
REQUIRED_OUTCOME = ("correlation_id", "clock", "outcome", "source")
OPTIONAL_OUTCOME = ("trust", "delay", "credited")

# Task intelligence must never cross the transducer boundary.
FORBIDDEN_KEYS = ("task_id", "phase", "target", "schedule", "answer", "label",
                  "policy", "correct", "reward_decoder")


def _require_mapping(payload, name):
    if not isinstance(payload, dict):
        raise ValueError("%s envelope must be a mapping" % name)
    return payload


def _check_required(payload, required, name):
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("%s envelope missing: %s" % (name, ",".join(missing)))


def _check_known(payload, required, optional, name):
    unknown = [key for key in payload if key not in required and key not in optional]
    if unknown:
        raise ValueError("%s envelope has unknown keys: %s" % (name, ",".join(sorted(unknown))))


def _check_clock(clock):
    if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(float(clock)):
        raise ValueError("clock must be a finite number")
    if float(clock) < 0:
        raise ValueError("clock must be nonnegative")


def _check_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be a mapping of signal name to number/list")
    forbidden = [key for key in payload if key in FORBIDDEN_KEYS]
    if forbidden:
        raise ValueError("payload carries forbidden task keys: %s" % ",".join(sorted(forbidden)))
    for key, value in payload.items():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError("payload signal %r must be finite numbers" % key)


def _check_shape(payload):
    shape = payload.get("shape")
    if shape is None:
        return
    if not isinstance(shape, (list, tuple)) or not shape or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError("shape must be a nonempty list of positive ints")


def _check_duration(payload):
    if payload.get("duration") is None:
        return
    duration = payload["duration"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) < 0:
        raise ValueError("duration must be a nonnegative finite number")


class Event(object):
    """One timestamped observation from a transducer channel."""

    VERSION = 2

    def __init__(self, channel, source, event_id, clock, payload, provenance, **optional):
        self.channel = str(channel)
        self.source = str(source)
        self.event_id = int(event_id)
        self.clock = float(clock)
        self.payload = {str(key): (list(value) if isinstance(value, (list, tuple)) else value)
                        for key, value in dict(payload).items()}
        self.provenance = dict(provenance)
        self.optional = {str(key): optional[key] for key in optional}
        self.validate()

    def validate(self):
        if not self.channel or not self.source:
            raise ValueError("channel and source must be nonempty")
        if self.event_id < 0:
            raise ValueError("event_id must be nonnegative")
        _check_clock(self.clock)
        _check_payload(self.payload)
        _check_shape(self.optional)
        _check_duration(self.optional)
        for key in self.optional:
            if key not in OPTIONAL_EVENT:
                raise ValueError("event has unknown optional key: %s" % key)

    def to_dict(self):
        payload = {
            "channel": self.channel,
            "source": self.source,
            "event_id": self.event_id,
            "clock": self.clock,
            "payload": {key: (list(value) if isinstance(value, (list, tuple)) else value)
                        for key, value in self.payload.items()},
            "provenance": dict(self.provenance),
        }
        payload.update(self.optional)
        return payload

    @classmethod
    def from_dict(cls, payload):
        _require_mapping(payload, "event")
        _check_required(payload, REQUIRED_EVENT, "event")
        _check_known(payload, REQUIRED_EVENT, OPTIONAL_EVENT, "event")
        optional = {key: payload[key] for key in OPTIONAL_EVENT if key in payload}
        return cls(payload["channel"], payload["source"], payload["event_id"],
                   payload["clock"], payload["payload"], payload["provenance"], **optional)


class ActionEvent(object):
    """One proposed action emitted toward an effector channel."""

    VERSION = 2

    def __init__(self, channel, event_id, clock, proposal, effector_schema, **optional):
        self.channel = str(channel)
        self.event_id = int(event_id)
        self.clock = float(clock)
        self.proposal = dict(proposal)
        self.effector_schema = str(effector_schema)
        self.optional = {str(key): optional[key] for key in optional}
        self.validate()

    def validate(self):
        if not self.channel or not self.effector_schema:
            raise ValueError("channel and effector_schema must be nonempty")
        if self.event_id < 0:
            raise ValueError("event_id must be nonnegative")
        _check_clock(self.clock)
        _check_payload(self.proposal)
        for key in self.optional:
            if key not in OPTIONAL_ACTION:
                raise ValueError("action has unknown optional key: %s" % key)

    def to_dict(self):
        payload = {
            "channel": self.channel,
            "event_id": self.event_id,
            "clock": self.clock,
            "proposal": dict(self.proposal),
            "effector_schema": self.effector_schema,
        }
        payload.update(self.optional)
        return payload

    @classmethod
    def from_dict(cls, payload):
        _require_mapping(payload, "action")
        _check_required(payload, REQUIRED_ACTION, "action")
        _check_known(payload, REQUIRED_ACTION, OPTIONAL_ACTION, "action")
        optional = {key: payload[key] for key in OPTIONAL_ACTION if key in payload}
        return cls(payload["channel"], payload["event_id"], payload["clock"],
                   payload["proposal"], payload["effector_schema"], **optional)


class OutcomeEvent(object):
    """One delayed outcome attributed to a prior action via correlation id."""

    VERSION = 2

    def __init__(self, correlation_id, clock, outcome, source, **optional):
        self.correlation_id = int(correlation_id)
        self.clock = float(clock)
        self.outcome = float(outcome)
        self.source = str(source)
        self.optional = {str(key): optional[key] for key in optional}
        self.validate()

    def validate(self):
        if self.correlation_id < 0:
            raise ValueError("correlation_id must be nonnegative")
        _check_clock(self.clock)
        if not math.isfinite(self.outcome):
            raise ValueError("outcome must be finite")
        if not self.source:
            raise ValueError("source must be nonempty")
        for key in self.optional:
            if key not in OPTIONAL_OUTCOME:
                raise ValueError("outcome has unknown optional key: %s" % key)
        if "credited" in self.optional and not isinstance(self.optional["credited"], bool):
            raise ValueError("outcome credited flag must be boolean")

    def to_dict(self):
        payload = {
            "correlation_id": self.correlation_id,
            "clock": self.clock,
            "outcome": self.outcome,
            "source": self.source,
        }
        payload.update(self.optional)
        return payload

    @classmethod
    def from_dict(cls, payload):
        _require_mapping(payload, "outcome")
        _check_required(payload, REQUIRED_OUTCOME, "outcome")
        _check_known(payload, REQUIRED_OUTCOME, OPTIONAL_OUTCOME, "outcome")
        optional = {key: payload[key] for key in OPTIONAL_OUTCOME if key in payload}
        return cls(payload["correlation_id"], payload["clock"], payload["outcome"], payload["source"], **optional)

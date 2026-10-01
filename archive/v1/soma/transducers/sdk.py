"""Transducer SDK base: declaration, packing, boundary enforcement."""

import math

from ..events.channel import ChannelSchema
from ..events.envelope import FORBIDDEN_KEYS


class TransducerSpec(object):
    """Static adapter declaration: channels, core vector width, bounds."""

    VERSION = 1

    def __init__(self, name, channels, input_size, action_schema, description="",
                 hardware="cpu", reversible=True):
        self.name = str(name)
        if not self.name:
            raise ValueError("transducer name must be nonempty")
        self.channels = list(channels)
        if not self.channels or any(not isinstance(schema, ChannelSchema) for schema in self.channels):
            raise ValueError("transducer must declare at least one ChannelSchema")
        self.input_size = int(input_size)
        if self.input_size < 1:
            raise ValueError("input_size must be positive")
        if not isinstance(action_schema, ChannelSchema):
            raise ValueError("action_schema must be a ChannelSchema")
        self.action_schema = action_schema
        self.description = str(description)
        self.hardware = str(hardware)
        self.reversible = bool(reversible)

    def to_dict(self):
        return {
            "version": self.VERSION,
            "name": self.name,
            "channels": [schema.to_dict() for schema in self.channels],
            "input_size": self.input_size,
            "action_schema": self.action_schema.to_dict(),
            "description": self.description,
            "hardware": self.hardware,
            "reversible": self.reversible,
        }


class Transducer(object):
    """Base class. Subclasses implement pack/unpack for one domain encoding."""

    SPEC_VERSION = 1

    def __init__(self, spec):
        if not isinstance(spec, TransducerSpec):
            raise ValueError("transducer requires a TransducerSpec")
        self.spec = spec
        self._next_event_id = {}

    @property
    def name(self):
        return self.spec.name

    @property
    def input_size(self):
        return self.spec.input_size

    def _sequenced(self, channel, source):
        key = "%s|%s" % (channel, source)
        event_id = self._next_event_id.get(key, 0)
        self._next_event_id[key] = event_id + 1
        return event_id

    @staticmethod
    def scan_boundary(mapping):
        """Reject evaluator/task metadata anywhere in a domain record."""
        stack = [mapping]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    if key in FORBIDDEN_KEYS:
                        raise ValueError("domain record carries forbidden key: %s" % key)
                    stack.append(value)
            elif isinstance(current, (list, tuple)):
                stack.extend(current)
        return True

    @staticmethod
    def check_finite(values):
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("transducer values must be finite numbers")
        return True

    def pack_frame(self, frame):
        """Map one joined channel frame to a fixed core input vector."""
        raise NotImplementedError

    def unpack_action(self, outputs):
        """Map core outputs to an action proposal payload for the action schema."""
        raise NotImplementedError

    def state_dict(self):
        return {"name": self.spec.name, "next_event_id": dict(sorted(self._next_event_id.items()))}

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or payload.get("name") != self.spec.name:
            raise ValueError("transducer state name mismatch")
        self._next_event_id = {str(key): int(value) for key, value in dict(payload.get("next_event_id", {})).items()}

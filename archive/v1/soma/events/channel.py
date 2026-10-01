"""Channel registry: declared transducer schemas with shape/rate/bounds."""

import math


class ChannelSchema(object):
    """One declared event channel: name, signal shape, rate, value bounds."""

    VERSION = 1

    def __init__(self, name, signals, rate=1.0, low=-1.0, high=1.0, description=""):
        self.name = str(name)
        self.signals = [str(signal) for signal in signals]
        if not self.signals:
            raise ValueError("channel must declare at least one signal")
        if len(set(self.signals)) != len(self.signals):
            raise ValueError("channel signals must be unique")
        self.rate = float(rate)
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("channel rate must be positive and finite")
        self.low = float(low)
        self.high = float(high)
        if not math.isfinite(self.low) or not math.isfinite(self.high) or self.low >= self.high:
            raise ValueError("channel bounds must satisfy low < high")
        self.description = str(description)

    def check_payload(self, payload):
        if set(payload.keys()) != set(self.signals):
            raise ValueError("payload signals %s do not match channel %s" % (
                sorted(payload.keys()), self.name))
        for key, value in payload.items():
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError("signal %r must be numeric" % key)
                if not math.isfinite(float(item)) or not self.low <= float(item) <= self.high:
                    raise ValueError("signal %r out of bounds [%s, %s]" % (key, self.low, self.high))
        return True

    def to_dict(self):
        return {
            "version": self.VERSION,
            "name": self.name,
            "signals": list(self.signals),
            "rate": self.rate,
            "low": self.low,
            "high": self.high,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported channel schema version")
        return cls(payload["name"], payload["signals"], rate=payload.get("rate", 1.0),
                   low=payload.get("low", -1.0), high=payload.get("high", 1.0),
                   description=payload.get("description", ""))


class ChannelRegistry(object):
    """Declared channels for one brain. Envelopes validate against it."""

    VERSION = 1

    def __init__(self):
        self.channels = {}

    def register(self, schema):
        if not isinstance(schema, ChannelSchema):
            raise ValueError("register requires a ChannelSchema")
        if schema.name in self.channels:
            raise ValueError("channel already registered: %s" % schema.name)
        self.channels[schema.name] = schema

    def check_event(self, event):
        schema = self.channels.get(event.channel)
        if schema is None:
            raise ValueError("envelope channel is not registered: %s" % event.channel)
        schema.check_payload(event.payload)
        return True

    def to_dict(self):
        return {"version": self.VERSION,
                "channels": {name: schema.to_dict() for name, schema in sorted(self.channels.items())}}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported channel registry version")
        registry = cls()
        for name in sorted(payload.get("channels", {})):
            registry.register(ChannelSchema.from_dict(payload["channels"][name]))
        return registry

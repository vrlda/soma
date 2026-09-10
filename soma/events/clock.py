"""Logical clocks and multi-rate synchronization.

One monotonic master tick per bridge. Each source has a monotone sequence
counter; gaps and duplicates are rejected. Multi-rate channels join by
explicit per-channel buffers with a bounded horizon: a joined frame emits
only when every required channel has advanced past the frame tick.
"""

import math


class LogicalClock(object):
    """Monotonic event clock with per-source sequence tracking."""

    VERSION = 1

    def __init__(self):
        self.tick = 0
        self.last_event_id = {}
        self.last_source_tick = {}

    def observe(self, event):
        """Validate ordering for one envelope; returns the envelope tick."""
        source = "%s|%s" % (event.channel, event.source)
        previous_id = self.last_event_id.get(source, -1)
        if int(event.event_id) != previous_id + 1:
            raise ValueError("event_id must advance by exactly one per source (got %s after %s)" % (
                event.event_id, previous_id))
        if float(event.clock) < float(self.last_source_tick.get(source, -1)):
            raise ValueError("source clock moved backwards")
        self.last_event_id[source] = int(event.event_id)
        self.last_source_tick[source] = float(event.clock)
        self.tick = max(self.tick, int(math.ceil(float(event.clock))))
        return self.tick

    def advance(self):
        self.tick += 1
        return self.tick

    def to_dict(self):
        return {
            "version": self.VERSION,
            "tick": self.tick,
            "last_event_id": dict(sorted(self.last_event_id.items())),
            "last_source_tick": dict(sorted(self.last_source_tick.items())),
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported logical clock version")
        clock = cls()
        clock.tick = int(payload.get("tick", 0))
        if clock.tick < 0:
            raise ValueError("clock tick must be nonnegative")
        clock.last_event_id = {str(key): int(value) for key, value in dict(payload.get("last_event_id", {})).items()}
        clock.last_source_tick = {str(key): float(value) for key, value in dict(payload.get("last_source_tick", {})).items()}
        return clock


class ChannelBuffer(object):
    """Bounded per-channel buffer joining multi-rate streams on master ticks."""

    VERSION = 1

    def __init__(self, channels, max_buffered=64):
        self.channels = [str(channel) for channel in channels]
        if not self.channels:
            raise ValueError("join requires at least one channel")
        self.max_buffered = int(max_buffered)
        if self.max_buffered < 1:
            raise ValueError("max_buffered must be positive")
        self.buffers = {channel: [] for channel in self.channels}

    def push(self, event):
        if event.channel not in self.buffers:
            raise ValueError("event channel is not joined: %s" % event.channel)
        buffer = self.buffers[event.channel]
        buffer.append(event.to_dict())
        if len(buffer) > self.max_buffered:
            raise ValueError("channel %s exceeded buffer bound %d" % (event.channel, self.max_buffered))

    def pop_frame(self):
        """Pop one joined frame when every channel holds at least one event."""
        if any(not buffer for buffer in self.buffers.values()):
            return None
        return {channel: self.buffers[channel].pop(0) for channel in self.channels}

    def pending(self):
        return {channel: len(buffer) for channel, buffer in self.buffers.items()}

    def to_dict(self):
        return {"version": self.VERSION, "channels": list(self.channels),
                "max_buffered": self.max_buffered,
                "buffers": {channel: list(buffer) for channel, buffer in self.buffers.items()}}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported channel buffer version")
        buffer = cls(payload["channels"], max_buffered=payload.get("max_buffered", 64))
        for channel, items in dict(payload.get("buffers", {})).items():
            if channel not in buffer.buffers:
                raise ValueError("buffer channel is not joined: %s" % channel)
            buffer.buffers[channel] = list(items)
        return buffer

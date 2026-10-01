"""History-buffer adapters: explicit lag taps around scalar/bit/base streams.

Buffers are transducer-side timing scaffolding (clocks and buffers are
explicitly allowed); they provide history without solving the task. The core
must still learn which lags and conjunctions predict. Depth and transient
handling are declared and frozen per fixture.
"""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec
from .symbol_bits import quantize

DEPTH = 3
TRANSIENT = 8


class BoundedLagWorkspaceTransducer(Transducer):
    """Domain-neutral bounded lag workspace around an existing transducer.

    The wrapped adapter supplies one current frame. Its declared non-bias
    signals are retained in newest-first lag order, padded with neutral zeros
    during the transient, signed-centered when their declared bounds are
    binary ``[0, 1]``, and followed by one bias. No domain labels, targets,
    text, or vocabulary are interpreted here. Both the wrapped adapter state
    and workspace are serialized for exact split/resume.
    """

    VERSION = 1

    def __init__(self, base, depth=8):
        if not isinstance(base, Transducer):
            raise ValueError("lag workspace requires a base transducer")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("lag workspace depth must be a positive integer")
        signal_width = sum(len(schema.signals) for schema in base.spec.channels)
        if base.input_size != signal_width + 1:
            raise ValueError("base transducer must expose one fixed bias after declared signals")
        super(BoundedLagWorkspaceTransducer, self).__init__(
            TransducerSpec(
                name="lag-workspace-v1:%s:%d" % (base.spec.name, depth),
                channels=list(base.spec.channels),
                input_size=int(depth) * signal_width + 1,
                action_schema=base.spec.action_schema,
                description="Bounded domain-neutral lag workspace around %s." % base.spec.name,
                hardware=base.spec.hardware,
                reversible=base.spec.reversible,
            )
        )
        self.base = base
        self.depth = int(depth)
        self.signal_width = int(signal_width)
        self._binary_slots = []
        for schema in base.spec.channels:
            for _signal in schema.signals:
                self._binary_slots.append(schema.low >= 0.0 and schema.high <= 1.0)
        self._history = []

    @staticmethod
    def _signed_value(value, binary):
        value = float(value)
        if binary:
            if value < 0.0 or value > 1.0:
                raise ValueError("binary lag signal is outside [0, 1]")
            return 2.0 * value - 1.0
        return max(-1.0, min(1.0, value))

    def pack_frame(self, frame):
        packed = list(self.base.pack_frame(frame))
        if len(packed) != self.signal_width + 1:
            raise ValueError("base adapter must return non-bias signals followed by one bias")
        values = [
            self._signed_value(value, binary)
            for value, binary in zip(packed[:-1], self._binary_slots)
        ]
        self.check_finite(values)
        self._history.insert(0, list(values))
        del self._history[self.depth:]
        rows = list(self._history)
        rows.extend([[0.0] * self.signal_width for _ in range(self.depth - len(rows))])
        return [value for row in rows for value in row] + [1.0]

    def unpack_action(self, outputs):
        return self.base.unpack_action(outputs)

    def state_dict(self):
        return {
            "version": self.VERSION,
            "name": self.spec.name,
            "depth": self.depth,
            "signal_width": self.signal_width,
            "history": [list(row) for row in self._history],
            "base": self.base.state_dict(),
            "next_event_id": dict(sorted(self._next_event_id.items())),
        }

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            raise ValueError("unsupported lag workspace state version")
        if payload.get("name") != self.spec.name or int(payload.get("depth", 0)) != self.depth:
            raise ValueError("lag workspace configuration mismatch on resume")
        if int(payload.get("signal_width", 0)) != self.signal_width:
            raise ValueError("lag workspace signal width mismatch on resume")
        history = payload.get("history", [])
        if not isinstance(history, list) or len(history) > self.depth:
            raise ValueError("lag workspace history exceeds declared depth")
        decoded = []
        for row in history:
            if not isinstance(row, list) or len(row) != self.signal_width:
                raise ValueError("lag workspace history row shape mismatch")
            self.check_finite(row)
            if any(float(value) < -1.0 or float(value) > 1.0 for value in row):
                raise ValueError("lag workspace history value is outside [-1, 1]")
            decoded.append([float(value) for value in row])
        base_state = payload.get("base")
        if not isinstance(base_state, dict) or base_state.get("name") != self.base.spec.name:
            raise ValueError("lag workspace base state mismatch on resume")
        self.base.load_state_dict(base_state)
        self._history = decoded
        self._next_event_id = {str(key): int(value) for key, value in dict(payload.get("next_event_id", {})).items()}


def history_scalar_spec(depth=DEPTH):
    return TransducerSpec(
        name="history-scalar-v1",
        channels=[ChannelSchema("lag%d" % lag, ["value"], rate=1.0, low=-1.0, high=1.0,
                                description="stream value %d steps ago" % lag)
                  for lag in range(depth)],
        input_size=depth + 1,
        action_schema=ChannelSchema("prediction", ["value"], rate=1.0, low=-1.0, high=1.0),
        description="Tapped delay line over the scalar stream plus bias.",
        reversible=True,
    )


def history_bits_spec(depth=DEPTH):
    channels = []
    for lag in range(depth):
        for bit in range(3):
            channels.append(ChannelSchema("lag%dbit%d" % (lag, bit), ["bit"],
                                          rate=1.0, low=0.0, high=1.0))
    return TransducerSpec(
        name="history-bits-v1",
        channels=channels,
        input_size=depth * 3 + 1,
        action_schema=ChannelSchema("prediction", ["value"], rate=1.0, low=-1.0, high=1.0),
        description="Tapped delay line over the 3-bit stream plus bias.",
        reversible=True,
    )


class HistoryScalarTransducer(Transducer):
    def __init__(self, depth=DEPTH):
        super(HistoryScalarTransducer, self).__init__(history_scalar_spec(depth))
        self.depth = int(depth)

    def pack_frame(self, frame):
        values = []
        for lag in range(self.depth):
            value = frame["lag%d" % lag]["payload"]["value"]
            if isinstance(value, (list, tuple)):
                value = value[0]
            values.append(max(-1.0, min(1.0, float(value))))
        self.check_finite(values)
        return values + [1.0]

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}


def bit_history_spec(depth=8):
    channels = [ChannelSchema("hist%d" % lag, ["bit"], rate=1.0, low=0.0, high=1.0,
                              description="stream bit %d steps ago" % lag)
                for lag in range(depth)]
    channels.append(ChannelSchema("boundary", ["byte_end"], rate=1.0, low=0.0, high=1.0))
    return TransducerSpec(
        name="bit-history-v1",
        channels=channels,
        input_size=depth + 2,
        action_schema=ChannelSchema("bit-prediction", ["value"], rate=1.0,
                                    low=0.0, high=1.0),
        description="Raw bit-history buffer plus byte framing plus bias.",
        reversible=True,
    )


class BitHistoryTransducer(Transducer):
    """Tapped delay line over raw bits. Buffer discloses history; the core
    must still select lags and conjunctions (adapter-only control stays)."""

    def __init__(self, depth=8):
        super(BitHistoryTransducer, self).__init__(bit_history_spec(depth))
        self.depth = int(depth)

    def pack_frame(self, frame):
        values = []
        for lag in range(self.depth):
            value = frame["hist%d" % lag]["payload"]["bit"]
            if isinstance(value, (list, tuple)):
                value = value[0]
            if value not in (0, 1, 0.0, 1.0):
                raise ValueError("history bits carry only 0/1")
            values.append(float(value))
        end = frame["boundary"]["payload"]["byte_end"]
        if isinstance(end, (list, tuple)):
            end = end[0]
        if end not in (0, 1, 0.0, 1.0):
            raise ValueError("byte_end carries only 0/1")
        values.append(float(end))
        return values + [1.0]

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        value = (float(outputs[0]) + 1.0) / 2.0
        return {"value": max(0.02, min(0.98, value))}


class HistoryBitsTransducer(Transducer):
    """Tapped delay line over the 3-bit stream plus bias, scalar prediction."""

    def __init__(self, depth=DEPTH):
        super(HistoryBitsTransducer, self).__init__(history_bits_spec(depth))
        self.depth = int(depth)

    def pack_frame(self, frame):
        values = []
        for lag in range(self.depth):
            for bit in range(3):
                value = frame["lag%dbit%d" % (lag, bit)]["payload"]["bit"]
                if isinstance(value, (list, tuple)):
                    value = value[0]
                if value not in (0, 1, 0.0, 1.0):
                    raise ValueError("bit channels carry only 0/1")
                values.append(float(value))
        return values + [1.0]

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}

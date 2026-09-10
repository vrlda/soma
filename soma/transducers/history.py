"""History-buffer adapters: explicit lag taps around the scalar/bit base.

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

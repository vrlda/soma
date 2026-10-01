"""Domain adapter B: 3-bit symbol encoding over three channels.

Non-equivalent to the scalar adapter: three binary channels instead of one
scalar channel, different shape and core packing, same temporal structure.
"""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec

LEVELS = 8


def quantize(value):
    clipped = max(-1.0, min(1.0, float(value)))
    return min(LEVELS - 1, max(0, int((clipped + 1.0) / 2.0 * LEVELS)))


def dequantize(symbol):
    return (min(LEVELS - 1, max(0, int(symbol))) + 0.5) / LEVELS * 2.0 - 1.0


def symbol_spec():
    return TransducerSpec(
        name="symbol-bits-v1",
        channels=[
            ChannelSchema("bit0", ["bit"], rate=1.0, low=0.0, high=1.0),
            ChannelSchema("bit1", ["bit"], rate=1.0, low=0.0, high=1.0),
            ChannelSchema("bit2", ["bit"], rate=1.0, low=0.0, high=1.0),
        ],
        input_size=4,
        action_schema=ChannelSchema("symbol", ["bit0", "bit1", "bit2"], rate=1.0,
                                    low=0.0, high=1.0,
                                    description="predicted next 3-bit symbol"),
        description="3-bit quantized encoding of the same scalar stream.",
        reversible=True,
    )


class SymbolBitsTransducer(Transducer):
    """Packs [b0, b1, b2, bias]; unpacks thresholded bits from core outputs."""

    def __init__(self):
        super(SymbolBitsTransducer, self).__init__(symbol_spec())

    def pack_frame(self, frame):
        bits = []
        for channel in ("bit0", "bit1", "bit2"):
            bit = frame[channel]["payload"]["bit"]
            if isinstance(bit, (list, tuple)):
                bit = bit[0]
            if bit not in (0, 1, 0.0, 1.0):
                raise ValueError("bit channels carry only 0/1")
            bits.append(float(bit))
        return bits + [1.0]

    def unpack_action(self, outputs):
        values = [float(item) for item in list(outputs)[:3]]
        self.check_finite(values)
        while len(values) < 3:
            values.append(0.0)
        return {"bit0": 1.0 if values[0] >= 0.5 else 0.0,
                "bit1": 1.0 if values[1] >= 0.5 else 0.0,
                "bit2": 1.0 if values[2] >= 0.5 else 0.0}

    @staticmethod
    def encode_value(value, clock, source="fixture"):
        """Helper: scalar to three bit-event payloads (fixture-side only)."""
        from ..events.envelope import Event
        symbol = quantize(value)
        bits = [(symbol >> index) & 1 for index in range(3)]
        return [
            {"channel": "bit%d" % index, "source": source, "payload": {"bit": float(bits[index])}}
            for index in range(3)
        ]

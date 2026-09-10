"""Domain adapter C: 3-bit symbol inputs, scalar prediction output.

Same input encoding as symbol-bits-v1, same scalar output binding as
scalar-stream-v1. Lets one proof task vary input encoding while holding the
action interface fixed.
"""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec
from .symbol_bits import SymbolBitsTransducer


def bit_scalar_spec():
    return TransducerSpec(
        name="bit-scalar-v1",
        channels=[
            ChannelSchema("bit0", ["bit"], rate=1.0, low=0.0, high=1.0),
            ChannelSchema("bit1", ["bit"], rate=1.0, low=0.0, high=1.0),
            ChannelSchema("bit2", ["bit"], rate=1.0, low=0.0, high=1.0),
        ],
        input_size=4,
        action_schema=ChannelSchema("prediction", ["value"], rate=1.0, low=-1.0, high=1.0,
                                    description="scalar prediction from bit-encoded inputs"),
        description="Bit-encoded inputs with a scalar prediction head.",
        reversible=True,
    )


class BitScalarTransducer(Transducer):
    """Reuses symbol bit packing; predicts one scalar like the scalar adapter."""

    def __init__(self):
        super(BitScalarTransducer, self).__init__(bit_scalar_spec())
        self._bits = SymbolBitsTransducer()

    def pack_frame(self, frame):
        return self._bits.pack_frame(frame)

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}

"""Cart transducer: position/velocity channels with a force head.

Two scalar channels plus bias; the safety envelope (force clamp, track
limits, emergency stop) lives in the simulator and the effector, never in
the learned core.
"""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec


def cart_spec():
    return TransducerSpec(
        name="cart-v1",
        channels=[
            ChannelSchema("position", ["value"], rate=1.0, low=-1.0, high=1.0),
            ChannelSchema("velocity", ["value"], rate=1.0, low=-1.0, high=1.0),
        ],
        input_size=3,
        action_schema=ChannelSchema("force", ["value"], rate=1.0, low=-1.0, high=1.0,
                                    description="requested force; clamped outside"),
        description="Cart position/velocity with bounded force output.",
        reversible=True,
    )


class CartTransducer(Transducer):
    def __init__(self):
        super(CartTransducer, self).__init__(cart_spec())

    def pack_frame(self, frame):
        values = []
        for channel in ("position", "velocity"):
            value = frame[channel]["payload"]["value"]
            if isinstance(value, (list, tuple)):
                value = value[0]
            values.append(max(-1.0, min(1.0, float(value))))
        self.check_finite(values)
        return values + [1.0]

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}

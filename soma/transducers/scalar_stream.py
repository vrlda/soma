"""Domain adapter A: single scalar series channel (minimally interpreted)."""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec


def scalar_spec():
    return TransducerSpec(
        name="scalar-stream-v1",
        channels=[ChannelSchema("series", ["value"], rate=1.0, low=-1.0, high=1.0,
                                description="one normalized sample per event")],
        input_size=2,
        action_schema=ChannelSchema("prediction", ["value"], rate=1.0, low=-1.0, high=1.0,
                                    description="one-step-ahead scalar prediction"),
        description="Minimally interpreted scalar stream: raw values, no features.",
        reversible=True,
    )


class ScalarStreamTransducer(Transducer):
    """Packs [value, bias] core inputs; unpacks the first output as prediction."""

    def __init__(self):
        super(ScalarStreamTransducer, self).__init__(scalar_spec())

    def pack_frame(self, frame):
        value = frame["series"]["payload"]["value"]
        if isinstance(value, (list, tuple)):
            value = value[0]
        self.check_finite([value])
        return [max(-1.0, min(1.0, float(value))), 1.0]

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}

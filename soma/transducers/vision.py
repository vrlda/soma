"""Minimal vision adapter: raw binary pixels, no features.

A SIZE x SIZE binary image maps to one channel per pixel (minimally
interpreted: no edges, no convolutions, no labels). The core must learn
any spatial structure itself through the universal event path.
"""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec

SIZE = 6


def glyph_spec(size=SIZE):
    channels = [ChannelSchema("px%d" % index, ["on"], rate=1.0, low=0.0, high=1.0,
                              description="raw pixel value")
                for index in range(size * size)]
    return TransducerSpec(
        name="glyph-vision-v1",
        channels=channels,
        input_size=size * size + 1,
        action_schema=ChannelSchema("prediction", ["value"], rate=1.0,
                                    low=-1.0, high=1.0,
                                    description="scalar class prediction"),
        description="Raw binary pixels plus bias. No visual features.",
        reversible=True,
    )


class GlyphTransducer(Transducer):
    def __init__(self, size=SIZE):
        super(GlyphTransducer, self).__init__(glyph_spec(size))
        self.size = int(size)

    def pack_frame(self, frame):
        values = []
        for index in range(self.size * self.size):
            value = frame["px%d" % index]["payload"]["on"]
            if isinstance(value, (list, tuple)):
                value = value[0]
            if value not in (0, 1, 0.0, 1.0):
                raise ValueError("pixels carry only 0/1")
            values.append(float(value))
        return values + [1.0]

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}


def render_vertical(size=SIZE, shift=(0, 0), noise=()):
    """Vertical center bar with translation shift and noise flips."""
    grid = [[0] * size for _ in range(size)]
    for row in range(size):
        grid[row][size // 2 + shift[0]] = 1
    for row, column in noise:
        grid[row % size][column % size] ^= 1
    return grid


def render_horizontal(size=SIZE, shift=(0, 0), noise=()):
    """Horizontal center bar with translation shift and noise flips."""
    grid = [[0] * size for _ in range(size)]
    for column in range(size):
        grid[size // 2 + shift[1]][column] = 1
    for row, column in noise:
        grid[row % size][column % size] ^= 1
    return grid

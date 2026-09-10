"""Canonical text transducer: reversible UTF-8 bytes plus stream boundaries.

No vocabulary, no segmentation, no language knowledge. Bytes are the neutral
low-level alphabet; all structure above bytes must be acquired, not given.
Invalid sequences are rejected, never silently repaired.
"""

from ..events.channel import ChannelSchema
from .sdk import Transducer, TransducerSpec


def text_bytes_spec():
    return TransducerSpec(
        name="text-utf8-v1",
        channels=[
            ChannelSchema("bit", ["value"], rate=1.0, low=0.0, high=1.0,
                          description="one bit of the UTF-8 stream, MSB first"),
            ChannelSchema("boundary", ["byte_end"], rate=1.0, low=0.0, high=1.0,
                          description="generic byte framing: 1 on last bit of a byte"),
        ],
        input_size=3,
        action_schema=ChannelSchema("bit-prediction", ["value"], rate=1.0, low=0.0, high=1.0,
                                    description="next-bit prediction in [0, 1]"),
        description="Reversible UTF-8 transducer, one binary prediction per bit.",
        reversible=True,
    )


def encode_bytes(data):
    """Bytes to bit lists, MSB first. Rejects non-bytes input."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("text transducer encodes bytes only")
    return [[float((byte >> index) & 1) for index in range(7, -1, -1)] for byte in data]


def decode_bits(bit_lists, strict=True):
    """Bit lists back to bytes. Strict mode rejects short rows and extras."""
    out = bytearray()
    for row in bit_lists:
        if len(row) != 8:
            raise ValueError("byte rows must hold exactly 8 bits")
        value = 0
        for bit in row:
            if bit not in (0, 1, 0.0, 1.0):
                raise ValueError("bits carry only 0/1")
            value = (value << 1) | int(bit)
        out.append(value)
    try:
        return bytes(out)
    except ValueError as error:
        raise ValueError("decoded bytes are invalid: %s" % error)


def validate_utf8(data):
    """Raise on malformed UTF-8. Never repair silently."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("validate_utf8 requires bytes")
    try:
        bytes(data).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("malformed UTF-8: %s" % error)
    return True


class TextBytesTransducer(Transducer):
    """Packs [bit, byte_end, bias] for bit prediction; unpacks bit forecasts."""

    def __init__(self):
        super(TextBytesTransducer, self).__init__(text_bytes_spec())

    def pack_frame(self, frame):
        bit = frame["bit"]["payload"]["value"]
        end = frame["boundary"]["payload"]["byte_end"]
        for name, value in (("bit", bit), ("byte_end", end)):
            if isinstance(value, (list, tuple)):
                value = value[0]
            if value not in (0, 1, 0.0, 1.0):
                raise ValueError("%s carries only 0/1" % name)
            if name == "bit":
                bit = float(value)
            else:
                end = float(value)
        return [bit, end, 1.0]

    # Probability floor: exact 0/1 claims make NLL catastrophic on any
    # error and hide learning signal under miscalibration. Fixed and
    # documented; not tuned per run.
    PROBABILITY_FLOOR = 0.02

    def unpack_action(self, outputs):
        self.check_finite(list(outputs)[:1])
        # Motor range is [-1, 1]; map it affinely onto P(next bit = 1).
        # Clipping raw outputs created a dead gradient half-space for mu < 0.
        value = (float(outputs[0]) + 1.0) / 2.0
        floor = self.PROBABILITY_FLOOR
        return {"value": max(floor, min(1.0 - floor, value))}  # P(next bit = 1)


class PhaseTextBytesTransducer(TextBytesTransducer):
    """Timing-infrastructure variant: free-running mod-8 phase channels.

    Phase is modality-neutral clock behavior declared by the adapter, not
    language knowledge: it counts events mod 8 and means nothing until the
    core learns what each phase predicts in this domain. The raw path
    (text-utf8-v1) remains the minimally interpreted baseline.
    """

    def __init__(self):
        super(PhaseTextBytesTransducer, self).__init__()
        channels = [
            ChannelSchema("bit", ["value"], rate=1.0, low=0.0, high=1.0),
            ChannelSchema("boundary", ["byte_end"], rate=1.0, low=0.0, high=1.0),
        ] + [ChannelSchema("phase%d" % index, ["on"], rate=1.0, low=0.0, high=1.0)
             for index in range(3)]
        action = ChannelSchema("bit-prediction", ["value"], rate=1.0, low=0.0, high=1.0)
        self.spec = TransducerSpec(
            "text-utf8-phase-v1", channels, 6, action,
            description="UTF-8 bits plus free-running mod-8 phase timing.",
            reversible=True)

    def pack_frame(self, frame):
        bit = frame["bit"]["payload"]["value"]
        end = frame["boundary"]["payload"]["byte_end"]
        values = []
        for name, value in (("bit", bit), ("byte_end", end)):
            if isinstance(value, (list, tuple)):
                value = value[0]
            if value not in (0, 1, 0.0, 1.0):
                raise ValueError("%s carries only 0/1" % name)
            values.append(float(value))
        for index in range(3):
            channel = "phase%d" % index
            if channel not in frame:
                raise ValueError("phase timing channel missing: %s" % channel)
            value = frame[channel]["payload"]["on"]
            if isinstance(value, (list, tuple)):
                value = value[0]
            if value not in (0, 1, 0.0, 1.0):
                raise ValueError("phase channels carry only 0/1")
            values.append(float(value))
        return values + [1.0]

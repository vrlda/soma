"""Background-memory variants for the frozen language gates.

The frozen gates (R3C, R3D, R7) were written against ``SequenceCircuitMemory``.
With ``--memory mixing`` they run the same checks against the
circuit-mixing memory as the product serves it: hosted by the Rust engine
(``EngineMixingMemory``), with the product's decoding settings (evidence
arbitration between dialogue and background tiers, deterministic decoding).
The default ``suffix`` path is unchanged and still writes the frozen reports.
"""

import os
import shutil
import tempfile

from .english import bit_stream

MEMORY_CHOICES = ("suffix", "mixing")
MIXING_CONFIG = {"max_circuits": 1 << 20}


def add_memory_argument(parser):
    parser.add_argument("--memory", choices=MEMORY_CHOICES, default="suffix",
                        help="background memory: 'mixing' runs the engine-served "
                             "circuit-mixing memory and writes a -mixing report")


def report_path(path, kind):
    if kind == "suffix":
        return path
    root, extension = os.path.splitext(path)
    return "%s-mixing%s" % (root, extension)


def respond_settings(kind):
    """Decoding settings the product uses for this memory kind."""
    if kind == "suffix":
        return {}
    return {"deterministic": True, "evidence_arbitration": True}


def _train_suffix(data):
    from ..memory import SequenceCircuitMemory
    memory = SequenceCircuitMemory((0, 1), max_order=16, max_circuits=131072,
                                   min_support=2, prior=0.5)
    bits, _ = bit_stream(data)
    for symbol in bits[:-1]:
        memory.observe(symbol)
    return memory


class BackgroundFactory(object):
    """Trained background memories; engine memories are trained once and
    reloaded from a saved state for every further copy."""

    def __init__(self, kind, data):
        if kind not in MEMORY_CHOICES:
            raise ValueError("unknown memory kind: %s" % kind)
        self.kind = kind
        self.data = bytes(data)
        self._directory = None
        self._state = None

    def __call__(self):
        if self.kind == "suffix":
            return _train_suffix(self.data)
        from ..memory.engine import EngineMixingMemory
        if self._state is None:
            self._directory = tempfile.mkdtemp(prefix="soma-background-")
            self._state = os.path.join(self._directory, "background.somamix")
            memory = EngineMixingMemory(config=MIXING_CONFIG, path=self._state)
            bits, _ = bit_stream(self.data)
            for symbol in bits[:-1]:
                memory.observe(symbol)
            memory.save()
            memory.close()
        return EngineMixingMemory(path=self._state)

    def untrained(self):
        """Same configuration with no learned circuits (the lesion control)."""
        if self.kind == "suffix":
            from ..memory import SequenceCircuitMemory
            return SequenceCircuitMemory((0, 1), max_order=16, max_circuits=131072,
                                         min_support=2, prior=0.5)
        from ..memory.engine import EngineMixingMemory
        return EngineMixingMemory(config=MIXING_CONFIG)

    def close(self):
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
            self._directory = None
            self._state = None


def learned_state(memory):
    """Bytes of everything a memory has learned, excluding stream position.

    For engine memories: the SOMAMIX1 body after the header (floats, weights,
    update counters, circuits, calibration, correction), without the header's
    history/partial/event counters or the checksum. For suffix memories: the
    circuit counts.
    """
    if not hasattr(memory, "save"):
        return [list(circuit["counts"]) for _, circuit in sorted(memory.circuits.items())]
    import struct
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "probe.somamix")
        memory.save(path)
        with open(path, "rb") as handle:
            data = handle.read()
    (length,) = struct.unpack_from("<I", data, 8)
    return data[12 + length:-8]

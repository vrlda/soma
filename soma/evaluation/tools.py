"""R7 tool calls: declared schemas, constrained decoding, stub executors.

Tools are explicit configuration, not learned knowledge: each tool declares
a name, string-enum arguments, and trigger substrings. Calls are generated
bit by bit with prefix-legality masking, so every emitted call parses
against its schema. Results return as attributed text the brain can learn.
"""

import json
import random

from ..transducers.text_bytes import encode_bytes
from .generate import _bits_form_valid_prefix


class ToolSpec(object):
    """One callable tool: name, enum args, trigger substrings, stub result."""

    VERSION = 1

    def __init__(self, name, args, triggers, results):
        self.name = str(name)
        if not self.name:
            raise ValueError("tool name must be nonempty")
        self.args = {str(key): [str(value) for value in values]
                     for key, values in dict(args).items()}
        if not self.args or any(not values for values in self.args.values()):
            raise ValueError("tool needs at least one enum argument")
        self.triggers = [str(trigger) for trigger in triggers]
        if not self.triggers:
            raise ValueError("tool needs at least one trigger")
        self.results = {str(key): str(value) for key, value in dict(results).items()}

    def instances(self, limit=10000):
        """Enumerate every valid call JSON (bounded subset schema)."""
        calls = [{}]
        for key in sorted(self.args):
            calls = [dict(call, **{key: value}) for call in calls for value in self.args[key]]
            if len(calls) > limit:
                raise ValueError("tool instance space exceeds bound")
        return [json.dumps({"tool": self.name, "arguments": call},
                           sort_keys=True).encode("utf-8") for call in calls]

    def to_dict(self):
        return {"version": self.VERSION, "name": self.name,
                "args": {k: list(v) for k, v in self.args.items()},
                "triggers": list(self.triggers),
                "results": dict(self.results)}

    def execute(self, arguments):
        """Stub executor: exact-argument lookup, else empty result."""
        key = json.dumps(arguments, sort_keys=True)
        return self.results.get(key, "")


class ToolRegistry(object):
    def __init__(self):
        self.tools = {}

    def register(self, spec):
        if not isinstance(spec, ToolSpec):
            raise ValueError("register requires a ToolSpec")
        if spec.name in self.tools:
            raise ValueError("tool already registered: %s" % spec.name)
        self.tools[spec.name] = spec

    def triggered(self, user_text):
        lowered = user_text.lower()
        return [spec for spec in self.tools.values()
                if any(trigger.lower() in lowered for trigger in spec.triggers)]

    def to_dict(self):
        return {"tools": {name: spec.to_dict() for name, spec in sorted(self.tools.items())}}


def _bits_of(data):
    rows = encode_bytes(bytes(data))
    return [int(bit) for row in rows for bit in row]


def propose_call(memory, tool, max_bytes=96, seed=0):
    """Sample a call bit by bit; only schema-legal prefixes survive.

    Returns (call_bytes, forced_bits). Every returned call parses and
    validates against the tool schema by construction.
    """
    rng = random.Random(seed)
    instances = tool.instances()
    out_bits = []
    forced = 0
    for _ in range(max_bytes * 8):
        distribution, _ = memory.distribution()
        probability = distribution[1]
        candidate = 1 if rng.random() < probability else 0
        trial = out_bits + [candidate]
        if _prefix_legal(trial, instances):
            chosen = candidate
        else:
            chosen = 1 - candidate
            forced += 1
            if not _prefix_legal(out_bits + [chosen], instances):
                raise ValueError("tool decoder reached a dead end")
        out_bits.append(chosen)
        memory.observe(chosen, learn=False)
        raw = _bits_to_bytes(out_bits)
        for instance in instances:
            if raw == instance[:len(raw)] and len(raw) == len(instance):
                return raw, {"forced_bits": forced, "complete": True}
    raw = _bits_to_bytes(out_bits)
    for instance in instances:
        if raw == instance:
            return raw, {"forced_bits": forced, "complete": True}
    raise ValueError("tool proposal exceeded budget without completing")


def _bits_to_bytes(out_bits):
    raw = bytearray()
    for index in range(0, len(out_bits), 8):
        value = 0
        for bit in out_bits[index:index + 8]:
            value = (value << 1) | bit
        raw.append(value)
    return bytes(raw)


def _prefix_legal(out_bits, instances):
    complete = (len(out_bits) // 8) * 8
    head = _bits_to_bytes(out_bits[:complete])
    tail = out_bits[complete:]
    for instance in instances:
        if not instance.startswith(head):
            continue
        if len(head) >= len(instance):
            # Head already covers an instance; only exact end is legal.
            if not tail:
                return True
            continue
        if not tail:
            return True
        # Some completion of the partial byte must match the next byte.
        for fill in range(1 << (8 - len(tail))):
            value = 0
            for bit in tail:
                value = (value << 1) | bit
            value = (value << (8 - len(tail))) | fill
            if instance[len(head)] == value:
                return True
    return False

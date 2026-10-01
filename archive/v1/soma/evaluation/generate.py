"""R3C generation: byte emission from ordered sequence memory.

Emitted bits never train protected state: generation observes with
learn=False always. Callers choose deterministic (argmax) or tempered
sampling; streaming yields whole bytes as they complete.
"""

import math
import random

from ..memory import SequenceCircuitMemory
from ..transducers.text_bytes import decode_bits, encode_bytes

FIXED_PREFIXES = (
    b"The ",
    b"Alice ",
    b"the rabbit ",
    b"Once upon a ",
    b"xyzzy ",
    b"\xff\x00\x7f",
    b"She said ",
    b"down the ",
)


def temper(probability, temperature):
    """Binary temperature scaling. Deterministic argmax at temperature 0."""
    probability = max(1e-6, min(1.0 - 1e-6, float(probability)))
    if temperature <= 0.0:
        return 1.0 if probability >= 0.5 else 0.0
    scaled = probability ** (1.0 / float(temperature))
    other = (1.0 - probability) ** (1.0 / float(temperature))
    return scaled / (scaled + other)


def generate(memory, prefix_bytes, max_bytes=50, deterministic=True,
             temperature=1.0, seed=0):
    """Emit bytes from memory. Returns (bytes, report). Never trains."""
    if not isinstance(prefix_bytes, (bytes, bytearray)):
        raise ValueError("generation prefix must be bytes")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    rng = random.Random(seed)
    memory.reset_history()
    for row in encode_bytes(bytes(prefix_bytes)):
        for bit in row:
            memory.observe(int(bit), learn=False)
    out_bits = []
    orders = []
    for _ in range(max_bytes * 8):
        distribution, order = memory.distribution()
        probability = temper(distribution[1], 0.0 if deterministic else temperature)
        if deterministic:
            bit = 1 if probability >= 0.5 else 0
        else:
            bit = 1 if rng.random() < probability else 0
        out_bits.append(bit)
        orders.append(order)
        memory.observe(bit, learn=False)
    raw = bytearray()
    for index in range(0, len(out_bits), 8):
        value = 0
        for bit in out_bits[index:index + 8]:
            value = (value << 1) | bit
        raw.append(value)
    raw_bytes = bytes(raw)
    try:
        raw_bytes.decode("utf-8", errors="strict")
        valid = True
    except UnicodeDecodeError:
        valid = False
    return raw_bytes, {
        "prefix": bytes(prefix_bytes),
        "bytes": len(raw_bytes),
        "valid_utf8": valid,
        "deterministic": deterministic,
        "temperature": temperature,
        "seed": seed,
        "mean_order": sum(orders) / max(1, len(orders)),
    }


def continuation_nll(memory, prefix_bytes, continuation_bytes):
    """Held-out likelihood of a fixed continuation given a prefix."""
    memory.reset_history()
    for row in encode_bytes(bytes(prefix_bytes)):
        for bit in row:
            memory.observe(int(bit), learn=False)
    total = 0.0
    count = 0
    for row in encode_bytes(bytes(continuation_bytes)):
        for bit in row:
            distribution, _ = memory.distribution()
            probability = temper(distribution[int(bit)], 1.0)
            total -= math.log(max(1e-9, probability), 2)
            count += 1
            memory.observe(int(bit), learn=False)
    return total / max(1, count)


def _valid_prefixes():
    return [bytes(prefix) for prefix in FIXED_PREFIXES if _is_valid_utf8(bytes(prefix))]


def run_generation_suite(memory_factory, max_bytes=50, samples=4):
    """Frozen generation gates over fixed prefixes.

    Constrained emission only (raw sampling documents the failure mode in
    tests, not here). Seeds vary per prefix so diversity is meaningful.
    """
    prefixes = _valid_prefixes()
    deterministic_reports = []
    sampled_reports = []
    for index, prefix in enumerate(prefixes):
        memory = memory_factory()
        output, report = generate_constrained(memory, prefix, max_bytes, deterministic=True)
        deterministic_reports.append((output, report))
        for sample in range(samples):
            memory = memory_factory()
            output, report = generate_constrained(memory, prefix, max_bytes, deterministic=False,
                                                  temperature=1.0, seed=index * 7919 + sample)
            sampled_reports.append((output, report))
    return {
        "prefixes": len(prefixes),
        "deterministic_valid_rate": sum(
            1.0 for _, report in deterministic_reports if report["valid_utf8"]) / max(1, len(deterministic_reports)),
        "sampled_valid_rate": sum(
            1.0 for _, report in sampled_reports if report["valid_utf8"]) / max(1, len(sampled_reports)),
        "deterministic_distinct": len({output for output, _ in deterministic_reports}),
        "sampled_distinct": len({output for output, _ in sampled_reports}),
        "deterministic_outputs": [output for output, _ in deterministic_reports],
    }


def horizon_sensitivity(memory_factory, validation, horizon_bytes=2):
    """Familiar junctions score better; deep history beyond the horizon is
    invisible. Returns (spread_within, spread_beyond).

    spread_within: true 60-byte prefix vs a novel prefix for the same 8-byte
    window (familiar junction vs backoff).
    spread_beyond: true prefix vs junk sharing its last ``horizon_bytes``
    bytes (identical histories within the memory's horizon must give
    identical likelihoods). The default 2 is the 16-bit suffix memory's
    horizon; circuit mixing sees 12 bytes.
    """
    window = bytes(validation[60:68])
    true_prefix = bytes(validation[:60])
    novel_prefix = b"xyzzy plugh frobnicate " * 3
    same_tail = b"Qz!#" * 10 + bytes(validation[60 - horizon_bytes:60])
    spread_within = abs(continuation_nll(memory_factory(), true_prefix, window)
                        - continuation_nll(memory_factory(), novel_prefix, window))
    spread_beyond = abs(continuation_nll(memory_factory(), true_prefix, window)
                        - continuation_nll(memory_factory(), same_tail, window))
    return spread_within, spread_beyond


def _is_valid_utf8(data):
    try:
        data.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


def _is_valid_utf8_prefix(data):
    """True when complete bytes are valid UTF-8 and any trailing partial
    sequence is a legal lead (ASCII byte, or the start of a 2/3/4-byte
    sequence with room left in the current byte budget is not checkable at
    bit level, so this validates byte completions plus lead-byte legality).
    """
    if not data:
        return True
    # Walk bytes directly: decoder reason strings are unreliable across
    # truncation shapes, so every completed character is strictly decoded
    # and any trailing partial sequence must satisfy lead-range rules.
    index = 0
    while index < len(data):
        byte = data[index]
        if byte < 0x80:
            width = 1
        elif 0xC2 <= byte <= 0xDF:
            width = 2
        elif 0xE0 <= byte <= 0xEF:
            width = 3
        elif 0xF0 <= byte <= 0xF4:
            width = 4
        else:
            return False
        chunk = data[index:index + width]
        if len(chunk) < width:
            for continuation in chunk[1:]:
                if not 0x80 <= continuation <= 0xBF:
                    return False
            # Lead-range rules apply even to partial sequences: overlongs,
            # surrogates, and out-of-range planes are uncompletable.
            lead = chunk[0]
            second = chunk[1] if len(chunk) > 1 else None
            if width == 2 and lead in (0xC0, 0xC1):
                return False
            if width == 3 and lead == 0xE0 and second is not None:
                if not 0xA0 <= second <= 0xBF:
                    return False
            if width == 3 and lead == 0xED and second is not None:
                # 0xED with a 0xA0-0xBF second byte is a surrogate: illegal.
                if not 0x80 <= second <= 0x9F:
                    return False
            if width == 4 and lead == 0xF0 and second is not None:
                if not 0x90 <= second <= 0xBF:
                    return False
            if width == 4 and lead == 0xF4 and second is not None:
                if not 0x80 <= second <= 0x8F:
                    return False
            return True
        try:
            chunk.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
        index += width
    return True


def generate_constrained(memory, prefix_bytes, max_bytes=50, deterministic=True,
                         temperature=1.0, seed=0):
    """Emit bytes with UTF-8 prefix legality enforced per bit.

    At each bit, the sampler's choice is kept when it preserves a valid
    UTF-8 prefix, otherwise the forced bit is emitted. Encoding law only;
    no language knowledge. Returns (bytes, report) like generate().
    """
    if not isinstance(prefix_bytes, (bytes, bytearray)):
        raise ValueError("generation prefix must be bytes")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if not _is_valid_utf8(bytes(prefix_bytes)):
        raise ValueError("generation prefix must be valid UTF-8")
    rng = random.Random(seed)
    memory.reset_history()
    for row in encode_bytes(bytes(prefix_bytes)):
        for bit in row:
            memory.observe(int(bit), learn=False)
    out_bits = []
    orders = []
    forced = 0
    for _ in range(max_bytes * 8):
        distribution, order = memory.distribution()
        probability = temper(distribution[1], 0.0 if deterministic else temperature)
        if deterministic:
            candidate = 1 if probability >= 0.5 else 0
        else:
            candidate = 1 if rng.random() < probability else 0
        trial = out_bits + [candidate]
        if _bits_form_valid_prefix(trial):
            chosen = candidate
        else:
            chosen = 1 - candidate
            forced += 1
            trial = out_bits + [chosen]
            if not _bits_form_valid_prefix(trial):
                raise ValueError("UTF-8 constrained decoder reached a dead end")
        out_bits.append(chosen)
        orders.append(order)
        memory.observe(chosen, learn=False)
    raw = bytearray()
    for index in range(0, len(out_bits), 8):
        value = 0
        for bit in out_bits[index:index + 8]:
            value = (value << 1) | bit
        raw.append(value)
    raw_bytes = bytes(raw)
    # Streaming semantics: a cut mid-character is truncation, not invalid
    # emission. Validity covers complete bytes; the tail is reported.
    try:
        raw_bytes.decode("utf-8", errors="strict")
        complete, truncated = raw_bytes, b""
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data":
            complete, truncated = raw_bytes[:error.start], raw_bytes[error.start:]
        else:
            complete, truncated = raw_bytes, raw_bytes
    return raw_bytes, {
        "prefix": bytes(prefix_bytes),
        "bytes": len(raw_bytes),
        "valid_utf8": _is_valid_utf8(complete),
        "truncated_tail_bytes": len(truncated),
        "deterministic": deterministic,
        "temperature": temperature,
        "seed": seed,
        "mean_order": sum(orders) / max(1, len(orders)),
        "forced_bits": forced,
    }


def _bits_form_valid_prefix(out_bits):
    """Exact check: some completion of the partial tail keeps UTF-8 prefix
    legality. Brute force over at most 128 completions; dead ends impossible
    because at least the taken path's own completion was legal on arrival."""
    complete = (len(out_bits) // 8) * 8
    head = bytearray()
    for index in range(0, complete, 8):
        value = 0
        for bit in out_bits[index:index + 8]:
            value = (value << 1) | bit
        head.append(value)
    tail = out_bits[complete:]
    if not _is_valid_utf8_prefix(bytes(head)):
        return False
    if not tail:
        return True
    for fill in range(1 << (8 - len(tail))):
        value = 0
        for bit in tail:
            value = (value << 1) | bit
        value = (value << (8 - len(tail))) | fill
        if _is_valid_utf8_prefix(bytes(head) + bytes([value])):
            return True
    return False

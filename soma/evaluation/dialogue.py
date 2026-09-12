"""R3D dialogue mechanics: attributed turns, episodic teaching, corrections.

Turns are literal USER/AGENT markers (content attribution) plus provenance
labels on stored entries. Teaching exposes fact text at background rate and
records an exact episodic trigger->completion rule. Asking emits completions
on exact trigger match and falls back to table sampling otherwise. All
emission is UTF-8 constrained and never trains protected state.
"""

import random

from ..memory.sequence import EpisodicBuffer
from ..transducers.text_bytes import decode_bits, encode_bytes
from .generate import _bits_form_valid_prefix, temper

USER = "USER"
AGENT = "AGENT"


def text_to_bits(text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    rows = encode_bytes(bytes(text))
    return [int(bit) for row in rows for bit in row]


def bits_to_bytes(out_bits):
    raw = bytearray()
    for index in range(0, len(out_bits), 8):
        value = 0
        for bit in out_bits[index:index + 8]:
            value = (value << 1) | bit
        raw.append(value)
    return bytes(raw)


DIALOGUE_ORDER = 256
DIALOGUE_BUDGET = 16384
DIALOGUE_FALLBACK_ORDER = 8


def fresh_dialogue():
    from ..memory import SequenceCircuitMemory
    return SequenceCircuitMemory((0, 1), max_order=DIALOGUE_ORDER,
                                 max_circuits=DIALOGUE_BUDGET)


def teach_fact(dialogue, episodic, fact_text, question, answer, provenance="correction"):
    """Expose fact text to turn-scoped dialogue memory; record the
    question->answer episodic rule. Background E0 knowledge stays frozen."""
    for bit in text_to_bits(fact_text):
        dialogue.observe(bit, learn=True)
    trigger = text_to_bits(question)
    completion = text_to_bits(answer)
    return episodic.add(trigger, completion, provenance)


def _sample(distribution, deterministic, rng):
    probability = temper(distribution[1], 0.0 if deterministic else 1.0)
    if deterministic:
        return 1 if probability >= 0.5 else 0
    return 1 if rng.random() < probability else 0


def _constrain(out_bits, candidate):
    if _bits_form_valid_prefix(out_bits + [candidate]):
        return candidate
    flipped = 1 - candidate
    if not _bits_form_valid_prefix(out_bits + [flipped]):
        raise ValueError("dialogue decoder reached a dead end")
    return flipped


def respond(background, dialogue, episodic, prefix_text, max_bytes=12,
            deterministic=False, seed=0):
    """Generate a turn across three tiers: episodic rule, dialogue table,
    background table. Dialogue history is turn-scoped; background is frozen
    E0 knowledge. Emissions never train protected state."""
    rng = random.Random(seed)
    background.reset_history()
    dialogue.reset_history()
    for bit in text_to_bits(prefix_text):
        background.observe(bit, learn=False)
        dialogue.observe(bit, learn=False)
    out_bits = []
    emitting = []
    for _ in range(max_bytes * 8):
        if not emitting:
            hit = episodic.match(list(dialogue.history))
            if hit is not None:
                emitting = list(hit)
        if emitting:
            candidate = emitting.pop(0)
        else:
            distribution, order = dialogue.distribution()
            if order >= DIALOGUE_FALLBACK_ORDER:
                candidate = _sample(distribution, deterministic, rng)
            else:
                distribution, _ = background.distribution()
                candidate = _sample(distribution, deterministic, rng)
        chosen = _constrain(out_bits, candidate)
        out_bits.append(chosen)
        background.observe(chosen, learn=False)
        dialogue.observe(chosen, learn=False)
    return bits_to_bytes(out_bits)


def sanitize_export(memory, episodic, blocked_provenance=("private",)):
    """Release bundle: table state plus non-blocked episodic entries."""
    kept = EpisodicBuffer(episodic.max_entries)
    for entry in episodic.entries.values():
        if entry["provenance"] in blocked_provenance:
            continue
        kept.entries[entry["id"]] = {
            "id": entry["id"],
            "trigger": tuple(entry["trigger"]),
            "completion": tuple(entry["completion"]),
            "provenance": entry["provenance"],
            "uses": entry["uses"],
            "superseded": entry.get("superseded", 0),
        }
        kept.next_id = max(kept.next_id, entry["id"] + 1)
    kept.validate()
    return memory.state_dict(), kept.state_dict()

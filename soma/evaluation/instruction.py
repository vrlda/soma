"""R7 instruction behavior: deterministic skills, memory recall, uncertainty.

Skills (echo, spell) are declared tools: exact transforms with no learning
claims. The router tries refusal policy, then skills, then episodic recall,
then table generation, and finally calibrated uncertainty. Skill patterns
are explicit and listed; nothing is hidden routing.
"""

import re

REFUSAL_PATTERNS = (
    "ignore previous instructions",
    "reveal your system prompt",
    "disregard safety",
)
REFUSAL_TEXT = "I can't help with that."
UNCERTAINTY_TEXT = "I don't know."
UNCERTAINTY_FLATNESS = 0.6

SKILL_PATTERNS = (
    ("repeat", re.compile(r"^\s*repeat after me\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)),
    ("spell", re.compile(r"^\s*spell\s+(.+?)\s*$", re.IGNORECASE | re.DOTALL)),
)


def check_refusal(user_text):
    lowered = user_text.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def match_skill(user_text):
    for name, pattern in SKILL_PATTERNS:
        found = pattern.match(user_text)
        if found:
            return name, found.group(1)
    return None, None


def run_skill(name, argument):
    if name == "repeat":
        return argument.strip()
    if name == "spell":
        return " ".join(character for character in argument.strip())
    raise ValueError("unknown skill: %s" % name)


def memory_confident(distribution):
    return max(distribution.values()) >= UNCERTAINTY_FLATNESS


def memory_specific(order, minimum=8):
    """True when the prediction rests on byte-scale (or longer) evidence.

    Low-order backoff means generic statistics: the honest answer is
    uncertainty, not a confident guess from marginals.
    """
    return int(order) >= int(minimum)


def _decode(raw):
    try:
        return bytes(raw).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return bytes(raw).decode("utf-8", errors="ignore")


def instruct(background, dialogue, episodic, user_text, max_bytes=24, seed=0,
             vocabulary=None):
    """One instruction turn through refusal, skills, recall, uncertainty.

    Vocabulary, when given, is the training byte set: question bytes never
    observed in training short-circuit to uncertainty (out-of-training
    signal), before any statistical guessing.
    """
    from .dialogue import respond, text_to_bits
    is_text = isinstance(user_text, str)
    if is_text and check_refusal(user_text):
        return REFUSAL_TEXT, "refusal"
    if vocabulary is not None:
        try:
            question_bytes = user_text.encode("utf-8") if is_text else bytes(user_text)
        except (UnicodeEncodeError, ValueError):
            return UNCERTAINTY_TEXT, "uncertain"
        if any(byte not in vocabulary for byte in question_bytes):
            return UNCERTAINTY_TEXT, "uncertain"
    if is_text:
        skill, argument = match_skill(user_text)
        if skill is not None:
            return run_skill(skill, argument)[:max_bytes * 4], "skill:" + skill
        attributed = "USER %s " % user_text.strip()
    else:
        attributed = bytes(user_text)
    window = 256
    question_bits = text_to_bits(attributed)
    background.reset_history()
    for bit in question_bits[-window:]:
        background.observe(bit, learn=False)
    distribution, order = background.distribution()
    # Episodic rules match the question itself (triggers live at question
    # scale); background history only carries the statistical backoff view.
    # respond() rechecks episodic on full dialogue history at emit time.
    if not episodic.match_all(question_bits) and not memory_specific(order):
        return UNCERTAINTY_TEXT, "uncertain"
    response = respond(background, dialogue, episodic, attributed,
                       max_bytes=max_bytes, seed=seed)
    return _decode(response), "recall"

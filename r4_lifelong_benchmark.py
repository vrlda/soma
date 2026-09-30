#!/usr/bin/env python3
"""Frozen R4 lifelong gates: growth, conflict, correction, delay, retention.

One system lives through all phases: E0 background, four novel facts,
a cross-source conflict, a correction, interference, then return.
G1 uptake 4/4. G2 latest-wins conflict with logged record. G3 correction.
G4 retention >= 3/4 after interference. G5 E0 held-out within +0.05.
G6 episodic capacity bounded.
"""

import argparse
import json

from soma.evaluation.dialogue import fresh_dialogue, respond, teach_fact
from soma.evaluation.english import (
    bit_stream, bits_per_bit, load_corpus, partition_documents, split_chapters,
)
from soma.memory import CircuitMixingMemory, EpisodicBuffer, SequenceCircuitMemory

FACTS = (
    ("USER alpha means first AGENT noted ", "USER alpha means ", "first"),
    ("USER beta means second AGENT noted ", "USER beta means ", "second"),
    ("USER gamma means third AGENT noted ", "USER gamma means ", "third"),
    ("USER delta means fourth AGENT noted ", "USER delta means ", "fourth"),
)
FILLERS = (
    "USER the sky is blue AGENT noted ",
    "USER grass is green AGENT noted ",
    "USER birds can fly AGENT noted ",
    "USER fish can swim AGENT noted ",
)


def starts_with(response, word):
    try:
        text = bytes(response).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return text.startswith(word)


def heldout_bits(memory, validation):
    bits, _ = bit_stream(validation)
    memory.reset_history()
    predictions, targets = [], []
    for step in range(max(0, len(bits) - 1)):
        memory.observe(bits[step], learn=False)
        distribution, _ = memory.distribution()
        predictions.append(distribution[1])
        targets.append(bits[step + 1])
    return bits_per_bit(predictions, targets)


def ask(background, dialogue, episodic, question):
    return respond(background, dialogue, episodic, question, 8, seed=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    parser.add_argument("--memory", choices=("suffix", "mixing"), default="suffix",
                        help="background memory; 'mixing' writes reports/r4-lifelong-mixing.json")
    args = parser.parse_args()

    data = load_corpus(args.corpus)
    acquisition, validation, _, manifest = partition_documents(split_chapters(data), 0)
    if args.memory == "mixing":
        background = CircuitMixingMemory(max_circuits=1 << 20)
    else:
        background = SequenceCircuitMemory((0, 1), max_order=16, max_circuits=131072,
                                           min_support=2, prior=0.5)
    bits, _ = bit_stream(acquisition)
    for symbol in bits[:-1]:
        background.observe(symbol)
    base_bits = heldout_bits(background, validation)
    dialogue = fresh_dialogue()
    episodic = EpisodicBuffer(max_entries=64)

    uptake = 0
    for fact_text, question, answer in FACTS:
        teach_fact(dialogue, episodic, fact_text, question, answer, provenance="userA")
        if starts_with(ask(background, dialogue, episodic, question), answer):
            uptake += 1

    teach_fact(dialogue, episodic, "USER alpha means zeroth AGENT noted ",
               "USER alpha means ", "zeroth", provenance="userB")
    conflict = starts_with(ask(background, dialogue, episodic, "USER alpha means "), "zeroth")
    conflict_logged = len(episodic.conflicts) > 0

    teach_fact(dialogue, episodic, "USER no, alpha means first AGENT fixed ",
               "USER alpha means ", "first", provenance="correction")
    corrected = starts_with(ask(background, dialogue, episodic, "USER alpha means "), "first")

    for filler in FILLERS:
        teach_fact(dialogue, episodic, filler, "USER filler ", "ok")

    expected = {"USER alpha means ": "first", "USER beta means ": "second",
                "USER gamma means ": "third", "USER delta means ": "fourth"}
    retained = sum(
        1 for question, answer in expected.items()
        if starts_with(ask(background, dialogue, episodic, question), answer))
    final_bits = heldout_bits(background, validation)
    result = {
        "protocol": "r4-lifelong-v1" if args.memory == "suffix" else "r4-lifelong-v1-mixing",
        "corpus": args.corpus,
        "manifest": manifest,
        "uptake": uptake,
        "conflict_latest_wins": bool(conflict),
        "conflict_logged": bool(conflict_logged),
        "corrected": bool(corrected),
        "retained": retained,
        "base_bits_per_bit": base_bits,
        "final_bits_per_bit": final_bits,
        "episodic_entries": len(episodic.entries),
        "gates": {
            "uptake": uptake == 4,
            "conflict": bool(conflict) and bool(conflict_logged),
            "correction": bool(corrected),
            "retention": retained >= 3,
            "no_forgetting": final_bits <= base_bits + 0.05,
            "bounded": len(episodic.entries) <= 64,
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    out = "reports/r4-lifelong.json" if args.memory == "suffix" else "reports/r4-lifelong-mixing.json"
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

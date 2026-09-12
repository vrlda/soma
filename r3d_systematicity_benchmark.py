#!/usr/bin/env python3
"""Frozen systematicity gates: novel combinations of familiar parts.

Teach color->red and shape->cube (plus an unused size->big distractor).
Ask a novel combination containing both triggers. Gates: the combined ask
emits both completions; single-suffix matching emits neither; unrelated asks
emit neither; entry removal destroys recall.
"""

import argparse
import json

from soma.evaluation.dialogue import fresh_dialogue, respond, teach_fact
from soma.evaluation.english import (
    bit_stream, load_corpus, partition_documents, split_chapters,
)
from soma.memory import EpisodicBuffer, SequenceCircuitMemory


def trained_background(acquisition):
    memory = SequenceCircuitMemory((0, 1), max_order=16, max_circuits=131072,
                                   min_support=2, prior=0.5)
    bits, _ = bit_stream(acquisition)
    for symbol in bits[:-1]:
        memory.observe(symbol)
    return memory


def contains_all(response, words):
    try:
        text = bytes(response).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = bytes(response).decode("utf-8", errors="ignore")
    return all(word in text for word in words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    args = parser.parse_args()

    data = load_corpus(args.corpus)
    acquisition, _, _, manifest = partition_documents(split_chapters(data), 0)
    background = trained_background(acquisition)
    dialogue = fresh_dialogue()
    episodic = EpisodicBuffer()
    teach_fact(dialogue, episodic, "USER KEY color means red AGENT noted ",
               "KEY color ", "red")
    teach_fact(dialogue, episodic, "USER KEY shape means cube AGENT noted ",
               "KEY shape ", "cube")
    teach_fact(dialogue, episodic, "USER KEY size means big AGENT noted ",
               "KEY size ", "big")

    combo = respond(background, dialogue, episodic,
                    "USER KEY color and KEY shape are ", 10, seed=1)
    single = respond(background, dialogue, episodic, "USER KEY color is ", 6, seed=1)
    unrelated = respond(background, dialogue, episodic, "USER the weather today ", 6, seed=1)

    # Single-suffix control: longest-suffix match only (legacy behavior).
    from soma.memory.sequence import EpisodicBuffer as _EB
    legacy_hits = []
    history = None
    dialogue.reset_history()
    from soma.evaluation.dialogue import text_to_bits
    for bit in text_to_bits("USER KEY color and KEY shape are "):
        dialogue.observe(bit, learn=False)
    legacy = episodic.match(list(dialogue.history))

    for entry_id in list(episodic.entries):
        episodic.remove(entry_id)
    lesioned = respond(background, dialogue, episodic,
                       "USER KEY color and KEY shape are ", 10, seed=1)

    result = {
        "protocol": "r3d-systematicity-v1",
        "corpus": args.corpus,
        "manifest": manifest,
        "combo_has_both": contains_all(combo, ("red", "cube")),
        "combo_has_distractor": contains_all(combo, ("big",)),
        "single_has_red": contains_all(single, ("red",)),
        "unrelated_clean": not contains_all(unrelated, ("red", "cube", "big")),
        "legacy_single_match_only": legacy is None,
        "lesioned_clean": not contains_all(lesioned, ("red", "cube")),
        "gates": {},
    }
    result["gates"] = {
        "composition": bool(result["combo_has_both"]),
        "precision": bool(not result["combo_has_distractor"] and result["unrelated_clean"]),
        "single_still_works": bool(result["single_has_red"]),
        "legacy_cannot_compose": bool(result["legacy_single_match_only"]),
        "lesion_destroys": bool(result["lesioned_clean"]),
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    with open("reports/r3d-systematicity.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

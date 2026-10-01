#!/usr/bin/env python3
"""Frozen R6 factual/provenance gates: taught facts answer with sources.

Teach 8 facts from mixed provenances, ask each back with provenance
attribution, correct one fact, verify the update, then lesion all and
verify recall collapses. This is an episodic exact-match factual-lookup and
provenance-plumbing gate, rather than a semantic QA benchmark. Gates: 8/8
lookup accuracy, 8/8 provenance, correction takes effect, lesion fails.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json

from soma.evaluation.dialogue import fresh_dialogue, respond, teach_fact
from soma.evaluation.english import (
    bit_stream, load_corpus, partition_documents, split_chapters,
)
from soma.memory import EpisodicBuffer, SequenceCircuitMemory

FACTS = (
    ("USER alpha means first AGENT noted ", "USER alpha means ", "first", "docA"),
    ("USER beta means second AGENT noted ", "USER beta means ", "second", "docA"),
    ("USER gamma means third AGENT noted ", "USER gamma means ", "third", "docB"),
    ("USER delta means fourth AGENT noted ", "USER delta means ", "fourth", "docB"),
    ("USER epsilon means fifth AGENT noted ", "USER epsilon means ", "fifth", "user"),
    ("USER zeta means sixth AGENT noted ", "USER zeta means ", "sixth", "user"),
    ("USER eta means seventh AGENT noted ", "USER eta means ", "seventh", "correction"),
    ("USER theta means eighth AGENT noted ", "USER theta means ", "eighth", "correction"),
)


def starts_with(response, word):
    """Check the answer prefix without requiring a valid trailing UTF-8 turn.

    ``respond`` emits a fixed byte budget, so the bytes after a correct answer
    can end in an incomplete UTF-8 sequence.  Prefix correctness is a byte
    property; decoding the entire response makes that unrelated tail a false
    negative.
    """
    response = response.encode("utf-8") if isinstance(response, str) else bytes(response)
    word = word.encode("utf-8") if isinstance(word, str) else bytes(word)
    return response.startswith(word)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    args = parser.parse_args()

    data = load_corpus(args.corpus)
    acquisition, _, _, manifest = partition_documents(split_chapters(data), 0)
    background = SequenceCircuitMemory((0, 1), max_order=16, max_circuits=131072,
                                       min_support=2, prior=0.5)
    bits, _ = bit_stream(acquisition)
    for symbol in bits[:-1]:
        background.observe(symbol)
    dialogue = fresh_dialogue()
    episodic = EpisodicBuffer()
    for fact_text, question, answer, provenance in FACTS:
        teach_fact(dialogue, episodic, fact_text, question, answer, provenance=provenance)

    accuracy, provenance = 0, 0
    for _, question, answer, source in FACTS:
        response, fired = respond(background, dialogue, episodic, question, 8,
                                  seed=2, trace=True)
        if starts_with(response, answer):
            accuracy += 1
        if any(record["provenance"] == source for record in fired):
            provenance += 1

    teach_fact(dialogue, episodic, "USER no, alpha means zeroth AGENT fixed ",
               "USER alpha means ", "zeroth", provenance="correction")
    corrected, fired = respond(background, dialogue, episodic, "USER alpha means ", 8,
                               seed=2, trace=True)
    correction_ok = starts_with(corrected, "zeroth")
    correction_source = any(record["provenance"] == "correction" for record in fired)

    for entry_id in list(episodic.entries):
        episodic.remove(entry_id)
    lesioned = sum(
        1 for _, question, answer, _ in FACTS
        if starts_with(respond(background, dialogue, episodic, question, 8, seed=2), answer))

    result = {
        "protocol": "r6-factual-v1",
        "evaluation_scope": "episodic exact-match factual lookup and provenance plumbing",
        "corpus": args.corpus,
        "manifest": manifest,
        "accuracy": "%d/%d" % (accuracy, len(FACTS)),
        "provenance": "%d/%d" % (provenance, len(FACTS)),
        "lesioned_recall": "%d/%d" % (lesioned, len(FACTS)),
        "gates": {
            "accuracy": accuracy == len(FACTS),
            "provenance": provenance == len(FACTS),
            "correction": bool(correction_ok and correction_source),
            "lesion_fails": lesioned == 0,
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    with open("research/reports/r6-factual.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

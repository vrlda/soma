#!/usr/bin/env python3
"""Frozen R3D dialogue gates: uptake, correction, interference, restart.

Background is frozen E0 knowledge; dialogue learns in a turn-scoped
order-256 memory plus exact episodic rules. G1 uptake, G2 supersede,
G3 interference, G4 restart, G5 controls fail, G6 release + new learning.
"""

import argparse
import json
import os
import tempfile

from soma.evaluation.dialogue import fresh_dialogue, respond, sanitize_export, teach_fact
from soma.evaluation.english import (
    bit_stream, load_corpus, partition_documents, split_chapters,
)
from soma.evaluation.lineage import clone_brain, verify_clone
from soma.evaluation.backgrounds import (
    BackgroundFactory, add_memory_argument, report_path, respond_settings,
)
from soma.memory import EpisodicBuffer, SequenceCircuitMemory



def starts_with(response, word):
    data = bytes(response)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data":
            return False
        text = data[:error.start].decode("utf-8", errors="strict")
    return text.startswith(word)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    add_memory_argument(parser)
    args = parser.parse_args()
    settings = respond_settings(args.memory)

    data = load_corpus(args.corpus)
    acquisition, _, _, manifest = partition_documents(split_chapters(data), 0)
    factory = BackgroundFactory(args.memory, acquisition)
    background = factory()
    dialogue = fresh_dialogue()
    episodic = EpisodicBuffer()

    teach_fact(dialogue, episodic, "USER the code is kite AGENT noted ",
               "USER the code is ", "kite")
    uptake = starts_with(respond(background, dialogue, episodic, "USER the code is ", 8, seed=2, **settings), "kite")

    teach_fact(dialogue, episodic, "USER the code is ball AGENT noted ",
               "USER the code is ", "ball")
    teach_fact(dialogue, episodic, "USER no, the code is kite AGENT fixed ",
               "USER the code is ", "kite", provenance="correction")
    corrected = starts_with(
        respond(background, dialogue, episodic, "USER the code is ", 8, seed=2, **settings), "kite")

    for filler in ("USER the sky is blue AGENT noted ",
                   "USER grass is green AGENT noted ",
                   "USER birds can fly AGENT noted "):
        teach_fact(dialogue, episodic, filler, "USER filler ", "ok")
    retained = starts_with(
        respond(background, dialogue, episodic, "USER the code is ", 8, seed=2, **settings), "kite")

    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "branch.json")
        with open(source, "w") as handle:
            json.dump(dialogue.state_dict(), handle, sort_keys=True)
        with open(source + ".episodic.json", "w") as handle:
            json.dump(episodic.state_dict(), handle, sort_keys=True)
        clone = os.path.join(directory, "restart.json")
        clone_brain(source, clone, "r3d-restart")
        verify_clone(source, clone)
        reloaded_dialogue = SequenceCircuitMemory.from_state_dict(json.load(open(clone)))
        reloaded_episodic = EpisodicBuffer.from_state_dict(json.load(open(source + ".episodic.json")))
    restarted = starts_with(
        respond(background, reloaded_dialogue, reloaded_episodic, "USER the code is ", 8, seed=2, **settings),
        "kite")

    bare_dialogue = fresh_dialogue()
    bare_episodic = EpisodicBuffer()
    table_alone = starts_with(
        respond(background, bare_dialogue, bare_episodic, "USER the code is ", 8, seed=2, **settings), "kite")

    memory2 = fresh_dialogue()
    episodic2 = EpisodicBuffer()
    entry = teach_fact(memory2, episodic2, "USER the code is kite AGENT noted ",
                       "USER the code is ", "kite")
    episodic2.remove(entry)
    lesioned = starts_with(
        respond(background, memory2, episodic2, "USER the code is ", 8, seed=2, **settings), "kite")

    table_state, kept_state = sanitize_export(reloaded_dialogue, reloaded_episodic)
    released_dialogue = SequenceCircuitMemory.from_state_dict(table_state)
    released_episodic = EpisodicBuffer.from_state_dict(kept_state)
    teach_fact(released_dialogue, released_episodic, "USER the key is opal AGENT noted ",
               "USER the key is ", "opal")
    continued = starts_with(
        respond(background, released_dialogue, released_episodic, "USER the key is ", 8, seed=2, **settings),
        "opal")

    result = {
        "protocol": "r3d-dialogue-v1" if args.memory == "suffix" else "r3d-dialogue-v1-mixing",
        "corpus": args.corpus,
        "manifest": manifest,
        "uptake": uptake,
        "corrected": corrected,
        "retained_after_interference": retained,
        "retained_after_restart": restarted,
        "table_alone_recalls": table_alone,
        "lesioned_recalls": lesioned,
        "continued_after_download": continued,
        "episodic_entries": len(released_episodic.entries),
        "gates": {
            "uptake": bool(uptake),
            "correction": bool(corrected),
            "interference": bool(retained),
            "restart": bool(restarted),
            "controls_fail": not table_alone and not lesioned,
            "release": bool(continued),
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    factory.close()
    with open(report_path("reports/r3d-dialogue.json", args.memory), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

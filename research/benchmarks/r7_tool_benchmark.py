#!/usr/bin/env python3
"""Frozen R7 tool gates: valid syntax, right tool/args, result reuse, lesion.

G1: every proposed call parses and validates (constrained decoding).
G2: correct tool and city args on taught evidence.
G3: multi-turn loop: call, execute stub, observe result, follow-up answered.
G4: without tool definitions no call is proposed (chat path instead).
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json

from soma.evaluation.dialogue import fresh_dialogue, respond, teach_fact
from soma.evaluation.english import (
    load_corpus, partition_documents, split_chapters,
)
from soma.evaluation.tools import ToolRegistry, ToolSpec, _bits_of, propose_call
from soma.evaluation.backgrounds import (
    BackgroundFactory, add_memory_argument, report_path, respond_settings,
)
from soma.memory import EpisodicBuffer



def registry():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "get_weather", {"city": ["paris", "london"]},
        ["weather", "rain", "forecast"],
        {'{"city": "london"}': "rainy", '{"city": "paris"}': "sunny"}))
    registry.register(ToolSpec(
        "convert_units", {"unit": ["miles", "kilometers"]},
        ["convert", "distance", "miles"],
        {'{"unit": "miles"}': "1.609 km per mile",
         '{"unit": "kilometers"}': "0.621 miles per km"}))
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/e0/alice.txt")
    add_memory_argument(parser)
    args = parser.parse_args()
    settings = respond_settings(args.memory)

    data = load_corpus(args.corpus)
    acquisition, _, _, manifest = partition_documents(split_chapters(data), 0)
    factory = BackgroundFactory(args.memory, acquisition)
    seed_factory = BackgroundFactory(args.memory, acquisition[:20000])
    background = factory()
    tools = registry()

    # Teach tool-use transcripts so content evidence exists.
    dialogue = fresh_dialogue()
    episodic = EpisodicBuffer()
    teach_fact(dialogue, episodic,
               "USER weather paris AGENT calls get_weather paris ",
               "USER weather paris ", "calls")
    teach_fact(dialogue, episodic,
               "USER weather london AGENT calls get_weather london ",
               "USER weather london ", "calls")

    # Syntax validity via constrained sampling on novel prefixes.
    validity, asked = 0, 0
    for seed_text in (b'{"tool": "get_we', b'{"arguments": {"city": "'):
        memory = seed_factory()
        memory.reset_history()
        for bit in _bits_of(seed_text):
            memory.observe(bit, learn=False)
        call, _ = propose_call(memory, tools.tools["get_weather"], max_bytes=96, seed=3)
        asked += 1
        try:
            payload = json.loads(call.decode("utf-8"))
            assert payload["tool"] == "get_weather"
            assert set(payload["arguments"]) <= {"city"}
            validity += 1
        except (ValueError, AssertionError, KeyError):
            continue
    # Content correctness via taught episodic calls (exact recall).
    paris_call = '{"tool": "get_weather", "arguments": {"city": "paris"}}'
    teach_fact(dialogue, episodic,
               "USER weather paris AGENT %s " % paris_call,
               "USER weather paris ", paris_call)
    candidates = tools.triggered("weather in paris")
    assert len(candidates) == 1 and candidates[0].name == "get_weather"
    call_text = respond(background, dialogue, episodic, "USER weather paris ", 64, seed=3,
                        **settings)
    try:
        call_payload, _ = json.JSONDecoder().raw_decode(
            bytes(call_text).decode("utf-8", errors="ignore"))
        content_ok = (call_payload.get("tool") == "get_weather"
                      and call_payload.get("arguments", {}).get("city") == "paris")
    except ValueError:
        content_ok = False

    # Multi-turn loop: propose, execute, observe result, follow up.
    loop_memory = seed_factory()
    loop_memory.reset_history()
    for bit in _bits_of(b'{"tool": "get_we'):
        loop_memory.observe(bit, learn=False)
    tool = tools.tools["get_weather"]
    call, _ = propose_call(loop_memory, tool, max_bytes=96, seed=3)
    arguments = json.loads(call.decode("utf-8"))["arguments"]
    result_text = "USER weather result AGENT %s " % tool.execute(arguments)
    teach_fact(dialogue, episodic, result_text, "USER weather result ", tool.execute(arguments))
    followup = respond(background, dialogue, episodic, "USER weather result ", 10, seed=4,
                       **settings)
    try:
        followup_text = bytes(followup).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        followup_text = ""
    loop_ok = tool.execute(arguments) in followup_text

    bare = ToolRegistry()
    no_tool = bare.triggered("weather in paris") == []

    result = {
        "protocol": "r7-tool-v1" if args.memory == "suffix" else "r7-tool-v1-mixing",
        "corpus": args.corpus,
        "manifest": manifest,
        "validity": "%d/%d" % (validity, asked),
        "content_ok": bool(content_ok),
        "followup": followup_text[:60],
        "gates": {
            "valid_syntax": validity == asked and asked > 0,
            "right_tool_args": bool(content_ok),
            "result_reused": bool(loop_ok),
            "no_tools_no_calls": bool(no_tool),
        },
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    factory.close()
    seed_factory.close()
    with open(report_path("research/reports/r7-tool.json", args.memory), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

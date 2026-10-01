#!/usr/bin/env python3
"""ADR 0010 stage 3: open-book question answering on SQuAD v1.1.

Format (one document per question; history resets between documents):

    <passage>\nQ: <question>\nA: <answer>\n

The model trains on SQuAD train documents (optionally plus Wikipedia). For a
held-out question it reads ``<passage>\nQ: <question>\nA: `` and answers
with the passage span (1–6 words) to which it assigns the highest
probability as ``span + newline`` (extractive; ``--generate`` instead
continues greedily until a newline). Scoring uses the official SQuAD normalization
(lowercase; strip punctuation, articles, extra space) for exact match and
token F1, against every reference answer.

Baseline (non-learning): the passage sentence sharing the most question
words, answered with up to three words after the last question word in it.

  python3 research/benchmarks/l3_squad_benchmark.py --split validation --limit 1000
"""

import argparse
import collections
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BINARY = os.path.join(ROOT, "engine", "soma-engine", "target", "release", "soma-longmix")
DATA = os.path.join(ROOT, "data", "modern")
STOP = set("a an the of in on at to for from by with and or is are was were be been what which "
           "who whom whose when where why how did do does that this these those it its as".split())


def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(prediction, truth):
    p, t = normalize(prediction).split(), normalize(truth).split()
    common = collections.Counter(p) & collections.Counter(t)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def score(predictions, examples):
    exact = sum(max(normalize(p) == normalize(a) for a in e["answers"])
                for p, e in zip(predictions, examples))
    token = sum(max(f1(p, a) for a in e["answers"]) for p, e in zip(predictions, examples))
    return {"exact_match": exact / len(examples), "f1": token / len(examples),
            "count": len(examples)}


def baseline_answer(example):
    question = {w for w in re.findall(r"\w+", example["question"].lower()) if w not in STOP}
    best, best_overlap = "", -1
    for sentence in re.split(r"(?<=[.!?])\s+", example["context"]):
        overlap = len(question & set(re.findall(r"\w+", sentence.lower())))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    tokens = re.findall(r"\S+", best)
    last = -1
    for index, token in enumerate(tokens):
        if re.sub(r"\W", "", token.lower()) in question:
            last = index
    answer = [t for t in tokens[last + 1:] if re.sub(r"\W", "", t.lower()) not in question][:3]
    return " ".join(answer).strip(".,;:")


def load(name, limit=None):
    rows = []
    with open(os.path.join(DATA, name), encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def prompt(example):
    return "%s\nQ: %s\nA: " % (example["context"].strip(), example["question"].strip())


def write_docs(path, texts):
    with open(path, "wb") as handle:
        for text in texts:
            handle.write(text.encode("utf-8") + b"\x1e")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--wiki-mb", type=float, default=0.0)
    parser.add_argument("--config", default="{}")
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scoring", default="pmi", choices=("pmi", "mean", "sum"))
    parser.add_argument("--generate", action="store_true",
                        help="free greedy generation instead of extractive span scoring")
    args = parser.parse_args()
    if args.split == "test" and not args.out:
        raise SystemExit("the sealed test split is scored only into a recorded --out report")
    held = load("squad-%s.jsonl" % args.split, args.limit)
    train = load("squad-train.jsonl", args.train_limit)
    config = dict({"max_circuits": 1 << 22}, **json.loads(args.config))
    started = time.time()
    with tempfile.TemporaryDirectory() as directory:
        files = []
        if args.wiki_mb > 0:
            wiki, total = [], 0
            with open(os.path.join(DATA, "wiki-train.jsonl"), encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    wiki.append(row["title"] + "\n" + row["text"])
                    total += len(wiki[-1])
                    if total >= args.wiki_mb * 1e6:
                        break
            write_docs(os.path.join(directory, "wiki.docs"), wiki)
            files.append(os.path.join(directory, "wiki.docs"))
        write_docs(os.path.join(directory, "squad.docs"),
                   [prompt(e) + e["answers"][0] + "\n" for e in train])
        files.append(os.path.join(directory, "squad.docs"))
        write_docs(os.path.join(directory, "prompts.docs"), [prompt(e) for e in held])
        answers_path = os.path.join(directory, "answers.jsonl")
        job = {"config": config, "train": files,
               "qa": {"prompts": os.path.join(directory, "prompts.docs"),
                      "out": answers_path, "max_bytes": 48,
                      "extractive": not args.generate, "max_words": 6,
                      "scoring": args.scoring}}
        completed = subprocess.run([args.binary], input=json.dumps(job), check=True,
                                   capture_output=True, text=True)
        engine = json.loads(completed.stdout)
        with open(answers_path, encoding="utf-8") as handle:
            predictions = [json.loads(line) for line in handle]
    baseline = [baseline_answer(e) for e in held]
    report = {
        "split": args.split, "config": config, "scoring": args.scoring, "train_questions": len(train),
        "wiki_mb": args.wiki_mb,
        "model": score(predictions, held), "baseline_lexical": score(baseline, held),
        "engine": {k: engine[k] for k in ("trained_bytes", "train_seconds", "circuits",
                                          "peak_rss_mb")},
        "samples": [{"question": e["question"], "answers": e["answers"], "model": p,
                     "baseline": b} for e, p, b in list(zip(held, predictions, baseline))[:25]],
        "seconds": time.time() - started,
    }
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({k: report[k] for k in ("model", "baseline_lexical", "engine", "seconds")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

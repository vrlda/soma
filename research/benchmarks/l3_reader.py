#!/usr/bin/env python3
"""ADR 0010 stage 3 (prototype): a span reader with local learning.

For a passage and question, every span of 1..5 tokens inside a sentence is
a candidate. Its score is a sum of weights of hashed evidence features:

- answer type: the question's wh-form crossed with the span's shape
  (number, year, month, capitalized, length) and with the question's head
  word ("what color" -> color);
- alignment: question words around the span (left and right windows,
  rarity-weighted), distance to the nearest matched question word,
  sentence overlap rank;
- local syntax proxies: the token before and after the span;
- penalties: span words that repeat the question.

Learning is online softmax regression with a local delta rule: after each
question, every active feature weight moves by learning_rate x (target -
probability) of its own candidate. No backpropagation through a network.

  python3 research/benchmarks/l3_reader.py --train-limit 20000 --limit 1000
"""

import argparse
import collections
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l3_squad_benchmark import DATA, load, normalize, score  # noqa: E402

STOP = set("a an the of in on at to for from by with and or is are was were be been being what "
           "which who whom whose when where why how did do does that this these those it its as "
           "has have had not but if than then there their they he she his her them also into "
           "about after before during can could would should will may might one many much".split())
MONTHS = set("january february march april may june july august september october november "
             "december".split())
TOKEN = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]")


def tokens(text):
    return TOKEN.findall(text)


def question_type(question):
    words = [w.lower() for w in tokens(question)]
    for index, word in enumerate(words):
        if word in ("how",) and index + 1 < len(words):
            return "how_" + words[index + 1]
        if word in ("what", "which") and index + 1 < len(words):
            return word
        if word in ("who", "whom", "whose", "when", "where", "why"):
            return word
    return "other"


def question_head(question):
    words = [w.lower() for w in tokens(question)]
    for index, word in enumerate(words):
        if word in ("what", "which") and index + 1 < len(words):
            follow = words[index + 1]
            if follow not in STOP:
                return follow
            if index + 2 < len(words) and words[index + 2] not in STOP:
                return words[index + 2]
    return "-"


def shape(span_tokens):
    text = " ".join(span_tokens)
    shapes = []
    if any(ch.isdigit() for ch in text):
        shapes.append("num")
    if any(re.fullmatch(r"1[0-9]{3}|20[0-9]{2}", t) for t in span_tokens):
        shapes.append("year")
    if any(t.lower() in MONTHS for t in span_tokens):
        shapes.append("month")
    caps = sum(t[:1].isupper() for t in span_tokens)
    shapes.append("caps%d" % min(caps, 3))
    shapes.append("allcaps" if caps == len(span_tokens) else "mixed")
    shapes.append("len%d" % min(len(span_tokens), 5))
    if "%" in text or "percent" in text.lower():
        shapes.append("pct")
    if "$" in text or "dollar" in text.lower():
        shapes.append("money")
    return shapes


class Reader(object):
    def __init__(self, learning_rate=0.1, max_span=5):
        self.weights = collections.defaultdict(float)
        self.learning_rate = learning_rate
        self.max_span = max_span
        self.document_frequency = collections.Counter()
        self.documents = 0

    def idf(self, word):
        return math.log((self.documents + 2.0) / (self.document_frequency[word] + 1.0))

    def candidates(self, context, question):
        q_words = [w.lower() for w in tokens(question)]
        q_content = {w for w in q_words if w not in STOP and w.isalnum()}
        qtype, head = question_type(question), question_head(question)
        sentences = re.split(r"(?<=[.!?])\s+", context)
        scored_sentences = []
        for sentence in sentences:
            toks = tokens(sentence)
            lower = [t.lower() for t in toks]
            overlap = sum(self.idf(w) for w in set(lower) & q_content)
            scored_sentences.append((overlap, toks, lower))
        ranking = sorted(range(len(scored_sentences)), key=lambda i: -scored_sentences[i][0])
        rank_of = {index: rank for rank, index in enumerate(ranking)}
        out = []
        for s_index, (overlap, toks, lower) in enumerate(scored_sentences):
            rank = rank_of[s_index]
            if rank > 3:
                continue
            matched = [i for i, w in enumerate(lower) if w in q_content]
            for start in range(len(toks)):
                if not toks[start][0].isalnum():
                    continue
                for end in range(start + 1, min(len(toks), start + self.max_span) + 1):
                    span = toks[start:end]
                    if not span[-1][0].isalnum():
                        continue
                    span_lower = lower[start:end]
                    if all(w in STOP or w in q_content for w in span_lower):
                        continue
                    features = ["bias", "rank%d" % rank, "olap%d" % min(int(overlap), 8)]
                    for sh in shape(span):
                        features.append("t:%s|%s" % (qtype, sh))
                        features.append("h:%s|%s" % (head, sh))
                    features.append("hl:%s|%s" % (head, span_lower[-1]))
                    features.append("tl:%s|%s" % (qtype, span_lower[-1]))
                    left = lower[max(0, start - 4):start]
                    right = lower[end:end + 4]
                    l_score = sum(self.idf(w) for w in set(left) & q_content)
                    r_score = sum(self.idf(w) for w in set(right) & q_content)
                    features.append("lw%d" % min(int(l_score), 6))
                    features.append("rw%d" % min(int(r_score), 6))
                    if matched:
                        distance = min(min(abs(start - m), abs(end - 1 - m)) for m in matched)
                        features.append("dist%d" % min(distance, 10))
                        features.append("tdist:%s|%d" % (qtype, min(distance, 6)))
                    else:
                        features.append("nomatch")
                    repeats = sum(w in q_content for w in span_lower)
                    features.append("rep%d" % min(repeats, 2))
                    before = lower[start - 1] if start > 0 else "<s>"
                    after = lower[end] if end < len(lower) else "</s>"
                    features.append("b:%s|%s" % (qtype, before))
                    features.append("a:%s|%s" % (qtype, after))
                    features.append("b:%s" % before)
                    features.append("a:%s" % after)
                    out.append((" ".join(span), features))
        return out

    def predict(self, context, question):
        candidates = self.candidates(context, question)
        if not candidates:
            return "", candidates, []
        scores = [sum(self.weights[f] for f in features) for _, features in candidates]
        best = max(range(len(candidates)), key=lambda i: scores[i])
        return candidates[best][0], candidates, scores

    def learn(self, context, question, answers):
        self.documents += 1
        for word in set(w.lower() for w in tokens(context)):
            self.document_frequency[word] += 1
        _, candidates, scores = self.predict(context, question)
        gold = {normalize(a) for a in answers}
        targets = [normalize(text) in gold for text, _ in candidates]
        if not any(targets):
            return False
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        positives = sum(targets)
        for (text, features), e, target in zip(candidates, exps, targets):
            probability = e / total
            error = (1.0 / positives if target else 0.0) - probability
            if abs(error) < 1e-4:
                continue
            for feature in features:
                self.weights[feature] += self.learning_rate * error
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train = load("squad-train.jsonl", args.train_limit)
    held = load("squad-%s.jsonl" % args.split, args.limit)
    reader = Reader()
    started = time.time()
    used = 0
    for _ in range(args.epochs):
        for example in train:
            used += reader.learn(example["context"], example["question"], example["answers"])
    predictions = [reader.predict(e["context"], e["question"])[0] for e in held]
    print(json.dumps({"reader": score(predictions, held), "trained_on": used,
                      "features": len(reader.weights),
                      "seconds": round(time.time() - started)}, indent=2))
    for e, p in list(zip(held, predictions))[:12]:
        print(repr(e["question"][:60]), "|", e["answers"][:1], "| reader:", repr(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())

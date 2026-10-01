#!/usr/bin/env python3
"""Build the modern English corpus (ADR 0010, stage 1). Standard library only.

Sources (downloaded once into data/modern/raw/, which is git-ignored):

  simplewiki  Simple English Wikipedia dump (CC BY-SA 4.0)
  squad       SQuAD v1.1 train + dev (CC BY-SA 4.0)
  oasst       OpenAssistant oasst1 ready trees (Apache 2.0)

Outputs in data/modern/ (git-ignored), one JSON object per line:

  wiki-{train,validation,test}.jsonl   {"title", "text"}
  squad-{train,validation,test}.jsonl  {"title", "context", "question", "answers"}
  dialog-{train,validation,test}.jsonl {"turns": [{"role", "text"}, ...]}

Split rule, fixed before any model sees the data: bucket =
int(sha256(key)[:8], 16) % 100; 0 → test (sealed), 1–2 → validation,
otherwise train. The key is the article title for wiki and the tree id for
dialogue. SQuAD keeps its own split: train articles train, and dev articles
are halved by title hash into validation and the sealed test. data/modern/MANIFEST.json (committed)
records source hashes, counts, and output hashes.

  python3 scripts/build_modern_corpus.py [--only simplewiki,squad,oasst]
"""

import argparse
import bz2
import gzip
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ElementTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "modern")
RAW = os.path.join(OUT, "raw")
SOURCES = {
    "simplewiki": [("simplewiki-20260901-pages-articles.xml.bz2",
                    "https://dumps.wikimedia.org/simplewiki/20260901/"
                    "simplewiki-20260901-pages-articles.xml.bz2")],
    "squad": [("squad-train-v1.1.json",
               "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"),
              ("squad-dev-v1.1.json",
               "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")],
    "oasst": [("oasst1-trees.jsonl.gz",
               "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
               "2023-04-12_oasst_ready.trees.jsonl.gz")],
}
DROP_SECTIONS = {"references", "other websites", "related pages", "notes", "sources",
                 "further reading", "external links", "gallery", "bibliography"}


def bucket(key):
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if value == 0 else ("validation" if value <= 2 else "train")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, url, attempts=4):
    """Download once; verify the size against Content-Length; retry on short reads."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    if os.path.exists(path):
        return path
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "soma-corpus/1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, \
                    open(path + ".part", "wb") as handle:
                expected = response.headers.get("Content-Length")
                size = 0
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
                    size += len(block)
            if expected is not None and size != int(expected):
                raise OSError("short download: %d of %s bytes" % (size, expected))
            os.replace(path + ".part", path)
            return path
        except OSError as error:
            print("download attempt %d failed: %s" % (attempt + 1, error), file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise SystemExit("could not download %s" % url)


# --- wikitext to plain text -----------------------------------------------------

def _strip_nested(text, opener, closer):
    """Remove balanced opener...closer spans (templates, tables)."""
    out = []
    depth = 0
    index = 0
    length = len(text)
    while index < length:
        if text.startswith(opener, index):
            depth += 1
            index += len(opener)
        elif depth and text.startswith(closer, index):
            depth -= 1
            index += len(closer)
        else:
            if not depth:
                out.append(text[index])
            index += 1
    return "".join(out)


_LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_FILE_LINK = re.compile(r"\[\[(?:File|Image|Category|Media|Wikipedia|Template|Help):", re.I)


def _remove_file_links(text):
    while True:
        match = _FILE_LINK.search(text)
        if not match:
            return text
        depth, index = 0, match.start()
        while index < len(text):
            if text.startswith("[[", index):
                depth += 1
                index += 2
            elif text.startswith("]]", index):
                depth -= 1
                index += 2
                if depth == 0:
                    break
            else:
                index += 1
        text = text[:match.start()] + text[index:]


def clean_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<(gallery|math|code|pre|timeline|score|syntaxhighlight)[^>]*>.*?</\1>",
                  "", text, flags=re.S | re.I)
    text = _strip_nested(text, "{{", "}}")
    text = _strip_nested(text, "{|", "|}")
    text = _remove_file_links(text)
    text = _LINK.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(1), text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]*\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = html.unescape(text)
    lines = []
    for line in text.split("\n"):
        heading = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", line)
        if heading:
            if heading.group(2).strip().lower() in DROP_SECTIONS:
                break
            title = heading.group(2).strip()
            if title:
                lines.append("")
                lines.append(title)
            continue
        line = re.sub(r"^[*#:;]+\s*", "", line).strip()
        if not line or line.startswith(("|", "!", "__")):
            continue
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\(\s*[,;]?\s*\)", "", line).replace(" ,", ",").replace(" .", ".")
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_simplewiki(writers, counts):
    (name, url), = SOURCES["simplewiki"]
    path = fetch(name, url)
    namespace = None
    with bz2.open(path, "rb") as stream:
        for event, element in ElementTree.iterparse(stream, events=("end",)):
            tag = element.tag
            if namespace is None and "}" in tag:
                namespace = tag[:tag.index("}") + 1]
            if tag != (namespace or "") + "page":
                continue
            ns = element.findtext((namespace or "") + "ns")
            redirect = element.find((namespace or "") + "redirect")
            title = element.findtext((namespace or "") + "title") or ""
            text = element.findtext((namespace or "") + "revision/" + (namespace or "") + "text")
            element.clear()
            if ns != "0" or redirect is not None or not text:
                continue
            if text.lstrip().lower().startswith("#redirect"):
                continue
            plain = clean_wikitext(text)
            if len(plain) < 200:
                counts["wiki_skipped_short"] += 1
                continue
            split = bucket(title)
            writers["wiki-" + split].write(json.dumps({"title": title, "text": plain},
                                                      ensure_ascii=False) + "\n")
            counts["wiki-" + split] += 1


def build_squad(writers, counts):
    for name, url in SOURCES["squad"]:
        path = fetch(name, url)
        with open(path) as handle:
            data = json.load(handle)["data"]
        dev = "dev" in name
        for article in data:
            title = article["title"]
            # SQuAD's own split is respected: train titles train; dev titles are
            # halved by title hash into validation and the sealed test.
            if dev:
                half = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:8], 16) % 2
                split = "test" if half == 0 else "validation"
            else:
                split = "train"
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    answers = sorted({answer["text"] for answer in qa["answers"]})
                    writers["squad-" + split].write(json.dumps({
                        "title": title.replace("_", " "), "context": paragraph["context"],
                        "question": qa["question"].strip(), "answers": answers},
                        ensure_ascii=False) + "\n")
                    counts["squad-" + split] += 1


def build_oasst(writers, counts):
    (name, url), = SOURCES["oasst"]
    path = fetch(name, url)

    def best_path(node, turns):
        turns = turns + [{"role": "user" if node["role"] == "prompter" else "assistant",
                          "text": node["text"].strip()}]
        replies = [reply for reply in node.get("replies", [])
                   if reply.get("lang") == "en" and not reply.get("deleted")
                   and reply.get("review_result", True)]
        if not replies:
            return turns
        replies.sort(key=lambda reply: (reply.get("rank") is None, reply.get("rank") or 0))
        return best_path(replies[0], turns)

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            tree = json.loads(line)
            root = tree["prompt"]
            if root.get("lang") != "en" or tree.get("tree_state") != "ready_for_export":
                continue
            turns = best_path(root, [])
            while turns and turns[-1]["role"] != "assistant":
                turns.pop()
            if len(turns) < 2:
                continue
            split = bucket(tree["message_tree_id"])
            writers["dialog-" + split].write(json.dumps({"turns": turns},
                                                        ensure_ascii=False) + "\n")
            counts["dialog-" + split] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="simplewiki,squad,oasst")
    args = parser.parse_args()
    wanted = args.only.split(",")
    builders = {"simplewiki": ("wiki", build_simplewiki), "squad": ("squad", build_squad),
                "oasst": ("dialog", build_oasst)}
    os.makedirs(OUT, exist_ok=True)
    manifest_path = os.path.join(OUT, "MANIFEST.json")
    manifest = {"sources": {}, "outputs": {}, "counts": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    for source in wanted:
        prefix, builder = builders[source]
        started = time.time()
        writers = {"%s-%s" % (prefix, split): open(os.path.join(OUT, "%s-%s.jsonl" % (prefix, split)),
                                                  "w", encoding="utf-8")
                   for split in ("train", "validation", "test")}
        counts = {key: 0 for key in writers}
        counts.setdefault("wiki_skipped_short", 0)
        try:
            builder(writers, counts)
        finally:
            for handle in writers.values():
                handle.close()
        for name, url in SOURCES[source]:
            manifest["sources"][name] = {"url": url,
                                         "sha256": sha256_file(os.path.join(RAW, name))}
        for key in writers:
            path = os.path.join(OUT, key + ".jsonl")
            manifest["outputs"][key + ".jsonl"] = {"bytes": os.path.getsize(path),
                                                   "sha256": sha256_file(path)}
        manifest["counts"].update({key: value for key, value in counts.items() if value})
        print(json.dumps({"source": source, "seconds": round(time.time() - started),
                          "counts": counts}), flush=True)
    manifest["split_rule"] = ("bucket = int(sha256(key)[:8], 16) % 100; 0 test, 1-2 validation, "
                              "else train; key = title (wiki) or tree id (dialog); SQuAD train "
                              "articles train, dev articles halved by title hash into "
                              "validation and test")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

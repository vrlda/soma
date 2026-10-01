#!/usr/bin/env python3
"""Build or re-fetch the E3-full corpus (about 100 MB) by a fixed rule.

Master plan §22 step 9. Acquisition = the 27 E3-lite books, plus new
Project Gutenberg books chosen by this rule from the official catalog
(``pg_catalog.csv``):

  1. Type "Text", Language exactly "en".
  2. LoCC starts with "PR" or "PS" (English or American literature) and
     Subjects mention "Fiction" (the register of the existing corpus).
  3. Not used anywhere before: no Gutenberg ID from the E1/E2/E3/E2u
     manifests (including the Jekyll validation, Time Machine test, and
     untouched Thursday books), and no other edition of those works (same
     normalized title).
  4. Not a collection: titles with "complete", "works of", "collected" or
     "anthology" are skipped.
  5. Seeded order: eligible IDs shuffled with random.Random(2026). Books are
     downloaded in that order (one request per second), CRLF normalized to
     LF, Gutenberg boilerplate stripped (required), valid UTF-8 required,
     and kept when the stripped text is 100 KB to 2 MB. Stop once the new
     books total at least TARGET_NEW_BYTES. Every skip is recorded.

Validation (Jekyll) and test (Time Machine) are unchanged.

  build  [--catalog pg_catalog.csv]   select, download, write the manifest
  fetch                              re-download into data/e3full/, verify hashes

Book texts live in data/e3full/ (git-ignored); the manifest is committed.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from soma.evaluation.english import strip_gutenberg_boilerplate  # noqa: E402

MANIFEST = "reports/e3-full-manifest.json"
BASE_MANIFEST = "reports/e3-manifest.json"
USED_MANIFESTS = ("reports/e1-manifest.json", "reports/e2-manifest.json",
                  "reports/e3-manifest.json", "reports/e2u-manifest.json")
DATA_DIRECTORY = "data/e3full"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
SEED = 2026
TARGET_NEW_BYTES = 90 * 1000 * 1000
MIN_BYTES, MAX_BYTES = 100 * 1000, 2 * 1000 * 1000
COLLECTION_WORDS = ("complete", "works of", "collected", "anthology")
USER_AGENT = "soma-research-corpus/1.0 (one request per second)"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalize_title(title):
    first = title.split("\n")[0].split(";")[0].split(":")[0].lower()
    first = re.sub(r"[^a-z0-9 ]+", " ", first)
    words = [word for word in first.split() if word not in ("the", "a", "an")]
    return " ".join(words)


def used_ids_and_books():
    ids, books = set(), []
    for path in USED_MANIFESTS:
        with open(os.path.join(ROOT, path)) as handle:
            for book in json.load(handle)["books"]:
                match = re.search(r"/(?:files|epub)/(\d+)/", book["source_url"])
                if match:
                    ids.add(match.group(1))
                if book not in books:
                    books.append(book)
    return ids, books


def eligible(catalog_rows, used_ids):
    used_titles = {normalize_title(row["Title"]) for row in catalog_rows
                   if row["Text#"] in used_ids}
    chosen = []
    for row in catalog_rows:
        if row["Type"] != "Text" or row["Language"] != "en":
            continue
        if not row["LoCC"].startswith(("PR", "PS")) or "Fiction" not in row["Subjects"]:
            continue
        if row["Text#"] in used_ids or normalize_title(row["Title"]) in used_titles:
            continue
        if any(word in row["Title"].lower() for word in COLLECTION_WORDS):
            continue
        chosen.append(row)
    chosen.sort(key=lambda row: int(row["Text#"]))
    random.Random(SEED).shuffle(chosen)
    return chosen


def download(url, attempts=3):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(5 * (attempt + 1))


def prepare(raw):
    """Normalize like the untouched test; return (normalized, stripped) or a reason."""
    normalized = raw.replace(b"\r\n", b"\n")
    try:
        normalized.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, "not valid UTF-8"
    stripped, did_strip = strip_gutenberg_boilerplate(normalized)
    if not did_strip:
        return None, "no Gutenberg boilerplate markers"
    if not MIN_BYTES <= len(stripped) <= MAX_BYTES:
        return None, "stripped size %d outside %d..%d" % (len(stripped), MIN_BYTES, MAX_BYTES)
    return (normalized, stripped), None


def build(args):
    catalog_path = args.catalog
    if not os.path.exists(catalog_path):
        with open(catalog_path, "wb") as handle:
            handle.write(download(CATALOG_URL))
    with open(catalog_path, "rb") as handle:
        catalog_bytes = handle.read()
    rows = list(csv.DictReader(catalog_bytes.decode("utf-8").splitlines()))
    used_ids, _ = used_ids_and_books()
    candidates = eligible(rows, used_ids)
    os.makedirs(os.path.join(ROOT, DATA_DIRECTORY), exist_ok=True)
    with open(os.path.join(ROOT, BASE_MANIFEST)) as handle:
        base = json.load(handle)
    books = [dict(book) for book in base["books"] if book["partition"] == "acquisition"]
    new_entries, skipped, new_bytes = [], [], 0
    for row in candidates:
        if new_bytes >= TARGET_NEW_BYTES:
            break
        book_id = row["Text#"]
        url = BOOK_URL.format(id=book_id)
        try:
            raw = download(url)
        except OSError as error:
            skipped.append({"id": book_id, "reason": "download failed: %s" % error})
            continue
        finally:
            time.sleep(1.0)
        prepared, reason = prepare(raw)
        if prepared is None:
            skipped.append({"id": book_id, "reason": reason})
            continue
        normalized, stripped = prepared
        path = os.path.join(DATA_DIRECTORY, "pg%s.txt" % book_id)
        with open(os.path.join(ROOT, path), "wb") as handle:
            handle.write(normalized)
        new_entries.append({
            "name": "pg%s" % book_id,
            "title": row["Title"].split("\n")[0][:120],
            "authors": row["Authors"][:120],
            "path": path,
            "partition": "acquisition",
            "license": "public-domain",
            "source_url": url,
            "downloaded_sha256": _sha256(raw),
            "line_endings_normalized": normalized != raw,
            "raw_bytes": len(normalized),
            "raw_sha256": _sha256(normalized),
            "bytes": len(stripped),
            "sha256": _sha256(stripped),
            "boilerplate_stripped": True,
            "retrieved_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        })
        new_bytes += len(stripped)
        print("%4d books  %6.1f MB  pg%s" % (len(new_entries), new_bytes / 1e6, book_id), flush=True)
    books.extend(new_entries)
    books.extend(dict(book) for book in base["books"] if book["partition"] != "acquisition")
    manifest = {
        "books": books,
        "acquisition_bytes": sum(b["bytes"] for b in books if b["partition"] == "acquisition"),
        "validation_bytes": sum(b["bytes"] for b in books if b["partition"] == "validation"),
        "test_bytes": sum(b["bytes"] for b in books if b["partition"] == "test"),
        "selection": {
            "rule": __doc__.split("Validation (Jekyll)")[0].strip(),
            "seed": SEED,
            "catalog_url": CATALOG_URL,
            "catalog_sha256": _sha256(catalog_bytes),
            "eligible": len(candidates),
            "new_books": len(new_entries),
            "skipped": skipped,
        },
    }
    with open(os.path.join(ROOT, MANIFEST), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"new_books": len(new_entries), "new_bytes": new_bytes,
                      "skipped": len(skipped), "eligible": len(candidates),
                      "acquisition_bytes": manifest["acquisition_bytes"]}, indent=2))
    return 0


def fetch(args):
    with open(os.path.join(ROOT, MANIFEST)) as handle:
        manifest = json.load(handle)
    os.makedirs(os.path.join(ROOT, DATA_DIRECTORY), exist_ok=True)
    failures = []
    for book in manifest["books"]:
        if not book["path"].startswith(DATA_DIRECTORY):
            continue
        target = os.path.join(ROOT, book["path"])
        if os.path.exists(target):
            with open(target, "rb") as handle:
                if _sha256(handle.read()) == book["raw_sha256"]:
                    continue
        raw = download(book["source_url"])
        time.sleep(1.0)
        normalized = raw.replace(b"\r\n", b"\n")
        if _sha256(normalized) != book["raw_sha256"]:
            failures.append(book["name"])
            continue
        with open(target, "wb") as handle:
            handle.write(normalized)
    print(json.dumps({"failures": failures}))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--catalog", default="pg_catalog.csv")
    commands.add_parser("fetch")
    args = parser.parse_args()
    return build(args) if args.command == "build" else fetch(args)


if __name__ == "__main__":
    sys.exit(main())

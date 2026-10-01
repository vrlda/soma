#!/usr/bin/env python3
"""Pre-registered, single-evaluation untouched test for R6 circuit mixing.

Master plan §22 step 2. The Time Machine test book was visible while the
circuit-mixing configuration was tuned, so its score is validation-grade.
This script scores one book that no manifest, experiment, or tuning run has
used, exactly once, with the configuration frozen below.

Two subcommands:

  prepare --raw FILE --name NAME --source-url URL
      Normalize a downloaded Project Gutenberg UTF-8 text to the acquisition
      conventions (LF line endings; ASCII-quote editions are refused), store
      it in data/e2u/, and
      write reports/e2u-manifest.json: the E2 acquisition and validation
      books unchanged, plus the new book as the only test partition.
      Prepare never reads scores.

  score
      Run every pre-registered comparison on the untouched book in one
      pass, write reports/r6-untouched-test.json, and refuse to run again
      if that report exists. The report is final whatever it says.

Pre-registered comparisons (all frozen, same acquisition bytes, frozen
scoring, bits per bit on the untouched book):
  - circuit mixing, full budget (FROZEN_CONFIG, 2**24 circuits)
  - circuit mixing, compact budget (FROZEN_CONFIG, 2**22 circuits)
  - prior E2 memory (`r3b_e1_benchmark.py`, protocol r3b-e1-v2, unchanged)
  - frozen Witten-Bell byte n-grams, orders 2 and 5
  - acquisition-fitted byte unigram

Pre-registered hypotheses (reported pass or fail, never re-scored):
  H1 full circuit mixing beats the order-5 n-gram.
  H2 full circuit mixing beats the prior E2 memory.
  H3 compact circuit mixing beats the order-5 n-gram.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

from r3b_reference_baselines import WittenBellByteModel
from r6_circuit_mixing_benchmark import build_engine, run_engine
from soma.evaluation.english import (
    byte_unigram_cross_bits, load_verified_book_corpus, strip_gutenberg_boilerplate,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = "r6-untouched-test-v1"
BASE_MANIFEST = "reports/e2-manifest.json"
MANIFEST = "reports/e2u-manifest.json"
REPORT = "reports/r6-untouched-test.json"
PRIOR_MANIFESTS = ("reports/e1-manifest.json", "reports/e2-manifest.json",
                   "reports/e3-manifest.json")
DATA_DIRECTORY = "data/e2u"

# Frozen at commit 53b4404 (the configuration reported in
# docs/r6-circuit-mixing.md). Written out in full so later default changes
# cannot alter this test.
FROZEN_CONFIG = {
    "orders": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
    "learning_rate": 0.002,
    "count_limit": 1023,
    "initial_weight": 0.3,
    "reclaim_fraction": 0.125,
    "arbitration": True,
    "halve_above": 1000000,
    "calibration": False,
    "calibration_limit": 255,
    "gate_bit_position": False,
    "gate_partial": True,
    "correction": True,
    "correction_rate": 0.02,
    "growth_threshold": 8,
    # Did not exist when this test was scored (r6-retention-policy-v1 later).
    "growth_pressure": 0,
    # Metaplasticity did not exist when this test was scored; 0 is the
    # constant-rate behavior that was actually evaluated.
    "plasticity_tau": 0.0,
}
FULL_BUDGET = 1 << 24
COMPACT_BUDGET = 1 << 22


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def used_content_hashes():
    """Stripped-content hashes of every book any earlier manifest used."""
    names, hashes = set(), set()
    for path in PRIOR_MANIFESTS:
        with open(os.path.join(ROOT, path)) as handle:
            for book in json.load(handle)["books"]:
                names.add(book["name"])
                hashes.add(book["sha256"])
    return names, hashes


def check_untouched(name, stripped, names, hashes):
    """Raise if the candidate book was used by any earlier manifest."""
    if name in names:
        raise ValueError("book name already used by a prior manifest: %s" % name)
    if _sha256(stripped) in hashes:
        raise ValueError("book content already used by a prior manifest: %s" % name)


def build_manifest(base, test_entry):
    books = [dict(book) for book in base["books"] if book["partition"] != "test"]
    books.append(test_entry)
    return {
        "books": books,
        "acquisition_bytes": sum(b["bytes"] for b in books if b["partition"] == "acquisition"),
        "validation_bytes": sum(b["bytes"] for b in books if b["partition"] == "validation"),
        "test_bytes": test_entry["bytes"],
        "protocol": PROTOCOL,
    }


def normalize_conventions(raw):
    """Match the acquisition books' byte conventions before anything else.

    Acquisition books are stored with LF line endings and curly quotes. A
    CRLF download is converted to LF (recorded in the manifest). An edition
    that uses ASCII double quotes is refused: those bytes never occur in
    acquisition, so scoring would measure novel-byte handling rather than
    English modeling.
    """
    normalized = raw.replace(b"\r\n", b"\n")
    ascii_quotes = normalized.count(b'"')
    curly_quotes = normalized.count("\u201c".encode("utf-8"))
    if ascii_quotes > curly_quotes:
        raise SystemExit("edition uses ASCII quotes (%d vs %d curly): use the UTF-8 edition"
                         % (ascii_quotes, curly_quotes))
    return normalized, normalized != raw


def prepare(args):
    with open(args.raw, "rb") as handle:
        downloaded = handle.read()
    raw, line_endings_normalized = normalize_conventions(downloaded)
    stripped, did_strip = strip_gutenberg_boilerplate(raw)
    if not did_strip:
        raise SystemExit("no Project Gutenberg boilerplate found: use the canonical UTF-8 edition")
    names, hashes = used_content_hashes()
    check_untouched(args.name, stripped, names, hashes)
    destination = os.path.join(DATA_DIRECTORY, args.name + ".txt")
    os.makedirs(os.path.join(ROOT, DATA_DIRECTORY), exist_ok=True)
    with open(os.path.join(ROOT, destination), "wb") as handle:
        handle.write(raw)
    entry = {
        "name": args.name,
        "path": destination,
        "partition": "test",
        "license": "public-domain",
        "source_url": args.source_url,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "bytes": len(stripped),
        "sha256": _sha256(stripped),
        "boilerplate_stripped": True,
        "downloaded_sha256": _sha256(downloaded),
        "line_endings_normalized": line_endings_normalized,
        "retrieved_utc": time.strftime("%Y-%m-%d", time.gmtime()),
    }
    with open(os.path.join(ROOT, BASE_MANIFEST)) as handle:
        manifest = build_manifest(json.load(handle), entry)
    with open(os.path.join(ROOT, MANIFEST), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"manifest": MANIFEST, "test": entry}, indent=2, sort_keys=True))
    return 0


def score(args):
    report_path = os.path.join(ROOT, REPORT)
    if os.path.exists(report_path):
        raise SystemExit("%s exists: the untouched test is scored exactly once" % REPORT)
    with open(os.path.join(ROOT, MANIFEST)) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, os.path.join(ROOT, MANIFEST))
    test_name, test_data = parts["test"][0]
    names, hashes = used_content_hashes()
    check_untouched(test_name, test_data, names, hashes)
    acquisition = parts["acquisition"]
    trained = b"".join(data for _, data in acquisition)
    started = time.time()
    build_engine()

    with tempfile.TemporaryDirectory() as directory:
        train_paths = []
        for index, (name, data) in enumerate(acquisition):
            path = os.path.join(directory, "acq-%02d-%s.bin" % (index, name))
            with open(path, "wb") as handle:
                handle.write(data)
            train_paths.append(path)
        test_path = os.path.join(directory, "test.bin")
        with open(test_path, "wb") as handle:
            handle.write(test_data)
        mixing = {}
        for tier, budget in (("full", FULL_BUDGET), ("compact", COMPACT_BUDGET)):
            config = dict(FROZEN_CONFIG, max_circuits=budget)
            result = run_engine(config, train_paths, {"test": test_path})
            mixing[tier] = {
                "bits_per_bit": result["bits_per_bit"]["test"],
                "circuits": result["circuits"],
                "circuits_reclaimed": result["circuits_reclaimed"],
                "peak_rss_mb": result["peak_rss_mb"],
                "seconds": result["total_seconds"],
            }
        prior_path = os.path.join(directory, "prior-e2.json")
        subprocess.run([sys.executable, os.path.join(ROOT, "r3b_e1_benchmark.py"),
                        "--manifest", os.path.join(ROOT, MANIFEST), "--report", prior_path],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        with open(prior_path) as handle:
            prior = json.load(handle)

    ngram = WittenBellByteModel(5).fit([data for _, data in acquisition])
    references = {
        "ngram_order_2": ngram.bits_per_byte(test_data, 2) / 8.0,
        "ngram_order_5": ngram.bits_per_byte(test_data, 5) / 8.0,
        "byte_unigram": byte_unigram_cross_bits(trained, test_data),
        "prior_e2_memory": prior["test_bits_per_bit"],
    }
    hypotheses = {
        "H1_full_beats_ngram_order5":
            mixing["full"]["bits_per_bit"] < references["ngram_order_5"],
        "H2_full_beats_prior_e2":
            mixing["full"]["bits_per_bit"] < references["prior_e2_memory"],
        "H3_compact_beats_ngram_order5":
            mixing["compact"]["bits_per_bit"] < references["ngram_order_5"],
    }
    report = {
        "protocol": PROTOCOL,
        "manifest": MANIFEST,
        "test_book": test_name,
        "test_bytes": len(test_data),
        "acquisition_bytes": len(trained),
        "frozen_config": FROZEN_CONFIG,
        "circuit_mixing": mixing,
        "references": references,
        "prior_e2_gates": prior.get("gates"),
        "hypotheses": hypotheses,
        "all_hypotheses_supported": all(hypotheses.values()),
        "total_seconds": time.time() - started,
        "single_evaluation": True,
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--raw", required=True)
    prepare_parser.add_argument("--name", required=True)
    prepare_parser.add_argument("--source-url", required=True)
    commands.add_parser("score")
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else score(args)


if __name__ == "__main__":
    sys.exit(main())

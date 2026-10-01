#!/usr/bin/env python3
"""Small candidate evaluation (protocol r8-small-candidate-v1).

Pre-registered in docs/r8-small-candidate.md. Inputs are two builds from
scripts/build_small_candidate.py: one straight, one with --split. The
sealed book is reports/e3u2-manifest.json, scored once per model.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from r3b_reference_baselines import WittenBellByteModel
from r6_circuit_mixing_benchmark import BINARY, run_engine
from soma.evaluation.english import byte_unigram_cross_bits, load_verified_book_corpus

PROTOCOL = "r8-small-candidate-v1"
SEALED = "reports/e3u2-manifest.json"
BUDGET = 1 << 24
NGRAM_MARGIN = 0.02
CEILINGS = {"engine_rss_mb": 8192, "cold_turn_seconds": 60.0, "warm_turn_median_seconds": 2.0}
PROMPTS = ("hello there", "what is the sea like?", "tell me about the night",
           "who lives in the old house?", "describe a storm", "where does the road go?")


def _load(path):
    with open(path) as handle:
        return load_verified_book_corpus(json.load(handle), path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _vm_hwm_mb(pid):
    with open("/proc/%d/status" % pid) as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def product_checks(root, name):
    from soma.service.chat import chat_turn
    from soma.service.store import BrainStore
    from soma.memory import engine
    store = BrainStore(root)
    began = time.time()
    replies = [chat_turn(store, name, PROMPTS[0])]
    cold = time.time() - began
    warm = []
    for prompt in PROMPTS[1:]:
        began = time.time()
        replies.append(chat_turn(store, name, prompt))
        warm.append(time.time() - began)
    live = engine._LIVE[os.path.abspath(os.path.join(root, name, "memory.somamix"))]
    return {"cold_turn_seconds": cold, "warm_turn_seconds": warm,
            "warm_turn_median_seconds": statistics.median(warm),
            "engine_rss_mb": _vm_hwm_mb(live._process.pid), "replies": replies}


def signed_round_trip(artifact, memory_sha256):
    from soma.persistence import signing
    from soma.service import distribution
    from soma.service.store import BrainStore
    with tempfile.TemporaryDirectory() as directory:
        os.environ["SOMA_HOME"] = os.path.join(directory, "home")
        copy = os.path.join(directory, "candidate.soma")
        shutil.copyfile(artifact, copy)
        secret = signing.generate_secret()  # throwaway: tests the flow, not the release key
        signing.sign_file(copy, secret)
        distribution.trust_key(signing.public_key(secret).hex(), "throwaway")
        store = BrainStore(os.path.join(directory, "brains"))
        result = distribution.import_signed(store, copy, "imported", require_signature=True)
        imported = _sha256(os.path.join(store.root, "imported", "memory.somamix"))
        del os.environ["SOMA_HOME"]
    return {"signature": result["provenance"]["signature"], "memory_identical":
            imported == memory_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--name", default="soma-small-candidate")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", default="reports/r8-small-candidate.json")
    parser.add_argument("--sealed", default=SEALED,
                        help="development smoke only; the protocol uses the default")
    args = parser.parse_args()

    def lineage(root):
        with open(os.path.join(root, args.name, "manifest.json")) as handle:
            return json.load(handle)["base_training"]

    built, split = lineage(args.root), lineage(args.split_root)
    memory_path = os.path.join(args.root, args.name, "memory.somamix")
    if _sha256(memory_path) != built["memory_sha256"]:
        raise SystemExit("candidate memory does not match its recorded lineage")
    sealed = _load(args.sealed)["test"][0]
    e2 = _load("reports/e2-manifest.json")["acquisition"]
    e3 = _load("reports/e3-full-manifest.json")["acquisition"]
    with tempfile.TemporaryDirectory() as directory:
        sealed_path = os.path.join(directory, "sealed.bin")
        with open(sealed_path, "wb") as handle:
            handle.write(sealed[1])
        job = {"load_state": memory_path, "train": [], "eval": {"sealed": sealed_path}}
        completed = subprocess.run([BINARY], input=json.dumps(job), check=True,
                                   capture_output=True, text=True)
        candidate = json.loads(completed.stdout)["bits_per_bit"]["sealed"]
        e2_paths = []
        for index, (_, data) in enumerate(e2):
            path = os.path.join(directory, "e2-%02d.bin" % index)
            with open(path, "wb") as handle:
                handle.write(data)
            e2_paths.append(path)
        control = run_engine({"max_circuits": BUDGET}, e2_paths,
                             {"sealed": sealed_path})["bits_per_bit"]["sealed"]
    ngram = WittenBellByteModel(5).fit([data for _, data in e2]).bits_per_byte(sealed[1], 5) / 8.0
    unigram = byte_unigram_cross_bits(b"".join(data for _, data in e3), sealed[1])
    product = product_checks(args.root, args.name)
    round_trip = signed_round_trip(args.artifact, built["memory_sha256"])
    gates = {
        "H1_beats_e2_trained_same_config": candidate < control,
        "H2_beats_order5_ngram_by_margin": candidate <= ngram - NGRAM_MARGIN,
        "P1_engine_rss": product["engine_rss_mb"] <= CEILINGS["engine_rss_mb"],
        "P2_cold_turn": product["cold_turn_seconds"] <= CEILINGS["cold_turn_seconds"],
        "P3_warm_turn_median": (product["warm_turn_median_seconds"]
                                <= CEILINGS["warm_turn_median_seconds"]),
        "P4_resume_identical": built["memory_sha256"] == split["memory_sha256"],
        "P5_signed_round_trip": (round_trip["signature"] == "verified"
                                 and round_trip["memory_identical"]),
        "P6_replies_nonempty": all(reply.strip() for reply in product["replies"]),
    }
    report = {
        "protocol": PROTOCOL if args.sealed == SEALED else PROTOCOL + "-smoke",
        "sealed_book": sealed[0],
        "sealed_bytes": len(sealed[1]),
        "scores": {"candidate": candidate, "control_e2_same_config": control,
                   "ngram_order5_e2": ngram, "byte_unigram_e3full": unigram},
        "lineage": built,
        "split_lineage_memory_sha256": split["memory_sha256"],
        "artifact": {"path": os.path.basename(args.artifact),
                     "bytes": os.path.getsize(args.artifact),
                     "sha256": _sha256(args.artifact)},
        "product": product,
        "signed_round_trip": round_trip,
        "ceilings": CEILINGS,
        "ngram_margin": NGRAM_MARGIN,
        "gates": gates,
        "all_passed": all(gates.values()),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"scores": report["scores"], "gates": gates}, indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

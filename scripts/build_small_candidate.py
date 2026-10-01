#!/usr/bin/env python3
"""Build the Small candidate brain reproducibly (master plan §22 step 12).

1. Verify the E3-full corpus (``scripts/build_e3_full_corpus.py fetch``).
2. ``BrainStore.create(memory="mixing", max_circuits=2**24)``: the canonical
   blank brain (seed 0) with an empty engine memory under current defaults.
3. Train that memory with the engine (``soma-mixer``): load the blank
   ``memory.somamix``, observe every acquisition book in manifest order
   (one document each, weight 1), and save it back. With ``--split N`` the
   training stops after book N, saves, and resumes from the file. The
   result must be byte-identical either way.
4. Record lineage in the brain manifest (``base_training``) and export
   ``dist/<name>.soma``.

The trained memory is deterministic; its SHA-256 is the reproducibility
check. The `.soma` file also holds a creation time, so it is signed as
built (``python3 -m soma.service.main sign dist/<name>.soma --key ...``).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from soma.evaluation.english import load_verified_book_corpus  # noqa: E402
from soma.service.store import BrainStore  # noqa: E402

MANIFEST = "reports/e3-full-manifest.json"
BUDGET = 1 << 24
BINARY = os.path.join(ROOT, "engine", "soma-engine", "target", "release", "soma-mixer")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train(load_state, files, save_state):
    job = {"load_state": load_state, "train": files, "eval": {}, "save_state": save_state}
    completed = subprocess.run([BINARY], input=json.dumps(job), check=True,
                               capture_output=True, text=True)
    report = json.loads(completed.stdout)
    return {key: report[key] for key in ("circuits", "circuits_created", "circuits_reclaimed",
                                         "events_seen", "peak_rss_mb", "total_seconds")}


def build(root, name, out_dir, split=None, manifest_path=MANIFEST):
    with open(os.path.join(ROOT, manifest_path)) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, manifest_path)
    store = BrainStore(root)
    created = store.create(name, description="SOMA Small candidate (circuit mixing, E3-full)",
                           memory="mixing", max_circuits=BUDGET)
    memory_path = os.path.join(root, name, "memory.somamix")
    blank_sha256 = _sha256(memory_path)
    started = time.time()
    with tempfile.TemporaryDirectory() as directory:
        files = []
        for index, (book, data) in enumerate(parts["acquisition"]):
            path = os.path.join(directory, "%04d-%s.bin" % (index, book))
            with open(path, "wb") as handle:
                handle.write(data)
            files.append(path)
        if split:
            middle = os.path.join(directory, "middle.somamix")
            runs = [_train(memory_path, files[:split], middle),
                    _train(middle, files[split:], memory_path)]
        else:
            runs = [_train(memory_path, files, memory_path)]
    lineage = {
        "corpus_manifest": manifest_path,
        "corpus_manifest_sha256": _sha256(os.path.join(ROOT, manifest_path)),
        "books": len(parts["acquisition"]),
        "bytes": sum(len(data) for _, data in parts["acquisition"]),
        "engine_config": {"max_circuits": BUDGET, "defaults": "retention-policy-v1"},
        "blank_memory_sha256": blank_sha256,
        "memory_sha256": _sha256(memory_path),
        "split_after_book": split,
        "runs": runs,
        "seconds": time.time() - started,
    }
    manifest_file = os.path.join(root, name, "manifest.json")
    with open(manifest_file) as handle:
        brain_manifest = json.load(handle)
    brain_manifest["base_training"] = lineage
    with open(manifest_file, "w") as handle:
        json.dump(brain_manifest, handle, indent=2, sort_keys=True)
    os.makedirs(out_dir, exist_ok=True)
    artifact = store.export_soma(name, os.path.join(out_dir, name + ".soma"))
    return {"identity": created["identity"], "lineage": lineage, "artifact": artifact}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="fresh brain store directory")
    parser.add_argument("--name", default="soma-small-candidate")
    parser.add_argument("--out", default="dist")
    parser.add_argument("--split", type=int, help="save and resume after this many books")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.name, args.out, args.split), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

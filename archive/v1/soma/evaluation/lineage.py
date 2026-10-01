"""Minimal brain lineage: clone with identity record (runbook 7A.1/14).

A clone copies the checkpoint bytes and records a new identity with its
parent hash and reason. The source is never modified. Promotion policy
(qualified generations) lives above this mechanism.
"""

import hashlib
import json
import os
import shutil
import time


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_brain(source, destination, reason):
    """Copy a checkpoint to a new identity; record lineage alongside it."""
    if not reason:
        raise ValueError("clone requires a reason")
    if os.path.exists(destination):
        raise ValueError("clone destination already exists: %s" % destination)
    shutil.copyfile(source, destination)
    record = {
        "identity": _sha256_file(destination)[:16],
        "parent_sha256": _sha256_file(source),
        "reason": str(reason),
        "created_utc": int(time.time()),
    }
    with open(destination + ".lineage.json", "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


def read_lineage(checkpoint):
    """Read the lineage sidecar for a checkpoint."""
    path = checkpoint + ".lineage.json"
    if not os.path.exists(path):
        raise ValueError("no lineage record for: %s" % checkpoint)
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "identity" not in payload:
        raise ValueError("lineage record is invalid")
    return payload


def verify_clone(source, destination):
    """Check the clone matches its source bytes and record."""
    if _sha256_file(source) != _sha256_file(destination):
        raise ValueError("clone bytes differ from source")
    record = read_lineage(destination)
    if record.get("parent_sha256") != _sha256_file(source):
        raise ValueError("lineage parent hash mismatch")
    return record

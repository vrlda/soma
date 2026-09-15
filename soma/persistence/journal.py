"""Crash-safe brain snapshots: dirty marker plus previous-generation rotation.

Every save writes a start marker, rotates the current generation aside,
writes the new generation, then clears the marker. A load that finds a
dirty marker validates the current generation and falls back to the
previous one when it fails. All renames are atomic; all appends fsynced.
"""

import json
import os

MARKER = "journal.log"
PREV_SUFFIX = ".prev"


def _fsync_append(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_marker(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        records = []
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _clear_marker(path):
    with open(path, "w") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def begin_save(directory):
    """Mark a save as dirty. Returns nothing; raises on I/O failure."""
    _fsync_append(os.path.join(directory, MARKER), {"op": "save-start"})


def end_save(directory):
    """Clear the dirty marker after a complete generation switch."""
    _clear_marker(os.path.join(directory, MARKER))


def is_dirty(directory):
    records = _read_marker(os.path.join(directory, MARKER))
    return bool(records) and records[-1].get("op") != "save-done"


def rotate_previous(paths):
    """Move current generation files aside (atomic renames per file)."""
    for key in ("brain", "episodic", "manifest", "dialogue"):
        source = paths.get(key)
        if source is not None and os.path.exists(source):
            os.rename(source, source + PREV_SUFFIX)


def restore_previous(paths):
    """Bring the previous generation back; raises when absent."""
    for key in ("brain", "episodic", "manifest", "dialogue"):
        backup = (paths.get(key) or "") + PREV_SUFFIX
        if not os.path.exists(backup):
            raise ValueError("no previous generation for crash recovery")
    for key in ("brain", "episodic", "manifest", "dialogue"):
        source = paths.get(key)
        if source is None:
            continue
        backup = source + PREV_SUFFIX
        if os.path.exists(backup):
            os.rename(backup, source)


def recover_if_needed(paths):
    """Validate current generation on a dirty marker; else restore previous.

    Returns "clean", "valid", or "restored".
    """
    from ..memory import EpisodicBuffer, SequenceCircuitMemory
    from ..organism import Organism
    directory = os.path.dirname(paths["brain"])
    if not is_dirty(directory):
        return "clean"
    try:
        organism = Organism.load(paths["brain"])
        organism.validate()
        with open(paths["episodic"]) as handle:
            EpisodicBuffer.from_state_dict(json.load(handle)).validate()
        if os.path.exists(paths.get("dialogue") or ""):
            with open(paths["dialogue"]) as handle:
                SequenceCircuitMemory.from_state_dict(json.load(handle)).validate()
        with open(paths["manifest"]) as handle:
            json.load(handle)
        end_save(directory)
        return "valid"
    except (OSError, ValueError, AssertionError, KeyError):
        restore_previous(paths)
        end_save(directory)
        return "restored"

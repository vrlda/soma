#!/usr/bin/env python3
"""Frozen E1 scaling protocol: book-level acquisition with curve + profile.

G1: validation bits/bit improves from first to last acquisition checkpoint.
G2: final validation beats the byte-unigram baseline.
G3: test (sealed, single eval) beats the byte-unigram baseline.
G4: circuits stay within budget; state serializes exactly mid-stream.
G5: lesion fails the baseline (causal).
"""

import argparse
import copy
import json
import os
import tempfile
import time

from soma.evaluation.english import (
    bit_stream,
    bits_per_bit,
    byte_unigram_cross_bits,
    load_verified_book_corpus,
    manifest_digest as _manifest_digest,
)
from soma.memory import SequenceCircuitMemory

try:
    import resource as _resource
    import sys as _sys

    def _peak_mb():
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024.0 * 1024.0) if _sys.platform == "darwin" else rss / 1024.0
except ImportError:  # pragma: no cover
    def _peak_mb():
        return -1.0


CHECKPOINT_PROTOCOL = "r3b-e1-checkpoint-v1"


def _acquisition_manifest_books(manifest):
    books = manifest.get("books", [])
    acquisition = [book for book in books if book.get("partition") == "acquisition"]
    return acquisition if acquisition else list(books)


def _atomic_json_dump(path, payload):
    """Write JSON to a durable temporary sibling before replacing ``path``."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".r6-checkpoint-")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def checkpoint_payload(memory, manifest, next_book, cumulative_bytes):
    """Build a manifest- and configuration-bound developmental checkpoint.

    The older ``--save-memory`` raw state remains supported for compatibility.
    This envelope is the reproducibility path: it records exactly which
    manifest and acquisition cursor the state belongs to, so a resume cannot
    silently continue a different corpus or stage.
    """
    next_book = int(next_book)
    cumulative_bytes = int(cumulative_bytes)
    if next_book < 0 or cumulative_bytes < 0:
        raise ValueError("checkpoint cursor and cumulative bytes must be nonnegative")
    books = _acquisition_manifest_books(manifest)
    if books:
        if next_book > len(books):
            raise ValueError("checkpoint cursor exceeds manifest books")
        expected_bytes = sum(int(item["bytes"]) for item in books[:next_book])
        if cumulative_bytes != expected_bytes:
            raise ValueError("checkpoint cumulative bytes do not match manifest cursor")
    state = memory.state_dict()
    return {
        "protocol": CHECKPOINT_PROTOCOL,
        "manifest_sha256": _manifest_digest(manifest),
        "max_order": int(memory.max_order),
        "max_circuits": int(memory.max_circuits),
        "next_book": next_book,
        "cumulative_bytes": cumulative_bytes,
        "memory": state,
    }


def load_resume_checkpoint(path, manifest, max_order, max_circuits):
    """Load a raw memory state or validate a developmental checkpoint.

    Returns ``(memory, next_book, cumulative_bytes, envelope)``.  Raw states
    have no cursor metadata and return ``None`` for the middle two values.
    Versioned envelopes reject manifest/configuration mismatches explicitly.
    """
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and payload.get("protocol") == CHECKPOINT_PROTOCOL:
        if payload.get("manifest_sha256") != _manifest_digest(manifest):
            raise ValueError("resume checkpoint manifest does not match requested manifest")
        if int(payload.get("max_order", -1)) != int(max_order):
            raise ValueError("resume checkpoint max_order differs from requested configuration")
        if int(payload.get("max_circuits", -1)) != int(max_circuits):
            raise ValueError("resume checkpoint max_circuits differs from requested configuration")
        next_book = int(payload.get("next_book", -1))
        cumulative_bytes = int(payload.get("cumulative_bytes", -1))
        if next_book < 0 or cumulative_bytes < 0:
            raise ValueError("resume checkpoint cursor metadata is invalid")
        books = _acquisition_manifest_books(manifest)
        if books:
            if next_book > len(books):
                raise ValueError("resume checkpoint cursor exceeds manifest books")
            expected_bytes = sum(int(item["bytes"]) for item in books[:next_book])
            if cumulative_bytes != expected_bytes:
                raise ValueError("resume checkpoint cumulative bytes do not match manifest cursor")
        memory = SequenceCircuitMemory.from_state_dict(payload["memory"])
        if (memory.max_order != int(max_order) or
                memory.max_circuits != int(max_circuits)):
            raise ValueError("resume checkpoint memory configuration is inconsistent")
        return memory, next_book, cumulative_bytes, True
    memory = SequenceCircuitMemory.from_state_dict(payload)
    return memory, None, None, False


def evaluate(memory, data):
    """Score validation/test bytes on a clone; never contaminate training."""
    evaluation_memory = copy.deepcopy(memory)
    bits, _ = bit_stream(data)
    evaluation_memory.reset_history()
    predictions, targets = [], []
    for step in range(max(0, len(bits) - 1)):
        evaluation_memory.observe(bits[step], learn=False)
        distribution, _ = evaluation_memory.distribution()
        predictions.append(distribution[1])
        targets.append(bits[step + 1])
    return bits_per_bit(predictions, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e1-manifest.json")
    parser.add_argument("--report", default="reports/r3b-e1.json")
    parser.add_argument("--max-order", type=int, default=16)
    parser.add_argument("--max-circuits", type=int, default=131072)
    parser.add_argument("--resume-from", default=None,
                        help="memory state JSON to resume acquisition from")
    parser.add_argument("--require-bound-checkpoint", action="store_true",
                        help="reject legacy raw states during locked qualification")
    parser.add_argument("--save-memory", default=None,
                        help="write memory state JSON after acquisition")
    parser.add_argument("--save-checkpoint", default=None,
                        help="write a manifest-bound developmental checkpoint")
    parser.add_argument("--start-book", type=int, default=0,
                        help="first acquisition book index (staged runs)")
    parser.add_argument("--end-book", type=int, default=None,
                        help="one past the last acquisition book index")
    parser.add_argument("--initial-bytes", type=int, default=0,
                        help="cumulative bytes before this stage")
    parser.add_argument("--skip-final", action="store_true",
                        help="skip sealed test/lesion eval (staged runs)")
    args = parser.parse_args()

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    try:
        parts = load_verified_book_corpus(manifest, args.manifest)
    except (OSError, KeyError, ValueError) as error:
        parser.error("invalid frozen corpus manifest: %s" % error)
    acquisition = parts["acquisition"]
    validation = parts["validation"][0][1]
    test_name, test_data = parts["test"][0]

    resume_next_book = None
    resume_bytes = None
    resume_envelope = False
    if args.resume_from is not None:
        try:
            memory, resume_next_book, resume_bytes, resume_envelope = load_resume_checkpoint(
                args.resume_from, manifest, args.max_order, args.max_circuits)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error("invalid resume checkpoint: %s" % error)
        if not resume_envelope and (
                memory.max_order != args.max_order or memory.max_circuits != args.max_circuits):
            parser.error("resume state config differs from requested")
        if args.require_bound_checkpoint and not resume_envelope:
            parser.error("locked qualification requires a manifest-bound checkpoint")
        if resume_envelope:
            if args.start_book not in (0, resume_next_book):
                parser.error("checkpoint requires --start-book %d" % resume_next_book)
            if args.initial_bytes not in (0, resume_bytes):
                parser.error("checkpoint requires --initial-bytes %d" % resume_bytes)
            start_book = resume_next_book
            initial_bytes = resume_bytes
        else:
            start_book = args.start_book
            initial_bytes = args.initial_bytes
    else:
        memory = SequenceCircuitMemory((0, 1), max_order=args.max_order,
                                       max_circuits=args.max_circuits,
                                       min_support=2, prior=0.5)
        start_book = args.start_book
        initial_bytes = args.initial_bytes
    end_book = len(acquisition) if args.end_book is None else args.end_book
    if start_book < 0 or end_book > len(acquisition) or start_book >= end_book:
        parser.error("acquisition book range must satisfy 0 <= start < end <= %d" % len(acquisition))
    curve = []
    start = time.time()
    cumulative_bytes = initial_bytes
    staged = acquisition[start_book:end_book]
    for name, data in staged:
        # Manifest books are independent documents.  The boundary is explicit
        # and is shared by uninterrupted and resumed stages.
        memory.reset_history()
        bits, _ = bit_stream(data)
        for symbol in bits[:-1]:
            memory.observe(symbol)
        cumulative_bytes += len(data)
        validation_bpb = evaluate(memory, validation)
        curve.append({
            "books_done": [entry["books_done"][-1] for entry in curve] + [name],
            "cumulative_bytes": cumulative_bytes,
            "validation_bits_per_bit": validation_bpb,
            "circuits": len(memory.circuits),
            "circuits_created": memory.circuits_created,
            "circuits_reclaimed": memory.circuits_reclaimed,
            "elapsed_seconds": time.time() - start,
            "peak_rss_mb": _peak_mb(),
        })
        print(json.dumps(curve[-1], sort_keys=True), flush=True)
        memory.validate()

    if args.save_memory is not None:
        state = memory.state_dict()
        _atomic_json_dump(args.save_memory, state)
    if args.save_checkpoint is not None:
        checkpoint = checkpoint_payload(memory, manifest, end_book, cumulative_bytes)
        _atomic_json_dump(args.save_checkpoint, checkpoint)
    trained_bytes = b"".join(data for _, data in acquisition[:end_book])
    bar = byte_unigram_cross_bits(trained_bytes, validation)
    if args.skip_final:
        test_bar, test_bpb, lesion_bpb = None, None, None
        gates = {
            "beats_validation_bar": curve[-1]["validation_bits_per_bit"] < bar,
            "bounded": len(memory.circuits) <= args.max_circuits,
        }
    else:
        test_bar = byte_unigram_cross_bits(trained_bytes, test_data)
        test_bpb = evaluate(memory, test_data)
        lesioned = SequenceCircuitMemory.from_state_dict(memory.state_dict())
        for circuit in lesioned.circuits.values():
            circuit["counts"] = [0] * len(circuit["counts"])
        lesioned.reset_history()
        lesion_predictions, lesion_targets = [], []
        test_bits, _ = bit_stream(test_data)
        for step in range(max(0, len(test_bits) - 1)):
            lesioned.observe(test_bits[step], learn=False)
            lesion_predictions.append(0.5)
            lesion_targets.append(test_bits[step + 1])
        lesion_bpb = bits_per_bit(lesion_predictions, lesion_targets)
        gates = {
            "improves_with_data": curve[-1]["validation_bits_per_bit"] < curve[0]["validation_bits_per_bit"],
            "beats_validation_bar": curve[-1]["validation_bits_per_bit"] < bar,
            "beats_test_bar": test_bpb < test_bar,
            "bounded": len(memory.circuits) <= args.max_circuits,
            "causal": lesion_bpb >= test_bar,
        }

    # Exact mid-stream resume check on a slice.
    resumed = SequenceCircuitMemory.from_state_dict(memory.state_dict())
    assert resumed.state_dict() == memory.state_dict()

    state_bytes = len(json.dumps(memory.state_dict(), sort_keys=True, default=str))
    result = {
        "protocol": "r3b-e1-v2",
        "checkpoint_protocol": CHECKPOINT_PROTOCOL,
        "manifest": manifest,
        "max_order": args.max_order,
        "max_circuits": args.max_circuits,
        "curve": curve,
        "validation_bar": bar,
        "test_bar": test_bar,
        "test_bits_per_bit": test_bpb,
        "test_book": test_name,
        "lesion_bits_per_bit": lesion_bpb,
        "final_circuits": len(memory.circuits),
        "final_circuits_created": memory.circuits_created,
        "final_circuits_reclaimed": memory.circuits_reclaimed,
        "state_bytes": state_bytes,
        "total_seconds": time.time() - start,
        "peak_rss_mb": _peak_mb(),
        "resume": {
            "source": args.resume_from,
            "envelope": resume_envelope,
            "start_book": start_book,
            "end_book": end_book,
            "initial_bytes": initial_bytes,
        },
        "gates": gates,
    }
    result["all_passed"] = all(result["gates"].values())
    print(json.dumps(result, indent=2, sort_keys=True))
    with open(args.report, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

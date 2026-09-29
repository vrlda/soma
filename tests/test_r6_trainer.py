import json
import hashlib
import os
import tempfile
import unittest

from r3b_e1_benchmark import (
    _atomic_json_dump,
    checkpoint_payload,
    evaluate,
    load_resume_checkpoint,
    load_verified_book_corpus,
)
from soma.memory import SequenceCircuitMemory


class R6TrainerCheckpointTests(unittest.TestCase):
    def test_checkpoint_json_write_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            _atomic_json_dump(path, {"version": 1, "value": "complete"})
            with open(path) as handle:
                self.assertEqual(json.load(handle)["value"], "complete")
            self.assertEqual([name for name in os.listdir(directory)
                              if name.startswith(".r6-checkpoint-")], [])

    def _memory(self):
        memory = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        for symbol in (0, 1, 1, 0, 1, 0):
            memory.observe(symbol)
        return memory

    def test_manifest_bound_checkpoint_round_trip(self):
        manifest = {"books": [{"bytes": 12}, {"bytes": 8}]}
        memory = self._memory()
        payload = checkpoint_payload(memory, manifest, next_book=1, cumulative_bytes=12)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            restored, next_book, cumulative_bytes, envelope = load_resume_checkpoint(
                handle.name, manifest, 4, 64)
        self.assertTrue(envelope)
        self.assertEqual(next_book, 1)
        self.assertEqual(cumulative_bytes, 12)
        self.assertEqual(restored.state_dict(), memory.state_dict())

    def test_checkpoint_rejects_wrong_manifest(self):
        manifest = {"books": [{"bytes": 12}, {"bytes": 8}]}
        payload = checkpoint_payload(self._memory(), manifest, next_book=1,
                                     cumulative_bytes=12)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_resume_checkpoint(handle.name,
                                       {"books": [{"bytes": 13}, {"bytes": 8}]},
                                       4, 64)

    def test_checkpoint_cursor_counts_acquisition_entries_only(self):
        manifest = {"books": [
            {"partition": "validation", "bytes": 99},
            {"partition": "acquisition", "bytes": 12},
        ]}
        payload = checkpoint_payload(self._memory(), manifest, next_book=1,
                                     cumulative_bytes=12)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            _, next_book, cumulative_bytes, envelope = load_resume_checkpoint(
                handle.name, manifest, 4, 64)
        self.assertTrue(envelope)
        self.assertEqual((next_book, cumulative_bytes), (1, 12))

    def test_checkpoint_rejects_nested_memory_configuration_mismatch(self):
        manifest = {"books": [{"bytes": 12}]}
        payload = checkpoint_payload(self._memory(), manifest, next_book=1,
                                     cumulative_bytes=12)
        payload["memory"]["max_circuits"] = 32
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_resume_checkpoint(handle.name, manifest, 4, 64)

    def test_raw_memory_resume_remains_compatible(self):
        manifest = {"books": [{"bytes": 12}]}
        memory = self._memory()
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(memory.state_dict(), handle)
            handle.flush()
            restored, next_book, cumulative_bytes, envelope = load_resume_checkpoint(
                handle.name, manifest, 4, 64)
        self.assertFalse(envelope)
        self.assertIsNone(next_book)
        self.assertIsNone(cumulative_bytes)
        self.assertEqual(restored.state_dict(), memory.state_dict())

    def test_e1_evaluation_does_not_mutate_training_state(self):
        memory = self._memory()
        before = memory.state_dict()
        evaluate(memory, b"ABCD" * 8)
        self.assertEqual(memory.state_dict(), before)

    def test_checkpoint_continuation_matches_uninterrupted_acquisition(self):
        manifest = {"books": [{"bytes": 5}, {"bytes": 6}]}
        books = ((0, 1, 1, 0, 1), (1, 1, 0, 0, 1, 0))
        full = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        for book in books:
            for symbol in book[:-1]:
                full.observe(symbol)
        staged = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        for symbol in books[0][:-1]:
            staged.observe(symbol)
        payload = checkpoint_payload(staged, manifest, next_book=1, cumulative_bytes=5)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            resumed, next_book, _, _ = load_resume_checkpoint(handle.name, manifest, 4, 64)
        self.assertEqual(next_book, 1)
        for symbol in books[next_book][:-1]:
            resumed.observe(symbol)
        self.assertEqual(resumed.state_dict(), full.state_dict())

    def test_corpus_loader_rejects_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "book.txt")
            raw = b"frozen acquisition bytes"
            with open(path, "wb") as handle:
                handle.write(raw)
            digest = hashlib.sha256(raw).hexdigest()
            manifest = {"books": [{
                "name": "book",
                "path": path,
                "partition": "acquisition",
                "raw_bytes": len(raw),
                "raw_sha256": digest,
                "bytes": len(raw),
                "sha256": digest,
                "boilerplate_stripped": False,
            }]}
            self.assertEqual(load_verified_book_corpus(manifest)["acquisition"][0][1], raw)
            with open(path, "wb") as handle:
                handle.write(raw + b" tampered")
            with self.assertRaisesRegex(ValueError, "raw"):
                load_verified_book_corpus(manifest)


if __name__ == "__main__":
    unittest.main()

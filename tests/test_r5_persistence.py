import json
import os
import tempfile
import unittest

from soma.persistence import (
    is_dirty,
    read_soma,
    recover_if_needed,
    write_soma,
)
from soma.service.store import BrainStore


def _chunks():
    return {
        "manifest.json": b'{"name": "t"}',
        "brain.json": b'{"version": 12}',
        "dialogue.json": b'{"version": 1}',
        "episodic.json": b'{"version": 1}',
    }


class SomaFormatTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.soma")
            info = write_soma(path, _chunks())
            self.assertTrue(info["bytes"] > 0)
            self.assertEqual(read_soma(path), _chunks())

    def test_rejects_wrong_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.soma")
            bad = dict(_chunks())
            del bad["brain.json"]
            with self.assertRaises(ValueError):
                write_soma(path, bad)

    def test_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.soma")
            write_soma(path, _chunks())
            for offset in (4, 20, 60):
                with open(path, "r+b") as handle:
                    handle.seek(offset)
                    byte = handle.read(1)
                    handle.seek(offset)
                    handle.write(bytes([byte[0] ^ 0xFF]))
                with self.assertRaises(ValueError):
                    read_soma(path)
                with open(path, "r+b") as handle:
                    handle.seek(offset)
                    handle.write(byte)

    def test_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "brain.soma")
            write_soma(path, _chunks())
            with open(path, "r+b") as handle:
                handle.truncate(os.path.getsize(path) // 2)
            with self.assertRaises(ValueError):
                read_soma(path)

    def test_rejects_bad_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "x.soma")
            with open(path, "wb") as handle:
                handle.write(b"NOPE0000" + b"\x00" * 64)
            with self.assertRaises(ValueError):
                read_soma(path)


class JournalTests(unittest.TestCase):
    def test_crash_recovery_restores_previous(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.chat import teach_text
            teach_text(store, "demo", b"first version here", provenance="t")
            before = store.inspect("demo")["identity"]
            paths = store._paths("demo")
            from soma.persistence import begin_save, rotate_previous
            begin_save(paths["dir"])
            rotate_previous(paths)
            # Simulate a crash: current files moved aside, nothing rewritten.
            self.assertTrue(is_dirty(paths["dir"]))
            info = store.inspect("demo")
            self.assertEqual(info["identity"], before)
            self.assertFalse(is_dirty(paths["dir"]))

    def test_clean_load_needs_no_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            paths = store._paths("demo")
            self.assertEqual(recover_if_needed(paths), "clean")


class BudgetProfileTests(unittest.TestCase):
    def test_state_budget_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root, state_budget_bytes=10)
            store.create("tiny")
            from soma.service.chat import teach_text
            with self.assertRaises(ValueError):
                teach_text(store, "tiny", b"x" * 100, provenance="t")

    def test_soma_export_import_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.chat import teach_text
            teach_text(store, "demo", b"hello world", provenance="t")
            bundle = os.path.join(root, "demo.soma")
            info = store.export_soma("demo", bundle)
            self.assertTrue(info["bytes"] > 0)
            store2 = BrainStore(os.path.join(root, "store2"))
            imported = store2.import_soma(bundle, "demo")
            self.assertEqual(imported["name"], "demo")
            first = store.load("demo")
            second = store2.load("demo")
            self.assertEqual(first[0].state_dict(), second[0].state_dict())
            self.assertEqual(first[1].state_dict(), second[1].state_dict())
            self.assertEqual(first[3].state_dict(), second[3].state_dict())

    def test_profiler_reports(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.profile import profile_brain
            report = profile_brain(store, "demo", steps=50)
            self.assertEqual(report["steps"], 50)
            self.assertGreater(report["organism_us_per_step"], 0)
            self.assertGreaterEqual(report["max_active_cells"], report["mean_active_cells"])
            self.assertLessEqual(report["mean_active_cells"], report["total_cells"])


if __name__ == "__main__":
    unittest.main()

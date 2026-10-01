import os
import tempfile
import unittest

from soma.memory.engine import EngineError, EngineMixingMemory, engine_available
from soma.memory.mixing import CircuitMixingMemory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = unittest.skipUnless(engine_available(), "Rust engine binary not built")


def alice(start, length):
    with open(os.path.join(ROOT, "data", "e0", "alice.txt"), "rb") as handle:
        return handle.read()[start:start + length]


@ENGINE
class EngineMixingMemoryTests(unittest.TestCase):
    def test_matches_python_reference(self):
        config = {"max_circuits": 4096}
        reference = CircuitMixingMemory(**config)
        with EngineMixingMemory(config=config) as engine:
            for memory in (reference, engine):
                memory.observe_bytes(alice(0, 1500))
                memory.reset_history()
                memory.observe_bytes(alice(2000, 200), learn=False)
                for bit in (1, 0, 1):
                    memory.observe(bit, learn=True, weight=3)
            self.assertEqual(engine.distribution(), reference.distribution())
            self.assertEqual(engine.summary()["circuits"], len(reference.circuits))

    def test_errors_are_reported_not_fatal(self):
        with EngineMixingMemory(config={"max_circuits": 4096}) as engine:
            with self.assertRaises(EngineError):
                engine._request({"op": "bogus"})
            with self.assertRaises(ValueError):
                engine.observe(2)
            engine.observe(1)
            self.assertEqual(set(engine.distribution()[0]), {0, 1})
        with self.assertRaises(EngineError):
            EngineMixingMemory(config={"orders": [3, 1]})
        with self.assertRaises(EngineError):
            EngineMixingMemory(binary="/nonexistent/soma-mixer-serve")

    def test_state_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.somamix")
            with EngineMixingMemory(config={"max_circuits": 4096}, path=path) as engine:
                engine.observe_bytes(alice(0, 1000))
                engine.save()
                expected = engine.distribution()
                reference = engine.state_dict()
            self.assertEqual(reference, {"kind": "circuit-mixing-file", "path": "m.somamix"})
            with EngineMixingMemory.from_state_dict(reference, base_dir=directory) as reloaded:
                self.assertEqual(reloaded.distribution(), expected)


@ENGINE
class EngineBrainServiceTests(unittest.TestCase):
    def test_engine_brain_lifecycle(self):
        from soma.service.chat import chat_turn, correct, teach_text
        from soma.service.store import BrainStore
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(os.path.join(root, "brains"))
            manifest = store.create("demo", memory="mixing", max_circuits=1 << 16)
            self.assertEqual(manifest["memory"], "mixing")
            teach_text(store, "demo", alice(0, 20000))
            correct(store, "demo", "USER the code is ", "kite")
            self.assertTrue(chat_turn(store, "demo", "the code is ").startswith("kite"))
            self.assertEqual(chat_turn(store, "demo", "spell hi"), "h i")
            info = store.inspect("demo")
            self.assertEqual(info["memory"], "mixing")
            self.assertEqual(info["sequence_events"], 20000 * 8)
            self.assertTrue(os.path.exists(os.path.join(store.root, "demo", "memory.somamix")))

            store.clone("demo", "twin", "test")
            self.assertEqual(store.inspect("twin")["sequence_events"], 20000 * 8)
            backup = store.backup("demo", os.path.join(root, "backups"))
            self.assertTrue(os.path.exists(os.path.join(backup, "memory.somamix")))
            soma_path = os.path.join(root, "demo.soma")
            store.export_soma("demo", soma_path)
            imported = store.import_soma(soma_path, "imported")
            self.assertEqual(imported["sequence_circuits"], info["sequence_circuits"])
            self.assertTrue(chat_turn(store, "imported", "the code is ").startswith("kite"))

    def test_crash_recovery_restores_engine_memory(self):
        from soma.persistence import begin_save, is_dirty, rotate_previous
        from soma.service.chat import teach_text
        from soma.service.store import BrainStore
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo", memory="mixing", max_circuits=1 << 16)
            teach_text(store, "demo", alice(0, 4000))
            before = store.inspect("demo")
            paths = store._paths("demo")
            begin_save(paths["dir"])
            rotate_previous(paths)
            self.assertFalse(os.path.exists(paths["memory"]))
            after = store.inspect("demo")
            self.assertEqual(after["identity"], before["identity"])
            self.assertEqual(after["sequence_events"], before["sequence_events"])
            self.assertFalse(is_dirty(paths["dir"]))


if __name__ == "__main__":
    unittest.main()


@ENGINE
class EngineReuseTests(unittest.TestCase):
    def test_live_engine_reused_only_when_clean_and_unchanged(self):
        from soma.memory.engine import reuse_live_engines
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.somamix")
            with EngineMixingMemory(config={"max_circuits": 4096}, path=path) as seed:
                seed.observe_bytes(alice(0, 2000))
                seed.save()
            payload = {"kind": "circuit-mixing-file", "path": "m.somamix"}
            with reuse_live_engines():
                first = EngineMixingMemory.from_state_dict(payload, base_dir=directory)
                first.observe_bytes(b"frozen prompt", learn=False)
                again = EngineMixingMemory.from_state_dict(payload, base_dir=directory)
                self.assertIs(again, first)
                fresh = EngineMixingMemory(path=path)
                fresh.reset_history()
                self.assertEqual(again.distribution(), fresh.distribution())
                fresh.close()
                first.observe_bytes(b"learned")  # dirty: never handed out again
                third = EngineMixingMemory.from_state_dict(payload, base_dir=directory)
                self.assertIsNot(third, first)
                with EngineMixingMemory(path=path) as writer:  # file changes
                    writer.observe_bytes(b"more")
                    writer.save()
                fourth = EngineMixingMemory.from_state_dict(payload, base_dir=directory)
                self.assertIsNot(fourth, third)
            outside = EngineMixingMemory.from_state_dict(payload, base_dir=directory)
            self.assertIsNot(outside, fourth)
            for memory in (first, third, fourth, outside):
                memory.close()

    def test_unchanged_memory_is_linked_not_rewritten(self):
        from soma.service.chat import chat_turn, teach_text
        from soma.service.store import BrainStore
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(os.path.join(root, "brains"))
            store.create("demo", memory="mixing", max_circuits=1 << 16)
            teach_text(store, "demo", alice(0, 5000))
            memory_path = os.path.join(store.root, "demo", "memory.somamix")
            for _ in range(3):  # chat learns only in the dialogue tier
                chat_turn(store, "demo", "hello")
            self.assertTrue(os.path.samefile(memory_path, memory_path + ".prev"))
            with open(memory_path, "rb") as handle:
                before = handle.read()
            teach_text(store, "demo", alice(5000, 2000))
            self.assertFalse(os.path.samefile(memory_path, memory_path + ".prev"))
            with open(memory_path + ".prev", "rb") as handle:
                self.assertEqual(handle.read(), before)
            self.assertEqual(store.inspect("demo")["sequence_events"], 7000 * 8)

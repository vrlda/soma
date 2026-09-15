import json
import os
import tempfile
import unittest
import urllib.request

from soma.service.main import main
from soma.service.store import BrainStore


def _run(*argv, root=None):
    arguments = list(argv)
    if root is not None:
        arguments = ["--root", root] + arguments
    return main(arguments)


class R8ServiceTests(unittest.TestCase):
    def test_store_crud_and_clone(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            self.assertEqual(store.list(), [])
            store.create("a", description="first")
            self.assertEqual(store.list(), ["a"])
            with self.assertRaises(ValueError):
                store.create("a")
            with self.assertRaises(ValueError):
                store.create("../evil")
            manifest = store.clone("a", "b", "test")
            self.assertEqual(sorted(store.list()), ["a", "b"])
            self.assertEqual(manifest["ancestor"], store.inspect("a")["identity"])
            info = store.inspect("b")
            self.assertEqual(info["episodic_entries"], 0)

    def test_teach_correct_forget_chat(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.chat import chat_turn, correct, forget_fact, teach_text
            result = teach_text(store, "demo", b"the code is kite here and there ", provenance="t")
            self.assertEqual(result["bytes"], 32)
            correct(store, "demo", "USER the code is ", "kite")
            reply = chat_turn(store, "demo", "the code is ", max_bytes=6, seed=2)
            self.assertTrue(reply.startswith("kite"))
            forgotten = forget_fact(store, "demo", "USER the code is ")
            self.assertEqual(forgotten["removed"], 1)
            self.assertEqual(forget_fact(store, "demo", "USER nope ")["removed"], 0)

    def test_backup_restore_export_import(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.chat import teach_text
            teach_text(store, "demo", b"hello world", provenance="t")
            with tempfile.TemporaryDirectory() as scratch:
                target = store.backup("demo", scratch)
                self.assertTrue(os.path.isdir(target))
                exported = store.export("demo", os.path.join(scratch, "demo.tgz"))
                self.assertTrue(os.path.exists(exported["path"]))
                store2 = BrainStore(os.path.join(scratch, "store2"))
                info = store2.restore("demo", scratch)
                self.assertEqual(info["name"], "demo")
                store3 = BrainStore(os.path.join(scratch, "store3"))
                info = store3.import_brain(os.path.join(scratch, "demo.tgz"), "demo")
                self.assertEqual(info["name"], "demo")
                with self.assertRaises(ValueError):
                    store3.import_brain(os.path.join(scratch, "demo.tgz"), "demo")

    def test_doctor_ok(self):
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.doctor import doctor
            report = doctor(root)
            self.assertTrue(report["ok"])
            self.assertTrue(report["brains"]["demo"]["valid"])
            self.assertTrue(report["smoke"])

    def test_serve_endpoints(self):
        import threading
        from soma.service.serve import _Handler
        from http.server import ThreadingHTTPServer
        with tempfile.TemporaryDirectory() as root:
            store = BrainStore(root)
            store.create("demo")
            from soma.service.chat import correct
            correct(store, "demo", "USER the code is ", "kite")
            _Handler.store = store
            server = ThreadingHTTPServer(("127.0.0.1", 18923), _Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base = "http://127.0.0.1:18923"
                health = json.load(urllib.request.urlopen(base + "/health"))
                self.assertTrue(health["ok"])
                models = json.load(urllib.request.urlopen(base + "/v1/models"))
                self.assertEqual(models["data"], [{"id": "demo"}])
                request = urllib.request.Request(
                    base + "/v1/chat/completions",
                    data=json.dumps({"model": "demo", "messages": [
                        {"role": "user", "content": "the code is "}]}).encode(),
                    headers={"Content-Type": "application/json"})
                reply = json.load(urllib.request.urlopen(request))
                content = reply["choices"][0]["message"]["content"]
                self.assertTrue(content.startswith("kite"))
                missing = urllib.request.Request(
                    base + "/v1/chat/completions",
                    data=json.dumps({"model": "nope", "messages": []}).encode(),
                    headers={"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(missing)
                    self.fail("expected 404")
                except urllib.error.HTTPError as error:
                    self.assertEqual(error.code, 404)
            finally:
                server.shutdown()
                thread.join()

    def test_cli_dispatch(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(_run("brain-create", "demo", root=root), 0)
            self.assertEqual(_run("brain-list", root=root), 0)
            self.assertEqual(_run("brain-inspect", "demo", root=root), 0)
            self.assertEqual(_run("brain-inspect", "missing", root=root), 1)
            self.assertEqual(_run("doctor", root=root), 0)
            self.assertEqual(_run("transducers", root=root), 0)


if __name__ == "__main__":
    unittest.main()

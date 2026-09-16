import os
import tempfile
import unittest

from soma.evaluation.preference import (
    load_json_list,
    make_pair,
    score,
    synthetic_rater,
    validate_judgment,
)
from soma.service.main import main
from soma.service.telemetry import bundle, record, set_consent, status, summary


class TelemetryTests(unittest.TestCase):
    def test_default_off(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(status(root), {"enabled": False, "content": False})
            self.assertFalse(record(root, "chat", 10, 0))
            self.assertEqual(summary(root)["events"], 0)

    def test_opt_in_records_metadata_only(self):
        with tempfile.TemporaryDirectory() as root:
            set_consent(root, True)
            self.assertTrue(record(root, "chat", 12, 0, brain="demo"))
            self.assertTrue(record(root, "chat", 5, 1, extra={"secret": 1}))
            with open(os.path.join(root, "telemetry.log")) as handle:
                content = handle.read()
            self.assertNotIn("secret", content)
            summary_report = summary(root)
            self.assertEqual(summary_report["events"], 2)
            self.assertEqual(summary_report["errors"], 1)
            set_consent(root, False)
            self.assertFalse(record(root, "chat", 1, 0))

    def test_content_flag(self):
        with tempfile.TemporaryDirectory() as root:
            set_consent(root, True, content=True)
            self.assertTrue(status(root)["content"])
            record(root, "chat", 1, 0, extra={"note": "hi"})
            with open(os.path.join(root, "telemetry.log")) as handle:
                self.assertIn("note", handle.read())

    def test_bundle_has_no_content(self):
        with tempfile.TemporaryDirectory() as root:
            from soma.service.store import BrainStore
            store = BrainStore(os.path.join(root, "brains"))
            store.create("demo")
            set_consent(root, True)
            record(root, "chat", 3, 0, brain="demo")
            path = os.path.join(root, "bundle.tgz")
            bundle(root, path)
            self.assertTrue(os.path.exists(path))
            import tarfile
            with tarfile.open(path) as archive:
                names = archive.getnames()
            self.assertEqual(names, ["bundle.json"])


class PreferenceTests(unittest.TestCase):
    def test_scoring(self):
        pairs = [make_pair("p1", "q", "good answer here", "bad", True),
                 make_pair("p2", "q", "bad", "good answer here", False)]
        judgments = [
            {"id": "p1", "choice": "B", "rater": "r1"},
            {"id": "p1", "choice": "B", "rater": "r2"},
            {"id": "p2", "choice": "B", "rater": "r1"},
            {"id": "p2", "choice": "tie", "rater": "r2"},
        ]
        report = score(pairs, judgments)
        self.assertAlmostEqual(report["win_rate"], (2 + 0.5) / 4)
        self.assertAlmostEqual(report["agreement"], 0.5)
        self.assertEqual(report["raters"], ["r1", "r2"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            validate_judgment({"id": "x", "choice": "A", "rater": "r"}, {"y"})
        with self.assertRaises(ValueError):
            validate_judgment({"id": "x", "choice": "C", "rater": "r"}, {"x"})
        with self.assertRaises(ValueError):
            load_json_list("/nonexistent")

    def test_synthetic_rater(self):
        pair = make_pair("p", "q", "fine", "bad\x00", True)
        self.assertIn(synthetic_rater(pair), ("A", "B", "tie"))

    def test_installer_smoke(self):
        import subprocess
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as root:
            environment = dict(os.environ, SOMA_ROOT=root)
            completed = subprocess.run(
                ["bash", "install.sh"], cwd=repo,
                env=environment, capture_output=True, text=True, timeout=300)
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            self.assertTrue(os.path.isdir(os.path.join(root, "brains")))

    def test_cli_telemetry_commands(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(main(["--root", root, "telemetry", "status"]), 0)
            self.assertEqual(main(["--root", root, "telemetry", "on"]), 0)
            self.assertEqual(main(["--root", root, "telemetry", "off"]), 0)
            self.assertEqual(main(["--root", root, "bundle", os.path.join(root, "b.tgz")]), 0)


if __name__ == "__main__":
    unittest.main()

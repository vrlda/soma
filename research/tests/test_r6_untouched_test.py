import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

import r6_untouched_test as untouched
from soma.evaluation.english import load_verified_book_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class UntouchedGuardTests(unittest.TestCase):
    def setUp(self):
        self.names, self.hashes = untouched.used_content_hashes()

    def test_prior_books_are_rejected_by_name_and_content(self):
        with self.assertRaises(ValueError):
            untouched.check_untouched("timemachine", b"anything", self.names, self.hashes)
        manifest_path = os.path.join(ROOT, "research", "reports", "e2-manifest.json")
        with open(manifest_path) as handle:
            parts = load_verified_book_corpus(json.load(handle), manifest_path)
        jekyll = parts["validation"][0][1]
        with self.assertRaises(ValueError):
            untouched.check_untouched("renamed", jekyll, self.names, self.hashes)

    def test_new_book_is_accepted(self):
        untouched.check_untouched("never-seen", b"fresh text", self.names, self.hashes)

    def test_manifest_keeps_acquisition_and_replaces_test(self):
        with open(os.path.join(ROOT, untouched.BASE_MANIFEST)) as handle:
            base = json.load(handle)
        entry = {"name": "new", "partition": "test", "bytes": 10}
        manifest = untouched.build_manifest(base, entry)
        tests = [b for b in manifest["books"] if b["partition"] == "test"]
        self.assertEqual(tests, [entry])
        self.assertEqual(manifest["acquisition_bytes"], base["acquisition_bytes"])
        self.assertEqual(manifest["validation_bytes"], base["validation_bytes"])

    def test_score_refuses_a_second_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.dirname(os.path.join(root, untouched.REPORT)))
            with open(os.path.join(root, untouched.REPORT), "w") as handle:
                handle.write("{}")
            with mock.patch.object(untouched, "ROOT", root):
                with self.assertRaises(SystemExit):
                    untouched.score(argparse.Namespace())

    def test_prepare_rejects_text_without_gutenberg_boilerplate(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as handle:
            handle.write("[A Book 1908]\n\n\u201cNo boilerplate,\u201d he said.\n".encode("utf-8"))
        try:
            with self.assertRaises(SystemExit):
                untouched.prepare(argparse.Namespace(
                    raw=handle.name, name="ascii-copy", source_url="x"))
        finally:
            os.unlink(handle.name)

    def test_conventions_normalize_crlf_and_refuse_ascii_quotes(self):
        curly = "\u201cHello,\u201d he said.\r\n".encode("utf-8")
        normalized, changed = untouched.normalize_conventions(curly)
        self.assertTrue(changed)
        self.assertNotIn(b"\r\n", normalized)
        same, changed = untouched.normalize_conventions(b"plain\n")
        self.assertFalse(changed)
        with self.assertRaises(SystemExit):
            untouched.normalize_conventions(b'"Hello," he said.\n')

    def test_frozen_config_is_complete(self):
        self.assertTrue(untouched.FROZEN_CONFIG["gate_partial"])
        self.assertFalse(untouched.FROZEN_CONFIG["calibration"])
        self.assertNotIn("max_circuits", untouched.FROZEN_CONFIG)


if __name__ == "__main__":
    unittest.main()

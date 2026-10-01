import json
import os
import random
import tempfile
import time
import unittest
from unittest import mock

import r6_tier_benchmark
from r6_tier_benchmark import (MIN_CONTROL_MARGIN, SuffixNgramControl,
                               CANONICAL_MANIFEST_SHA256,
                               CANONICAL_MANIFEST_TOTALS, TIER_CONFIGS,
                               _aggregate_qualification, _canonical_manifest_match,
                               _evaluate, load_verified_manifest, run_tier)
from soma.memory import SequenceCircuitMemory


class SlowSuffixReference:
    """Pre-optimization reference implementation for semantic lock tests."""

    def __init__(self, order, max_contexts, prior=0.5):
        self.order = order
        self.max_contexts = max_contexts
        self.prior = prior
        self.history = []
        self.contexts = {(): {"counts": [0, 0], "last_used": 0}}
        self.events_seen = 0

    def observe(self, symbol, learn=True):
        contexts = [tuple(self.history[-length:]) if length else ()
                    for length in range(min(self.order, len(self.history)) + 1)]
        if learn:
            for context in contexts:
                if context not in self.contexts:
                    if len(self.contexts) >= self.max_contexts:
                        candidates = [(item["last_used"], len(key), key)
                                      for key, item in self.contexts.items() if key]
                        del self.contexts[min(candidates)[-1]]
                    if len(self.contexts) < self.max_contexts:
                        self.contexts[context] = {"counts": [0, 0],
                                                  "last_used": self.events_seen}
                item = self.contexts.get(context)
                if item is not None:
                    item["counts"][symbol] += 1
                    item["last_used"] = self.events_seen
        self.history.append(symbol)
        if len(self.history) > self.order:
            del self.history[:-self.order]
        self.events_seen += 1

    def state_dict(self):
        return {
            "version": 1,
            "order": self.order,
            "max_contexts": self.max_contexts,
            "prior": self.prior,
            "history": list(self.history),
            "events_seen": self.events_seen,
            "contexts": [{"context": list(context), "counts": list(item["counts"]),
                          "last_used": item["last_used"]}
                         for context, item in sorted(self.contexts.items())],
        }


class R6TierHarnessTests(unittest.TestCase):
    def test_tiers_use_canonical_runtime_presets(self):
        self.assertEqual(TIER_CONFIGS["Micro"]["max_circuits"], 512)
        self.assertEqual(TIER_CONFIGS["Tiny"]["max_circuits"], 16384)
        self.assertEqual(TIER_CONFIGS["Small"]["max_circuits"], 131072)
        self.assertIsNone(TIER_CONFIGS["Micro"]["frozen_test_floor"])
        self.assertIsNone(TIER_CONFIGS["Tiny"]["frozen_test_floor"])
        self.assertIsNotNone(TIER_CONFIGS["Small"]["frozen_test_floor"])

    def test_canonical_manifest_identity_is_frozen(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "reports",
                               "e1-manifest.json")) as handle:
            manifest = json.load(handle)
        self.assertTrue(_canonical_manifest_match(manifest))
        self.assertEqual(CANONICAL_MANIFEST_TOTALS, {
            field: manifest[field] for field in CANONICAL_MANIFEST_TOTALS})
        self.assertEqual(r6_tier_benchmark.manifest_digest(manifest),
                         CANONICAL_MANIFEST_SHA256)
        manifest["test_bytes"] += 1
        self.assertFalse(_canonical_manifest_match(manifest))

    def test_manifest_loader_rejects_aggregate_total_mismatch(self):
        manifest = {"acquisition_bytes": 3, "validation_bytes": 2,
                    "test_bytes": 1}
        parts = {"acquisition": [("a", b"abcd")],
                 "validation": [("v", b"ef")], "test": [("t", b"g")]}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(manifest, handle)
            handle.flush()
            with mock.patch.object(r6_tier_benchmark, "load_verified_book_corpus",
                                   return_value=parts):
                with self.assertRaisesRegex(ValueError, "acquisition_bytes"):
                    load_verified_manifest(handle.name)

    def test_suffix_control_respects_context_budget(self):
        control = SuffixNgramControl(order=3, max_contexts=5)
        for symbol in (0, 1, 1, 0, 1, 0, 0, 1) * 4:
            control.observe(symbol)
        self.assertLessEqual(len(control.contexts), 5)
        self.assertEqual(len(control.history), 3)
        state = control.state_dict()
        self.assertEqual(state["order"], 3)
        self.assertEqual(state["max_contexts"], 5)

    def test_suffix_control_matches_reference_and_roundtrips_index(self):
        source = random.Random(11)
        symbols = [source.getrandbits(1) for _ in range(400)]
        fast = SuffixNgramControl(order=7, max_contexts=19)
        slow = SlowSuffixReference(order=7, max_contexts=19)
        for symbol in symbols:
            fast.observe(symbol)
            slow.observe(symbol)
            self.assertEqual(fast.state_dict(), slow.state_dict())
        restored = SuffixNgramControl.from_state_dict(fast.state_dict())
        self.assertEqual(restored.state_dict(), fast.state_dict())
        for symbol in symbols[:100]:
            fast.observe(symbol)
            restored.observe(symbol)
        self.assertEqual(restored.state_dict(), fast.state_dict())

    def test_suffix_control_index_avoids_reference_scan_regression(self):
        source = random.Random(7)
        symbols = [source.getrandbits(1) for _ in range(5000)]
        fast = SuffixNgramControl(order=12, max_contexts=256)
        started = time.perf_counter()
        for symbol in symbols:
            fast.observe(symbol)
        fast_seconds = time.perf_counter() - started
        slow = SlowSuffixReference(order=12, max_contexts=256)
        started = time.perf_counter()
        for symbol in symbols:
            slow.observe(symbol)
        slow_seconds = time.perf_counter() - started
        self.assertEqual(fast.state_dict(), slow.state_dict())
        self.assertLess(fast_seconds, slow_seconds * 1.5 + 0.05)

    def test_tier_report_has_effect_controls_and_resources(self):
        acquisition = [
            ("a", b"ABCD" * 32),
            ("b", b"ABCD" * 32),
            ("c", b"ABCD" * 32),
        ]
        result = run_tier("Micro", acquisition, b"ABCD" * 16, b"ABCD" * 16)
        self.assertEqual(result["tier"], "Micro")
        self.assertEqual(len(result["curve"]), 3)
        self.assertIn("meaningful_model_gain", result["gates"])
        self.assertIn("beats_suffix_control_margin", result["gates"])
        self.assertIn("test_beats_suffix_control_margin", result["gates"])
        self.assertIn("suffix_control_test_bits_per_bit", result["metrics"])
        self.assertEqual(
            result["gates"]["test_beats_suffix_control_margin"],
            result["metrics"]["test_bits_per_bit"] <=
            result["metrics"]["suffix_control_test_bits_per_bit"] - MIN_CONTROL_MARGIN)
        self.assertGreater(result["metrics"]["state_bytes"], 0)
        self.assertGreater(result["metrics"]["peak_rss_mb"], 0)
        self.assertLessEqual(result["metrics"]["circuits"],
                             result["config"]["max_circuits"])

    def test_manifest_bound_midpoint_resume_is_exact(self):
        acquisition = [("a", b"ABCD" * 8), ("b", b"BCDA" * 8),
                       ("c", b"CDAB" * 8)]
        manifest = {"books": [{"partition": "acquisition", "bytes": len(data)}
                               for _, data in acquisition]}
        result = run_tier("Micro", acquisition, b"ABCD" * 8, b"ABCD" * 8,
                          checkpoint_manifest=manifest)
        self.assertTrue(result["resume"]["manifest_bound"])
        self.assertTrue(result["resume"]["state_digest_equal"])
        self.assertGreater(result["resume"]["resume_continuation_elapsed_seconds"], 0.0)
        self.assertIsNotNone(result["resume"]["full_worker_elapsed_seconds"])

    def test_evaluation_is_observational(self):
        memory = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        for symbol in (0, 1, 1, 0, 1, 0):
            memory.observe(symbol)
        before = memory.state_dict()
        _evaluate(memory, b"ABCD" * 8)
        self.assertEqual(memory.state_dict(), before)
        control = SuffixNgramControl(order=4, max_contexts=64)
        for symbol in (0, 1, 1, 0, 1, 0):
            control.observe(symbol)
        before_control = control.state_dict()
        _evaluate(control, b"ABCD" * 8)
        self.assertEqual(control.state_dict(), before_control)

    def test_intermediate_evaluation_cannot_change_acquisition_state(self):
        books = [b"ABCD" * 8, b"BCDA" * 8, b"CDAB" * 8]
        with_eval = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        without_eval = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        for data in books:
            with_eval.reset_history()
            without_eval.reset_history()
            bits, _ = r6_tier_benchmark.bit_stream(data)
            for symbol in bits[:-1]:
                with_eval.observe(symbol)
                without_eval.observe(symbol)
            _evaluate(with_eval, b"ABCD" * 8)
        self.assertEqual(with_eval.state_dict(), without_eval.state_dict())

    def test_micro_tiny_quality_is_diagnostic_but_execution_is_required(self):
        acquisition = [("a", b"ABCD" * 8), ("b", b"BCDA" * 8), ("c", b"CDAB" * 8)]
        result = run_tier("Micro", acquisition, b"ABCD" * 8, b"ABCD" * 8)
        self.assertIsNone(result["config"]["frozen_test_floor"])
        self.assertIn("quality_passed", result)
        self.assertIn("execution_passed", result)

    def test_qualification_uses_small_quality_and_all_execution_gates(self):
        measured = [
            {"tier": "Micro", "execution_passed": True, "quality_passed": False},
            {"tier": "Tiny", "execution_passed": True, "quality_passed": False},
            {"tier": "Small", "execution_passed": True, "quality_passed": True},
        ]
        aggregate = _aggregate_qualification(measured)
        self.assertTrue(aggregate["execution_gates_passed"])
        self.assertTrue(aggregate["small_quality_gates_passed"])
        self.assertTrue(aggregate["all_gates_passed"])
        measured[0]["execution_passed"] = False
        self.assertFalse(_aggregate_qualification(measured)["all_gates_passed"])

    def test_isolated_worker_is_terminated_at_hard_timeout(self):
        acquisition = [("a", b"ABCD" * 8)]
        with mock.patch.dict(TIER_CONFIGS["Micro"], {"max_seconds": 0}), \
                mock.patch.object(r6_tier_benchmark, "WORKER_GRACE_SECONDS", 0):
            result = r6_tier_benchmark._run_isolated(
                "Micro", acquisition, b"ABCD" * 8, b"ABCD" * 8)
        self.assertFalse(result["all_passed"])
        self.assertIn("timeout", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()

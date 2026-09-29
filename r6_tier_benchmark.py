#!/usr/bin/env python3
"""Locked R6 Micro/Tiny/Small acquisition qualification harness.

The default run uses the frozen E1 public-domain manifest untouched.  Use
``--max-books``/``--validation-bytes``/``--test-bytes`` only for development
smoke runs; those reports are explicitly ineligible for R6 acceptance.

Each tier is evaluated in an isolated worker with fixed order/circuit budgets,
acquisition-fitted byte-unigram and capacity-matched variable-order suffix
controls, resource measurements, and a meaningful-effect gate.  No held-out statistic is used to
fit a baseline or to select a tier configuration.
"""

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import platform
import queue as _queue
import resource
import sys
import tempfile
import time

from r3b_e1_benchmark import (
    checkpoint_payload,
    load_resume_checkpoint,
    _atomic_json_dump,
)
from soma.evaluation.english import (
    bit_stream,
    bits_per_bit,
    byte_unigram_cross_bits,
    load_verified_book_corpus,
    manifest_digest,
)
from soma.memory import SequenceCircuitMemory


PROTOCOL = "r6-tier-v2"
CHECKPOINTS = ("Micro", "Tiny", "Small")
CANONICAL_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "configs", "presets.json")
CANONICAL_MANIFEST_PATH = os.path.join(os.path.dirname(__file__),
                                       "reports", "e1-manifest.json")
# Frozen identity of the exact untouched E1 acquisition/validation/test split.
CANONICAL_MANIFEST_SHA256 = "bb3eec0d2e42e762967169153bb0f007c47e102770a0c97150502df307a04d5b"
CANONICAL_MANIFEST_TOTALS = {
    "acquisition_bytes": 3852331,
    "validation_bytes": 141077,
    "test_bytes": 181481,
}
EXECUTION_GATE_NAMES = (
    "at_least_three_checkpoints",
    "staged_resume_exact",
    "circuit_budget",
    "peak_model_state_budget",
    "peak_suffix_state_budget",
    "rss_budget",
    "time_budget",
)
QUALITY_GATE_NAMES = (
    "meaningful_model_gain",
    "negative_validation_slope",
    "beats_byte_unigram_margin",
    "beats_suffix_control_margin",
    "test_beats_suffix_control_margin",
    "gain_over_suffix_control",
    "test_beats_byte_unigram_margin",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_metadata():
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": os.path.realpath(sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _load_tier_configs():
    with open(CANONICAL_PRESETS_PATH) as handle:
        presets = json.load(handle)["presets"]
    limits = {"micro": (4 * 1024 * 1024, 180),
              "tiny": (16 * 1024 * 1024, 600),
              "small": (32 * 1024 * 1024, 1800)}
    floors = {"micro": None, "tiny": None, "small": 0.45}
    configs = {}
    for name in CHECKPOINTS:
        key = name.lower()
        preset = presets[key]
        state_bytes, max_seconds = limits[key]
        configs[name] = {
            "preset": key,
            "max_order": int(preset["max_order"]),
            "max_circuits": int(preset["max_circuits"]),
            "max_state_bytes": state_bytes,
            "max_rss_mb": int(preset["ram_mb"]),
            "max_seconds": max_seconds,
            "frozen_test_floor": floors[key],
        }
    return configs


TIER_CONFIGS = _load_tier_configs()

# Frozen effect-size bars.  The endpoint must improve materially, not merely
# move by an uncalibrated last<first comparison.
MIN_ENDPOINT_GAIN = 0.002
MIN_CONTROL_MARGIN = 0.001
MIN_SLOPE = -0.0001
WORKER_GRACE_SECONDS = 5
THRESHOLD_RATIONALE = {
    "endpoint_gain": "two-thousandths bits/bit exceeds the locked small-run measurement jitter bar",
    "control_margin": "one-thousandth bits/bit is a predeclared practical separation from matched controls",
    "slope": "negative least-squares trend across at least three fixed book checkpoints",
    "variance_handling": "ordered multi-checkpoint slope plus endpoint and control margins must all agree; no result-dependent threshold tuning",
}


def _peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def load_verified_manifest(path):
    with open(path) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, path)
    measured = {
        "acquisition_bytes": sum(len(data) for _, data in parts["acquisition"]),
        "validation_bytes": sum(len(data) for _, data in parts["validation"]),
        "test_bytes": sum(len(data) for _, data in parts["test"]),
    }
    for field, actual in measured.items():
        if field not in manifest:
            raise ValueError("manifest is missing aggregate %s" % field)
        try:
            expected = int(manifest[field])
        except (TypeError, ValueError):
            raise ValueError("manifest aggregate %s is not an integer" % field)
        if expected != actual:
            raise ValueError("manifest aggregate %s mismatch: %s != %s" %
                             (field, expected, actual))
    return manifest, parts


def _canonical_manifest_match(manifest):
    if manifest_digest(manifest) != CANONICAL_MANIFEST_SHA256:
        return False
    try:
        return all(int(manifest[field]) == value
                   for field, value in CANONICAL_MANIFEST_TOTALS.items())
    except (KeyError, TypeError, ValueError):
        return False


class SuffixNgramControl:
    """Conventional variable-order suffix table with a matched budget.

    It stores the same family of suffix contexts and the same hard resident
    context ceiling as the learned memory, but uses ordinary count-table LRU
    replacement without SOMA's grace-window/reuse policy.  This makes the
    control capacity-matched without treating the learned implementation as
    its own baseline.
    """

    VERSION = 1

    def __init__(self, order, max_contexts, prior=0.5):
        if int(order) < 1 or int(max_contexts) < 2:
            raise ValueError("suffix context bounds are invalid")
        self.order = int(order)
        self.max_contexts = int(max_contexts)
        self.prior = float(prior)
        self.history = []
        self.contexts = {(): {"counts": [0, 0], "last_used": 0}}
        # Indexed min-heap keyed exactly like the former reference scan.
        # Position bookkeeping keeps it bounded by the resident contexts and
        # makes eviction/update O(log n), retaining deterministic
        # (last_used, length, context) ties.
        self._eviction_heap = []
        self._eviction_positions = {}
        self.events_seen = 0

    def reset_history(self):
        self.history = []

    def _touch_context(self, context, item):
        if context:
            entry = (item["last_used"], len(context), context)
            position = self._eviction_positions.get(context)
            if position is None:
                self._eviction_positions[context] = len(self._eviction_heap)
                self._eviction_heap.append(entry)
                self._heap_sift_up(len(self._eviction_heap) - 1)
            else:
                previous = self._eviction_heap[position]
                self._eviction_heap[position] = entry
                if entry < previous:
                    self._heap_sift_up(position)
                else:
                    self._heap_sift_down(position)

    def _heap_swap(self, left, right):
        self._eviction_heap[left], self._eviction_heap[right] = (
            self._eviction_heap[right], self._eviction_heap[left])
        self._eviction_positions[self._eviction_heap[left][2]] = left
        self._eviction_positions[self._eviction_heap[right][2]] = right

    def _heap_sift_up(self, position):
        while position:
            parent = (position - 1) // 2
            if self._eviction_heap[parent] <= self._eviction_heap[position]:
                break
            self._heap_swap(parent, position)
            position = parent

    def _heap_sift_down(self, position):
        length = len(self._eviction_heap)
        while True:
            left = position * 2 + 1
            right = left + 1
            smallest = position
            if left < length and self._eviction_heap[left] < self._eviction_heap[smallest]:
                smallest = left
            if right < length and self._eviction_heap[right] < self._eviction_heap[smallest]:
                smallest = right
            if smallest == position:
                return
            self._heap_swap(position, smallest)
            position = smallest

    def _evict_one(self):
        if not self._eviction_heap:
            raise RuntimeError("suffix context eviction index is inconsistent")
        context = self._eviction_heap[0][2]
        del self.contexts[context]
        del self._eviction_positions[context]
        replacement = self._eviction_heap.pop()
        if self._eviction_heap:
            self._eviction_heap[0] = replacement
            self._eviction_positions[replacement[2]] = 0
            self._heap_sift_down(0)

    def observe(self, symbol, learn=True):
        symbol = int(symbol)
        if symbol not in (0, 1):
            raise ValueError("suffix context symbols must be bits")
        contexts = [tuple(self.history[-length:]) if length else ()
                    for length in range(min(self.order, len(self.history)) + 1)]
        if learn:
            for context in contexts:
                if context not in self.contexts:
                    if len(self.contexts) >= self.max_contexts:
                        self._evict_one()
                    if len(self.contexts) < self.max_contexts:
                        self.contexts[context] = {"counts": [0, 0],
                                                  "last_used": self.events_seen}
                item = self.contexts.get(context)
                if item is not None:
                    item["counts"][symbol] += 1
                    item["last_used"] = self.events_seen
                    self._touch_context(context, item)
        self.history.append(symbol)
        if len(self.history) > self.order:
            del self.history[:-self.order]
        self.events_seen += 1

    def distribution(self):
        chosen = self.contexts[()]
        chosen_order = 0
        for length in range(min(self.order, len(self.history)), 0, -1):
            context = tuple(self.history[-length:])
            item = self.contexts.get(context)
            if item is not None and sum(item["counts"]) >= 2:
                chosen = item
                chosen_order = length
                break
        counts = chosen["counts"]
        denominator = sum(counts) + self.prior * 2.0
        return {0: (counts[0] + self.prior) / denominator,
                1: (counts[1] + self.prior) / denominator}, chosen_order

    def state_dict(self):
        return {
            "version": self.VERSION,
            "order": self.order,
            "max_contexts": self.max_contexts,
            "prior": self.prior,
            "history": list(self.history),
            "events_seen": self.events_seen,
            "contexts": [{"context": list(context), "counts": list(item["counts"]),
                          "last_used": item["last_used"]}
                         for context, item in sorted(self.contexts.items())],
        }

    @classmethod
    def from_state_dict(cls, state):
        """Restore canonical state and rebuild the non-persisted LRU index."""
        if not isinstance(state, dict) or int(state.get("version", -1)) != cls.VERSION:
            raise ValueError("unsupported suffix control state")
        control = cls(state["order"], state["max_contexts"], state["prior"])
        control.history = [int(symbol) for symbol in state.get("history", [])]
        control.events_seen = int(state.get("events_seen", 0))
        contexts = {}
        for entry in state.get("contexts", []):
            context = tuple(int(symbol) for symbol in entry["context"])
            if context in contexts or len(context) > control.order:
                raise ValueError("invalid suffix context state")
            counts = [int(value) for value in entry["counts"]]
            if len(counts) != 2 or any(value < 0 for value in counts):
                raise ValueError("invalid suffix context counts")
            last_used = int(entry["last_used"])
            if last_used < 0:
                raise ValueError("invalid suffix context recency")
            contexts[context] = {"counts": counts, "last_used": last_used}
        if () not in contexts or len(contexts) > control.max_contexts:
            raise ValueError("invalid suffix context table")
        control.contexts = contexts
        control._eviction_heap = []
        control._eviction_positions = {}
        for context, item in contexts.items():
            control._touch_context(context, item)
        return control


def _evaluate(memory, data):
    """Score a stream without changing acquisition state.

    Evaluation uses a state clone because ``observe(..., learn=False)`` still
    advances history and event clocks.  Resetting the clone gives each
    document a deliberate boundary while leaving the training object exactly
    as it was before the score.
    """
    evaluation_memory = copy.deepcopy(memory)
    bits, _ = bit_stream(data)
    evaluation_memory.reset_history()
    predictions, targets = [], []
    for index in range(max(0, len(bits) - 1)):
        evaluation_memory.observe(bits[index], learn=False)
        predictions.append(evaluation_memory.distribution()[0][1])
        targets.append(bits[index + 1])
    return bits_per_bit(predictions, targets)


def _train(memory, acquisition):
    for _, data in acquisition:
        memory.reset_history()
        bits, _ = bit_stream(data)
        for symbol in bits[:-1]:
            memory.observe(symbol)


def _state_bytes(memory):
    return len(json.dumps(memory.state_dict(), sort_keys=True,
                          separators=(",", ":"), default=str).encode("utf-8"))


def _state_digest(memory):
    payload = json.dumps(memory.state_dict(), sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _slope(values):
    if len(values) < 2:
        return 0.0
    mean_x = (len(values) - 1) / 2.0
    mean_y = sum(values) / float(len(values))
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    return sum((index - mean_x) * (value - mean_y)
               for index, value in enumerate(values)) / denominator


def run_tier(tier, acquisition, validation, test_data, checkpoint_manifest=None):
    config = dict(TIER_CONFIGS[tier])
    started = time.perf_counter()
    memory = SequenceCircuitMemory((0, 1), max_order=config["max_order"],
                                   max_circuits=config["max_circuits"],
                                   min_support=2, prior=0.5)
    suffix = SuffixNgramControl(config["max_order"], config["max_circuits"])
    curve = []
    seen = []
    peak_model_state_bytes = 0
    peak_suffix_state_bytes = 0
    checkpoint_payload_value = None
    checkpoint_index = max(1, len(acquisition) // 2)
    for name, data in acquisition:
        seen.append(data)
        # Books are independent manifest documents; never carry a validation
        # or prior-book suffix across this boundary.
        memory.reset_history()
        suffix.reset_history()
        bits, _ = bit_stream(data)
        for symbol in bits[:-1]:
            memory.observe(symbol)
            suffix.observe(symbol)
        train_data = b"".join(seen)
        model_state_bytes = _state_bytes(memory)
        suffix_state_bytes = _state_bytes(suffix)
        peak_model_state_bytes = max(peak_model_state_bytes, model_state_bytes)
        peak_suffix_state_bytes = max(peak_suffix_state_bytes, suffix_state_bytes)
        curve.append({
            "book": name,
            "cumulative_bytes": len(train_data),
            "model_validation_bits_per_bit": _evaluate(memory, validation),
            "suffix_control_validation_bits_per_bit": _evaluate(suffix, validation),
            "byte_unigram_validation_bits_per_bit": byte_unigram_cross_bits(train_data, validation),
            "circuits": len(memory.circuits),
            "suffix_contexts": len(suffix.contexts),
            "model_state_bytes": model_state_bytes,
            "suffix_state_bytes": suffix_state_bytes,
        })
        memory.validate()
        if checkpoint_manifest is not None and len(seen) == checkpoint_index:
            checkpoint_payload_value = checkpoint_payload(
                memory, checkpoint_manifest, len(seen), len(train_data))
    train_data = b"".join(seen)
    final_training_digest = _state_digest(memory)
    model_test = _evaluate(memory, test_data)
    suffix_test = _evaluate(suffix, test_data)
    byte_test = byte_unigram_cross_bits(train_data, test_data)
    model_validation = [row["model_validation_bits_per_bit"] for row in curve]
    suffix_validation = [row["suffix_control_validation_bits_per_bit"] for row in curve]
    byte_validation = [row["byte_unigram_validation_bits_per_bit"] for row in curve]
    model_gain = model_validation[0] - model_validation[-1]
    suffix_gain = suffix_validation[0] - suffix_validation[-1]
    resume_info = {
        "manifest_bound": checkpoint_payload_value is not None,
        "state_digest_equal": False,
        "checkpoint_index": None,
        "full_worker_elapsed_seconds": None,
        "resume_continuation_elapsed_seconds": None,
    }
    if checkpoint_payload_value is not None:
        resume_info["checkpoint_index"] = checkpoint_index
        with tempfile.TemporaryDirectory(prefix="r6-tier-resume-") as directory:
            path = os.path.join(directory, "checkpoint.json")
            _atomic_json_dump(path, checkpoint_payload_value)
            resumed, next_book, _, envelope = load_resume_checkpoint(
                path, checkpoint_manifest, config["max_order"], config["max_circuits"])
            resume_started = time.perf_counter()
            for _, data in acquisition[next_book:]:
                resumed.reset_history()
                bits, _ = bit_stream(data)
                for symbol in bits[:-1]:
                    resumed.observe(symbol)
            resume_info["resume_continuation_elapsed_seconds"] = time.perf_counter() - resume_started
            resume_info["state_digest_equal"] = bool(
                envelope and _state_digest(resumed) == final_training_digest)
        resume_info["full_worker_elapsed_seconds"] = time.perf_counter() - started
    gates = {
        "at_least_three_checkpoints": len(curve) >= 3,
        "meaningful_model_gain": model_gain >= MIN_ENDPOINT_GAIN,
        "negative_validation_slope": _slope(model_validation) <= MIN_SLOPE,
        "beats_byte_unigram_margin": model_validation[-1] <= byte_validation[-1] - MIN_CONTROL_MARGIN,
        "beats_suffix_control_margin": model_validation[-1] <= suffix_validation[-1] - MIN_CONTROL_MARGIN,
        "test_beats_suffix_control_margin": model_test <= suffix_test - MIN_CONTROL_MARGIN,
        "gain_over_suffix_control": model_gain - max(0.0, suffix_gain) >= MIN_CONTROL_MARGIN,
        "test_beats_byte_unigram_margin": model_test <= byte_test - MIN_CONTROL_MARGIN,
        "staged_resume_exact": resume_info["state_digest_equal"],
        "circuit_budget": len(memory.circuits) <= config["max_circuits"],
        "peak_model_state_budget": peak_model_state_bytes <= config["max_state_bytes"],
        "peak_suffix_state_budget": peak_suffix_state_bytes <= config["max_state_bytes"],
    }
    if config["frozen_test_floor"] is not None:
        gates["frozen_test_floor"] = model_test < config["frozen_test_floor"]
    elapsed = time.perf_counter() - started
    gates["rss_budget"] = _peak_rss_mb() <= config["max_rss_mb"]
    gates["time_budget"] = elapsed <= config["max_seconds"]
    execution_passed = all(gates[name] for name in EXECUTION_GATE_NAMES)
    quality_passed = all(gates[name] for name in QUALITY_GATE_NAMES)
    if config["frozen_test_floor"] is not None:
        quality_passed = quality_passed and gates["frozen_test_floor"]
    return {
        "tier": tier,
        "config": config,
        "curve": curve,
        "metrics": {
            "validation_bits_per_bit": model_validation[-1],
            "test_bits_per_bit": model_test,
            "suffix_control_validation_bits_per_bit": suffix_validation[-1],
            "suffix_control_test_bits_per_bit": suffix_test,
            "byte_unigram_validation_bits_per_bit": byte_validation[-1],
            "byte_unigram_test_bits_per_bit": byte_test,
            "model_gain": model_gain,
            "suffix_control_gain": suffix_gain,
            "gain_over_suffix_control": model_gain - max(0.0, suffix_gain),
            "validation_slope": _slope(model_validation),
            "circuits": len(memory.circuits),
            "suffix_contexts": len(suffix.contexts),
            "state_bytes": _state_bytes(memory),
            "peak_model_state_bytes": peak_model_state_bytes,
            "peak_suffix_state_bytes": peak_suffix_state_bytes,
            "peak_rss_mb": _peak_rss_mb(),
            "elapsed_seconds": elapsed,
        },
        "resume": resume_info,
        "gates": gates,
        "gate_groups": {
            "execution": list(EXECUTION_GATE_NAMES),
            "quality": list(QUALITY_GATE_NAMES) + (
                ["frozen_test_floor"] if config["frozen_test_floor"] is not None else []),
        },
        "execution_passed": execution_passed,
        "quality_passed": quality_passed,
        # Micro/Tiny quality is diagnostic only.  Their execution/resource
        # gates remain required for qualification aggregation below.
        "all_passed": execution_passed and (
            quality_passed if config["frozen_test_floor"] is not None else True),
    }


def _worker(queue, tier, acquisition, validation, test_data, checkpoint_manifest):
    try:
        queue.put(run_tier(tier, acquisition, validation, test_data, checkpoint_manifest))
    except BaseException as error:  # parent turns this into a failed report
        queue.put({"tier": tier, "error": "%s: %s" % (type(error).__name__, error),
                   "all_passed": False})


def _run_isolated(tier, acquisition, validation, test_data, checkpoint_manifest=None):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker,
                              args=(queue, tier, acquisition, validation, test_data,
                                    checkpoint_manifest))
    process.start()
    timeout = TIER_CONFIGS[tier]["max_seconds"] + WORKER_GRACE_SECONDS
    process.join(timeout)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(WORKER_GRACE_SECONDS)
    try:
        result = queue.get(timeout=2.0)
    except _queue.Empty:
        reason = "tier worker timeout" if timed_out else "tier worker exited without a report"
        result = {"tier": tier, "error": reason,
                  "all_passed": False}
    if timed_out:
        result["error"] = "tier worker timeout after %s seconds" % timeout
        result["all_passed"] = False
    elif process.exitcode != 0 and "error" not in result:
        result["error"] = "tier worker exit code %s" % process.exitcode
        result["all_passed"] = False
    return result


def _aggregate_qualification(results):
    """Apply the R6 plan's tier roles to measured gate groups.

    Every tier must execute within its resource/resume envelope.  Only Small's
    quality/control group is an acceptance gate; Micro/Tiny quality numbers are
    diagnostic and cannot veto qualification.
    """
    execution_gates_passed = all(result.get("execution_passed", False)
                                 for result in results)
    small = next(item for item in results if item.get("tier") == "Small")
    small_quality_gates_passed = bool(small.get("quality_passed", False))
    return {
        "execution_gates_passed": execution_gates_passed,
        "small_quality_gates_passed": small_quality_gates_passed,
        "all_gates_passed": execution_gates_passed and small_quality_gates_passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/e1-manifest.json")
    parser.add_argument("--report", default="reports/r6-tier.json")
    parser.add_argument("--max-books", type=int, default=None,
                        help="development-only acquisition prefix")
    parser.add_argument("--acquisition-bytes", type=int, default=None,
                        help="development-only acquisition byte prefix")
    parser.add_argument("--validation-bytes", type=int, default=None,
                        help="development-only validation prefix")
    parser.add_argument("--test-bytes", type=int, default=None,
                        help="development-only test prefix")
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in
           (args.max_books, args.acquisition_bytes, args.validation_bytes, args.test_bytes)):
        parser.error("development subset sizes must be positive")
    try:
        manifest, parts = load_verified_manifest(args.manifest)
    except (OSError, KeyError, ValueError) as error:
        parser.error("invalid frozen manifest: %s" % error)
    acquisition = parts["acquisition"]
    validation = b"".join(data for _, data in parts["validation"])
    test_data = b"".join(data for _, data in parts["test"])
    full = (args.max_books is None and args.acquisition_bytes is None and
            args.validation_bytes is None and args.test_bytes is None)
    if args.max_books is not None:
        acquisition = acquisition[:args.max_books]
    if args.acquisition_bytes is not None:
        remaining = args.acquisition_bytes
        limited = []
        for name, data in acquisition:
            if remaining <= 0:
                break
            chunk = data[:remaining]
            limited.append((name, chunk))
            remaining -= len(chunk)
        acquisition = limited
    if args.validation_bytes is not None:
        validation = validation[:args.validation_bytes]
    if args.test_bytes is not None:
        test_data = test_data[:args.test_bytes]
    if not acquisition or not validation or not test_data:
        parser.error("manifest subsets must retain acquisition, validation, and test bytes")
    checkpoint_manifest = manifest if full else None
    results = [_run_isolated(tier, acquisition, validation, test_data,
                             checkpoint_manifest)
               for tier in CHECKPOINTS]
    aggregate = _aggregate_qualification(results)
    canonical_manifest = _canonical_manifest_match(manifest)
    eligible_for_r6 = bool(full and canonical_manifest)
    if not full:
        acceptance_scope = "development-smoke"
        ineligibility_reason = "development subset requested"
    elif not canonical_manifest:
        acceptance_scope = "noncanonical-full-manifest"
        ineligibility_reason = "full manifest is not the frozen canonical E1 manifest"
    else:
        acceptance_scope = "untouched-manifest"
        ineligibility_reason = None
    with open(CANONICAL_PRESETS_PATH, "rb") as preset_handle:
        preset_sha256 = hashlib.sha256(preset_handle.read()).hexdigest()
    result = {
        "protocol": PROTOCOL,
        "harness_sha256": _sha256_file(__file__),
        "preset_path": CANONICAL_PRESETS_PATH,
        "preset_sha256": preset_sha256,
        "manifest_sha256": manifest_digest(manifest),
        "manifest_identity": {
            "canonical_path": CANONICAL_MANIFEST_PATH,
            "canonical_sha256": CANONICAL_MANIFEST_SHA256,
            "canonical_totals": CANONICAL_MANIFEST_TOTALS,
            "matches_canonical": canonical_manifest,
        },
        "environment": _environment_metadata(),
        "thresholds": {
            "endpoint_gain": MIN_ENDPOINT_GAIN,
            "control_margin": MIN_CONTROL_MARGIN,
            "slope": MIN_SLOPE,
            "rationale": THRESHOLD_RATIONALE,
        },
        "manifest": manifest,
        "acceptance": {
            "scope": acceptance_scope,
            "eligible_for_r6": eligible_for_r6,
            "ineligibility_reason": ineligibility_reason,
            "qualification_passed": bool(eligible_for_r6 and aggregate["all_gates_passed"]),
        },
        "subset": {
            "acquisition_books": len(acquisition),
            "acquisition_bytes": sum(len(data) for _, data in acquisition),
            "validation_bytes": len(validation),
            "test_bytes": len(test_data),
        },
        "tiers": {item["tier"]: item for item in results},
        **aggregate,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    with open(args.report, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if eligible_for_r6 and aggregate["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

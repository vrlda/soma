"""Non-gating English byte/event pilot for the R6 mechanism review.

This fixture is intentionally small and development-only.  It streams frozen
UTF-8 bytes through the existing TextBytesTransducer and EventBridge, trains
the generic Organism on acquisition bytes only, and scores validation/test
bytes from independent read-only checkpoint clones.  It does not alter or
qualify the locked r6-tier-v2 evidence.
"""

from __future__ import annotations


import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from r6_tier_benchmark import SuffixNgramControl
from soma.events import Event, EventBridge
from soma.events.envelope import OutcomeEvent
from soma.evaluation.english import (
    bit_stream,
    bits_per_bit,
    byte_unigram_cross_bits,
    make_english_organism,
)
from soma.organism import Organism
from soma.transducers.text_bytes import TextBytesTransducer, encode_bytes, validate_utf8


PROTOCOL = "r6-english-event-pilot-v1"
MODES = ("variable_order", "nonlinear_additive_context")
K3_KEY = "k3:input-000|input-001|input-002"

# Frozen, public-domain-style development text.  Partitions are deliberately
# separate byte strings; no validation/test bytes are used for training or
# control fitting.
ACQUISITION_DOCUMENTS = (
    b"The river moved quietly beneath the morning bridge. The lanterns faded as the town woke.\n",
    b"A careful keeper marked each door, copied every number, and returned before the rain.\n",
    b"Beyond the hill, a narrow road crossed the field and joined the old market path.\n",
    b"Small changes in weather altered the garden, but the patient work continued each day.\n",
)
VALIDATION_BYTES = b"The quiet road crossed the garden before the morning rain.\n"
TEST_BYTES = b"A careful river keeper marked the old bridge beyond the hill.\n"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _state_bytes(state: Mapping[str, object]) -> int:
    return len(json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _nll(predictions: Sequence[float], targets: Sequence[int]) -> float:
    return bits_per_bit(predictions, targets)


def _envelopes(bits: Sequence[int], ends: Sequence[int], index: int, event_id: int) -> List[dict]:
    clock = float(index)
    return [
        Event("bit", "english-pilot", event_id, clock, {"value": float(bits[index])}, {"trust": "signed-base"}).to_dict(),
        Event("boundary", "english-pilot", event_id, clock, {"byte_end": float(ends[index])}, {"trust": "signed-base"}).to_dict(),
    ]


def _new_organism(seed: int, mode: str) -> Organism:
    if mode not in MODES:
        raise ValueError("unsupported English pilot mode: %s" % mode)
    organism = make_english_organism(int(seed), frozen=False, input_size=3)
    if mode == "variable_order":
        organism.enable_variable_order_learning(int(seed) + 7919, max_order=3)
    # Keep mode comparison about the generic variable-order substrate, with
    # identical actor/context topology and exploration stream.
    organism.actor_learning_rate = 0.10
    return organism


def _train(organism: Organism, data: bytes, seed: int) -> Tuple[dict, Dict[str, object]]:
    validate_utf8(data)
    bits, ends = bit_stream(data)
    if len(bits) < 2:
        raise ValueError("pilot acquisition must contain at least two bits")
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xE11).uniform(-1.0, 1.0) for _ in range(len(bits))))
    bridge = EventBridge(organism, TextBytesTransducer(), novelty=0.10, exploration=0.20)
    predictions: List[float] = []
    targets: List[int] = []
    peak_state = 0
    for index in range(len(bits) - 1):
        action = bridge.ingest(_envelopes(bits, ends, index, index))
        prediction = float(action["proposal"]["value"])
        target = int(bits[index + 1])
        predictions.append(prediction)
        targets.append(target)
        # State serialization is intentionally sampled, not performed on
        # every bit; the checkpoint itself remains the exact resource record.
        if index % 128 == 0:
            peak_state = max(peak_state, _state_bytes(organism.state_dict()))
        reward = 1.0 - abs(prediction - target)
        bridge.outcome(OutcomeEvent(action["correlation_id"], float(index) + 0.5, reward, "english-pilot").to_dict())
    # The final action is intentionally not requested: the final acquisition
    # bit has no next-bit target.
    bridge.close()
    state = organism.state_dict()
    return state, {
        "acquisition_bits": len(predictions),
        "acquisition_nll_bits_per_bit": _nll(predictions, targets),
        "peak_state_bytes": peak_state,
        "checkpoint_state_bytes": _state_bytes(state),
        "cells": len(state["cells"]),
        "synapses": len(state["synapses"]),
        "variable_order_install_count": int(state.get("variable_order_install_count", 0)),
    }


def _frozen_score(checkpoint: Mapping[str, object], data: bytes, seed: int) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Score bytes from independent clones; never deliver target-derived reward.

    One clone handles one byte so the transducer's eight-bit framing remains
    intact while keeping this MacBook-sized pilot practical.  Zero outcomes
    are supplied only to satisfy EventBridge's delayed-action contract; they
    contain no target or evaluation statistic, and the clone is discarded
    after the byte.
    """
    validate_utf8(data)
    bits, ends = bit_stream(data)
    source_digest = _digest(checkpoint)
    predictions: List[float] = []
    targets: List[int] = []
    clone_count = 0
    for byte_start in range(0, len(bits), 8):
        clone = json.loads(json.dumps(checkpoint))
        organism = Organism.from_state_dict(clone)
        # Each byte gets a fresh bridge and clone.  A fixed local tape keeps
        # score noise independent of prior evaluation bytes.
        organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xF00D + byte_start).uniform(-1.0, 1.0) for _ in range(8)))
        bridge = EventBridge(organism, TextBytesTransducer(), novelty=0.10, exploration=0.20)
        for local_index in range(8):
            index = byte_start + local_index
            if index >= len(bits) - 1:
                break
            action = bridge.ingest(_envelopes(bits, ends, index, local_index))
            predictions.append(float(action["proposal"]["value"]))
            targets.append(int(bits[index + 1]))
            if local_index < 7 and index + 1 < len(bits):
                bridge.outcome(OutcomeEvent(action["correlation_id"], float(local_index) + 0.5, 0.0, "english-pilot").to_dict())
        clone_count += 1
        if _digest(checkpoint) != source_digest:
            raise AssertionError("frozen English evaluation mutated its source checkpoint")
    return {"predictions": predictions, "targets": targets}, {
        "nll_bits_per_bit": _nll(predictions, targets),
        "checkpoint_unchanged": _digest(checkpoint) == source_digest,
        "frozen_read_only": True,
        "bits": len(targets),
        "independent_clone_count": clone_count,
    }


def _lesion_product_path(state: Mapping[str, object]) -> Tuple[dict, Optional[str]]:
    cloned = json.loads(json.dumps(state))
    owners = dict(cloned.get("variable_order_feature_owners", {}))
    # Prefer the ternary path required by the mechanism review, but retain a
    # causal lesion for a lower-order product if this small corpus does not
    # install k3 under a particular seed.
    feature_key = K3_KEY if K3_KEY in owners else next(iter(sorted(owners)), None)
    owner_id = owners.get(feature_key) if feature_key is not None else None
    if owner_id is None or feature_key is None:
        return cloned, None
    owner = dict(cloned.get("motor_modules", {}).get(owner_id, {}))
    owner_cell = owner.get("cell_id")
    feature_sources = tuple(feature_key.split(":", 1)[1].split("|"))
    feature_cell = None
    for cell_id, payload in dict(cloned.get("cells", {})).items():
        if payload.get("activation_type") in ("dendritic_product", "dendritic_product_n") and tuple(payload.get("dendritic_sources", ())) == feature_sources:
            feature_cell = cell_id
            break
    if feature_cell is None or owner_cell is None:
        return cloned, None
    changed = False
    for edge in cloned.get("synapses", []):
        if edge.get("source") == feature_cell and edge.get("destination") == owner_cell:
            edge["strength"] = 0.0
            edge["plasticity"] = 0.0
            changed = True
    return cloned, feature_key if changed else None


def _suffix_train(data: bytes, order: int = 3, max_contexts: int = 4096) -> SuffixNgramControl:
    control = SuffixNgramControl(order, max_contexts)
    bits, _ = bit_stream(data)
    control.reset_history()
    for bit in bits[:-1]:
        control.observe(bit)
    return control


def _suffix_score(control: SuffixNgramControl, data: bytes) -> float:
    evaluation = SuffixNgramControl.from_state_dict(control.state_dict())
    evaluation.reset_history()
    bits, _ = bit_stream(data)
    predictions, targets = [], []
    for index in range(len(bits) - 1):
        evaluation.observe(bits[index], learn=False)
        predictions.append(evaluation.distribution()[0][1])
        targets.append(bits[index + 1])
    return _nll(predictions, targets)


def _dependency_metadata() -> Dict[str, object]:
    files = [
        "soma/organism.py", "soma/events/bridge.py", "soma/events/envelope.py",
        "soma/events/channel.py", "soma/transducers/sdk.py", "soma/transducers/text_bytes.py",
        "soma/evaluation/english.py", "research/benchmarks/r6_tier_benchmark.py",
    ]
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *files], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"sha256": hashes, "git_revision": revision, "dirty": dirty}


def run_english_pilot(seeds: Sequence[int] = (0, 1, 2, 3)) -> Dict[str, object]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("at least one pilot seed is required")
    started = time.monotonic()
    acquisition = b"".join(ACQUISITION_DOCUMENTS)
    validate_utf8(acquisition)
    runs: Dict[str, List[Dict[str, object]]] = {mode: [] for mode in MODES}
    for seed in normalized:
        suffix = _suffix_train(acquisition)
        controls = {
            "suffix_acquisition_fitted": {
                "validation_nll_bits_per_bit": _suffix_score(suffix, VALIDATION_BYTES),
                "test_nll_bits_per_bit": _suffix_score(suffix, TEST_BYTES),
            },
            "byte_unigram_acquisition_fitted": {
                "validation_nll_bits_per_bit": byte_unigram_cross_bits(acquisition, VALIDATION_BYTES),
                "test_nll_bits_per_bit": byte_unigram_cross_bits(acquisition, TEST_BYTES),
            },
        }
        for mode in MODES:
            organism = _new_organism(seed, mode)
            checkpoint, resources = _train(organism, acquisition, seed)
            checkpoint_digest = _digest(checkpoint)
            validation, validation_metrics = _frozen_score(checkpoint, VALIDATION_BYTES, seed)
            test, test_metrics = _frozen_score(checkpoint, TEST_BYTES, seed + 10000)
            lesion_state, lesion_feature_key = _lesion_product_path(checkpoint)
            lesion_installed = lesion_feature_key is not None
            lesion_validation = _frozen_score(lesion_state, VALIDATION_BYTES, seed)[1] if lesion_installed else None
            lesion_test = _frozen_score(lesion_state, TEST_BYTES, seed + 10000)[1] if lesion_installed else None
            resumed = Organism.from_state_dict(json.loads(json.dumps(checkpoint)))
            resume_digest = _digest(resumed.state_dict())
            runs[mode].append({
                "seed": seed,
                "mode": mode,
                "resources": resources,
                "checkpoint_digest": checkpoint_digest,
                "resume_digest": resume_digest,
                "resume_digest_equal": resume_digest == checkpoint_digest,
                "validation": validation_metrics,
                "test": test_metrics,
                "validation_nll_bits_per_bit": validation_metrics["nll_bits_per_bit"],
                "test_nll_bits_per_bit": test_metrics["nll_bits_per_bit"],
                "product_feature_key": lesion_feature_key,
                "k3_product_feature_installed": K3_KEY in dict(checkpoint.get("variable_order_feature_owners", {})),
                "product_feature_installed": lesion_installed,
                "lesion_validation_nll_bits_per_bit": lesion_validation["nll_bits_per_bit"] if lesion_validation else None,
                "lesion_test_nll_bits_per_bit": lesion_test["nll_bits_per_bit"] if lesion_test else None,
                "lesion_read_only": bool(lesion_validation and lesion_test and lesion_validation["checkpoint_unchanged"] and lesion_test["checkpoint_unchanged"]) if lesion_installed else None,
                "lesion_validation_delta_bits": (lesion_validation["nll_bits_per_bit"] - validation_metrics["nll_bits_per_bit"]) if lesion_validation else None,
                "lesion_test_delta_bits": (lesion_test["nll_bits_per_bit"] - test_metrics["nll_bits_per_bit"]) if lesion_test else None,
                "controls": controls,
            })
    variable = runs["variable_order"]
    additive = runs["nonlinear_additive_context"]
    deltas = [a["test_nll_bits_per_bit"] - v["test_nll_bits_per_bit"] for v, a in zip(variable, additive)]
    suffix_deltas = [
        r["test_nll_bits_per_bit"] - r["controls"]["suffix_acquisition_fitted"]["test_nll_bits_per_bit"]
        for r in variable
    ]
    unigram_deltas = [
        r["test_nll_bits_per_bit"] - r["controls"]["byte_unigram_acquisition_fitted"]["test_nll_bits_per_bit"]
        for r in variable
    ]
    lesion_deltas = [r["lesion_test_delta_bits"] for r in variable if r["lesion_test_delta_bits"] is not None]
    return {
        "protocol": PROTOCOL,
        "development_only": True,
        "acceptance_gating": False,
        "r6_qualification": "not_evaluated",
        "corpus": {
            "acquisition_documents": len(ACQUISITION_DOCUMENTS),
            "acquisition_bytes": len(acquisition),
            "validation_bytes": len(VALIDATION_BYTES),
            "test_bytes": len(TEST_BYTES),
            "acquisition_sha256": hashlib.sha256(acquisition).hexdigest(),
            "validation_sha256": hashlib.sha256(VALIDATION_BYTES).hexdigest(),
            "test_sha256": hashlib.sha256(TEST_BYTES).hexdigest(),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": _dependency_metadata(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "event_path": {"transducer": TextBytesTransducer().spec.to_dict(), "target_in_payload": False, "heldout_target_derived_outcomes_sent": False, "heldout_zero_outcomes_for_bridge": True},
        "controls": {"fit_partition": "acquisition_only", "suffix_order": 3, "suffix_max_contexts": 4096},
        "runs": runs,
        "comparison": {
            "variable_order_minus_additive_test_nll_gain_bits": deltas,
            "mean_variable_order_minus_additive_test_nll_gain_bits": sum(deltas) / float(len(deltas)),
            "variable_order_lesion_test_delta_bits": lesion_deltas,
            "mean_variable_order_lesion_test_delta_bits": sum(lesion_deltas) / float(len(lesion_deltas)) if lesion_deltas else None,
            "variable_order_minus_suffix_test_nll_bits": suffix_deltas,
            "variable_order_minus_byte_unigram_test_nll_bits": unigram_deltas,
            "mean_variable_order_minus_suffix_test_nll_bits": sum(suffix_deltas) / float(len(suffix_deltas)),
            "mean_variable_order_minus_byte_unigram_test_nll_bits": sum(unigram_deltas) / float(len(unigram_deltas)),
        },
        "diagnostics": {
            "frozen_read_only_all": all(r["validation"]["frozen_read_only"] and r["test"]["frozen_read_only"] for mode in MODES for r in runs[mode]),
            "checkpoint_resume_digest_all": all(r["resume_digest_equal"] for mode in MODES for r in runs[mode]),
            "k3_installed_any_variable_order": any(r["k3_product_feature_installed"] for r in variable),
            "product_path_installed_all_variable_order": all(r["product_feature_installed"] for r in variable),
            "lesion_read_only_all": all(r["lesion_read_only"] for r in variable),
            "causal_heldout_nll_gain_observed": bool(lesion_deltas) and sum(lesion_deltas) / len(lesion_deltas) > 0.01,
            "causal_heldout_nll_gain_consistent_all_seeds": bool(lesion_deltas) and all(delta > 0.0 for delta in lesion_deltas),
            "variable_order_beats_suffix_control": bool(suffix_deltas) and all(delta < 0.0 for delta in suffix_deltas),
            "variable_order_beats_byte_unigram": bool(unigram_deltas) and all(delta < 0.0 for delta in unigram_deltas),
            "elapsed_seconds": time.monotonic() - started,
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--report", default="/tmp/r6-english-event-pilot.json")
    args = parser.parse_args()
    report = run_english_pilot(tuple(args.seeds) if args.seeds else (0, 1, 2, 3))
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostics"], sort_keys=True))

"""Exploratory continuous English pilot using a generic bounded lag workspace.

This is development evidence only.  UTF-8 bytes enter the existing
TextBytesTransducer; the lag modes add only a serialized, signed-centered
bounded history.  Next-bit outcomes are supplied for acquisition bytes.  The
validation and test streams are scored continuously from one independent
clone each, with neutral bridge outcomes and no target-derived feedback.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from r6_tier_benchmark import SuffixNgramControl
from soma.events import Event, EventBridge
from soma.events.envelope import OutcomeEvent
from soma.evaluation.english import (
    bit_stream,
    bits_per_bit,
    byte_unigram_cross_bits,
)
from soma.organism import Organism
from soma.synapses import Synapse
from soma.transducers import BoundedLagWorkspaceTransducer, TextBytesTransducer
from soma.transducers.text_bytes import validate_utf8


PROTOCOL = "r6-english-lag-pilot-v1"
DEPTH = 8
MODES = ("lag_persistent", "raw_text", "lag_additive")
PERSISTENT_EVIDENCE_BUDGET = 512

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


def _state_bytes(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _events(bits, ends, index, event_id):
    return [
        Event("bit", "english-lag-pilot", event_id, float(index), {"value": float(bits[index])}, {"trust": "raw"}).to_dict(),
        Event("boundary", "english-lag-pilot", event_id, float(index), {"byte_end": float(ends[index])}, {"trust": "raw"}).to_dict(),
    ]


def _new_organism(seed: int, mode: str) -> Tuple[Organism, object]:
    if mode not in MODES:
        raise ValueError("unsupported English lag mode: %s" % mode)
    wrapped = mode != "raw_text"
    input_size = DEPTH * 2 + 1 if wrapped else 3
    # Keep all modes on the same organism/resource topology; only the input
    # dimensionality and transducer wrapper differ.
    organism = Organism.create_default(input_size=input_size, hidden_size=6, output_size=1,
                                       seed=int(seed), max_synapses=1024, energy_per_step=100.0)
    for synapse in organism.graph.iter_synapses():
        if synapse.destination in organism.output_ids:
            synapse.strength = 0.0
    for input_id in organism.input_ids:
        if not organism.graph.has(input_id, organism.output_ids[0]):
            organism.graph.add(Synapse(input_id, organism.output_ids[0], 0.0, plasticity=1.0))
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    organism.enable_context_modules(max_modules=4)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    organism.representation_learning_enabled = False
    organism.actor_learning_rate = 0.10
    candidate_input_ids = tuple("input-%03d" % index for index in range(input_size - 1))
    if mode == "lag_persistent":
        organism.enable_variable_order_learning(int(seed) + 7919, max_order=3,
                                                 candidate_input_ids=candidate_input_ids)
        organism.enable_variable_order_persistent_scout(
            evidence_budget=PERSISTENT_EVIDENCE_BUDGET, min_count=32)
    elif mode == "raw_text":
        organism.enable_variable_order_learning(int(seed) + 7919, max_order=3,
                                                 candidate_input_ids=candidate_input_ids)
    transducer = BoundedLagWorkspaceTransducer(TextBytesTransducer(), DEPTH) if wrapped else TextBytesTransducer()
    return organism, transducer


def _train(organism: Organism, transducer, data: bytes, seed: int):
    validate_utf8(data)
    bits, ends = bit_stream(data)
    if len(bits) < 2:
        raise ValueError("acquisition must contain at least two bits")
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xE11).uniform(-1.0, 1.0) for _ in range(len(bits))))
    bridge = EventBridge(organism, transducer, novelty=0.10, exploration=0.20)
    peak_state = 0
    predictions, targets = [], []
    for index in range(len(bits) - 1):
        action = bridge.ingest(_events(bits, ends, index, index))
        prediction = float(action["proposal"]["value"])
        target = int(bits[index + 1])
        predictions.append(prediction)
        targets.append(target)
        reward = 1.0 - abs(prediction - target)
        bridge.outcome(OutcomeEvent(action["correlation_id"], float(index) + 0.5, reward, "english-lag-pilot").to_dict())
        if index % 128 == 0:
            peak_state = max(peak_state, _state_bytes(organism.state_dict()))
    bridge.close()
    checkpoint = {"organism": organism.state_dict(), "transducer": transducer.state_dict()}
    return checkpoint, {
        "acquisition_bits": len(bits) - 1,
        "acquisition_nll_bits_per_bit": bits_per_bit(predictions, targets),
        "peak_state_bytes": peak_state,
        "checkpoint_state_bytes": _state_bytes(checkpoint),
        "cells": len(checkpoint["organism"]["cells"]),
        "synapses": len(checkpoint["organism"]["synapses"]),
        "persistent_evidence_count": int(checkpoint["organism"].get("variable_order_persistent_evidence_count", 0)),
        "persistent_episode_count": int(checkpoint["organism"].get("variable_order_persistent_episode_count", 0)),
        "persistent_terminal_state": checkpoint["organism"].get("variable_order_persistent_terminal_state", "inactive"),
        "persistent_telemetry": checkpoint["organism"].get("variable_order_persistent_last_telemetry", {}),
    }


def _restore(checkpoint: Mapping[str, object], mode: str):
    organism, transducer = _new_organism(0, mode)
    organism = Organism.from_state_dict(json.loads(json.dumps(checkpoint["organism"])))
    transducer.load_state_dict(json.loads(json.dumps(checkpoint["transducer"])))
    return organism, transducer


def _frozen_score(checkpoint: Mapping[str, object], mode: str, data: bytes, seed: int):
    validate_utf8(data)
    bits, ends = bit_stream(data)
    source_digest = _digest(checkpoint)
    organism, transducer = _restore(checkpoint, mode)
    # Freeze the scoring clone: outcomes are neutral and no learning signal is
    # allowed to update the checkpoint being measured.
    organism.learning_rate = 0.0
    organism.actor_learning_rate = 0.0
    organism.representation_learning_enabled = False
    organism.structural_plasticity_enabled = False
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xF00D).uniform(-1.0, 1.0) for _ in range(len(bits))))
    bridge = EventBridge(organism, transducer, novelty=0.10, exploration=0.20)
    predictions, targets = [], []
    for index in range(len(bits) - 1):
        action = bridge.ingest(_events(bits, ends, index, index))
        predictions.append(float(action["proposal"]["value"]))
        targets.append(int(bits[index + 1]))
        bridge.no_credit()
    bridge.close()
    return {
        "nll_bits_per_bit": bits_per_bit(predictions, targets),
        "bits": len(targets),
        "checkpoint_unchanged": _digest(checkpoint) == source_digest,
        "frozen_read_only": True,
        "continuous_single_clone": True,
        "neutral_outcomes_only": True,
    }


def _lesion_winner(checkpoint: Mapping[str, object]):
    cloned = json.loads(json.dumps(checkpoint))
    state = cloned["organism"]
    telemetry = dict(state.get("variable_order_persistent_last_telemetry", {}))
    winner = telemetry.get("winner_key")
    owners = dict(state.get("variable_order_feature_owners", {}))
    owner_id = owners.get(winner)
    if not winner or not owner_id:
        return cloned, None
    owner_cell = dict(state.get("motor_modules", {}).get(owner_id, {})).get("cell_id")
    sources = tuple(str(winner).split(":", 1)[1].split("|")) if ":" in str(winner) else ()
    feature_cell = next((cell_id for cell_id, payload in dict(state.get("cells", {})).items()
                         if payload.get("activation_type") in ("dendritic_product", "dendritic_product_n")
                         and tuple(payload.get("dendritic_sources", ())) == sources), None)
    if not owner_cell or not feature_cell:
        return cloned, None
    for edge in state.get("synapses", []):
        if edge.get("source") == feature_cell and edge.get("destination") == owner_cell:
            edge["strength"] = 0.0
            edge["plasticity"] = 0.0
            return cloned, winner
    return cloned, None


def _suffix_fit(data: bytes, order: int = 3):
    control = SuffixNgramControl(order, 4096)
    bits, _ = bit_stream(data)
    control.reset_history()
    for bit in bits:
        control.observe(bit)
    return control


def _suffix_score(control, data: bytes):
    evaluation = SuffixNgramControl.from_state_dict(control.state_dict())
    evaluation.reset_history()
    bits, _ = bit_stream(data)
    predictions, targets = [], []
    for index in range(len(bits) - 1):
        evaluation.observe(bits[index], learn=False)
        predictions.append(evaluation.distribution()[0][1])
        targets.append(bits[index + 1])
    return bits_per_bit(predictions, targets)


def _dependencies():
    files = ["soma/organism.py", "soma/events/bridge.py", "soma/events/envelope.py",
             "soma/transducers/history.py", "soma/transducers/text_bytes.py",
             "soma/evaluation/english.py", "r6_tier_benchmark.py"]
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *files], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"sha256": hashes, "git_revision": revision, "dirty": dirty}


def run_english_lag_pilot(seeds: Sequence[int] = (0,), acquisition: Optional[bytes] = None,
                          validation: bytes = VALIDATION_BYTES, test: bytes = TEST_BYTES):
    started = time.monotonic()
    acquisition = b"".join(ACQUISITION_DOCUMENTS) if acquisition is None else bytes(acquisition)
    validate_utf8(acquisition); validate_utf8(validation); validate_utf8(test)
    controls = {"fit_partition": "acquisition_only", "constant_controls": "oracle_diagnostics",
                "byte_unigram": {"validation_nll_bits_per_bit": byte_unigram_cross_bits(acquisition, validation),
                                  "test_nll_bits_per_bit": byte_unigram_cross_bits(acquisition, test)},
                "suffix_order3": {"validation_nll_bits_per_bit": _suffix_score(_suffix_fit(acquisition), validation),
                                   "test_nll_bits_per_bit": _suffix_score(_suffix_fit(acquisition), test)}}
    runs = {mode: [] for mode in MODES}
    for seed in tuple(int(value) for value in seeds):
        for mode in MODES:
            organism, transducer = _new_organism(seed, mode)
            checkpoint, resources = _train(organism, transducer, acquisition, seed)
            validation_metrics = _frozen_score(checkpoint, mode, validation, seed + 1000)
            test_metrics = _frozen_score(checkpoint, mode, test, seed + 2000)
            resumed_org, resumed_trans = _restore(checkpoint, mode)
            resumed = {"organism": resumed_org.state_dict(), "transducer": resumed_trans.state_dict()}
            lesioned, winner = _lesion_winner(checkpoint) if mode == "lag_persistent" else (checkpoint, None)
            lesion_test = _frozen_score(lesioned, mode, test, seed + 2000) if winner else None
            runs[mode].append({"seed": seed, "mode": mode, "resources": resources,
                               "checkpoint_digest": _digest(checkpoint),
                               "resume_digest_equal": _digest(resumed) == _digest(checkpoint),
                               "validation": validation_metrics, "test": test_metrics,
                               "winner_feature_key": winner, "lesion_test": lesion_test,
                               "lesion_test_delta_bits": lesion_test["nll_bits_per_bit"] - test_metrics["nll_bits_per_bit"] if lesion_test else None})
    persistent = runs["lag_persistent"]; raw = runs["raw_text"]; additive = runs["lag_additive"]
    raw_delta = [p["test"]["nll_bits_per_bit"] - r["test"]["nll_bits_per_bit"] for p, r in zip(persistent, raw)]
    additive_delta = [p["test"]["nll_bits_per_bit"] - a["test"]["nll_bits_per_bit"] for p, a in zip(persistent, additive)]
    lesions = [row["lesion_test_delta_bits"] for row in persistent if row["lesion_test_delta_bits"] is not None]
    return {"protocol": PROTOCOL, "development_only": True, "acceptance_gating": False,
            "r6_qualification": "not_evaluated", "persistent_evidence_budget": PERSISTENT_EVIDENCE_BUDGET,
            "corpus": {"acquisition_bytes": len(acquisition), "validation_bytes": len(validation), "test_bytes": len(test),
                       "acquisition_sha256": hashlib.sha256(acquisition).hexdigest(),
                       "validation_sha256": hashlib.sha256(validation).hexdigest(), "test_sha256": hashlib.sha256(test).hexdigest()},
            "transport": {"base": TextBytesTransducer().spec.to_dict(), "depth": DEPTH,
                          "lag": BoundedLagWorkspaceTransducer(TextBytesTransducer(), DEPTH).spec.to_dict()},
            "controls": controls, "runs": runs, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": _dependencies(), "environment": {"python": sys.version, "platform": platform.platform()},
            "comparison": {"persistent_minus_raw_text_test_nll_bits": raw_delta,
                           "persistent_minus_lag_additive_test_nll_bits": additive_delta,
                           "mean_persistent_minus_raw_text_test_nll_bits": sum(raw_delta) / max(1, len(raw_delta)),
                           "mean_persistent_minus_lag_additive_test_nll_bits": sum(additive_delta) / max(1, len(additive_delta)),
                           "lesion_test_delta_bits": lesions},
            "diagnostics": {"continuous_frozen_scoring": True,
                            "frozen_read_only_all": all(row["validation"]["frozen_read_only"] and row["test"]["frozen_read_only"] for rows in runs.values() for row in rows),
                            "single_clone_all": all(row["validation"]["continuous_single_clone"] and row["test"]["continuous_single_clone"] for rows in runs.values() for row in rows),
                            "resume_digest_all": all(row["resume_digest_equal"] for rows in runs.values() for row in rows),
                            "persistent_install_coverage": sum(row["resources"]["persistent_terminal_state"] == "accepted" for row in persistent) / max(1, len(persistent)),
                            "lesion_available": bool(lesions), "lesion_positive_all": bool(lesions) and all(value > 0.0 for value in lesions),
                            "elapsed_seconds": time.monotonic() - started}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--report", default="/tmp/r6-english-lag-pilot-v1.json")
    args = parser.parse_args()
    report = run_english_lag_pilot(tuple(args.seeds) if args.seeds else (0,))
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostics"], sort_keys=True))

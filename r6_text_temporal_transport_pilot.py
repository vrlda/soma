"""Development-only TextBytes + bounded lag-workspace temporal pilot.

The event path is the existing TextBytesTransducer wrapped by the generic
BoundedLagWorkspaceTransducer. The evaluator target is a sparse product of
three nonadjacent lagged bit taps; it is never present in event payloads.
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from soma.events import Event, EventBridge
from soma.events.envelope import OutcomeEvent
from soma.organism import Organism
from soma.transducers import BoundedLagWorkspaceTransducer, TextBytesTransducer


PROTOCOL = "r6-text-temporal-transport-pilot-v1"
DEPTH = 8
SIGNAL_WIDTH = 2  # TextBytes: bit, byte boundary
TARGET_INDICES = (0, 4, 10)  # lag0.bit, lag2.bit, lag5.bit
TARGET_KEY = "k3:input-000|input-004|input-010"
MODES = ("lag_persistent", "lag_shuffled", "lag_additive", "raw_text")


def _dependency_metadata() -> Dict[str, object]:
    """Record the implementation surface used by this development fixture."""
    files = [
        "soma/organism.py", "soma/events/bridge.py", "soma/events/envelope.py",
        "soma/events/channel.py", "soma/transducers/sdk.py",
        "soma/transducers/history.py", "soma/transducers/text_bytes.py",
    ]
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *files],
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"sha256": hashes, "git_revision": revision, "dirty": dirty}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _target(workspace: Sequence[float]) -> float:
    product = 1.0
    for index in TARGET_INDICES:
        product *= 1.0 if float(workspace[index]) > 0.0 else -1.0
    return 0.8 * product


def _frame(bit: int, boundary: int, event_id: int) -> List[dict]:
    return [
        Event("bit", "text-transport", int(event_id), float(event_id), {"value": int(bit)}, {"trust": "raw"}).to_dict(),
        Event("boundary", "text-transport", int(event_id), float(event_id), {"byte_end": int(boundary)}, {"trust": "raw"}).to_dict(),
    ]


def _workspace(history: Sequence[Tuple[int, int]]) -> Tuple[float, ...]:
    rows = []
    for bit, boundary in list(history)[:DEPTH]:
        rows.append((2.0 * float(bit) - 1.0, 2.0 * float(boundary) - 1.0))
    rows.extend([(0.0, 0.0)] * (DEPTH - len(rows)))
    return tuple(value for row in rows for value in row)


def _stream(
    seed: int,
    heldout_combo: int,
    wanted_eligible: int,
    require_heldout: bool = False,
    initial_history: Optional[Sequence[Tuple[int, int]]] = None,
    start_position: int = 0,
) -> List[Dict[str, object]]:
    rng = random.Random(int(seed))
    history = list(initial_history or [])[:DEPTH]
    stream: List[Dict[str, object]] = []
    eligible_count = 0
    position = int(start_position)
    while eligible_count < int(wanted_eligible):
        bit = rng.randrange(2)
        boundary = 1 if position % 8 == 7 else 0
        current_history = [(bit, boundary)] + history
        workspace = _workspace(current_history)
        combo = sum((1 if workspace[index] > 0.0 else 0) << shift for shift, index in zip((2, 1, 0), TARGET_INDICES))
        eligible = combo == int(heldout_combo) if require_heldout else combo != int(heldout_combo)
        entry = {
            "bit": bit,
            "boundary": boundary,
            "workspace": workspace,
            "combo": combo,
            "target": _target(workspace),
            "eligible": bool(eligible),
            "byte_index": position // 8,
            "bit_index": position % 8,
            "byte_value": None,
        }
        stream.append(entry)
        if eligible:
            eligible_count += 1
        history = current_history[:DEPTH]
        position += 1
    return stream


def _new_organism(seed: int, mode: str, persistent_budget: int = 512) -> Tuple[Organism, object]:
    if mode not in MODES:
        raise ValueError("unsupported transport mode: %s" % mode)
    wrapped = mode != "raw_text"
    input_size = DEPTH * SIGNAL_WIDTH + 1 if wrapped else 3
    organism = Organism.create_default(input_size=input_size, hidden_size=6, output_size=1,
                                       seed=int(seed), max_synapses=1024, energy_per_step=100.0)
    for edge in organism.graph.iter_synapses():
        if edge.destination in organism.output_ids:
            edge.strength = 0.0
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    organism.enable_context_modules(max_modules=4)
    organism.resources.energy_per_step = max(100.0, organism.resources.energy_per_step)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    organism.representation_learning_enabled = False
    organism.actor_learning_rate = 0.10
    candidate_input_ids = tuple("input-%03d" % index for index in range(input_size - 1))
    if mode in ("lag_persistent", "lag_shuffled"):
        organism.enable_variable_order_learning(int(seed) + 7919, max_order=3,
                                                 candidate_input_ids=candidate_input_ids)
        organism.enable_variable_order_persistent_scout(evidence_budget=int(persistent_budget), min_count=32)
    transducer = BoundedLagWorkspaceTransducer(TextBytesTransducer(), depth=DEPTH) if wrapped else TextBytesTransducer()
    return organism, transducer


def _signed_prediction(action: Mapping[str, object]) -> float:
    return 2.0 * float(action["proposal"]["value"]) - 1.0


def _train(organism: Organism, transducer, stream: Sequence[Mapping[str, object]], seed: int, shuffled: bool = False):
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xE17).uniform(-1.0, 1.0) for _ in range(len(stream))))
    bridge = EventBridge(organism, transducer, novelty=0.10, exploration=0.20)
    order = list(range(len(stream)))
    random.Random(int(seed) + 0xBAD5EED).shuffle(order)
    predictions = []
    eligible_examples = 0
    withheld_examples = 0
    shuffled_targets = [float(stream[index]["target"]) for index in order]
    for index, entry in enumerate(stream):
        action = bridge.ingest(_frame(int(entry["bit"]), int(entry["boundary"]), index))
        prediction = _signed_prediction(action)
        predictions.append(prediction)
        if bool(entry["eligible"]):
            eligible_examples += 1
            # The shuffled null preserves the same target multiset but breaks
            # event-to-outcome alignment.  Ineligible held-out combinations
            # always receive neutral reward and never enter the scout.
            target = shuffled_targets[index] if shuffled else float(entry["target"])
            reward = 1.0 - abs(prediction - target)
            bridge.outcome(OutcomeEvent(action["correlation_id"], float(index) + 0.5, reward, "text-transport").to_dict())
        else:
            withheld_examples += 1
            # Keep the withheld frame in the event/lag stream without turning
            # it into a numeric zero reward that would update learner state.
            bridge.no_credit()
    bridge.close()
    checkpoint = {"organism": organism.state_dict(), "transducer": transducer.state_dict()}
    return checkpoint, {
        "state_bytes": len(json.dumps(checkpoint, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")),
        "eligible_training_examples": eligible_examples,
        "withheld_no_credit_examples": withheld_examples,
        "evidence_count": int(checkpoint["organism"].get("variable_order_persistent_evidence_count", 0)),
        "episode_count": int(checkpoint["organism"].get("variable_order_persistent_episode_count", 0)),
        "variable_order_install_count": int(checkpoint["organism"].get("variable_order_install_count", 0)),
        "terminal_state": checkpoint["organism"].get("variable_order_persistent_terminal_state", "inactive"),
        "telemetry": checkpoint["organism"].get("variable_order_persistent_last_telemetry", {}),
    }


def _restore(checkpoint: Mapping[str, object], mode: str, persistent_budget: int = 512):
    organism, transducer = _new_organism(0, mode, persistent_budget=persistent_budget)
    organism = Organism.from_state_dict(json.loads(json.dumps(checkpoint["organism"])))
    transducer.load_state_dict(json.loads(json.dumps(checkpoint["transducer"])))
    return organism, transducer


def _frozen_score(checkpoint: Mapping[str, object], mode: str, stream: Sequence[Mapping[str, object]], seed: int,
                  persistent_budget: int = 512):
    source_digest = _digest(checkpoint)
    organism, transducer = _restore(checkpoint, mode, persistent_budget=persistent_budget)
    organism.set_exploration_tape(tuple(random.Random(int(seed) + 0xF00D + index).uniform(-1.0, 1.0) for index in range(len(stream))))
    bridge = EventBridge(organism, transducer, novelty=0.10, exploration=0.20)
    predictions = []
    targets = []
    for index, entry in enumerate(stream):
        action = bridge.ingest(_frame(int(entry["bit"]), int(entry["boundary"]), index))
        bridge.no_credit()
        if bool(entry["eligible"]):
            predictions.append(_signed_prediction(action))
            targets.append(float(entry["target"]))
    bridge.close()
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / float(max(1, len(targets)))
    sse0 = sum(t * t for t in targets)
    skill = 1.0 - sum((p - t) ** 2 for p, t in zip(predictions, targets)) / sse0 if sse0 > 1e-12 else 0.0
    return {"mae": mae, "skill": skill, "eligible_bits": len(targets),
            "checkpoint_unchanged": _digest(checkpoint) == source_digest,
            "frozen_read_only": True, "neutral_outcomes_only": True}


def _lesion_target(checkpoint: Mapping[str, object]) -> Tuple[dict, bool]:
    cloned = json.loads(json.dumps(checkpoint))
    state = cloned["organism"]
    owner_id = dict(state.get("variable_order_feature_owners", {})).get(TARGET_KEY)
    if owner_id is None:
        return cloned, False
    owner_cell = dict(state.get("motor_modules", {}).get(owner_id, {})).get("cell_id")
    feature = tuple(TARGET_KEY.split(":", 1)[1].split("|"))
    feature_cell = next((cell_id for cell_id, payload in dict(state.get("cells", {})).items()
                         if payload.get("activation_type") in ("dendritic_product", "dendritic_product_n")
                         and tuple(payload.get("dendritic_sources", ())) == feature), None)
    changed = False
    if owner_cell and feature_cell:
        for edge in state.get("synapses", []):
            if edge.get("source") == feature_cell and edge.get("destination") == owner_cell:
                edge["strength"] = 0.0
                edge["plasticity"] = 0.0
                changed = True
    return cloned, changed


class _TargetControl:
    def __init__(self, kind: str, order: int = 0):
        self.kind, self.order = kind, int(order)
        self.table = {}
        self.global_values = []

    def fit(self, stream):
        for entry in stream:
            if not entry["eligible"]:
                continue
            if self.kind == "byte_unigram":
                key = (entry["byte_index"], entry["bit_index"])
            else:
                key = tuple(int(value > 0.0) for value in entry["workspace"][: self.order * SIGNAL_WIDTH : SIGNAL_WIDTH])
            cell = self.table.setdefault(key, [0.0, 0.0])
            cell[0] += float(entry["target"])
            cell[1] += 1.0
            self.global_values.append(float(entry["target"]))

    def predict(self, entry):
        if self.kind == "byte_unigram":
            key = (entry["byte_index"], entry["bit_index"])
        else:
            key = tuple(int(value > 0.0) for value in entry["workspace"][: self.order * SIGNAL_WIDTH : SIGNAL_WIDTH])
        cell = self.table.get(key)
        return cell[0] / cell[1] if cell else sum(self.global_values) / float(max(1, len(self.global_values)))


def _control_score(control, stream):
    values = [control.predict(entry) for entry in stream if entry["eligible"]]
    targets = [float(entry["target"]) for entry in stream if entry["eligible"]]
    sse0 = sum(t * t for t in targets)
    return {"mae": sum(abs(v - t) for v, t in zip(values, targets)) / float(max(1, len(targets))),
            "skill": 1.0 - sum((v - t) ** 2 for v, t in zip(values, targets)) / sse0 if sse0 > 1e-12 else 0.0}


def run_transport_pilot(seeds: Sequence[int] = (0,), acquisition_examples: int = 512, evaluation_examples: int = 64,
                        persistent_budget: int = 512, protocol: str = PROTOCOL):
    started = time.monotonic()
    runs = {mode: [] for mode in MODES}
    controls = []
    manifest = {}
    for seed in tuple(int(value) for value in seeds):
        per_mode = []
        for heldout in range(8):
            acquisition = _stream(seed, heldout, acquisition_examples)
            tail = [(int(entry["bit"]), int(entry["boundary"])) for entry in acquisition[-DEPTH:]]
            evaluation = _stream(seed + 10000, heldout, evaluation_examples, require_heldout=True,
                                 initial_history=tail, start_position=len(acquisition))
            manifest["%s:%s" % (seed, heldout)] = _digest({"acquisition": acquisition, "evaluation": evaluation})
            byte_control = _TargetControl("byte_unigram"); byte_control.fit(acquisition)
            suffix_control = _TargetControl("suffix", order=8); suffix_control.fit(acquisition)
            controls.append({"seed": seed, "heldout_combo": heldout,
                             "heldout_sign": 1 if evaluation[0]["target"] > 0 else -1,
                             "byte_unigram": _control_score(byte_control, evaluation),
                             "suffix8": _control_score(suffix_control, evaluation),
                             "eligible_acquisition_combos": sorted({entry["combo"] for entry in acquisition if entry["eligible"]}),
                             "eligible_evaluation_combos": sorted({entry["combo"] for entry in evaluation if entry["eligible"]})})
            for mode in MODES:
                mode_started = time.monotonic()
                organism, transducer = _new_organism(seed, mode, persistent_budget=persistent_budget)
                checkpoint, resources = _train(organism, transducer, acquisition, seed, shuffled=(mode == "lag_shuffled"))
                metrics = _frozen_score(checkpoint, mode, evaluation, seed + heldout,
                                        persistent_budget=persistent_budget)
                # The shuffled control must traverse the identical persistent
                # scout and target-inspection path; only reward/event pairing
                # is shuffled during _train.  This makes null installation and
                # lesion telemetry symmetric rather than structurally absent.
                lesioned, installed = _lesion_target(checkpoint) if mode in ("lag_persistent", "lag_shuffled") else (checkpoint, False)
                lesion_metrics = _frozen_score(lesioned, mode, evaluation, seed + heldout,
                                               persistent_budget=persistent_budget) if installed else None
                resumed_org, resumed_trans = _restore(checkpoint, mode, persistent_budget=persistent_budget)
                resumed = {"organism": resumed_org.state_dict(), "transducer": resumed_trans.state_dict()}
                telemetry = resources.get("telemetry", {})
                target_winner = telemetry.get("winner_key") == TARGET_KEY
                target_rank_available = resources.get("evidence_count", 0) >= int(persistent_budget)
                per_mode.append({"seed": seed, "heldout_combo": heldout, "heldout_sign": 1 if evaluation[0]["target"] > 0 else -1,
                                 "mode": mode, "resources": resources, "checkpoint_digest": _digest(checkpoint),
                                 "resume_digest_equal": _digest(resumed) == _digest(checkpoint),
                                 "frozen_metrics": metrics, "target_k3_installed": installed,
                                 "lesion_metrics": lesion_metrics,
                                 "lesion_skill_drop": metrics["skill"] - lesion_metrics["skill"] if lesion_metrics else None,
                                 "persistent_telemetry": telemetry,
                                 "target_winner_key": telemetry.get("winner_key"),
                                 "target_winner": target_winner,
                                 "target_rank_available": target_rank_available,
                                 "target_rank": telemetry.get("winner_rank") if target_rank_available and target_winner else None,
                                 "target_feature_rank": telemetry.get("winner_feature_rank") if target_rank_available and target_winner else None})
                per_mode[-1]["case_elapsed_seconds"] = time.monotonic() - mode_started
        for mode in MODES:
            runs[mode].append({"seed": seed, "cases": [case for case in per_mode if case["mode"] == mode],
                               "frozen_read_only_all": all(case["frozen_metrics"]["frozen_read_only"] for case in per_mode if case["mode"] == mode),
                               "resume_digest_all": all(case["resume_digest_equal"] for case in per_mode if case["mode"] == mode)})
    persistent = [case for run in runs["lag_persistent"] for case in run["cases"]]
    shuffled = [case for run in runs["lag_shuffled"] for case in run["cases"]]
    additive = [case for run in runs["lag_additive"] for case in run["cases"]]
    byte_unigram = [case["byte_unigram"] for case in controls]
    suffix = [case["suffix8"] for case in controls]
    margins = [persistent[index]["frozen_metrics"]["skill"] - additive[index]["frozen_metrics"]["skill"] for index in range(len(persistent))]
    null_margins = [persistent[index]["frozen_metrics"]["skill"] - shuffled[index]["frozen_metrics"]["skill"] for index in range(len(persistent))]
    lesion_drops = [case["lesion_skill_drop"] for case in persistent if case["lesion_skill_drop"] is not None]
    return {"protocol": protocol, "development_only": True, "acceptance_gating": False,
            "r6_qualification": "not_evaluated", "transport": {"base": TextBytesTransducer().spec.to_dict(), "depth": DEPTH,
            "wrapper": BoundedLagWorkspaceTransducer(TextBytesTransducer(), DEPTH).spec.to_dict(),
            "candidate_input_ids": ["input-%03d" % index for index in range(DEPTH * SIGNAL_WIDTH)],
            "target_indices": list(TARGET_INDICES), "target_key": TARGET_KEY},
            "seeds": list(seeds), "acquisition_examples": int(acquisition_examples),
            "persistent_evidence_budget": int(persistent_budget),
            "evaluation_examples": int(evaluation_examples), "manifest": manifest,
            "manifest_digest": _digest(manifest), "controls": controls,
            "runs": runs, "comparison": {"persistent_minus_additive_skill": margins,
            "mean_persistent_minus_additive_skill": sum(margins) / float(max(1, len(margins))),
            "persistent_minus_shuffled_null_skill": null_margins,
            "mean_persistent_minus_shuffled_null_skill": sum(null_margins) / float(max(1, len(null_margins))),
            "persistent_minus_byte_unigram_skill": [persistent[i]["frozen_metrics"]["skill"] - byte_unigram[i]["skill"] for i in range(len(persistent))],
            "persistent_minus_suffix8_skill": [persistent[i]["frozen_metrics"]["skill"] - suffix[i]["skill"] for i in range(len(persistent))],
            "mean_persistent_minus_byte_unigram_skill": sum(persistent[i]["frozen_metrics"]["skill"] - byte_unigram[i]["skill"] for i in range(len(persistent))) / float(max(1, len(persistent))),
            "mean_persistent_minus_suffix8_skill": sum(persistent[i]["frozen_metrics"]["skill"] - suffix[i]["skill"] for i in range(len(persistent))) / float(max(1, len(persistent)))},
            "diagnostics": {"balanced_leave_one_out": all(sorted(c["heldout_combo"] for c in run["cases"]) == list(range(8)) for run in runs["lag_persistent"]),
            "both_target_signs_covered": {-1, 1} == {c["heldout_sign"] for c in persistent},
            "frozen_read_only_all": all(run["frozen_read_only_all"] for mode in MODES for run in runs[mode]),
            "resume_digest_all": all(run["resume_digest_all"] for mode in MODES for run in runs[mode]),
            "withheld_frames_no_credit_all": all(
                case["resources"]["withheld_no_credit_examples"] > 0
                and case["resources"]["evidence_count"] == case["resources"]["eligible_training_examples"]
                for run in runs["lag_persistent"] for case in run["cases"]
            ),
            "candidate_family_excludes_fixed_bias": all(
                case["resources"]["telemetry"].get("family_size") == 680
                for run in runs["lag_persistent"] for case in run["cases"]
            ),
            "target_k3_installed_any": any(c["target_k3_installed"] for c in persistent),
            "target_k3_install_coverage": sum(c["target_k3_installed"] for c in persistent) / float(max(1, len(persistent))),
            "target_rank_values": [c["target_rank"] for c in persistent],
            "lesion_consistent_positive": bool(lesion_drops) and all(value > 0.05 for value in lesion_drops),
            "target_discovery_observed": any(c["target_k3_installed"] for c in persistent),
            "target_winner_observed": any(c["target_winner"] for c in persistent),
            "target_accepted_install_observed": any(c["target_k3_installed"] for c in persistent),
            "shuffled_null_observed": bool(shuffled),
            "lesion_skill_drops": lesion_drops, "elapsed_seconds": time.monotonic() - started},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": _dependency_metadata(),
            "environment": {"python": sys.version, "platform": platform.platform()}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--acquisition-examples", type=int, default=512)
    parser.add_argument("--evaluation-examples", type=int, default=64)
    parser.add_argument("--report", default="/tmp/r6-text-temporal-transport-pilot-v1.json")
    args = parser.parse_args()
    report = run_transport_pilot(tuple(args.seeds) if args.seeds else (0,), args.acquisition_examples, args.evaluation_examples)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostics"], sort_keys=True))

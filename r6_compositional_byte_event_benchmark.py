"""Development-only held-out product experiment for the SOMA event path.

This is deliberately not an R6 acceptance benchmark.  It is a small causal
discriminator for the target-blind variable-order substrate: three signed
sensor bits arrive in one universal event envelope, the evaluator rotates a
leave-one-combination-out case over all eight combinations, and acquisition
receives only the byte and a delayed scalar outcome. Held-out scoring emits
one byte event from an independent checkpoint clone and sends no outcome. The
target is never placed in an event payload.

The experiment is useful because an exact table/episodic lookup and a
nonlinear-additive context control cannot infer the withheld parity/product
combination.  It
does not establish tier qualification, and its diagnostics are never used by
the locked ``r6-tier-v2`` report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from soma.events.bridge import EventBridge
from soma.events.channel import ChannelSchema
from soma.events.envelope import Event, OutcomeEvent
from soma.lifetime import _skill
from soma.organism import Organism
from soma.transducers.sdk import Transducer, TransducerSpec


PROTOCOL = "r6-compositional-byte-event-dev-v2"
MODES = ("variable_order", "nonlinear_additive_context", "shuffled_reward")
SENSOR_COUNT = 3
COMBINATION_COUNT = 1 << SENSOR_COUNT
HELDOUT_SYMBOL = COMBINATION_COUNT - 1  # retained as the historical default only


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def symbol_to_values(symbol: int) -> Tuple[float, ...]:
    """Decode one opaque byte symbol into simultaneous signed sensor values."""
    value = int(symbol)
    if value < 0 or value >= COMBINATION_COUNT:
        raise ValueError("triple symbol is outside the frozen 3-bit alphabet")
    return tuple(1.0 if value & (1 << index) else -1.0 for index in range(SENSOR_COUNT))


def product_target(symbol: int) -> float:
    """Evaluator-owned product target; never sent through the event boundary."""
    product = 1.0
    for value in symbol_to_values(symbol):
        product *= value
    return 0.8 * product


class ByteTripleTransducer(Transducer):
    """One-byte event adapter with a fixed, reversible 3-bit decoding."""

    SPEC_NAME = "r6-byte-triple-v1"

    def __init__(self) -> None:
        super().__init__(TransducerSpec(
            self.SPEC_NAME,
            [ChannelSchema("byte", ["value"], rate=1.0, low=0.0, high=float(HELDOUT_SYMBOL))],
            SENSOR_COUNT,
            ChannelSchema("action", ["value"], rate=1.0, low=-1.0, high=1.0),
            description="Opaque 3-bit signed simultaneous sensor frame",
            reversible=True,
        ))

    def pack_frame(self, frame: Mapping[str, object]) -> Tuple[float, ...]:
        try:
            payload = frame["byte"]["payload"]  # type: ignore[index]
            raw = payload["value"]  # type: ignore[index]
        except (KeyError, TypeError):
            raise ValueError("byte frame is missing its declared value")
        if isinstance(raw, bool) or int(raw) != float(raw) or not 0 <= int(raw) < COMBINATION_COUNT:
            raise ValueError("byte frame value must be an integer in the frozen alphabet")
        return symbol_to_values(int(raw))

    def unpack_action(self, outputs: Sequence[float]) -> Dict[str, float]:
        if not outputs:
            raise ValueError("organism produced no action output")
        return {"value": max(-1.0, min(1.0, float(outputs[0])))}


@dataclass(frozen=True)
class ProductExperimentConfig:
    """Frozen development manifest; no field is an R6 acceptance threshold."""

    acquisition_repetitions: int = 192
    exploration: float = 0.20
    novelty: float = 0.10
    actor_learning_rate: float = 0.10
    max_modules: int = 6

    def validate(self) -> None:
        if self.acquisition_repetitions < 8:
            raise ValueError("development acquisition repetitions are too small")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("modulators must be in [0, 1]")
        if self.actor_learning_rate < 0.0 or self.max_modules < 1:
            raise ValueError("learning rate and module limit must be nonnegative/positive")


def frozen_manifest(seed: int, config: ProductExperimentConfig, heldout_symbol: int = HELDOUT_SYMBOL) -> Dict[str, object]:
    """Return one frozen leave-one-combination-out acquisition manifest."""
    config.validate()
    heldout_symbol = int(heldout_symbol)
    if not 0 <= heldout_symbol < COMBINATION_COUNT:
        raise ValueError("heldout symbol is outside the frozen alphabet")
    acquisition = [
        symbol for symbol in range(COMBINATION_COUNT) if symbol != heldout_symbol
        for _ in range(config.acquisition_repetitions)
    ]
    random.Random(int(seed) + 0x5EED + 131 * heldout_symbol).shuffle(acquisition)
    return {
        "seed": int(seed),
        "alphabet_size": COMBINATION_COUNT,
        "sensor_count": SENSOR_COUNT,
        "acquisition_symbols": acquisition,
        "heldout_symbol": heldout_symbol,
        "acquisition_symbol_set": sorted(set(acquisition)),
        "heldout_sign": 1 if product_target(heldout_symbol) > 0 else -1,
    }


def _reward(action: float, target: float) -> float:
    clipped = max(-1.0, min(1.0, float(action)))
    return max(-1.0, min(1.0, (target * target - (clipped - target) ** 2) / 4.0))


def _new_organism(seed: int, mode: str, config: ProductExperimentConfig) -> Organism:
    if mode not in MODES:
        raise ValueError("unsupported product experiment mode: %s" % mode)
    # Every mode uses the same hidden-afferent context topology.  The
    # variable-order substrate is the only mechanism difference; using the
    # older input-afferent topology for the control would confound the causal
    # comparison with a different initial graph.
    afferent_kind = "hidden"
    organism = Organism.create_default(input_size=SENSOR_COUNT, hidden_size=8, output_size=1, seed=int(seed))
    organism.enable_context_modules(max_modules=config.max_modules, afferent_kind=afferent_kind)
    organism.resources.energy_per_step = max(30.0, organism.resources.energy_per_step)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    organism.representation_learning_enabled = False
    organism.representation_learning_rate = 0.0
    organism.adaptive_dendritic_enabled = False
    organism.adaptive_dendritic_shadow_enabled = False
    organism.adaptive_dendritic_fingerprint_enabled = False
    if mode == "variable_order":
        organism.enable_variable_order_learning(int(seed) + 7919, max_order=3)
    organism.actor_learning_rate = config.actor_learning_rate
    return organism


def _event(symbol: int, event_id: int) -> Event:
    return Event(
        "byte", "product-dev", int(event_id), float(event_id),
        {"value": int(symbol)}, {"trust": "signed-development-frame"},
    )


def _run_acquisition(
    organism: Organism,
    symbols: Sequence[int],
    config: ProductExperimentConfig,
    reward_symbols: Optional[Sequence[int]] = None,
) -> Tuple[List[float], List[float]]:
    bridge = EventBridge(organism, ByteTripleTransducer(), novelty=config.novelty, exploration=config.exploration)
    actions: List[float] = []
    targets: List[float] = []
    rewards = list(reward_symbols) if reward_symbols is not None else list(symbols)
    if len(rewards) != len(symbols):
        raise ValueError("reward stream length must match event stream length")
    for index, symbol in enumerate(symbols):
        target = product_target(int(symbol))
        action_event = bridge.ingest([_event(int(symbol), index)])
        if action_event is None:
            raise AssertionError("single byte channel must yield one action")
        action = float(action_event["proposal"]["value"])
        actions.append(action)
        targets.append(target)
        if index + 1 < len(symbols):
            reward_target = product_target(int(rewards[index]))
            bridge.outcome(OutcomeEvent(
                int(action_event["correlation_id"]), float(index) + 0.5,
                _reward(action, reward_target), "product-dev",
            ))
    if symbols:
        reward_target = product_target(int(rewards[-1]))
        # close() applies the terminal delayed outcome without a new action.
        bridge.outcome(OutcomeEvent(
            int(action_event["correlation_id"]), float(len(symbols)) - 0.5,
            _reward(actions[-1], reward_target), "product-dev",
        ))
    bridge.close()
    return actions, targets


def _frozen_action(
    checkpoint: Mapping[str, object], symbol: int, config: ProductExperimentConfig,
) -> Tuple[float, str]:
    """Score exactly one frame from a disposable checkpoint clone.

    No outcome is delivered and the clone is discarded immediately.  This is
    intentionally stricter than merely setting a learning rate to zero: every
    held-out action starts from the identical acquisition state, so delayed
    reward, routing, and exploration state cannot leak between test symbols.
    """
    source_digest = _canonical_digest(checkpoint)
    clone = json.loads(json.dumps(checkpoint))
    organism = Organism.from_state_dict(clone)
    bridge = EventBridge(organism, ByteTripleTransducer(), novelty=config.novelty, exploration=config.exploration)
    action_event = bridge.ingest([_event(int(symbol), 0)])
    if action_event is None:
        raise AssertionError("single held-out byte must yield one action")
    action = float(action_event["proposal"]["value"])
    # There is deliberately no OutcomeEvent and no bridge.close(): this object
    # is disposable, and the source checkpoint is never mutated.
    if _canonical_digest(checkpoint) != source_digest:
        raise AssertionError("held-out evaluation mutated the acquisition checkpoint")
    return action, source_digest


def _mae(actions: Sequence[float], targets: Sequence[float]) -> float:
    if not targets:
        return 0.0
    return sum(abs(float(action) - float(target)) for action, target in zip(actions, targets)) / float(len(targets))


def _sign_accuracy(actions: Sequence[float], targets: Sequence[float]) -> float:
    if not targets:
        return 0.0
    return sum(1.0 if float(action) * float(target) > 0.0 else 0.0 for action, target in zip(actions, targets)) / float(len(targets))


def _metrics(actions: Sequence[float], targets: Sequence[float]) -> Dict[str, float]:
    return {
        "skill": _skill(actions, targets, 0, len(targets)),
        "mae": _mae(actions, targets),
        "sign_accuracy": _sign_accuracy(actions, targets),
    }


def _lesion_product_path(state: Mapping[str, object]) -> Tuple[Dict[str, object], Optional[str]]:
    """Zero only the installed k3 feature's motor edge in a state clone."""
    cloned = json.loads(json.dumps(state))
    owners = dict(cloned.get("variable_order_feature_owners", {}))
    key = "k3:input-000|input-001|input-002"
    owner_id = owners.get(key)
    if owner_id is None:
        return cloned, None
    modules = dict(cloned.get("motor_modules", {}))
    owner = dict(modules.get(owner_id, {}))
    owner_cell = owner.get("cell_id")
    cells = dict(cloned.get("cells", {}))
    feature_cell = None
    for cell_id, payload in cells.items():
        if payload.get("activation_type") in ("dendritic_product", "dendritic_product_n") and tuple(payload.get("dendritic_sources", ())) == ("input-000", "input-001", "input-002"):
            feature_cell = cell_id
            break
    if feature_cell is None or owner_cell is None:
        return cloned, None
    changed = False
    for payload in cloned.get("synapses", []):
        if payload.get("source") == feature_cell and payload.get("destination") == owner_cell:
            payload["strength"] = 0.0
            payload["plasticity"] = 0.0
            changed = True
    return cloned, key if changed else None


def _table_diagnostic(acquisition: Sequence[int], heldout: Sequence[int]) -> Dict[str, object]:
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for symbol in acquisition:
        sums[symbol] = sums.get(symbol, 0.0) + product_target(symbol)
        counts[symbol] = counts.get(symbol, 0) + 1
    actions = [sums.get(symbol, 0.0) / counts[symbol] if symbol in counts else 0.0 for symbol in heldout]
    targets = [product_target(symbol) for symbol in heldout]
    return {"seen_acquisition_symbols": sorted(counts), "metrics": _metrics(actions, targets)}


def _run_mode(seed: int, mode: str, config: ProductExperimentConfig, manifests: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not manifests:
        raise ValueError("at least one leave-one-out manifest is required")
    cases: List[Dict[str, object]] = []
    shared_initial_structure = None
    for manifest in manifests:
        acquisition = tuple(int(value) for value in manifest["acquisition_symbols"])
        heldout_symbol = int(manifest["heldout_symbol"])
        organism = _new_organism(seed, mode, config)
        tape_rng = random.Random(int(seed) + 0xA11CE)
        organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(len(acquisition) + 1)))
        initial_structure = {
            "cells": tuple(sorted(organism.cells)),
            "synapses": tuple(edge.id for edge in organism.graph.iter_synapses()),
        }
        if shared_initial_structure is None:
            shared_initial_structure = initial_structure
        reward_symbols = None
        if mode == "shuffled_reward":
            reward_symbols = list(acquisition)
            random.Random(int(seed) + 0xBAD5EED + heldout_symbol).shuffle(reward_symbols)
        train_actions, train_targets = _run_acquisition(organism, acquisition, config, reward_symbols)
        checkpoint = organism.state_dict()
        checkpoint_digest = _canonical_digest(checkpoint)
        intact_actions = []
        lesion_actions = []
        source_digests = []
        for symbol in range(COMBINATION_COUNT):
            action, source_digest = _frozen_action(checkpoint, symbol, config)
            intact_actions.append(action)
            source_digests.append(source_digest)
        lesioned_state, lesion_key = _lesion_product_path(checkpoint)
        if lesion_key is not None:
            for symbol in range(COMBINATION_COUNT):
                action, source_digest = _frozen_action(lesioned_state, symbol, config)
                lesion_actions.append(action)
                if source_digest == checkpoint_digest:
                    # A lesion must be a state change in its own clone while
                    # the intact acquisition checkpoint remains untouched.
                    raise AssertionError("lesion clone unexpectedly equals intact checkpoint")
        targets = [product_target(symbol) for symbol in range(COMBINATION_COUNT)]
        intact_metrics = _metrics(intact_actions, targets)
        lesion_metrics = _metrics(lesion_actions, targets) if lesion_actions else None
        feature_key = "k3:input-000|input-001|input-002"
        installed = feature_key in dict(checkpoint.get("variable_order_feature_owners", {}))
        cases.append({
            "heldout_symbol": heldout_symbol,
            "heldout_sign": 1 if product_target(heldout_symbol) > 0 else -1,
            "acquisition_stream_sha256": _canonical_digest(list(acquisition)),
            "acquisition_metrics": _metrics(train_actions, train_targets),
            "frozen_all_symbol_actions": intact_actions,
            "frozen_all_symbol_metrics": intact_metrics,
            "frozen_heldout_action": intact_actions[heldout_symbol],
            "frozen_heldout_metrics": _metrics([intact_actions[heldout_symbol]], [targets[heldout_symbol]]),
            "product_path_lesion_actions": lesion_actions,
            "product_path_lesion_metrics": lesion_metrics,
            "product_path_lesion_drop": (intact_metrics["skill"] - lesion_metrics["skill"]) if lesion_metrics else None,
            "frozen_evaluation_symbols": list(range(COMBINATION_COUNT)),
            "intact_lesion_protocol_match": lesion_key is None or len(lesion_actions) == COMBINATION_COUNT,
            "k3_product_feature_installed": installed,
            "product_path_lesion_applied": lesion_key is not None,
            "checkpoint_digest": checkpoint_digest,
            "heldout_checkpoint_unchanged": all(digest == checkpoint_digest for digest in source_digests),
            "frozen_read_only": True,
            "event_count_acquisition": len(acquisition),
            "initial_structure": initial_structure,
        })
    return {
        "seed": int(seed),
        "mode": mode,
        "target_blind_event_path": True,
        "initial_structure": shared_initial_structure,
        "cases": cases,
        "all_cases_frozen_read_only": all(bool(case["frozen_read_only"]) and bool(case["heldout_checkpoint_unchanged"]) for case in cases),
        "k3_product_feature_installed_all_cases": all(bool(case["k3_product_feature_installed"]) for case in cases),
        "product_path_lesion_applied_all_cases": all(bool(case["product_path_lesion_applied"]) for case in cases),
        "intact_lesion_protocol_match_all_cases": all(bool(case["intact_lesion_protocol_match"]) for case in cases),
    }


def run_product_experiment(seeds: Sequence[int] = (0, 1, 2, 3), config: Optional[ProductExperimentConfig] = None) -> Dict[str, object]:
    config = config or ProductExperimentConfig()
    config.validate()
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("at least one development seed is required")
    started = time.monotonic()
    manifests = {
        str(seed): [frozen_manifest(seed, config, heldout_symbol=symbol) for symbol in range(COMBINATION_COUNT)]
        for seed in normalized_seeds
    }
    runs = {mode: [_run_mode(seed, mode, config, manifests[str(seed)]) for seed in normalized_seeds] for mode in MODES}
    variable = runs["variable_order"]
    additive = runs["nonlinear_additive_context"]
    shuffled = runs["shuffled_reward"]
    def _case_metric(run: Mapping[str, object], field: str) -> List[float]:
        return [float(case["frozen_heldout_metrics"][field]) for case in run["cases"]]
    def _flatten_metric(mode: str, field: str) -> List[float]:
        return [value for run in runs[mode] for value in _case_metric(run, field)]
    margins = {
        "variable_order_minus_nonlinear_additive_skill": [
            a - b for run_v, run_a in zip(variable, additive)
            for a, b in zip(_case_metric(run_v, "skill"), _case_metric(run_a, "skill"))
        ],
        "variable_order_minus_shuffled_reward_skill": [
            a - b for run_v, run_s in zip(variable, shuffled)
            for a, b in zip(_case_metric(run_v, "skill"), _case_metric(run_s, "skill"))
        ],
    }
    # These are development diagnostics only.  Requiring two seeds, a
    # predeclared margin, and an explicit lesion prevents a single lucky run
    # from being described as a mechanism result.
    margin_floor = 0.10
    lesion_floor = 0.05
    margin_values = margins["variable_order_minus_nonlinear_additive_skill"]
    positive = bool(margin_values) and sum(margin_values) / float(len(margin_values)) >= margin_floor
    lesion_values = [float(case["product_path_lesion_drop"]) for run in variable for case in run["cases"] if case["product_path_lesion_drop"] is not None]
    lesion_mean = sum(lesion_values) / float(len(lesion_values)) if lesion_values else 0.0
    lesion_positive = len(lesion_values) == len(normalized_seeds) * COMBINATION_COUNT and lesion_mean >= lesion_floor
    protocol_manifest = {
        "seeds": list(normalized_seeds),
        "leave_one_combination_out_symbols": list(range(COMBINATION_COUNT)),
        "positive_sign_symbols": [symbol for symbol in range(COMBINATION_COUNT) if product_target(symbol) > 0],
        "negative_sign_symbols": [symbol for symbol in range(COMBINATION_COUNT) if product_target(symbol) < 0],
        "acquisition_repetitions": config.acquisition_repetitions,
        "manifests": manifests,
    }
    dependency_files = [
        "soma/organism.py", "soma/events/bridge.py", "soma/events/envelope.py",
        "soma/events/channel.py", "soma/transducers/sdk.py",
    ]
    dependency_hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in dependency_files
    }
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *dependency_files], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    constant_controls = {}
    representative_targets = [product_target(symbol) for symbol in range(COMBINATION_COUNT)]
    for name, value in (("constant_positive", 0.8), ("constant_negative", -0.8), ("zero", 0.0)):
        constant_controls[name] = _metrics([value] * COMBINATION_COUNT, representative_targets)
    table_cases = []
    for symbol in range(COMBINATION_COUNT):
        manifest = manifests[str(normalized_seeds[0])][symbol]
        table_cases.append({
            "heldout_symbol": symbol,
            **_table_diagnostic(tuple(manifest["acquisition_symbols"]), [symbol]),
        })
    return {
        "protocol": PROTOCOL,
        "development_only": True,
        "acceptance_gating": False,
        "r6_qualification": "not_evaluated",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": dependency_hashes,
        "git_revision": revision,
        "dependency_files_dirty": dirty,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "manifest_digest": _canonical_digest(protocol_manifest),
        "manifest": protocol_manifest,
        "event_boundary": {"payload_channels": ["byte.value"], "forbidden_target_metadata": True, "transducer": ByteTripleTransducer().spec.to_dict()},
        "matched_streams": True,
        "matched_initial_structure": all(run["initial_structure"] == variable[0]["initial_structure"] for mode in MODES for run in runs[mode]),
        "controls": {
            "matched": ["nonlinear_additive_context"],
            "shuffled_reward": True,
            "constant_controls": constant_controls,
            "table_diagnostic": {"cases": table_cases},
        },
        "runs": runs,
        "comparison": {
            "leave_one_out_skill_margins": margins,
            "margin_floor": margin_floor,
            "lesion_floor": lesion_floor,
            "mean_variable_order_minus_nonlinear_additive_skill": sum(margin_values) / float(len(margin_values)),
            "mean_product_path_lesion_drop": lesion_mean,
            "variable_order_mae": _flatten_metric("variable_order", "mae"),
            "variable_order_sign_accuracy": _flatten_metric("variable_order", "sign_accuracy"),
            "nonlinear_additive_mae": _flatten_metric("nonlinear_additive_context", "mae"),
            "nonlinear_additive_sign_accuracy": _flatten_metric("nonlinear_additive_context", "sign_accuracy"),
        },
        "diagnostics": {
            "minimum_seed_count_for_positive_result": 2,
            "balanced_leave_one_out_coverage": all(
                sorted(int(case["heldout_symbol"]) for case in run["cases"]) == list(range(COMBINATION_COUNT))
                for mode in MODES for run in runs[mode]
            ),
            "both_product_signs_covered": all(
                {int(case["heldout_sign"]) for case in run["cases"]} == {-1, 1}
                for mode in MODES for run in runs[mode]
            ),
            "frozen_read_only_all_cases": all(run["all_cases_frozen_read_only"] for mode in MODES for run in runs[mode]),
            "intact_lesion_protocol_match": all(run["intact_lesion_protocol_match_all_cases"] for mode in MODES for run in runs[mode]),
            "heldout_margin_positive": len(normalized_seeds) >= 2 and positive,
            "product_path_lesion_positive": len(normalized_seeds) >= 2 and lesion_positive,
            "k3_installed_all_variable_order_cases": all(run["k3_product_feature_installed_all_cases"] for run in variable),
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds", help="development seed (repeatable; default 0,1,2,3)")
    parser.add_argument("--report", default="reports/r6-compositional-byte-event-dev.json")
    args = parser.parse_args(argv)
    report = run_product_experiment(tuple(args.seeds) if args.seeds else (0, 1, 2, 3))
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

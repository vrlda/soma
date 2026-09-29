"""Development-only temporal composition continuation at the 512 budget.

Protocol v4 reuses the frozen v3 task, streams, controls, thresholds, lesion,
resume, and read-only evaluation.  It changes only the already-predeclared
scout budget from the v3 transportability probe (256) to the exploratory 512
budget.  This is not R6 acceptance evidence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import r6_temporal_composition_pilot as base


PROTOCOL = "r6-temporal-composition-pilot-v4"
PERSISTENT_EVIDENCE_BUDGET = 512
TRANSPORTABILITY_NOTE = (
    "v3 budget 256 failed transportability in the frozen leave-one-out pilot; "
    "v4 budget 512 is exploratory pending matched calibration and is not an R6 gate."
)

_examples = base._examples
_frame = base._frame
_digest = base._digest
_frozen_score = base._frozen_score
_lesion_feature = base._lesion_feature
_lesion_target_k3 = base._lesion_target_k3
SignedHistoryBitsTransducer = base.SignedHistoryBitsTransducer
TARGET_KEY = base.TARGET_KEY
MODES = base.MODES


def _new_organism(seed: int, mode: str):
    return base._new_organism(seed, mode, persistent_budget=PERSISTENT_EVIDENCE_BUDGET)


def _mean(values):
    return sum(float(value) for value in values) / float(max(1, len(values)))


def run_temporal_pilot(
    seeds: Sequence[int] = (0,),
    acquisition_examples: int = 512,
    evaluation_examples: int = 16,
) -> Dict[str, object]:
    report = base.run_temporal_pilot(
        seeds=seeds,
        acquisition_examples=acquisition_examples,
        evaluation_examples=evaluation_examples,
        persistent_budget=PERSISTENT_EVIDENCE_BUDGET,
    )
    report["protocol"] = PROTOCOL
    report["persistent_scout_budget"] = PERSISTENT_EVIDENCE_BUDGET
    report["transportability_note"] = TRANSPORTABILITY_NOTE
    report["v3_reference_protocol"] = base.PROTOCOL

    additive_by_seed = {}
    persistent_by_seed = {}
    for run in report["runs"]["nonlinear_additive_context"]:
        additive_by_seed[int(run["seed"])] = {int(case["heldout_combo"]): case for case in run["cases"]}
    for run in report["runs"]["persistent_variable_order"]:
        persistent_by_seed[int(run["seed"])] = {int(case["heldout_combo"]): case for case in run["cases"]}

    controls_by_combo = {int(case["heldout_combo"]): case for case in report["controls"]["cases"]}
    heldout_rows = []
    target_ranks = []
    lesion_drops = []
    for seed, cases in persistent_by_seed.items():
        for combo, case in sorted(cases.items()):
            additive = additive_by_seed[seed][combo]
            control = controls_by_combo[combo]
            persistent_skill = float(case["frozen_metrics"]["skill"])
            case["persistent_target_rank"] = (
                int(case["persistent_winner_rank"])
                if case.get("persistent_winner_key") == TARGET_KEY and case.get("persistent_winner_rank") is not None
                else None
            )
            case["persistent_vs_additive_skill"] = persistent_skill - float(additive["frozen_metrics"]["skill"])
            case["persistent_vs_suffix3_skill"] = persistent_skill - float(control["suffix3"]["skill"])
            case["persistent_vs_suffix5_skill"] = persistent_skill - float(control["suffix5"]["skill"])
            target_ranks.append(case["persistent_target_rank"])
            if case.get("lesion_skill_drop") is not None:
                lesion_drops.append(float(case["lesion_skill_drop"]))
            heldout_rows.append({
                "seed": seed,
                "heldout_combo": combo,
                "heldout_sign": case["heldout_sign"],
                "target_rank": case["persistent_target_rank"],
                "target_k3_installed": case["target_k3_installed"],
                "persistent_skill": persistent_skill,
                "persistent_vs_additive_skill": case["persistent_vs_additive_skill"],
                "persistent_vs_suffix3_skill": case["persistent_vs_suffix3_skill"],
                "persistent_vs_suffix5_skill": case["persistent_vs_suffix5_skill"],
                "lesion_skill_drop": case.get("lesion_skill_drop"),
            })

    installed = [row for row in heldout_rows if row["target_k3_installed"]]
    report["v4_heldout_performance"] = {
        "cases": heldout_rows,
        "persistent_target_install_count": len(installed),
        "persistent_target_install_coverage": len(installed) / float(max(1, len(heldout_rows))),
        "target_rank_values": target_ranks,
        "target_rank_mean": _mean([value for value in target_ranks if value is not None]),
        "persistent_vs_additive_skill": [row["persistent_vs_additive_skill"] for row in heldout_rows],
        "persistent_vs_suffix3_skill": [row["persistent_vs_suffix3_skill"] for row in heldout_rows],
        "persistent_vs_suffix5_skill": [row["persistent_vs_suffix5_skill"] for row in heldout_rows],
        "mean_persistent_vs_additive_skill": _mean([row["persistent_vs_additive_skill"] for row in heldout_rows]),
        "mean_persistent_vs_suffix3_skill": _mean([row["persistent_vs_suffix3_skill"] for row in heldout_rows]),
        "mean_persistent_vs_suffix5_skill": _mean([row["persistent_vs_suffix5_skill"] for row in heldout_rows]),
        "lesion_skill_drops": lesion_drops,
        "lesion_consistent_positive": bool(lesion_drops) and all(value > 0.05 for value in lesion_drops),
    }
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--acquisition-examples", type=int, default=512)
    parser.add_argument("--evaluation-examples", type=int, default=16)
    parser.add_argument("--report", default="/tmp/r6-temporal-composition-pilot-v4.json")
    args = parser.parse_args()
    report = run_temporal_pilot(
        tuple(args.seeds) if args.seeds else (0,),
        args.acquisition_examples,
        args.evaluation_examples,
    )
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["diagnostics"], sort_keys=True))

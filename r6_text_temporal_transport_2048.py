"""Frozen development-only TextBytes transport-2048 protocol.

This is a separately versioned continuation of ``r6_text_temporal_transport_pilot``.
It changes only the predeclared persistent-scout evidence budget and the
independent seeds/acquisition size.  The event schema, lag wrapper, candidate
family, reward thresholds, controls, no-credit path, frozen evaluation,
lesion, and resume checks are inherited from the v1 runner.  This protocol is
diagnostic evidence and never qualifies the locked R6 tier report.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

import r6_text_temporal_transport_pilot as base


PROTOCOL = "r6-text-temporal-transport-pilot-v2-2048"
PERSISTENT_EVIDENCE_BUDGET = 2048
SEEDS = (2137, 2271)
HELDOUT_CASES_PER_SEED = 8
ACQUISITION_EXAMPLES = 2048
EVALUATION_EXAMPLES = 64
CASE_TIME_LIMIT_SECONDS = 180.0
STATE_BYTES_LIMIT = 2_000_000
TARGET_CASE_FLOOR = 0.75
NULL_TARGET_INSTALL_MAX = 0
LESION_SKILL_THRESHOLD = 0.05

# These are frozen before executing the matrix.  "Null install" is explicitly
# a target-feature install in the paired shuffled-reward control; unrelated
# learner features are reported separately and are not silently conflated with
# discovery of the evaluator-only target.
PREDECLARED_CRITERIA = {
    "matrix": {
        "seeds": list(SEEDS),
        "heldout_cases_per_seed": HELDOUT_CASES_PER_SEED,
        "total_cases": len(SEEDS) * HELDOUT_CASES_PER_SEED,
        "acquisition_examples": ACQUISITION_EXAMPLES,
        "frozen_evaluation_examples": EVALUATION_EXAMPLES,
        "candidate_family_size": 680,
    },
    "persistent_quality": {
        "case_pass": "target_k3_installed and lesion_skill_drop > 0.05",
        "minimum_case_pass_rate": TARGET_CASE_FLOOR,
        "rationale": "A target installation is only useful evidence when its targeted lesion removes held-out skill; 0.05 is the v1 development threshold and is not retuned here.",
    },
    "paired_shuffled_null": {
        "null_install_definition": "target_k3_installed in lag_shuffled",
        "maximum_null_target_installs": NULL_TARGET_INSTALL_MAX,
        "rationale": "The shuffled-reward stream preserves event and reward marginals while breaking target alignment; no evaluator-target feature may be accepted from it.",
    },
    "execution": {
        "max_case_seconds": CASE_TIME_LIMIT_SECONDS,
        "max_checkpoint_state_bytes": STATE_BYTES_LIMIT,
        "required": [
            "all 16 persistent cases execute",
            "all modes satisfy frozen read-only evaluation and checkpoint resume digests",
            "withheld frames use genuine no-credit and do not update evidence",
            "all persistent cases expose the 680-member bias-excluded family",
            "both target signs and all eight leave-one-combination-out cases are covered",
        ],
    },
    "status": "development_only; not an R6 acceptance gate",
}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_metadata() -> Dict[str, object]:
    paths = ["r6_text_temporal_transport_2048.py", "r6_text_temporal_transport_pilot.py"]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "sha256": {path: _sha256(path) for path in paths},
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
    }


def _case_pass(case: Mapping[str, object]) -> bool:
    drop = case.get("lesion_skill_drop")
    return bool(case.get("target_k3_installed")) and drop is not None and float(drop) > LESION_SKILL_THRESHOLD


def _cases(report: Mapping[str, object], mode: str) -> Iterable[Mapping[str, object]]:
    for run in report["runs"][mode]:
        yield from run["cases"]


def _summarize(report: Dict[str, object]) -> Dict[str, object]:
    persistent = list(_cases(report, "lag_persistent"))
    shuffled = list(_cases(report, "lag_shuffled"))
    modes = tuple(base.MODES)
    all_cases = [case for mode in modes for case in _cases(report, mode)]
    per_seed = []
    for seed in SEEDS:
        rows = [case for case in persistent if int(case["seed"]) == seed]
        null_rows = [case for case in shuffled if int(case["seed"]) == seed]
        passed = [_case_pass(case) for case in rows]
        per_seed.append({
            "seed": seed,
            "persistent_cases": len(rows),
            "target_install_count": sum(bool(case["target_k3_installed"]) for case in rows),
            "positive_lesion_count": sum(
                case.get("lesion_skill_drop") is not None
                and float(case["lesion_skill_drop"]) > LESION_SKILL_THRESHOLD
                for case in rows
            ),
            "case_pass_count": sum(passed),
            "case_pass_rate": sum(passed) / float(max(1, len(passed))),
            "null_target_install_count": sum(bool(case["target_k3_installed"]) for case in null_rows),
            "null_any_feature_install_count": sum(
                int(case["resources"].get("variable_order_install_count", 0)) > 0
                for case in null_rows
            ),
        })

    null_target_installs = sum(bool(case["target_k3_installed"]) for case in shuffled)
    resource_guard = all(
        float(case.get("case_elapsed_seconds", float("inf"))) <= CASE_TIME_LIMIT_SECONDS
        and int(case["resources"].get("state_bytes", STATE_BYTES_LIMIT + 1)) <= STATE_BYTES_LIMIT
        for case in all_cases
    )
    execution_gates = {
        "exact_case_count": len(persistent) == len(SEEDS) * HELDOUT_CASES_PER_SEED,
        "balanced_leave_one_out": bool(report["diagnostics"].get("balanced_leave_one_out")),
        "both_target_signs": bool(report["diagnostics"].get("both_target_signs_covered")),
        "frozen_read_only": bool(report["diagnostics"].get("frozen_read_only_all")),
        "resume_digest": bool(report["diagnostics"].get("resume_digest_all")),
        "genuine_no_credit": bool(report["diagnostics"].get("withheld_frames_no_credit_all")),
        "bias_excluded_family": bool(report["diagnostics"].get("candidate_family_excludes_fixed_bias")),
        "resource_guard": resource_guard,
    }
    quality_gates = {
        "persistent_case_pass_rate": sum(_case_pass(case) for case in persistent) / float(max(1, len(persistent))) >= TARGET_CASE_FLOOR,
        "paired_shuffled_zero_target_installs": null_target_installs <= NULL_TARGET_INSTALL_MAX,
    }
    report["predeclared_criteria"] = PREDECLARED_CRITERIA
    report["per_seed_summary"] = per_seed
    report["qualification_summary"] = {
        "development_only": True,
        "execution_gates": execution_gates,
        "quality_gates": quality_gates,
        "persistent_case_pass_count": sum(_case_pass(case) for case in persistent),
        "persistent_case_pass_rate": sum(_case_pass(case) for case in persistent) / float(max(1, len(persistent))),
        "null_target_install_count": null_target_installs,
        "null_any_feature_install_count": sum(
            int(case["resources"].get("variable_order_install_count", 0)) > 0 for case in shuffled
        ),
        "qualifies_development_protocol": all(execution_gates.values()) and all(quality_gates.values()),
        "r6_acceptance_impact": "none",
    }
    report["source_metadata"] = _source_metadata()
    return report


def run_transport_2048() -> Dict[str, object]:
    """Run exactly the frozen two-seed/16-case development matrix."""
    report = base.run_transport_pilot(
        SEEDS,
        acquisition_examples=ACQUISITION_EXAMPLES,
        evaluation_examples=EVALUATION_EXAMPLES,
        persistent_budget=PERSISTENT_EVIDENCE_BUDGET,
        protocol=PROTOCOL,
    )
    return _summarize(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="/tmp/r6-text-temporal-transport-pilot-v2-2048.json")
    args = parser.parse_args()
    output = run_transport_2048()
    Path(args.report).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["qualification_summary"], sort_keys=True))

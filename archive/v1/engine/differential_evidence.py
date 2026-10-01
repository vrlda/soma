#!/usr/bin/env python3
"""Differential evidence check: R1 pure functions, Python vs Rust engine.

Generates a seeded op tape (evidence updates, posteriors, switches, probe
verdicts, calibration), runs both engines, and compares return values plus
final per-circuit state within tolerance.
"""

import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOLERANCE = 1e-9


def build_tape(seed=0, ops=300):
    rng = random.Random(seed + 555)
    circuits = ["m%d" % index for index in range(4)]
    tape = []
    for _ in range(ops):
        kind = rng.random()
        if kind < 0.6:
            tape.append({
                "op": "update",
                "circuit": rng.choice(circuits),
                "outcome": rng.uniform(-3.0, 3.0),
                "probe": rng.random() < 0.2,
                "decay": 0.999,
                "step": rng.randint(0, 1000),
                "accumulate": True,
            })
        elif kind < 0.75:
            tape.append({
                "op": "posterior",
                "scores": {name: rng.uniform(-8.0, 2.0) for name in circuits},
                "novelty": 0.05,
            })
        elif kind < 0.85:
            tape.append({
                "op": "switch",
                "c": rng.random(), "i": rng.random(),
                "cn": rng.randint(0, 40), "in": rng.randint(0, 40),
            })
        elif kind < 0.95:
            tape.append({
                "op": "probe",
                "cll": rng.uniform(-8.0, 2.0), "cn": rng.randint(0, 20),
                "ill": rng.uniform(-8.0, 2.0), "in": rng.randint(0, 20),
            })
        else:
            size = rng.randint(1, 30)
            tape.append({
                "op": "calibration",
                "probs": [rng.random() for _ in range(size)],
                "outcomes": [rng.randint(0, 1) for _ in range(size)],
            })
    return {"ops": tape}


def run_python(tape):
    from soma.routing.evidence import (
        CircuitEvidence, calibration_metrics, normalize_posterior, probe_decision,
        should_switch, update_evidence,
    )
    circuits = {}
    returns = []
    for op in tape["ops"]:
        kind = op["op"]
        if kind == "update":
            evidence = circuits.setdefault(op["circuit"], CircuitEvidence(op["circuit"]))
            returns.append(repr(update_evidence(
                evidence, op["outcome"], op["probe"], op["decay"], op["step"], op["accumulate"])))
        elif kind == "posterior":
            posterior = normalize_posterior(op["scores"], op["novelty"])
            returns.append(",".join("%r:%r" % (key, posterior[key]) for key in sorted(posterior)))
        elif kind == "switch":
            returns.append(str(should_switch(op["c"], op["i"], op["cn"], op["in"])))
        elif kind == "probe":
            returns.append(probe_decision(op["cll"], op["cn"], op["ill"], op["in"]))
        elif kind == "calibration":
            metrics = calibration_metrics(op["probs"], op["outcomes"])
            returns.append("%r,%r,%r" % (metrics["nll"], metrics["brier"], metrics["ece"]))
    state = []
    for name in sorted(circuits):
        evidence = circuits[name]
        state.append("%r:%r,%r,%r,%r,%r,%r" % (
            name, evidence.outcome_mean, evidence.outcome_scale, evidence.eff_n,
            evidence.log_evidence, evidence.normal_outcomes, evidence.probe_outcomes))
    return returns, state


def compare_float_strings(left, right):
    import math
    try:
        lval, rval = float(left), float(right)
    except ValueError:
        return left == right
    if math.isnan(lval) or math.isnan(rval):
        return math.isnan(lval) and math.isnan(rval)
    return abs(lval - rval) <= TOLERANCE * max(1.0, abs(lval), abs(rval))


def _strip_quotes(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def compare_structured(left, right):
    left_parts, right_parts = left.split(","), right.split(",")
    if len(left_parts) != len(right_parts):
        return False, abs(len(left_parts) - len(right_parts))
    worst = 0.0
    for left_item, right_item in zip(left_parts, right_parts):
        lkey, sep, lval = left_item.partition(":")
        rkey, _, rval = right_item.partition(":")
        if not sep:
            if left_item.lower() in ("true", "false") or right_item.lower() in ("true", "false"):
                if left_item.lower() != right_item.lower():
                    return False, float("inf")
                continue
            if not compare_float_strings(left_item, right_item):
                return False, float("inf")
            try:
                worst = max(worst, abs(float(left_item) - float(right_item)))
            except ValueError:
                pass
            continue
        if _strip_quotes(lkey) != _strip_quotes(rkey):
            return False, float("inf")
        if not compare_float_strings(lval, rval):
            # Non-numeric payloads (switch/probe verdicts) compare exactly.
            if _strip_quotes(lval) != _strip_quotes(rval):
                return False, float("inf")
            continue
        try:
            worst = max(worst, abs(float(lval) - float(rval)))
        except ValueError:
            pass
    return True, worst


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tape = build_tape(seed)
    python_returns, python_state = run_python(tape)
    binary = "./engine/soma-engine/target/release/soma-evidence"
    raw = subprocess.run([binary], input=json.dumps(tape), check=True,
                         capture_output=True, text=True).stdout
    rust = json.loads(raw)
    worst, mismatches = 0.0, 0
    for left, right in zip(python_returns, rust["returns"]):
        if left != right:
            ok, gap = compare_structured(left, right)
            if not ok:
                mismatches += 1
            else:
                worst = max(worst, gap)
    for left, right in zip(python_state, rust["state"]):
        ok, gap = compare_structured(left, right)
        if not ok:
            mismatches += 1
        else:
            worst = max(worst, gap)
    result = {
        "seed": seed,
        "returns": len(python_returns),
        "worst_numeric_gap": worst,
        "mismatches": mismatches,
        "tolerance": TOLERANCE,
    }
    result["all_passed"] = mismatches == 0 and worst < TOLERANCE
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

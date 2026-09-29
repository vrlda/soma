#!/usr/bin/env python3
"""Frozen R6b reclamation-under-interference protocol (v3, diagnostic).

SCOPE: mechanism qualification only. This does NOT close R6 and does not
alter any v2 acceptance gate. It tests the one SOMA-specific mechanism that
v2 left unengaged (grace-window/reuse reclamation never fired: budgets never
bound), in the regime where it must fire.

Hypothesis (predeclared): under forced eviction with interfering streams, a
grace-window + reuse reclamation policy retains consolidated contexts better
than deterministic LRU replacement, at comparable acquisition cost.

Design (frozen before first run):
- Subjects: SequenceCircuitMemory(order=8, budget=256) vs SuffixNgramControl
  (order=8, budget=256). Budget 256 < 511 (full order-8 binary space), so
  eviction MUST engage. Same order, same context budget, same prior.
- Stream (frozen E1 manifest bytes, manifest itself untouched): acquisition
  book[0] = skill A; books[1..3] = interferers; book[4] = second skill B.
- Metrics (bits per input bit, observational clones, never mutating state):
  A_pre (after A), A_post (after interferers), B_post (acquisition check).
  retention_loss = A_post - A_pre per subject.
- Gates (frozen, rationale inline):
  - engagement: both subjects resident within ENGAGEMENT_BAND of cap AND
    model reclamations >= MIN_RECLAMATIONS. If unfired, the experiment is
    VOID (not a pass): no mechanism engaged. (v1 required exactly >= cap;
    corrected in v2: reclaim-batch steady state sits at cap-batch. Only
    this gate changed; retention/acquisition/determinism identical.)
  - retention: model_loss + 0.001 <= control_loss (practical separation,
    same philosophy as v2 control margins; direction favors consolidation).
  - acquisition: model B_post <= control B_post + 0.005 (non-inferiority
    tolerance: reclamation must not wreck new learning; 0.005 is 5x the
    v2 margin, deliberately lenient because this gate guards catastrophe,
    not equivalence).
  - determinism: full-stream digest identical across two in-harness runs.

One locked run. Result reported to reports/r6-reclamation.json.
"""

import copy
import hashlib
import json
import sys
import time


def _log(phase):
    print("[r6b] %s %.1fs" % (phase, time.perf_counter() - _log.start),
          flush=True)


_log.start = time.perf_counter()

from r6_tier_benchmark import SuffixNgramControl, _evaluate as _tier_evaluate
from soma.evaluation.english import bit_stream, load_verified_book_corpus
from soma.memory import SequenceCircuitMemory

PROTOCOL = "r6b-reclamation-v2"
ORDER = 8
BUDGET = 256
RETENTION_MARGIN = 0.001
ACQUISITION_TOLERANCE = 0.005
# v1 engagement required model_circuits >= BUDGET and returned VOID on a
# locked run where the mechanism demonstrably fired (23M reclamations, both
# subjects at cap). Root cause: _reclaim evicts a batch BEFORE insert, so the
# model's steady state sits at BUDGET-batch (batch=2 here), never exactly at
# BUDGET. v2 corrects the gate to the steady-state band with rationale, not
# from the retention outcome (retention/acquisition gates and margins are
# byte-identical to v1).
ENGAGEMENT_BAND = 2
MIN_RECLAMATIONS = 1000
MANIFEST = "reports/e1-manifest.json"


def _digest(memory):
    payload = json.dumps(memory.state_dict(), sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _train(memory, data):
    memory.reset_history()
    bits, _ = bit_stream(data)
    for symbol in bits[:-1]:
        memory.observe(symbol)


def _evaluate(memory, data):
    clone = copy.deepcopy(memory)
    return _tier_evaluate(clone, data)


def run_stream():
    with open(MANIFEST) as handle:
        manifest = json.load(handle)
    parts = load_verified_book_corpus(manifest, MANIFEST)
    acquisition = parts["acquisition"]
    if len(acquisition) < 5:
        raise ValueError("E1 manifest must provide at least 5 acquisition books")
    book_a = acquisition[0][1]
    interferers = [data for _, data in acquisition[1:4]]
    book_b = acquisition[4][1]

    model = SequenceCircuitMemory((0, 1), max_order=ORDER, max_circuits=BUDGET,
                                  min_support=2, prior=0.5)
    control = SuffixNgramControl(ORDER, BUDGET)
    _train(model, book_a)
    _train(control, book_a)
    _log("trained A")
    a_pre = (_evaluate(model, book_a), _evaluate(control, book_a))
    _log("scored A_pre")
    for data in interferers:
        _train(model, data)
        _train(control, data)
    _log("trained interferers")
    a_post = (_evaluate(model, book_a), _evaluate(control, book_a))
    _log("scored A_post")
    _train(model, book_b)
    _train(control, book_b)
    _log("trained B")
    b_post = (_evaluate(model, book_b), _evaluate(control, book_b)
    )
    _log("scored B_post")
    return {
        "a_pre": a_pre,
        "a_post": a_post,
        "b_post": b_post,
        "model_circuits": len(model.circuits),
        "control_contexts": len(control.contexts),
        "model_reclaimed": model.circuits_reclaimed,
        "model_digest": _digest(model),
        "control_digest": _digest(control),
    }


def main():
    _log.start = time.perf_counter()
    first = run_stream()
    _log("stream 1 done")
    second = run_stream()
    _log("stream 2 done")
    model_loss = first["a_post"][0] - first["a_pre"][0]
    control_loss = first["a_post"][1] - first["a_pre"][1]
    gates = {
        "engagement": (first["model_circuits"] >= BUDGET - ENGAGEMENT_BAND
                       and first["control_contexts"] >= BUDGET - ENGAGEMENT_BAND
                       and first["model_reclaimed"] >= MIN_RECLAMATIONS),
        "retention": model_loss + RETENTION_MARGIN <= control_loss,
        "acquisition": first["b_post"][0] <= first["b_post"][1] + ACQUISITION_TOLERANCE,
        "determinism": (first["model_digest"] == second["model_digest"]
                        and first["control_digest"] == second["control_digest"]),
    }
    result = {
        "protocol": PROTOCOL,
        "order": ORDER,
        "budget": BUDGET,
        "margins": {"retention": RETENTION_MARGIN,
                    "acquisition_tolerance": ACQUISITION_TOLERANCE},
        "metrics": {
            "model_a_pre": first["a_pre"][0],
            "control_a_pre": first["a_pre"][1],
            "model_a_post": first["a_post"][0],
            "control_a_post": first["a_post"][1],
            "model_retention_loss": model_loss,
            "control_retention_loss": control_loss,
            "model_b_post": first["b_post"][0],
            "control_b_post": first["b_post"][1],
            "model_circuits": first["model_circuits"],
            "control_contexts": first["control_contexts"],
            "model_reclaimed": first["model_reclaimed"],
        },
        "gates": gates,
        "all_passed": all(gates.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    with open("reports/r6-reclamation.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

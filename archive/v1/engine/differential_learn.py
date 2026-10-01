#!/usr/bin/env python3
"""Differential learning kernels: Hebbian + actor updates, Rust vs Python.

Builds a small organism, seeds traces/plasticity/strengths deterministically,
exports the fixture, runs both engines once, and compares strengths.
Routing-coupled module updates stay Python-side by design (documented).
"""

import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOLERANCE = 1e-9


def build_fixture(seed=0):
    from soma.organism import Organism
    organism = Organism.create_default(
        input_size=3, hidden_size=4, output_size=2, seed=seed,
        max_cells=64, max_synapses=256)
    rng = random.Random(seed + 913)
    for synapse in organism.graph.iter_synapses():
        synapse.strength = rng.uniform(-0.9, 0.9)
        synapse.plasticity = rng.uniform(0.1, 1.0)
        synapse.eligibility_trace = rng.uniform(-1.0, 1.0)
        synapse.actor_eligibility_trace = rng.uniform(-1.0, 1.0)
    cells = []
    for identifier in sorted(organism.cells):
        cell = organism.cells[identifier]
        cells.append({
            "id": identifier,
            "kind": cell.kind,
            "threshold": cell.threshold,
            "adaptation": cell.adaptation,
            "bias": cell.representation_bias,
            "activation": cell.activation,
            "activation_type": cell.activation_type,
            "dendritic_sources": list(cell.dendritic_sources),
            "dendritic_normalizer": cell.dendritic_normalizer,
            "dormant": False,
        })
    synapses = [{"source": s.source, "destination": s.destination, "strength": s.strength,
                 "plasticity": s.plasticity, "trace": s.eligibility_trace,
                 "actor_trace": s.actor_eligibility_trace}
                for s in organism.graph.iter_synapses()]
    return organism, {
        "cells": cells,
        "synapses": synapses,
        "input_ids": list(organism.input_ids),
        "output_ids": list(organism.output_ids),
        "steps": [],
        "passes": 2,
        "learning_rate": 0.08,
        "actor_learning_rate": 0.40,
        "prediction_error": 0.35,
    }


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    organism, fixture = build_fixture(seed)
    binary = "./engine/soma-engine/target/release/soma-learn"
    raw = subprocess.run([binary], input=json.dumps(fixture), check=True,
                         capture_output=True, text=True).stdout
    rust = {(source, destination): strength
            for source, destination, strength in json.loads(raw)["strengths"]}
    organism.learning_rate = 0.08
    organism.actor_learning_rate = 0.40
    organism._apply_reward(0.35)
    organism._apply_actor_reward(0.35)
    worst = 0.0
    count = 0
    for synapse in organism.graph.iter_synapses():
        key = (synapse.source, synapse.destination)
        worst = max(worst, abs(synapse.strength - rust[key]))
        count += 1
    result = {
        "seed": seed,
        "synapses": count,
        "worst_strength_gap": worst,
        "tolerance": TOLERANCE,
    }
    result["all_passed"] = worst < TOLERANCE
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

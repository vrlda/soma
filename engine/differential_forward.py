#!/usr/bin/env python3
"""Differential forward check: frozen organism in Python vs Rust engine.

Builds a small organism with context modules (one dormant), exports a
ForwardFixture, runs both engines over a seeded input tape, and compares
per-step outputs plus final activations within tolerance. Learning,
noise, and proposals stay off: this validates forward semantics only.
"""

import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOLERANCE = 1e-9


def build_fixture(seed=0, steps=200):
    from soma.organism import Organism
    from soma.synapses import Synapse
    organism = Organism.create_default(
        input_size=3, hidden_size=5, output_size=1, seed=seed,
        max_cells=64, max_synapses=256)
    organism.enable_context_modules(max_modules=2)
    dormant = sorted(organism.motor_modules)[-1]
    organism.motor_modules[dormant].dormant = True
    rng = random.Random(seed + 77)
    for synapse in organism.graph.iter_synapses():
        synapse.strength = rng.uniform(-0.8, 0.8)
    dormant_cells = {organism.motor_modules[identifier].cell_id
                     for identifier, module in organism.motor_modules.items() if module.dormant}
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
            "dormant": identifier in dormant_cells,
        })
    synapses = [{"source": s.source, "destination": s.destination, "strength": s.strength}
                for s in organism.graph.iter_synapses()]
    tape = [[rng.uniform(-1.0, 1.0) for _ in organism.input_ids] for _ in range(steps)]
    return organism, {
        "cells": cells,
        "synapses": synapses,
        "input_ids": list(organism.input_ids),
        "output_ids": list(organism.output_ids),
        "steps": tape,
        "passes": organism._propagation_steps(),
    }


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    organism, fixture = build_fixture(seed, steps)
    binary = "./engine/soma-engine/target/release/soma-forward"
    raw = subprocess.run([binary], input=json.dumps(fixture), check=True,
                         capture_output=True, text=True).stdout
    rust = json.loads(raw)
    python_outputs = []
    for values in fixture["steps"]:
        organism._propagate(values)
        python_outputs.append([organism.cells[identifier].activation
                               for identifier in organism.output_ids])
    worst_output, worst_final = 0.0, 0.0
    for left, right in zip(python_outputs, rust["outputs"]):
        for a, b in zip(left, right):
            worst_output = max(worst_output, abs(a - b))
    python_final = {identifier: organism.cells[identifier].activation
                    for identifier in sorted(organism.cells)}
    for identifier, value in python_final.items():
        worst_final = max(worst_final, abs(value - rust["final"][identifier]))
    result = {
        "seed": seed,
        "steps": steps,
        "worst_output_gap": worst_output,
        "worst_final_gap": worst_final,
        "tolerance": TOLERANCE,
    }
    result["all_passed"] = worst_output < TOLERANCE and worst_final < TOLERANCE
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

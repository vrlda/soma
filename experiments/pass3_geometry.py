"""Experiment-only nearest-exemplar control: never used by SOMA."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.nonlinear import label, stream
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                 ('experiments/pass3_geometry.py', 'experiments/nonlinear.py')}


def nearest_control(seed, task):
    train, test, seen = stream(seed, task['episodes'], task['noise'], task['condition'])
    # This explicitly labelled baseline stores raw training views. It is not
    # a neural substrate, cannot teach SOMA, and is not used for its outputs.
    templates = [(frames[10], label(pair)) for pair, frames in train]
    def probe(records):
        scores = []
        for pair, frames in records:
            x = frames[10]
            _, target = min(templates, key=lambda entry: sum((a - b) ** 2 for a, b in zip(x, entry[0])))
            probabilities = [.01, .01]
            probabilities[target] = .99
            scores.append(score(probabilities, label(pair)))
        return metrics(scores)
    return dict(suite='pass3_geometry', condition=task['condition'], variant='nearest_exemplar',
                seed=seed, task=task, metrics=probe(test), seen_probe=probe(seen), complete=True,
                stored_examples=len(templates), source_hashes=SOURCE_HASHES,
                brain=dict(hidden=0, births=0, deaths=0, synapses=0),
                interpretation='Raw-memory baseline, not SOMA; perfect scores would invalidate an exact-template-only exclusion.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    task = json.loads(Path(args.selection).read_text())['task']
    with Path(args.output).open('x') as output:
        for seed in range(10, 20):
            row = nearest_control(seed, task)
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps({key: row[key] for key in ('seed', 'metrics', 'seen_probe')}), flush=True)


if __name__ == '__main__':
    main()

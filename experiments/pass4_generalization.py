"""Contingent transfer on the conventionally calibrated original LEAKY task."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

from brain.soma import SharedSoma, SharedConfig
from experiments import nonlinear
from experiments.pass4 import PROVENANCE, baseline, episode
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = dict(PROVENANCE)
PROVENANCE['source_hashes'] = dict(PROVENANCE['source_hashes'], **{
    'experiments/pass4_generalization.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def experiment(seed, variant, settings):
    if variant in ('nearest', 'mlp'):
        refs = json.loads((ROOT / 'experiments/SELECTED_REFERENCES.json').read_text())['tasks']['original']
        row = baseline(seed, 'original', variant, refs['size'], refs['rate'])
        row.update(suite='pass4_generalization', condition='original_product_view_after_query', **PROVENANCE)
        return row
    start = time.monotonic()
    train, test, seen = nonlinear.stream(seed)
    config = dict(settings, inputs=13, outputs=2, seed=seed)
    if variant == 'prediction_off':
        config['prediction_learning'] = False
    brain = SharedSoma(SharedConfig(**config))
    scores = []
    trajectory = []
    for i, (pair, frames) in enumerate(train):
        p = episode(brain, frames, nonlinear.label(pair))
        scores.append(score(p, nonlinear.label(pair)))
        if i % 100 == 99:
            trajectory.append(brain.summary())
    checkpoint = brain.state()
    def probe(records, frozen=False):
        clone = SharedSoma.from_state(checkpoint)
        if frozen:
            clone.config = replace(clone.config, plasticity=False)
        results = []
        probabilities = []
        for pair, frames in records:
            p = episode(clone, frames)
            results.append(score(p, nonlinear.label(pair)))
            probabilities.append(p)
        return metrics(results), probabilities, clone.summary()
    held, probabilities, final = probe(test)
    seen_scores, _, _ = probe(seen)
    frozen_scores, frozen_probabilities, _ = probe(test, True)
    return dict(suite='pass4_generalization', seed=seed, variant=variant,
                condition='original_product_view_after_query', config=asdict(brain.config),
                metrics=held, seen_probe=seen_scores, frozen_probe=frozen_scores,
                final_window=metrics(scores[-400:]), probabilities=probabilities,
                frozen_probabilities=frozen_probabilities, trajectory=trajectory, brain=final,
                state_hash=hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest(),
                stream_hash=hashlib.sha256(json.dumps([train, test, seen]).encode()).hexdigest(),
                complete=True, seconds=time.monotonic() - start, **PROVENANCE)


def work(args):
    row = experiment(*args)
    print(json.dumps({k: row[k] for k in ('seed', 'variant', 'metrics', 'seen_probe')}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--confirmation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    candidates = [json.loads(line) for line in Path(args.confirmation).read_text().splitlines() if json.loads(line)['variant'] == 'candidate']
    assert {r['seed'] for r in candidates} == set(range(10, 20))
    assert all(r['complete'] for r in candidates)
    assert sum(r['final_window']['accuracy'] >= .95 for r in candidates) >= 9
    assert sum(r['frozen_probe']['accuracy'] >= .95 for r in candidates) >= 9
    jobs = [(s, v, selection['brain']) for s in range(10, 20) for v in ('candidate', 'prediction_off', 'nearest', 'mlp')]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, jobs):
            output.write(json.dumps(row) + '\n')
            output.flush()


if __name__ == '__main__':
    main()

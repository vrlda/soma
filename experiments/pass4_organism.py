"""Slow raw-input pass-1 reference on the identical leak-free stream."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from brain import Config, Organism
from experiments.pass4 import acquisition_stream, acquisition_label, episode
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('brain/organism.py', 'experiments/pass4_organism.py', 'experiments/pass4.py', 'experiments/run.py')
PROVENANCE = dict(source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
                  source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  runtime=dict(implementation=platform.python_implementation(), version=platform.python_version()))


def experiment(seed):
    start = time.monotonic()
    train, test, _ = acquisition_stream(seed)
    settings = json.loads((ROOT / 'experiments/SELECTED_COMBINED.json').read_text())['overrides']
    brain = Organism(Config(**dict(settings, inputs=13, outputs=2, seed=seed)))
    scores = []
    probabilities = []
    trajectory = []
    for i, (pair, frames) in enumerate(train):
        p = episode(brain, frames, acquisition_label(pair))
        scores.append(score(p, acquisition_label(pair)))
        probabilities.append(p)
        if i % 100 == 99:
            trajectory.append(brain.summary())
        if i % 200 == 199:
            print(json.dumps(dict(progress=True, seed=seed, episodes=i + 1, seconds=time.monotonic() - start, brain=brain.summary())), flush=True)
    checkpoint = brain.state()
    clone = Organism.from_state(checkpoint)
    clone.config = replace(clone.config, learning=False)
    probe_scores = []
    probe_probabilities = []
    for pair, frames in test:
        p = episode(clone, frames)
        probe_scores.append(score(p, acquisition_label(pair)))
        probe_probabilities.append(p)
    row = dict(suite='pass4_acquisition', variant='pass1', seed=seed, config=asdict(brain.config),
               final_window=metrics(scores[-400:]), frozen_probe=metrics(probe_scores),
               learning_curve=[metrics(scores[i:i + 200]) for i in range(0, len(scores), 200)],
               probabilities=probabilities, probe_probabilities=probe_probabilities,
               trajectory=trajectory, brain=brain.summary(), complete=True, trained_episodes=len(scores),
               state_hash=hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest(),
               stream_hash=hashlib.sha256(json.dumps([train, test]).encode()).hexdigest(),
               seconds=time.monotonic() - start, **PROVENANCE)
    print(json.dumps({k: row[k] for k in ('seed', 'variant', 'final_window', 'frozen_probe', 'seconds')}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(experiment, map(int, args.seeds.split(','))):
            output.write(json.dumps(row) + '\n')
            output.flush()


if __name__ == '__main__':
    main()

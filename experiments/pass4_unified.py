"""One frozen shared rule set across old interfaces, explicit specialist controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

from brain import Config, Organism
from brain.soma import SharedSoma, SharedConfig
from experiments import composition
from experiments.pass4 import PROVENANCE
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = dict(PROVENANCE)
PROVENANCE['source_hashes'] = dict(PROVENANCE['source_hashes'], **{
    'experiments/pass4_unified.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'experiments/composition.py': hashlib.sha256((ROOT / 'experiments/composition.py').read_bytes()).hexdigest()})


def temporal(seed, settings, lifetime=False, gap=8, specialist=False):
    start = time.monotonic()
    inputs = 7 if lifetime else 6
    if specialist:
        selection = 'SELECTED_COMBINED.json' if lifetime else 'SELECTED_HOMEOSTASIS.json'
        overrides = json.loads((ROOT / 'experiments' / selection).read_text())['overrides']
        brain = Organism(Config(**dict(overrides, inputs=inputs, outputs=2, seed=seed)))
    else:
        brain = SharedSoma(SharedConfig(**dict(settings, inputs=inputs, outputs=2, seed=seed)))
    rng = random.Random(seed + (70000 if lifetime else 60000))
    phases = []
    trajectory = []
    digest = hashlib.sha256()
    def experience(signal, target=None):
        digest.update(json.dumps([signal, target]).encode())
        return brain.experience(signal, target)
    for context, count in ((0, 1200), (1, 1200), (0, 400)) if lifetime else ((0, 1200),):
        results = []
        probabilities = []
        for episode in range(count):
            def signal(index=None):
                x = [0.] * inputs
                if lifetime:
                    x[6] = float(context)
                if index is not None:
                    x[index] = 1.
                return x
            for _ in range(8):
                experience(signal())
            first, second = rng.randrange(2), rng.randrange(2)
            experience(signal(first))
            for _ in range(gap):
                experience(signal(4))
            experience(signal(2 + second))
            target = first ^ second ^ context
            p = experience(signal(5), target)
            results.append(score(p, target))
            probabilities.append(p)
            if episode % 100 == 99:
                trajectory.append(dict(context=context, **brain.summary()))
        phases.append(dict(context=context, episodes=len(results), first_1=metrics(results[:1]),
                           first_50=metrics(results[:50]), last_400=metrics(results[-400:]),
                           probabilities=probabilities))
    acquired = phases[0]['last_400']['accuracy'] >= .9
    return dict(metrics=phases[2]['first_50'] if lifetime else phases[0]['last_400'],
                phases=phases, acquired=acquired, retention_eligible=bool(lifetime and acquired),
                complete=True, config=asdict(brain.config), brain=brain.summary(), trajectory=trajectory,
                state_hash=hashlib.sha256(json.dumps(brain.state(), sort_keys=True).encode()).hexdigest(),
                stream_hash=digest.hexdigest(), seconds=time.monotonic() - start)


def linear(seed, settings, specialist=False):
    selection = json.loads((ROOT / 'experiments/SELECTED_PREDICTIVE.json').read_text())['task']
    if specialist:
        row, _, _ = composition.experiment(seed, 'predictive', selection)
        return row
    start = time.monotonic()
    train, test, seen = composition.stream(seed, selection['episodes'], selection['noise'], selection['amplitude'])
    brain = SharedSoma(SharedConfig(**dict(settings, inputs=25, outputs=2, seed=seed)))
    trajectory = []
    for i, (pair, frames) in enumerate(train):
        for index, frame in enumerate(frames):
            brain.experience(frame, composition.label(pair) if index == len(frames) - 1 else None)
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
            for frame in frames:
                p = clone.experience(frame)
            results.append(score(p, composition.label(pair)))
            probabilities.append(p)
        return metrics(results), probabilities
    held, probabilities = probe(test)
    seen_scores, _ = probe(seen)
    frozen_scores, _ = probe(test, True)
    return dict(metrics=held, seen_probe=seen_scores, frozen_probe=frozen_scores,
                probabilities=probabilities, complete=True, brain=brain.summary(), trajectory=trajectory,
                config=asdict(brain.config), seconds=time.monotonic() - start,
                state_hash=hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest(),
                stream_hash=hashlib.sha256(json.dumps([train, test, seen]).encode()).hexdigest())


def work(args):
    seed, settings, task, variant = args
    config = dict(settings)
    if variant == 'trace_off':
        config['trace_decay'] = 0.
    specialist = variant == 'specialist'
    if task == 'linear':
        row = linear(seed, config, specialist)
    else:
        row = temporal(seed, config, task == 'context', 16 if task == 'grammar16' else 8, specialist)
    row.update(suite='pass4_unified', seed=seed, task=task, variant=variant, shared_config=settings, **PROVENANCE)
    print(json.dumps({k: row[k] for k in ('seed', 'task', 'variant', 'metrics', 'brain', 'complete')}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--tasks', default='linear,grammar8,context')
    parser.add_argument('--variants', default='candidate,specialist')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    settings = json.loads(Path(args.selection).read_text())['brain']
    jobs = [(s, settings, t, v) for s in map(int, args.seeds.split(',')) for t in args.tasks.split(',') for v in args.variants.split(',')]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, jobs):
            output.write(json.dumps(row) + '\n')
            output.flush()


if __name__ == '__main__':
    main()

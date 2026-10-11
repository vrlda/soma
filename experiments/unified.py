"""One frozen SomaConfig on four interfaces, plus an acquired-first lifetime."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from brain.soma import Soma, SomaConfig
from experiments import composition, run

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                 ('brain/soma.py', 'experiments/unified.py', 'experiments/composition.py', 'experiments/run.py')}
SOURCE_REVISION = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def temporal(seed, config, lifetime=False, gap=8, budget=300):
    brain = Soma(SomaConfig(**dict(config, inputs=7 if lifetime else 6, outputs=2, seed=seed)))
    rng = random.Random(seed + (70000 if lifetime else 60000))
    phases = []
    trajectory = []
    start = time.monotonic()
    complete = True
    for context, count in ((0, 1200), (1, 1200), (0, 400)) if lifetime else ((0, 1200),):
        scores = []
        for episode in range(count):
            if time.monotonic() - start > budget:
                complete = False
                break
            def signal(index=None):
                x = [0.] * brain.config.inputs
                if lifetime:
                    x[6] = float(context)
                if index is not None:
                    x[index] = 1.
                return x
            for _ in range(8):
                brain.experience(signal())
            first, second = rng.randrange(2), rng.randrange(2)
            brain.experience(signal(first))
            for _ in range(gap):
                brain.experience(signal(4))
            brain.experience(signal(2 + second))
            target = first ^ second ^ context
            scores.append(run.score(brain.experience(signal(5), target), target))
            if episode % 100 == 99:
                trajectory.append(dict(context=context, **brain.summary()))
        phases.append(dict(context=context, episodes=len(scores), first_1=run.metrics(scores[:1]),
                           first_50=run.metrics(scores[:50]), last_400=run.metrics(scores[-400:])))
        if not complete:
            break
    acquired = phases[0]['last_400']['accuracy'] >= .9
    return dict(metrics=phases[2]['first_50'] if lifetime and len(phases) == 3 else phases[-1]['last_400'],
                phases=phases, acquired=acquired, retention_eligible=bool(lifetime and acquired),
                complete=complete, brain=brain.summary(), config=asdict(brain.config),
                trajectory=trajectory, gap=gap, seconds=time.monotonic() - start)


def linear(seed, config):
    selection = json.loads((ROOT / 'experiments/SELECTED_PREDICTIVE.json').read_text())
    task = selection['task']
    train, test, seen = composition.stream(seed, task['episodes'], task['noise'], task['amplitude'])
    brain = Soma(SomaConfig(**dict(config, inputs=25, outputs=2, seed=seed)))
    start = time.monotonic()
    for pair, frames in train:
        for index, frame in enumerate(frames):
            brain.experience(frame, composition.label(pair) if index == len(frames) - 1 else None)
    checkpoint = brain.state()
    clone = Soma.from_state(checkpoint)
    seen_scores = []
    for pair, frames in seen:
        for frame in frames:
            p = clone.experience(frame)
        seen_scores.append(run.score(p, composition.label(pair)))
    scores = []
    for pair, frames in test:
        for frame in frames:
            p = brain.experience(frame)
        scores.append(run.score(p, composition.label(pair)))
    return dict(metrics=run.metrics(scores), seen_probe=run.metrics(seen_scores), complete=True,
                brain=brain.summary(), config=asdict(brain.config), seconds=time.monotonic() - start)


def work(args):
    seed, config, task, variant = args
    if variant == 'specialist':
        if task == 'linear_composition':
            preset = json.loads((ROOT / 'experiments/SELECTED_PREDICTIVE.json').read_text())['task']
            row, _, _ = composition.experiment(seed, 'predictive', preset)
        else:
            name = 'SELECTED_COMBINED.json' if task == 'context_lifetime' else 'SELECTED_HOMEOSTASIS.json'
            settings = json.loads((ROOT / 'experiments' / name).read_text())['overrides']
            row = (run.lifetime(seed, 'candidate', settings, budget_seconds=600, gap=8)
                   if task == 'context_lifetime' else run.grammar(seed, 'candidate', settings, gap=16 if task == 'grammar16' else 8, budget_seconds=300))
            if task == 'context_lifetime':
                row['acquired'] = row['phases'][0]['last_400']['accuracy'] >= .9
                row['retention_eligible'] = row['acquired']
    else:
        settings = dict(config)
        if variant == 'trace_off':
            settings['trace_decay'] = 0.
        if variant == 'protection_off':
            settings['protection'] = False
        if task == 'linear_composition':
            row = linear(seed, settings)
        else:
            row = temporal(seed, settings, task == 'context_lifetime', gap=16 if task == 'grammar16' else 8)
    row.update(suite='pass3_unified', seed=seed, condition=task, variant=variant,
               shared_config=config, source_hashes=SOURCE_HASHES, source_revision=SOURCE_REVISION,
               runtime=dict(implementation=platform.python_implementation(), version=platform.python_version()))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--tasks', default='linear_composition,grammar8,context_lifetime')
    parser.add_argument('--variants', default='unified,specialist')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    config = json.loads(Path(args.selection).read_text())['task']['brain']
    jobs = [(seed, config, task, variant) for seed in map(int, args.seeds.split(','))
            for task in args.tasks.split(',') for variant in args.variants.split(',')]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(work, jobs):
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps({k: row[k] for k in ('seed', 'condition', 'variant', 'metrics', 'brain', 'complete')}), flush=True)


if __name__ == '__main__':
    main()

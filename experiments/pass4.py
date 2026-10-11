"""Pass-4 environments and conventional references. Backprop is BASELINE ONLY."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

from experiments import nonlinear, nonlinear_hard
from experiments.linear_memory import LinearMemoryBaseline
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
GRID = ((16, .03), (16, .15), (64, .03), (64, .15))


def provenance():
    names = ('brain/soma.py', 'brain/organism.py', 'brain/predictive.py',
             'experiments/pass4.py', 'experiments/nonlinear.py',
             'experiments/nonlinear_hard.py', 'experiments/linear_memory.py')
    return dict(source_hashes={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names},
                source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                runtime=dict(implementation=platform.python_implementation(), version=platform.python_version()))


PROVENANCE = provenance()


def acquisition_label(pair):
    return int(pair[0] == pair[1])


def acquisition_stream(seed, episodes=1200, noise=.3):
    rng = random.Random(seed + 410002)
    a, b, _ = nonlinear.axes(seed + 410000)
    def frames(pair):
        result = [[0.] * 13 for _ in range(8)]
        x, y = (2 * value - 1 for value in pair)
        for marker in (0., 0., 1., 0.):
            n = [rng.uniform(-noise, noise) for _ in a]
            for axis in (a, b):
                parallel = sum(v * w for v, w in zip(n, axis))
                n = [v - parallel * w for v, w in zip(n, axis)]
            result.append([.6 * (x * aa + y * bb) + nn for aa, bb, nn in zip(a, b, n)] + [marker])
        return result
    def records(count):
        result = []
        for _ in range(math.ceil(count / 4)):
            pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
            rng.shuffle(pairs)
            result.extend((pair, frames(pair)) for pair in pairs)
        return result[:count]
    return records(episodes), records(200), records(200)


class MLPReference:
    """One-hidden-layer online BACKPROP baseline; never imported by brain/."""
    def __init__(self, inputs=13, size=16, rate=.03, seed=0):
        rng = random.Random(seed + 410003)
        self.config = dict(inputs=inputs, size=size, rate=rate, seed=seed)
        self.previous = [0.] * inputs
        self.hidden = [[rng.gauss(0, 1 / math.sqrt(inputs)) for _ in range(inputs)] + [rng.uniform(-.2, .2)] for _ in range(size)]
        self.output = [rng.gauss(0, 1 / math.sqrt(size)) for _ in range(size)] + [0.]
        self.plasticity = True
    def experience(self, signal, target=None):
        x = self.previous + [1.]
        activities = [math.tanh(sum(w * v for w, v in zip(row, x))) for row in self.hidden]
        logit = sum(w * v for w, v in zip(self.output, activities + [1.]))
        positive = 1 / (1 + math.exp(-max(-700., min(700., logit))))
        probabilities = [1 - positive, positive]
        if self.plasticity and target is not None:
            error = target - positive
            # Explicit transported output credit: forbidden in SOMA, allowed here.
            deltas = [error * w * (1 - a * a) for w, a in zip(self.output, activities)]
            for row, delta in zip(self.hidden, deltas):
                for i, value in enumerate(x):
                    row[i] += self.config['rate'] * delta * value
            for i, value in enumerate(activities + [1.]):
                self.output[i] += self.config['rate'] * error * value
        self.previous = list(signal)
        return probabilities
    def state(self):
        return json.loads(json.dumps(vars(self)))
    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(json.loads(json.dumps(state)))
        return obj
    def summary(self):
        return dict(hidden=len(self.hidden), synapses=len(self.hidden) * (self.config['inputs'] + 2) + 1, births=0, deaths=0)


def records_for(task, seed):
    if task == 'acquisition':
        return acquisition_stream(seed), acquisition_label
    generator = nonlinear if task == 'original' else nonlinear_hard
    task_settings = dict(episodes=1200, noise=.3)
    records = generator.stream(seed) if task == 'original' else generator.stream(seed, task_settings)
    return records, generator.label


def episode(brain, frames, target=None, teacher=None):
    probabilities = None
    for i, frame in enumerate(frames):
        kwargs = dict(observation_signal=teacher[i]) if teacher is not None else {}
        p = brain.experience(frame, target if i == 10 else None, **kwargs)
        if i == 10:
            probabilities = p
    return probabilities


def baseline(seed, task, variant, size=16, rate=.03):
    start = time.monotonic()
    (train, test, seen), label = records_for(task, seed)
    if variant == 'mlp':
        brain = MLPReference(size=size, rate=rate, seed=seed)
    elif variant == 'linear':
        brain = LinearMemoryBaseline(inputs=13, rate=.3)
    else:
        brain = None
    examples = []
    def nearest(frames):
        if not examples:
            return [.5, .5]
        _, answer = min(examples, key=lambda entry: sum((a - b) ** 2 for a, b in zip(entry[0], frames[9])))
        return [.99, .01] if answer == 0 else [.01, .99]
    scores = []
    for pair, frames in train:
        target = label(pair)
        p = episode(brain, frames, target) if brain else nearest(frames)
        scores.append(score(p, target))
        if brain is None:
            examples.append((frames[9], target))
    checkpoint = brain.state() if brain else None
    def probe(records):
        clone = type(brain).from_state(checkpoint) if brain else None
        if isinstance(clone, MLPReference):
            clone.plasticity = False
        results = []
        probabilities = []
        for pair, frames in records:
            p = episode(clone, frames) if clone else nearest(frames)
            results.append(score(p, label(pair)))
            probabilities.append(p)
        return metrics(results), probabilities
    test_metrics, probabilities = probe(test)
    seen_metrics, _ = probe(seen)
    return dict(suite='pass4_calibration', seed=seed, task=task, variant=variant,
                reference_config=dict(size=size, rate=rate) if variant == 'mlp' else None,
                metrics=test_metrics, seen_probe=seen_metrics, final_window=metrics(scores[-400:]),
                learning_curve=[metrics(scores[i:i + 200]) for i in range(0, len(scores), 200)],
                probabilities=probabilities, stream_hash=hashlib.sha256(json.dumps([train, test, seen]).encode()).hexdigest(),
                complete=True, seconds=time.monotonic() - start, **PROVENANCE)


def baseline_job(args):
    row = baseline(*args)
    print(json.dumps({k: row[k] for k in ('seed', 'task', 'variant', 'reference_config', 'metrics', 'final_window')}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--tasks', default='original,stronger,acquisition')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    jobs = [(s, t, 'mlp', size, rate) for s in map(int, args.seeds.split(',')) for t in args.tasks.split(',') for size, rate in GRID]
    jobs += [(s, t, v) for s in map(int, args.seeds.split(',')) for t in args.tasks.split(',') for v in ('linear', 'nearest')]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(baseline_job, jobs):
            output.write(json.dumps(row) + '\n')
            output.flush()


if __name__ == '__main__':
    main()

"""Nonlinear predictive environment and paired controls, never brain rules."""
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

from brain.soma import Soma, SomaConfig
from brain.predictive import PredictiveBrain, PredictiveConfig
from experiments.linear_memory import LinearMemoryBaseline
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
VALUES = (-3., -1., 1., 3.)
HELD = ((0, 1), (0, 2), (3, 1), (3, 2))
SEEN = tuple((i, j) for i in range(4) for j in range(4) if (i, j) not in HELD)
VARIANTS = ('candidate', 'frozen', 'shuffled', 'temporal_shuffled', 'matched_capacity', 'fixed_final', 'pass2_linear', 'linear_memory')
SOURCE_HASHES = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                 ('brain/soma.py', 'brain/predictive.py', 'experiments/nonlinear.py', 'experiments/linear_memory.py')}
SOURCE_REVISION = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def label(pair):
    return int(VALUES[pair[0]] * VALUES[pair[1]] > 0)


def axes(seed, dimension=12):
    rng = random.Random(seed + 300001)
    result = []
    for _ in range(3):
        vector = [rng.gauss(0, 1) for _ in range(dimension)]
        for old in result:
            parallel = sum(a * b for a, b in zip(vector, old))
            vector = [a - parallel * b for a, b in zip(vector, old)]
        norm = math.sqrt(sum(value * value for value in vector))
        result.append([value / norm for value in vector])
    return result


def stream(seed, episodes=1200, noise=.3, condition='interaction'):
    rng = random.Random(seed + 300002)
    a, b, c = axes(seed)
    def nuisance():
        vector = [rng.uniform(-noise, noise) for _ in a]
        for axis in (a, b, c):
            parallel = sum(x * y for x, y in zip(vector, axis))
            vector = [x - parallel * y for x, y in zip(vector, axis)]
        return vector
    def frames(pair):
        x, y = (VALUES[i] for i in pair)
        result = [[0.] * 13 for _ in range(8)]
        for marker in (0., 0., 1.):
            n = nuisance()
            result.append([.6 * (x * aa + y * bb) + nn for aa, bb, nn in zip(a, b, n)] + [marker])
        interaction = x * y / 3 if condition == 'interaction' else (x * x + y * y) / 6
        n = nuisance()
        result.append([interaction * cc + nn for cc, nn in zip(c, n)] + [0.])
        return result
    train_pairs = (list(SEEN) * math.ceil(episodes / len(SEEN)))[:episodes]
    rng.shuffle(train_pairs)
    train = [(pair, frames(pair)) for pair in train_pairs]
    test_pairs = list(HELD) * 40
    rng.shuffle(test_pairs)
    test = [(pair, frames(pair)) for pair in test_pairs]
    seen_pairs = list(SEEN) * 10
    rng.shuffle(seen_pairs)
    seen = [(pair, frames(pair)) for pair in seen_pairs]
    assert not {p for p, _ in train} & {p for p, _ in test}
    assert all({p[i] for p, _ in train} == set(range(4)) for i in (0, 1))
    return train, test, seen


def symmetry_check():
    results = {}
    for name, pairs in (('seen', SEEN), ('held', HELD)):
        means = [[sum(VALUES[p[k]] for p in pairs if label(p) == y) /
                  sum(label(p) == y for p in pairs) for k in (0, 1)] for y in (0, 1)]
        prior = sum(label(p) for p in pairs) / len(pairs)
        assert means == [[0., 0.], [0., 0.]] and prior == .5
        results[name] = dict(class_means=means, positive_prior=prior,
                             affine_logistic_optimum_loss=math.log(2), accuracy=.5)
    return results


def episode(brain, frames, target=None, schedule=None, offset=0, teacher_frames=None):
    events = []
    probabilities = None
    residuals = []
    for index, signal in enumerate(frames):
        kwargs = {}
        if isinstance(brain, Soma) and schedule is not None:
            kwargs['structure_override'] = schedule[offset + index]
        if isinstance(brain, Soma) and teacher_frames is not None:
            kwargs['observation_signal'] = teacher_frames[index]
        output = brain.experience(signal, target if index == 10 else None, **kwargs)
        if index == 10:
            probabilities = output
        if isinstance(brain, Soma):
            events.append(dict(births=brain.last_event['births'], deaths=brain.last_event['deaths'][:]))
            if index == 11:
                residuals.append(sum(n.error ** 2 for n in brain.neurons) / max(1, len(brain.neurons)))
    return probabilities, events, residuals


def make_brain(seed, variant, config, final_size=None):
    if variant == 'linear_memory':
        return LinearMemoryBaseline(inputs=13, rate=.3)
    if variant == 'pass2_linear':
        settings = json.loads((ROOT / 'experiments/SELECTED_PREDICTIVE.json').read_text())['task']['brain']
        return PredictiveBrain(PredictiveConfig(**dict(settings, inputs=13, seed=seed)))
    settings = dict(config, inputs=13, outputs=2, seed=seed)
    if variant == 'frozen':
        settings['hidden_learning'] = False
    if variant == 'shuffled':
        settings['shuffled'] = True
    if variant == 'matched_capacity':
        settings['hidden_rule'] = 'oja'
    if variant == 'fixed_final':
        settings.update(initial_hidden=max(1, final_size), growth=False, pruning=False)
    return Soma(SomaConfig(**settings))


def experiment(seed, variant, task, schedule=None, final_size=None, budget=300):
    train, test, seen = stream(seed, task['episodes'], task['noise'], task['condition'])
    brain = make_brain(seed, variant, task['brain'], final_size)
    shuffle_rng = random.Random(seed + 771001)
    def teachers(records):
        if variant != 'temporal_shuffled':
            return [None] * len(records)
        shuffled = [frames for _, frames in records]
        shuffle_rng.shuffle(shuffled)
        return shuffled
    train_teachers, test_teachers = teachers(train), teachers(test)
    stream_hash = hashlib.sha256(json.dumps([train, test, seen]).encode()).hexdigest()
    start = time.monotonic()
    train_scores = []
    events = []
    residuals = []
    trajectory = []
    for index, (pair, frames) in enumerate(train):
        if time.monotonic() - start > budget:
            break
        p, ee, errors = episode(brain, frames, label(pair), schedule, index * 12, train_teachers[index])
        events.extend(ee)
        residuals.extend(errors)
        train_scores.append(score(p, label(pair)))
        if index % 100 == 99:
            trajectory.append(brain.summary())
    checkpoint = brain.state()
    trained = len(train_scores)
    def probe(records, frozen=False):
        clone = type(brain).from_state(checkpoint)
        if frozen and isinstance(clone, Soma):
            clone.config = replace(clone.config, plasticity=False, growth=False, pruning=False)
        scores = []
        probe_teachers = teachers(records)
        for index, (pair, frames) in enumerate(records):
            p, _, _ = episode(clone, frames, teacher_frames=probe_teachers[index])
            scores.append(score(p, label(pair)))
        return metrics(scores)
    seen_metrics = probe(seen)
    frozen_metrics = probe(test, frozen=True) if isinstance(brain, Soma) else None
    tests = []
    probabilities = []
    for index, (pair, frames) in enumerate(test):
        if time.monotonic() - start > budget:
            break
        p, ee, _ = episode(brain, frames, schedule=schedule, offset=(trained + index) * 12, teacher_frames=test_teachers[index])
        events.extend(ee)
        tests.append(score(p, label(pair)))
        probabilities.append(dict(pair=pair, target=label(pair), probabilities=p))
    row = dict(suite='pass3_nonlinear', seed=seed, variant=variant, condition=task['condition'],
               task=task, config=asdict(brain.config) if variant != 'linear_memory' else vars(brain.config),
               metrics=metrics(tests), seen_probe=seen_metrics, frozen_probe=frozen_metrics,
               train_last_400=metrics(train_scores[-400:]), trained_episodes=trained,
               complete=trained == len(train) and len(tests) == len(test), brain=brain.summary(),
               trajectory=trajectory, probes=probabilities, stream_hash=stream_hash,
               source_hashes=SOURCE_HASHES, source_revision=SOURCE_REVISION,
               runtime=dict(implementation=platform.python_implementation(), version=platform.python_version()),
               seconds=time.monotonic() - start, chance=.5,
               prediction_first_200=sum(residuals[:200]) / max(1, len(residuals[:200])),
               prediction_last_200=sum(residuals[-200:]) / max(1, len(residuals[-200:])))
    if isinstance(brain, Soma):
        row['schedule_hash'] = hashlib.sha256(json.dumps(events).encode()).hexdigest()
        row['structure_events'] = [dict(tick=i + 1, **event) for i, event in enumerate(events)
                                   if event['births'] or event['deaths']]
        if schedule is not None:
            assert events == schedule
    return row, events, checkpoint


def seed_run(args):
    seed, task, variants, budget = args
    rows = []
    schedule = None
    final_size = None
    for variant in variants:
        row, ee, _ = experiment(seed, variant, task,
                 schedule if variant in ('frozen', 'shuffled', 'temporal_shuffled', 'matched_capacity') else None, final_size, budget)
        if variant == 'candidate':
            schedule = ee
            final_size = row['brain']['hidden']
        rows.append(row)
        print(json.dumps(dict(progress=True, seed=seed, variant=variant, accuracy=row['metrics']['accuracy'], complete=row['complete'])), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=1200)
    parser.add_argument('--brain', default='{}')
    parser.add_argument('--condition', choices=('interaction', 'predictable_nuisance'), default='interaction')
    args = parser.parse_args()
    task = dict(episodes=args.episodes, noise=.3, condition=args.condition, brain=json.loads(args.brain))
    if args.selection:
        task = json.loads(Path(args.selection).read_text())['task']
        task = dict(task, condition=args.condition)
    jobs = [(seed, task, args.variants.split(','), 300) for seed in map(int, args.seeds.split(','))]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as executor:
        for results in executor.map(seed_run, jobs):
            for row in results:
                output.write(json.dumps(row) + '\n')
                output.flush()
                print(json.dumps({k: row[k] for k in ('seed', 'variant', 'condition', 'metrics', 'seen_probe', 'brain', 'complete', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()

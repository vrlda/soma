"""Frozen-config falsification with alternating factor properties.

Reuses existing brain/episode primitives without changing any earlier runner.
All new task rules live here. Raw-memory controls never teach or answer for SOMA.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from brain.soma import Soma
from experiments import nonlinear as original
from experiments.run import metrics, score

ROOT = original.ROOT
SOURCE_HASHES = dict(original.SOURCE_HASHES)
SOURCE_HASHES['experiments/nonlinear_hard.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_REVISION = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
CONDITION = 'alternating_factor_property'


def property_value(value):
    return value * (value * value - 5) / 6


def label(pair):
    return int(property_value(original.VALUES[pair[0]]) * property_value(original.VALUES[pair[1]]) > 0)


def stream(seed, task):
    groups = original.stream(seed, task['episodes'], task['noise'], 'interaction')
    axis = original.axes(seed)[2]
    for records in groups:
        for pair, frames in records:
            x, y = (original.VALUES[i] for i in pair)
            change = property_value(x) * property_value(y) - x * y / 3
            for index, weight in enumerate(axis):
                frames[11][index] += change * weight
    return groups


def nearest_metrics(train, records):
    templates = [(frames[10], label(pair)) for pair, frames in train]
    scores = []
    for pair, frames in records:
        x = frames[10]
        _, target = min(templates, key=lambda item: sum((a - b) ** 2 for a, b in zip(x, item[0])))
        probabilities = [.01, .01]
        probabilities[target] = .99
        scores.append(score(probabilities, label(pair)))
    return metrics(scores)


def experiment(seed, variant, task, schedule=None, final_size=None, budget=300):
    train, test, seen = stream(seed, task)
    stream_hash = hashlib.sha256(json.dumps([train, test, seen]).encode()).hexdigest()
    if variant == 'nearest_exemplar':
        return dict(suite='pass3_hard', seed=seed, variant=variant, condition=CONDITION, task=task,
                    metrics=nearest_metrics(train, test), seen_probe=nearest_metrics(train, seen),
                    frozen_probe=None, stored_examples=len(train), complete=True, stream_hash=stream_hash,
                    brain=dict(hidden=0, births=0, deaths=0, synapses=0),
                    source_hashes=SOURCE_HASHES, source_revision=SOURCE_REVISION), None
    brain = original.make_brain(seed, variant, task['brain'], final_size)
    rng = random.Random(seed + 771001)
    def teachers(records):
        if variant != 'temporal_shuffled':
            return [None] * len(records)
        values = [frames for _, frames in records]
        rng.shuffle(values)
        return values
    train_teachers, test_teachers = teachers(train), teachers(test)
    events = []
    scores = []
    residuals = []
    trajectory = []
    start = time.monotonic()
    for index, (pair, frames) in enumerate(train):
        if time.monotonic() - start > budget:
            break
        p, ee, losses = original.episode(brain, frames, label(pair), schedule, index * 12, train_teachers[index])
        events.extend(ee)
        residuals.extend(losses)
        scores.append(score(p, label(pair)))
        if index % 100 == 99:
            trajectory.append(brain.summary())
    checkpoint = brain.state()
    def probe(records, frozen=False):
        clone = type(brain).from_state(checkpoint)
        if frozen and isinstance(clone, Soma):
            clone.config = replace(clone.config, plasticity=False, growth=False, pruning=False)
        values = []
        supplied = teachers(records)
        for index, (pair, frames) in enumerate(records):
            p, _, _ = original.episode(clone, frames, teacher_frames=supplied[index])
            values.append(score(p, label(pair)))
        return metrics(values)
    seen_metrics = probe(seen)
    frozen_metrics = probe(test, True) if isinstance(brain, Soma) else None
    tests = []
    probabilities = []
    for index, (pair, frames) in enumerate(test):
        if time.monotonic() - start > budget:
            break
        p, ee, _ = original.episode(brain, frames, schedule=schedule, offset=(len(scores) + index) * 12,
                                     teacher_frames=test_teachers[index])
        events.extend(ee)
        tests.append(score(p, label(pair)))
        probabilities.append(dict(pair=pair, target=label(pair), probabilities=p))
    row = dict(suite='pass3_hard', seed=seed, variant=variant, condition=CONDITION, task=task,
               config=asdict(brain.config) if variant != 'linear_memory' else vars(brain.config),
               metrics=metrics(tests), seen_probe=seen_metrics, frozen_probe=frozen_metrics,
               trained_episodes=len(scores), complete=len(scores) == len(train) and len(tests) == len(test),
               train_last_400=metrics(scores[-400:]), brain=brain.summary(), trajectory=trajectory, probes=probabilities,
               stream_hash=stream_hash, source_hashes=SOURCE_HASHES, source_revision=SOURCE_REVISION,
               runtime=dict(implementation=platform.python_implementation(), version=platform.python_version()),
               seconds=time.monotonic() - start,
               prediction_first_200=sum(residuals[:200]) / max(1, len(residuals[:200])),
               prediction_last_200=sum(residuals[-200:]) / max(1, len(residuals[-200:])))
    if isinstance(brain, Soma):
        row['schedule_hash'] = hashlib.sha256(json.dumps(events).encode()).hexdigest()
        row['structure_events'] = [dict(tick=index + 1, **event) for index, event in enumerate(events)
                                   if event['births'] or event['deaths']]
        if schedule is not None:
            assert events == schedule
    return row, events


def work(args):
    seed, task = args
    rows = []
    schedule = None
    final_size = None
    for variant in original.VARIANTS + ('nearest_exemplar',):
        row, events = experiment(seed, variant, task,
            schedule if variant in ('frozen', 'shuffled', 'temporal_shuffled', 'matched_capacity') else None, final_size)
        if variant == 'candidate':
            schedule = events
            final_size = row['brain']['hidden']
        rows.append(row)
        print(json.dumps(dict(progress=True, seed=seed, variant=variant, metrics=row['metrics'], complete=row['complete'])), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    task = dict(json.loads(Path(args.selection).read_text())['task'], condition=CONDITION)
    jobs = [(seed, task) for seed in map(int, args.seeds.split(','))]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=2) as executor:
        for rows in executor.map(work, jobs):
            for row in rows:
                output.write(json.dumps(row) + '\n')
                output.flush()


if __name__ == '__main__':
    main()

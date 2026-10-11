"""Identical-stream acquisition, structural controls and finite noise probes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

from brain import Config, Organism
from brain.soma import SharedSoma, SharedConfig, Soma, SomaConfig
from experiments.pass4 import PROVENANCE, acquisition_stream, acquisition_label, episode, MLPReference
from experiments.linear_memory import LinearMemoryBaseline
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = dict(PROVENANCE)
PROVENANCE['source_hashes'] = dict(PROVENANCE['source_hashes'], **{
    'experiments/pass4_acquisition.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
VARIANTS = ('candidate', 'recruitment_off', 'prediction_off', 'fixed_final', 'shuffled', 'pass3', 'pass1', 'linear', 'mlp')


def make_brain(seed, variant, settings, layout=None):
    if variant == 'linear':
        return LinearMemoryBaseline(inputs=13, rate=.3)
    if variant == 'mlp':
        selected = json.loads((ROOT / 'experiments/SELECTED_REFERENCES.json').read_text())['tasks']['acquisition']
        return MLPReference(seed=seed, size=selected['size'], rate=selected['rate'])
    if variant == 'pass1':
        selected = json.loads((ROOT / 'experiments/SELECTED_COMBINED.json').read_text())['overrides']
        return Organism(Config(**dict(selected, inputs=13, outputs=2, seed=seed)))
    if variant == 'pass3':
        selected = json.loads((ROOT / 'experiments/SELECTED_SOMA.json').read_text())['task']['brain']
        return Soma(SomaConfig(**dict(selected, inputs=13, outputs=2, seed=seed)))
    config = dict(settings, inputs=13, outputs=2, seed=seed)
    if variant == 'recruitment_off':
        config['recruitment'] = False
    if variant == 'prediction_off':
        config['prediction_learning'] = False
    if variant == 'fixed_final':
        config.update(initial_layout=layout, initial_fields=0, recruitment=False, pruning=False)
    return SharedSoma(SharedConfig(**config))


def freeze(brain):
    if isinstance(brain, SharedSoma):
        brain.config = replace(brain.config, plasticity=False)
    elif isinstance(brain, Soma):
        brain.config = replace(brain.config, plasticity=False)
    elif isinstance(brain, Organism):
        brain.config = replace(brain.config, learning=False)
    elif isinstance(brain, MLPReference):
        brain.plasticity = False
    # Linear weights only update on labels; freeze its incidental statistics
    # by a separate reference raw-query evaluation not a brain modification.


def experiment(seed, variant, settings, layout=None, episodes=1200, budget=1200):
    start = time.monotonic()
    train, test, _ = acquisition_stream(seed, episodes)
    brain = make_brain(seed, variant, settings, layout)
    teachers = [None] * len(train)
    if variant == 'shuffled':
        teachers = [frames for _, frames in train]
        random.Random(seed + 410004).shuffle(teachers)
    scores = []
    trajectory = []
    probabilities = []
    events = []
    for i, (pair, frames) in enumerate(train):
        if time.monotonic() - start > budget:
            break
        p = episode(brain, frames, acquisition_label(pair), teachers[i])
        scores.append(score(p, acquisition_label(pair)))
        probabilities.append(p)
        if isinstance(brain, SharedSoma):
            events.append(dict(episode=i, **brain.last_event))
        if i % 100 == 99:
            trajectory.append(brain.summary())
    checkpoint = brain.state()
    clone = type(brain).from_state(checkpoint)
    freeze(clone)
    probes = []
    probe_probabilities = []
    for pair, frames in test:
        p = episode(clone, frames)
        probes.append(score(p, acquisition_label(pair)))
        probe_probabilities.append(p)
    final_layout = ([(n.id, [s.source for s in b.outcomes]) for n, b in brain._branches()]
                    if isinstance(brain, SharedSoma) else None)
    row = dict(suite='pass4_acquisition', seed=seed, variant=variant, shared_config=settings,
               config=asdict(brain.config) if not isinstance(brain, (LinearMemoryBaseline, MLPReference)) else (vars(brain.config) if variant == 'linear' else brain.config),
               final_window=metrics(scores[-400:]), frozen_probe=metrics(probes),
               learning_curve=[metrics(scores[i:i + 200]) for i in range(0, len(scores), 200)],
               probabilities=probabilities, probe_probabilities=probe_probabilities,
               trained_episodes=len(scores), complete=len(scores) == episodes,
               trajectory=trajectory, brain=brain.summary(), final_layout=final_layout,
               state_hash=hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest(),
               stream_hash=hashlib.sha256(json.dumps([train, test]).encode()).hexdigest(),
               seconds=time.monotonic() - start, **PROVENANCE)
    return row, checkpoint


def noise(seed, settings, condition):
    start = time.monotonic()
    brain = SharedSoma(SharedConfig(**dict(settings, seed=seed)))
    trajectory = []
    scores = []
    if condition == 'independent':
        rng = random.Random(seed + 410005)
        for tick in range(12000):
            brain.experience([rng.uniform(-1, 1) for _ in range(13)])
            if tick % 1000 == 999:
                trajectory.append(brain.summary())
    else:
        records, _, _ = acquisition_stream(seed, 2400)
        rng = random.Random(seed + 410006)
        for i, (pair, frames) in enumerate(records):
            target = rng.randrange(2) if condition == 'random_labels' else acquisition_label(pair)
            scores.append(score(episode(brain, frames, target), target))
            if i % 200 == 199:
                trajectory.append(brain.summary())
    return dict(suite='pass4_noise', seed=seed, condition=condition, config=asdict(brain.config),
                trajectory=trajectory, brain=brain.summary(), metrics=metrics(scores),
                hidden_hash=hidden_hash(brain), complete=True, seconds=time.monotonic() - start, **PROVENANCE)


def hidden_hash(brain):
    state = brain.state()
    state.pop('bias')
    for neuron in state['neurons']:
        neuron.pop('outcomes')
        for branch in neuron['branches']:
            branch.pop('outcomes')
    # Outcome connection counts are allowed to depend on local outcome error.
    for key in ('synapse_births', 'synapse_deaths'):
        state.pop(key)
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def seed_run(args):
    seed, settings, variants, mode = args
    if mode == 'noise':
        return [noise(seed, settings, c) for c in ('independent', 'random_labels', 'ordinary_labels')]
    rows = []
    layout = None
    for variant in variants:
        row, _ = experiment(seed, variant, settings, layout)
        if variant == 'candidate':
            layout = row['final_layout']
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('seed', 'variant', 'final_window', 'frozen_probe', 'brain', 'complete', 'seconds')}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection')
    parser.add_argument('--brain', default='{}')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--mode', choices=('acquisition', 'noise'), default='acquisition')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    settings = (json.loads(Path(args.selection).read_text())['brain'] if args.selection else json.loads(args.brain))
    jobs = [(s, settings, args.variants.split(','), args.mode) for s in map(int, args.seeds.split(','))]
    with Path(args.output).open('x') as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(seed_run, jobs):
            for row in rows:
                output.write(json.dumps(row) + '\n')
                output.flush()


if __name__ == '__main__':
    main()

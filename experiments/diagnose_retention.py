"""Replay unchanged lifetimes; diagnose return errors without retuning.

Final saved states have already experienced 400 return episodes. They cannot
alone establish what failed before that return, so reconstruct exact streams.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

from brain import Organism
from experiments.run import config, metrics, score


def signal(context, index=None):
    x = [0.] * 7
    x[6] = float(context)
    if index is not None:
        x[index] = 1.
    return x


def episode(brain, context, first, second, feedback=True):
    for _ in range(8):
        brain.experience(signal(context))
    brain.experience(signal(context, first))
    for _ in range(8):
        brain.experience(signal(context, 4))
    brain.experience(signal(context, second + 2))
    snapshot = {}
    def observe(data):
        snapshot.update(winners=[n for n in brain.hidden if brain.neurons[n].activity > .5],
                        novelty=brain.current_representation_surprise,
                        hidden=len(brain.hidden))
    label = first ^ second ^ context
    p = brain.experience(signal(context, 5), label if feedback else None, observer=observe)
    return dict(first=first, second=second, probabilities=p, **snapshot,
                accuracy=score(p, label)[0])


def probe(brain):
    result = []
    for first in range(2):
        for second in range(2):
            clone = Organism.from_state(brain.state())
            clone.config = replace(clone.config, learning=False)
            r = episode(clone, 0, first, second, feedback=False)
            winner = r['winners'][0] if r['winners'] else None
            r['prediction_edges'] = [asdict(s) for s in brain.synapses
                                     if s.kind == 'prediction' and s.source == winner]
            r['afferents'] = [asdict(s) for s in brain.synapses
                              if s.kind == 'drive' and s.target == winner]
            result.append(r)
    return result


def diagnose(seed):
    selected = json.loads(Path('experiments/SELECTED_COMBINED.json').read_text())['overrides']
    brain = Organism(config(seed, 7, 2, 'candidate', selected))
    stream = random.Random(seed + 70000)
    phases, checkpoints, first_return = [], [], []
    populations = []
    winner_counts = []
    profiles = []
    for phase, (context, count) in enumerate(((0,1200),(1,1200),(0,400))):
        scores = []
        counts = {}
        for i in range(count):
            first, second = stream.randrange(2), stream.randrange(2)
            r = episode(brain, context, first, second)
            scores.append(score(r['probabilities'], first ^ second ^ context))
            if i >= count - 400:
                key = f'{first},{second}'
                bucket = counts.setdefault(key,{})
                winner = str(r['winners'][0]) if r['winners'] else 'none'
                bucket[winner] = bucket.get(winner,0) + 1
            if phase == 2 and i < 50:
                first_return.append(r)
        phases.append(dict(context=context, episodes=count, first_1=metrics(scores[:1]),
                           first_50=metrics(scores[:50]), last_400=metrics(scores[-400:])))
        winner_counts.append(counts)
        if phase < 2:
            checkpoints.append(probe(brain))
            populations.append(set(brain.hidden))
            profiles.append({str(n):[asdict(edge) for edge in brain.synapses
                             if edge.target == n or edge.source == n] for n in brain.hidden})
    saved = Path(f'experiments/checkpoints/combined_confirmation/lifetime_candidate_8_{seed}.json')
    exact = brain.state() == Organism.load(saved).state()
    if not exact:
        raise AssertionError('Replay differs from saved final lifetime')
    original = checkpoints[0]
    after_b = checkpoints[1]
    diagnostics=[]
    for old, current in zip(original, after_b):
        old_id = old['winners'][0] if old['winners'] else None
        new_id = current['winners'][0] if current['winners'] else None
        diagnostics.append(dict(first=old['first'], second=old['second'],
                                old_winner=old_id, old_winner_survives=old_id in populations[1],
                                return_winner=new_id, same_winner=old_id == new_id,
                                old_accuracy=old['accuracy'], frozen_return_accuracy=current['accuracy'],
                                old_prediction_edges=old['prediction_edges'],
                                return_prediction_edges=current['prediction_edges']))
    return dict(suite='retention_diagnostic', variant='unchanged', condition=8, seed=seed,
                config=asdict(brain.config), exact_final_checkpoint=exact, phases=phases,
                first_return=first_return, probes=checkpoints, comparisons=diagnostics,
                winner_counts=winner_counts, boundary_profiles=profiles,
                hidden_after_a=len(populations[0]), hidden_after_b=len(populations[1]),
                surviving_old_hidden=len(populations[0] & populations[1]), brain=brain.summary(),
                source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',default='10,11,12')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    with Path(a.output).open('x') as stream:
        for seed in map(int,a.seeds.split(',')):
            r=diagnose(seed)
            stream.write(json.dumps(r)+'\n');stream.flush()
            print(json.dumps(dict(seed=seed, exact=r['exact_final_checkpoint'],
                  return_accuracy=r['phases'][2]['first_50']['accuracy'],
                  comparisons=[{k:v for k,v in x.items() if 'edges' not in k}
                               for x in r['comparisons']])),flush=True)

if __name__ == '__main__':
    main()

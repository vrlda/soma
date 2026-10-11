"""Observer-only failure diagnosis; no signal is returned to the shared brain."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random

from brain.soma import SharedSoma, SharedConfig
from experiments import composition
from experiments.pass4 import PROVENANCE
from experiments.run import metrics, score

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = dict(PROVENANCE)
PROVENANCE['source_hashes'] = dict(PROVENANCE['source_hashes'], **{
    'experiments/pass4_diagnose.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


class NormalizedReadoutReference:
    """Labelled observer baseline over actual axons, not a SOMA component."""
    def __init__(self, dimension=8):
        self.means = [0.] * dimension
        self.variances = [1.] * dimension
        self.weights = [[0.] * (dimension + 1) for _ in range(2)]
        self.plasticity = True
    def experience(self, sources, target=None):
        x = [(v - m) / math.sqrt(max(.0001, s)) for v, m, s in zip(sources, self.means, self.variances)] + [1.]
        logits = [sum(w * v for w, v in zip(row, x)) for row in self.weights]
        exps = [math.exp(v - max(logits)) for v in logits]
        p = [v / sum(exps) for v in exps]
        if self.plasticity:
            if target is not None:
                energy = 1 + sum(v * v for v in x)
                for k, row in enumerate(self.weights):
                    error = float(k == target) - p[k]
                    for i, v in enumerate(x):
                        row[i] += .2 * v * error / energy
            for i, v in enumerate(sources):
                old = self.means[i]
                self.means[i] += .01 * (v - old)
                self.variances[i] = max(0., .99 * self.variances[i] + .01 * (v - old) * (v - self.means[i]))
        return p
    def state(self):
        return json.loads(json.dumps(vars(self)))
    @classmethod
    def from_state(cls, data):
        obj = cls.__new__(cls)
        obj.__dict__.update(json.loads(json.dumps(data)))
        return obj


def linear(seed, settings, reference):
    task = json.loads((ROOT / 'experiments/SELECTED_PREDICTIVE.json').read_text())['task']
    train, test, _ = composition.stream(seed, task['episodes'], task['noise'], task['amplitude'])
    brain = SharedSoma(SharedConfig(**dict(settings, inputs=25, outputs=2, seed=seed)))
    observer = NormalizedReadoutReference(brain.config.predictors)
    train_results = []
    for pair, frames in train:
        for i, frame in enumerate(frames):
            target = composition.label(pair) if i == len(frames) - 1 else None
            brain.experience(frame, target)
            p = observer.experience([n.axon for n in brain.neurons], target)
        train_results.append(score(p, composition.label(pair)))
    checkpoint = brain.state()
    state_hash = hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest()
    assert state_hash == reference['state_hash']
    def probe(frozen=False):
        candidate = SharedSoma.from_state(checkpoint)
        shadow = NormalizedReadoutReference.from_state(observer.state())
        if frozen:
            candidate.config = replace(candidate.config, plasticity=False)
            shadow.plasticity = False
        original = []
        observed = []
        original_probabilities = []
        for pair, frames in test:
            for frame in frames:
                p = candidate.experience(frame)
                q = shadow.experience([n.axon for n in candidate.neurons])
            original.append(score(p, composition.label(pair)))
            observed.append(score(q, composition.label(pair)))
            original_probabilities.append(p)
        return metrics(original), metrics(observed), original_probabilities
    original, observed, probabilities = probe()
    frozen_original, frozen_observed, _ = probe(True)
    assert original == reference['metrics'] and probabilities == reference['probabilities']
    return dict(suite='pass4_diagnosis', condition='linear_normalized_observer', seed=seed,
                original=original, observer=observed, frozen_original=frozen_original,
                frozen_observer=frozen_observed, observer_train_final=metrics(train_results[-200:]),
                original_exact_match=True, state_hash=state_hash, complete=True, **PROVENANCE)


def temporal(settings, reference, checkpoint_path=None):
    seed = 0
    brain = SharedSoma(SharedConfig(**dict(settings, inputs=7, outputs=2, seed=seed)))
    rng = random.Random(seed + 70000)
    results = []
    probabilities = []
    winners = defaultdict(Counter)
    pairs = defaultdict(Counter)
    oracle = 0
    distances = []
    def signal(index=None):
        x = [0.] * 7
        if index is not None:
            x[index] = 1.
        return x
    for i in range(1200):
        for _ in range(8):
            brain.experience(signal())
        first, second = rng.randrange(2), rng.randrange(2)
        brain.experience(signal(first))
        for _ in range(8):
            brain.experience(signal(4))
        brain.experience(signal(2 + second))
        target = first ^ second
        p = brain.experience(signal(5), target)
        probabilities.append(p)
        results.append(score(p, target))
        if i >= 800:
            active = [b.id for _, b in brain._branches() if b.activity]
            winner = str(active[0]) if active else 'abstention'
            winners[winner][target] += 1
            pairs[winner][str((first, second))] += 1
            # Evaluator ORACLE ONLY. Never a source or teacher for SOMA.
            decoded = int(brain.traces[1] > brain.traces[0]) ^ int(brain.traces[3] > brain.traces[2])
            oracle += int(decoded == target)
            distances.append(min((b.mismatch for _, b in brain._branches()), default=None))
    assert probabilities == reference['phases'][0]['probabilities']
    assert metrics(results[-400:]) == reference['phases'][0]['last_400']
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        brain.save(path)
    return dict(suite='pass4_diagnosis', condition='initial_context_branch_collision', seed=seed,
                metrics=metrics(results[-400:]), oracle_accuracy=oracle / 400,
                winner_class_counts={k: dict(v) for k, v in winners.items()},
                winner_pair_counts={k: dict(v) for k, v in pairs.items()},
                winner_best_constant_assignment=sum(max(v.values()) for v in winners.values()) / 400,
                minimum_query_mismatches=distances, brain=brain.summary(),
                original_exact_match=True, complete=True, **PROVENANCE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--unified', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    settings = json.loads(Path(args.selection).read_text())['brain']
    records = [json.loads(x) for x in Path(args.unified).read_text().splitlines()]
    with Path(args.output).open('x') as output:
        for seed in range(5):
            reference = next(r for r in records if r['seed'] == seed and r['task'] == 'linear' and r['variant'] == 'candidate')
            row = linear(seed, settings, reference)
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps(row), flush=True)
        reference = next(r for r in records if r['seed'] == 0 and r['task'] == 'context' and r['variant'] == 'candidate')
        row = temporal(settings, reference, args.checkpoint)
        output.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ('minimum_query_mismatches', 'source_hashes')}), flush=True)


if __name__ == '__main__':
    main()

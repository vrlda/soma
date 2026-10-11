"""Registered structural noise probe and exact new-substrate replay receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess

from brain.soma import Soma, SomaConfig
from experiments.nonlinear import experiment, symmetry_check

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                 ('brain/soma.py', 'experiments/pass3_diagnostics.py', 'experiments/nonlinear.py', 'experiments/unified.py')}


def matches_recorded(value, recorded):
    """Compare exact serialized values, including tuples encoded as JSON lists."""
    return json.loads(json.dumps(value, allow_nan=False)) == recorded


def noise_probe(seed, config, condition):
    brain = Soma(SomaConfig(**dict(config, inputs=13, outputs=2, seed=seed)))
    rng = random.Random(seed + 440001)
    trajectory = []
    last = [rng.uniform(-1, 1) for _ in range(13)]
    for tick in range(4000):
        if condition == 'independent' or tick % 4 == 0:
            last = [rng.uniform(-1, 1) for _ in range(13)]
        brain.experience(last)
        if tick % 400 == 399:
            trajectory.append(brain.summary())
    return dict(suite='pass3_structure', variant='candidate', condition=condition, seed=seed,
                config=vars(brain.config), brain=brain.summary(), trajectory=trajectory, complete=True,
                source_hashes=SOURCE_HASHES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--mode', choices=('replay', 'noise', 'symmetry'), required=True)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    with Path(args.output).open('x') as output:
        def emit(row):
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps(row), flush=True)
        if args.mode == 'symmetry':
            emit(dict(suite='pass3_symmetry', checks=symmetry_check(), complete=True))
        elif args.mode == 'noise':
            for seed in range(5):
                for condition in ('independent', 'repeated'):
                    row = noise_probe(seed, selection['task']['brain'], condition)
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    print(json.dumps({k: row[k] for k in ('seed', 'condition', 'brain')}), flush=True)
        else:
            for name in ('pass3_nonlinear_initial.jsonl', 'pass3_nonlinear_underfit.jsonl', 'pass3_nonlinear_covariance.jsonl'):
                row = next(json.loads(line) for line in (ROOT / 'experiments/results' / name).read_text().splitlines()
                           if json.loads(line)['seed'] == 0 and json.loads(line)['variant'] == 'candidate')
                replay, _, _ = experiment(0, 'candidate', row['task'])
                checks = {key: matches_recorded(replay[key], row[key]) for key in
                          ('metrics', 'seen_probe', 'frozen_probe', 'brain', 'trajectory', 'probes', 'schedule_hash')}
                emit(dict(suite='pass3_reproducibility', source=name, variant='candidate', seed=0,
                          checks=checks, exact_match=all(checks.values()), source_hashes=SOURCE_HASHES))
                if not all(checks.values()):
                    raise SystemExit('Exploratory replay changed')
            records = [json.loads(line) for line in (ROOT / 'experiments/results/pass3_nonlinear_confirmation.jsonl').read_text().splitlines()]
            candidate, schedule, checkpoint = experiment(13, 'candidate', selection['task'])
            for variant in ('candidate', 'frozen', 'shuffled', 'temporal_shuffled'):
                row = next(r for r in records if r['seed'] == 13 and r['variant'] == variant)
                replay = candidate if variant == 'candidate' else experiment(13, variant, selection['task'], schedule)[0]
                checks = {key: matches_recorded(replay[key], row[key]) for key in
                          ('metrics', 'seen_probe', 'frozen_probe', 'brain', 'trajectory', 'probes', 'schedule_hash')}
                emit(dict(suite='pass3_reproducibility', source='pass3_nonlinear_confirmation.jsonl',
                          variant=variant, seed=13, checks=checks, exact_match=all(checks.values()),
                          source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                          source_hashes=SOURCE_HASHES, runtime=dict(implementation=platform.python_implementation(), version=platform.python_version())))
                if not all(checks.values()):
                    raise SystemExit('Unified nonlinear replay changed')
            if args.checkpoint:
                path = Path(args.checkpoint)
                if path.exists():
                    raise FileExistsError(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                Soma.from_state(checkpoint).save(path)
            from experiments.unified import work
            records = [json.loads(line) for line in (ROOT / 'experiments/results/pass3_unified_comparison.jsonl').read_text().splitlines()]
            for condition in ('linear_composition', 'grammar8', 'context_lifetime'):
                row = next(r for r in records if r['seed'] == 0 and r['variant'] == 'unified' and r['condition'] == condition)
                replay = work((0, selection['task']['brain'], condition, 'unified'))
                checks = {key: matches_recorded(replay[key], row[key]) for key in ('metrics', 'brain', 'config', 'complete')}
                if 'phases' in row:
                    checks['phases'] = matches_recorded(replay['phases'], row['phases'])
                emit(dict(suite='pass3_reproducibility', source='pass3_unified_comparison.jsonl',
                          variant='unified', condition=condition, seed=0, checks=checks, exact_match=all(checks.values())))
                if not all(checks.values()):
                    raise SystemExit('Unified cross-task replay changed')


if __name__ == '__main__':
    main()

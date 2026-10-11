"""Exact scientific replay, pinned exploratory sources and immutable-file audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a1b9d04a48edb83a995b743c7933109b9752a493'
CHILD = r'''
import json,sys
request=json.load(sys.stdin)
row=request['row'];kind=request['kind']
if kind=='acquisition':
 from experiments.pass4_acquisition import experiment
 replay,_=experiment(row['seed'],row['variant'],row['shared_config'])
 keys=('final_window','frozen_probe','probabilities','probe_probabilities','learning_curve','brain','trajectory','state_hash','stream_hash','complete')
elif kind=='unified':
 from experiments.pass4_unified import work
 # Suppress progress output, returning only the receipt below.
 import contextlib,io
 with contextlib.redirect_stdout(io.StringIO()):
  replay=work((row['seed'],row['shared_config'],row['task'],row['variant']))
 keys=('metrics','brain','config','state_hash','stream_hash','complete')
 if 'phases' in row:keys+=('phases',)
 if 'probabilities' in row:keys+=('probabilities',)
else:
 from experiments.pass4_organism import experiment
 import contextlib,io
 with contextlib.redirect_stdout(io.StringIO()):replay=experiment(row['seed'])
 keys=('final_window','frozen_probe','probabilities','probe_probabilities','learning_curve','brain','trajectory','state_hash','stream_hash','complete')
checks={k:json.loads(json.dumps(replay[k],allow_nan=False))==row[k] for k in keys}
print(json.dumps(dict(checks=checks,exact_match=all(checks.values()))))
'''


def rows(name):
    return [json.loads(x) for x in (ROOT / 'experiments/results' / name).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw-organism', action='store_true')
    args = parser.parse_args()
    cases = [('pass4_shared_hypothesis1.jsonl', 'candidate', 0, 'acquisition', True, None),
             ('pass4_shared_hypothesis2.jsonl', 'candidate', 0, 'acquisition', True, None),
             ('pass4_acquisition_confirmation.jsonl', 'candidate', 10, 'acquisition', False, None),
             ('pass4_acquisition_confirmation.jsonl', 'candidate', 18, 'acquisition', False, None),
             ('pass4_acquisition_confirmation.jsonl', 'prediction_off', 11, 'acquisition', False, None),
             ('pass4_unified.jsonl', 'candidate', 0, 'unified', False, 'linear'),
             ('pass4_unified.jsonl', 'candidate', 0, 'unified', False, 'grammar8'),
             ('pass4_unified.jsonl', 'candidate', 0, 'unified', False, 'context')]
    if args.raw_organism:
        cases = [('pass4_raw_organism_confirmation.jsonl', 'pass1', 10, 'organism', False, None)]
    with Path(args.output).open('x') as output, tempfile.TemporaryDirectory(prefix='soma-pass4-pinned-') as temp:
        def emit(row):
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps(row), flush=True)
            if not row['exact_match']:
                raise SystemExit('Exact pass-4 replay/audit failed')
        for filename, variant, seed, kind, pinned, task in cases:
            reference = next(r for r in rows(filename) if r['variant'] == variant and r['seed'] == seed
                             and (kind != 'unified' or r['task'] == task))
            cwd = ROOT
            if pinned:
                cwd = Path(temp) / reference['source_revision']
                subprocess.run(['git', 'worktree', 'add', '--detach', str(cwd), reference['source_revision']], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                result = subprocess.run([sys.executable, '-B', '-c', CHILD], input=json.dumps(dict(row=reference, kind=kind)), cwd=cwd, text=True, check=True, stdout=subprocess.PIPE)
                receipt = json.loads(result.stdout)
            finally:
                if pinned:
                    subprocess.run(['git', 'worktree', 'remove', str(cwd)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            emit(dict(suite='pass4_verification', kind='scientific_replay', source=filename,
                      variant=variant, seed=seed, task=task, pinned_source=reference['source_revision'] if pinned else None, **receipt))
        if args.raw_organism:
            return
        complete = rows('pass4_calibration_complete.jsonl')
        partial = rows('pass4_calibration.jsonl')
        for reference in partial:
            duplicate = next(r for r in complete if r['seed'] == reference['seed'] and r['task'] == reference['task'] and r['variant'] == reference['variant'] and r['reference_config'] == reference['reference_config'])
            keys = ('metrics', 'seen_probe', 'final_window', 'learning_curve', 'probabilities', 'stream_hash')
            checks = {k: duplicate[k] == reference[k] for k in keys}
            emit(dict(suite='pass4_verification', kind='unchanged_calibration_restart', seed=reference['seed'],
                      config=reference['reference_config'], checks=checks, exact_match=all(checks.values())))
        checks = {}
        for name in ('brain/organism.py', 'brain/predictive.py', 'VISION.md'):
            original = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
            checks[name] = original == (ROOT / name).read_bytes()
        original = subprocess.check_output(['git', 'show', BASE + ':brain/soma.py'], cwd=ROOT)
        checks['legacy_soma_prefix'] = (ROOT / 'brain/soma.py').read_bytes().startswith(original)
        changed = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=MDR', BASE, '--', 'experiments/results'], cwd=ROOT, text=True)
        checks['historic_results_immutable'] = not changed.strip()
        confirmation = rows('pass4_acquisition_confirmation.jsonl')
        for seed in range(10, 20):
            hashes = {r['stream_hash'] for r in confirmation if r['seed'] == seed}
            checks['paired_stream_' + str(seed)] = len(hashes) == 1
        frozen = json.loads((ROOT / 'experiments/SELECTED_SHARED.json').read_text())['brain']
        for row in confirmation:
            if row['variant'] == 'candidate':
                config = {k: v for k, v in row['config'].items() if k not in ('inputs', 'outputs', 'seed')}
                checks['frozen_configuration_' + str(row['seed'])] = config == frozen
        # Check each executed source hash against its recorded immutable Git blob.
        checked = set()
        for filename in ('pass4_shared_hypothesis1.jsonl', 'pass4_shared_hypothesis2.jsonl',
                         'pass4_shared_hypothesis3.jsonl', 'pass4_acquisition_confirmation.jsonl', 'pass4_unified.jsonl'):
            for row in rows(filename):
                for name, digest in row['source_hashes'].items():
                    key = (row['source_revision'], name, digest)
                    if key in checked:
                        continue
                    checked.add(key)
                    source = subprocess.check_output(['git', 'show', row['source_revision'] + ':' + name], cwd=ROOT)
                    checks['source:' + row['source_revision'][:7] + ':' + name] = hashlib.sha256(source).hexdigest() == digest
        emit(dict(suite='pass4_verification', kind='immutability_pairing_selection_provenance',
                  checks=checks, exact_match=all(checks.values())))


if __name__ == '__main__':
    main()

"""Replay selected pass-2 rows with current and pinned research source versions.

New mechanisms evolved during exploration. Exact numerical replay needs the
source revision as well as seed/config; the original organism stays unchanged.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CASES=(('pass2_composition_initial.jsonl','predictive','b3b6d3f'),
       ('pass2_composition_initial.jsonl','frozen','b3b6d3f'),
       ('pass2_composition_homeostasis_slow.jsonl','predictive','b3b6d3f'),
       ('pass2_composition_hebbian_control.jsonl','hebbian_capacity','9bd06b5'),
       ('pass2_composition_confirmation.jsonl','predictive','935aae1'))

CHILD='''
import json,sys
from experiments.composition import experiment
row=json.loads(sys.argv[1])
_,events,_=experiment(row['seed'],'predictive',row['task'])
r,_,_=experiment(row['seed'],row['variant'],row['task'],events if row['variant']!='predictive' else None)
checks={k:r[k]==row[k] for k in ('metrics','seen_probe','initial_transfer','brain')}
print(json.dumps({'checks':checks,'exact_match':all(checks.values())}))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with Path(args.output).open('x') as output, tempfile.TemporaryDirectory(prefix='soma-source-replay-') as tmp:
        for name,variant,revision in CASES:
            row=next(r for r in map(json.loads,(ROOT/'experiments/results'/name).read_text().splitlines())
                     if r['variant']==variant)
            for mode in ('current','pinned'):
                cwd=ROOT
                if mode=='pinned':
                    cwd=Path(tmp)/revision
                    if not cwd.exists():
                        cwd.mkdir()
                        archive=subprocess.check_output(['git','archive',revision,'brain','experiments'],cwd=ROOT)
                        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                            tar.extractall(cwd,filter='data')
                result=json.loads(subprocess.check_output([sys.executable,'-B','-c',CHILD,json.dumps(row)],cwd=cwd))
                receipt=dict(suite='pass2_reproducibility',variant=variant,condition=mode,
                             seed=row['seed'],source_file=name,source_revision=revision if mode=='pinned' else 'working_tree',
                             config=row['config'],metrics=row['metrics'],brain=row['brain'],complete=True,**result)
                output.write(json.dumps(receipt)+'\n');output.flush()
                print(json.dumps({k:receipt[k] for k in ('source_file','variant','condition','exact_match','checks')}),flush=True)
                if mode=='pinned' and not receipt['exact_match']:
                    raise SystemExit('Recorded-source replay failed')

if __name__=='__main__':main()

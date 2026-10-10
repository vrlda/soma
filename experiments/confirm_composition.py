"""Confirm a frozen composition selection on independent, paired seed groups.

Processes are experiment workers, not research agents. Each seed owns its
brains/streams. All variants in a seed share the candidate's structural events.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.composition import experiment

VARIANTS=('predictive','frozen','shuffled','matched_capacity','hebbian_capacity',
          'linear_memory','competitive')


def seed_group(seed,options):
    schedule=None;rows=[]
    for variant in VARIANTS:
        row,events,_=experiment(seed,variant,options,
                               schedule if variant not in ('predictive','linear_memory','competitive') else None)
        if variant=='predictive':schedule=events
        rows.append(row)
        print(json.dumps(dict(seed=seed,variant=variant,complete=row['complete'],
                              accuracy=row['metrics']['accuracy'])),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    selection=json.loads(Path(args.selection).read_text())
    if not selection['selected_before_confirmation']:
        raise SystemExit('Confirmation requires a frozen selection')
    with Path(args.output).open('x') as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(seed_group,seed,selection['task']) for seed in selection['seeds']]
            for task in as_completed(pending):
                for row in task.result():stream.write(json.dumps(row)+'\n')
                stream.flush()

if __name__=='__main__':main()

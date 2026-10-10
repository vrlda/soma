"""Environment-aware geometry diagnostic; never a brain learning signal."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from brain.predictive import PredictiveBrain
from experiments.composition import directions, dot, experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    options=dict(episodes=600,noise=2.,amplitude=.6,
                 brain=dict(hidden_rate=.02,normalize_activity=True))
    with Path(args.output).open('x') as f:
        for seed in range(5):
            schedule=None
            for variant in ('predictive','frozen','shuffled','hebbian_capacity'):
                row,events,state=experiment(seed,variant,options,schedule)
                if variant=='predictive':schedule=events
                brain=PredictiveBrain.from_state(state)
                a,b=directions(seed);fractions=[];coefficient_a=[];coefficient_b=[]
                for unit in brain.units:
                    w=[s.strength for s in unit.basal[:24]]
                    coefficient_a.append(dot(w,a));coefficient_b.append(dot(w,b))
                    total=dot(w,w)
                    fractions.append((dot(w,a)**2+dot(w,b)**2)/max(1e-30,total))
                aa=dot(coefficient_a,coefficient_a);bb=dot(coefficient_b,coefficient_b)
                ab=dot(coefficient_a,coefficient_b)
                gap=((aa-bb)**2+4*ab*ab)**.5
                eigenvalues=[max(0.,(aa+bb+gap)/2),max(0.,(aa+bb-gap)/2)]
                result=dict(signal_eigenvalues=eigenvalues,signal_rank_ratio=eigenvalues[1]/max(1e-30,eigenvalues[0]),suite='predictive_geometry',condition='factor_span',seed=seed,
                            variant=variant,config=asdict(brain.config),complete=True,
                            metrics=dict(accuracy=None,log_loss=None,n=len(fractions)),
                            signal_fraction=statistics.mean(fractions),fractions=fractions,
                            brain=brain.summary(),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                f.write(json.dumps(result)+'\n');f.flush()
                print(json.dumps(dict(seed=seed,variant=variant,signal_fraction=result['signal_fraction'])),flush=True)

if __name__=='__main__':main()

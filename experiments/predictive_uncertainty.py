"""Unlabelled checkpoint probes; uncertainty signals evaluated externally."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

from brain.predictive import PredictiveBrain
from experiments.composition import SEEN, HELD, VALUES, directions, dot, experiment, label
from experiments.run import metrics, score


def auc(samples):
    """Mann-Whitney AUC with tie credit; None when either class is absent."""
    correct=[v for v,error in samples if not error]
    wrong=[v for v,error in samples if error]
    if not correct or not wrong:return None
    return sum(float(w>c)+.5*float(w==c) for w in wrong for c in correct)/(len(wrong)*len(correct))


def frames_for(seed,condition,count=200):
    rng=random.Random(seed+280001)
    a,b=directions(seed)
    pairs=HELD if condition=='withheld' else SEEN
    noise={'noise4':4.,'noise8':8.}.get(condition,2.)
    amplitude=.2 if condition=='low_signal' else .6
    for _ in range(count):
        pair=rng.choice(pairs);views=[]
        for _ in range(3):
            n=[rng.uniform(-noise,noise) for _ in a]
            da,db=dot(n,a),dot(n,b)
            views.append([amplitude*(VALUES[pair[0]]*x+VALUES[pair[1]]*y)+v-da*x-db*y
                          for x,y,v in zip(a,b,n)]+[0.])
        yield pair,views+[[0.]*24+[1.]]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--selection',required=True)
    args=p.parse_args()
    options=json.loads(Path(args.selection).read_text())['task']
    with Path(args.output).open('x') as f:
        for seed in range(5):
            _,_,state=experiment(seed,'predictive',options)
            for condition in ('familiar','withheld','noise4','noise8','low_signal','reversed_labels'):
                brain=PredictiveBrain.from_state(state)
                scores=[];trials=[]
                for pair,frames in frames_for(seed,condition):
                    view_errors=[]
                    for frame in frames[:-1]:
                        brain.experience(frame)
                        view_errors.append(sum(u.error*u.error for u in brain.units)/max(1,len(brain.units)))
                    probabilities=brain.experience(frames[-1])
                    target=label(pair)^int(condition=='reversed_labels')
                    ss=score(probabilities,target);scores.append(ss)
                    trials.append(dict(error=int(ss[0]<1.),view_residual=sum(view_errors)/len(view_errors),
                         query_residual=sum(u.error*u.error for u in brain.units)/max(1,len(brain.units)),
                         uncertainty=-sum(x*math.log(max(x,1e-300)) for x in probabilities),
                         confidence=max(probabilities),probabilities=probabilities,target=target,pair=pair))
                result=dict(suite='predictive_uncertainty',variant='predictive',condition=condition,
                            seed=seed,complete=True,config=asdict(brain.config),metrics=metrics(scores),
                            auc={key:auc([(t[key],t['error']) for t in trials]) for key in
                                 ('view_residual','query_residual','uncertainty')},
                            errors=sum(t['error'] for t in trials),
                            mean_confidence=sum(t['confidence'] for t in trials)/len(trials),
                            first_20=metrics(scores[:20]),trials=trials,brain=brain.summary(),
                            source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                f.write(json.dumps(result)+'\n');f.flush()
                print(json.dumps({k:result[k] for k in ('seed','condition','metrics','errors','auc','mean_confidence')}),flush=True)

if __name__=='__main__':main()

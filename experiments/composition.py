"""Held-out factor combinations; task logic never enters the brain module."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

from brain import Config, Organism
from brain.predictive import PredictiveBrain, PredictiveConfig
from experiments.run import metrics, score

VALUES=(-3.,-1.,1.,3.)
HELD=((0,1),(1,3),(2,0),(3,2))
SEEN=tuple((i,j) for i in range(4) for j in range(4) if (i,j) not in HELD)


def dot(a,b):return sum(x*y for x,y in zip(a,b))

def directions(seed,dimension=24):
    rng=random.Random(seed+180001)
    a=[rng.gauss(0,1) for _ in range(dimension)]
    norm=math.sqrt(dot(a,a));a=[x/norm for x in a]
    b=[rng.gauss(0,1) for _ in range(dimension)]
    parallel=dot(a,b);b=[x-parallel*y for x,y in zip(b,a)]
    norm=math.sqrt(dot(b,b));b=[x/norm for x in b]
    return a,b


def stream(seed,episodes=600,noise=2.,amplitude=.6):
    rng=random.Random(seed+180002);a,b=directions(seed)
    def view(pair):
        nuisance=[rng.uniform(-noise,noise) for _ in a]
        da,db=dot(nuisance,a),dot(nuisance,b)
        return [amplitude*(VALUES[pair[0]]*x+VALUES[pair[1]]*y)+n-da*x-db*y
                for x,y,n in zip(a,b,nuisance)]+[0.]
    def frames(pair):return [view(pair) for _ in range(3)]+[[0.]*24+[1.]]
    train=[]
    for _ in range(episodes):
        pair=rng.choice(SEEN);train.append((pair,frames(pair)))
    testpairs=list(HELD)*40;rng.shuffle(testpairs)
    test=[(pair,frames(pair)) for pair in testpairs]
    seenpairs=list(SEEN)*10;rng.shuffle(seenpairs)
    seen=[(pair,frames(pair)) for pair in seenpairs]
    return train,test,seen


def label(pair):return int(VALUES[pair[0]]+VALUES[pair[1]]>=0.)


def clone(brain):return type(brain).from_state(brain.state())


def episode(brain,frames,target=None,events=None,event_offset=0):
    probabilities=None;new_events=[];losses=[]
    for i,x in enumerate(frames):
        kwargs={}
        if isinstance(brain,PredictiveBrain):
            if events is not None:kwargs['structure_override']=events[event_offset+i]
        probabilities=brain.experience(x,target if i==len(frames)-1 else None,**kwargs)
        if isinstance(brain,PredictiveBrain):
            new_events.append(dict(births=brain.last_event['births'],deaths=brain.last_event['deaths'][:]))
            # Predicted before this sensory observation arrived.
            losses.append(sum(u.error*u.error for u in brain.units)/max(1,len(brain.units)))
    return probabilities,new_events,losses


def experiment(seed,variant,options,schedule=None,budget=600):
    train,test,seen=stream(seed,options['episodes'],options['noise'],options['amplitude'])
    if variant=='competitive':
        preset=json.loads(Path('experiments/SELECTED_COMBINED.json').read_text())['overrides']
        brain=Organism(Config(**dict(preset,inputs=25,outputs=2,seed=seed)))
    else:
        settings=dict(options.get('brain',{}),seed=seed)
        if variant in ('frozen','matched_capacity'):settings['hidden_learning']=False
        if variant=='shuffled':settings['shuffled']=True
        brain=PredictiveBrain(PredictiveConfig(**settings))
    start=time.monotonic();scores=[];trajectory=[];events=[];loss=[]
    train_complete=True
    for index,(pair,frames) in enumerate(train):
        if time.monotonic()-start>budget:train_complete=False;break
        p,ee,ll=episode(brain,frames,label(pair),schedule,index*4)
        scores.append(score(p,label(pair)));events.extend(ee);loss.extend(ll)
        if index%100==99:trajectory.append(brain.summary())
    saved=brain.state();trained=len(scores)
    initial=[]
    frozen_transfer=[]
    # Independent checkpoint clones: one held-out episode each, no feedback.
    for pair in HELD:
        first_frames=next(f for p,f in test if p==pair)
        p,_,_=episode(clone(brain),first_frames)
        initial.append(score(p,label(pair)))
        frozen=clone(brain)
        if isinstance(frozen,PredictiveBrain):
            frozen.config=replace(frozen.config,hidden_learning=False,growth=False,pruning=False,activity_rate=0.)
            p,_,_=episode(frozen,first_frames)
            frozen_transfer.append(score(p,label(pair)))
    probe=clone(brain);seen_scores=[]
    for pair,frames in seen:
        p,_,_=episode(probe,frames);seen_scores.append(score(p,label(pair)))
    held_scores=[];first=[];visited=set();probes=[]
    for pair,frames in test:
        if time.monotonic()-start>budget:break
        p,_,_=episode(brain,frames)
        ss=score(p,label(pair));held_scores.append(ss)
        if pair not in visited:first.append(ss);visited.add(pair)
        probes.append(dict(pair=pair,target=label(pair),probabilities=p))
    complete=train_complete and len(held_scores)==len(test)
    result=dict(suite='composition',condition='noisy_factor_sum',variant=variant,seed=seed,
        config=asdict(brain.config),task=options,complete=complete,
        metrics=metrics(held_scores),seen_probe=metrics(seen_scores),first_occurrences=metrics(first),
        initial_transfer=metrics(initial),frozen_hidden_transfer=metrics(frozen_transfer),train_last_200=metrics(scores[-200:]),trained_episodes=trained,
        hidden_prediction_first_200=sum(loss[:200])/max(1,len(loss[:200])),
        hidden_prediction_last_200=sum(loss[-200:])/max(1,len(loss[-200:])),
        brain=brain.summary(),trajectory=trajectory,probes=probes,seconds=time.monotonic()-start,
        controls={'chance':.5,'majority':.5,'exact_unseen_template':.5},
        source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       ('brain/predictive.py','brain/organism.py','experiments/composition.py')})
    if variant!='competitive':
        result['structure_events']=events
        result['capacity_schedule_matched']=schedule is None or events==schedule
        assert result['capacity_schedule_matched']
    return result,events,saved


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parser.add_argument('--variants',default='predictive,frozen,shuffled,matched_capacity,competitive')
    parser.add_argument('--selection')
    parser.add_argument('--budget',type=float,default=600.)
    parser.add_argument('--episodes',type=int,default=600)
    parser.add_argument('--noise',type=float,default=2.)
    args=parser.parse_args()
    options=dict(episodes=args.episodes,noise=args.noise,amplitude=.6,brain={})
    if args.selection:options=json.loads(Path(args.selection).read_text())['task']
    with Path(args.output).open('x') as f:
        for seed in map(int,args.seeds.split(',')):
            schedule=None
            for variant in args.variants.split(','):
                if variant!='predictive' and variant!='competitive' and schedule is None:
                    raise ValueError('Paired controls require predictive first')
                row,ee,_=experiment(seed,variant,options,schedule if variant not in ('predictive','competitive') else None,args.budget)
                if variant=='predictive':schedule=ee
                f.write(json.dumps(row)+'\n');f.flush()
                print(json.dumps({k:row[k] for k in ('seed','variant','complete','metrics','seen_probe','initial_transfer','brain','seconds')}),flush=True)

if __name__=='__main__':main()

"""Distributed next-input predictors with strictly postsynaptic-local learning.

Observation neurons project sensory activity. Predictor neurons predict their
observation partner from previous sensory activity, then receive that partner's
activity through an observation dendrite. No outcome error enters either hidden
population. This is a hypothesis about representation learning, not backprop.
"""
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import random
import tempfile


@dataclass
class PredictiveConfig:
    inputs: int = 25
    outputs: int = 2
    initial_hidden: int = 8
    seed: int = 0
    hidden_rate: float = .2
    outcome_rate: float = .2
    hidden_learning: bool = True
    shuffled: bool = False
    normalize_activity: bool = False
    activity_rate: float = .01
    activity_floor: float = .0001
    growth: bool = True
    growth_interval: int = 200
    growth_threshold: float = .5
    growth_cost: float = .2
    pruning: bool = True
    prune_interval: int = 200
    prune_age: int = 2000
    prune_threshold: float = .00001

    def __post_init__(self):
        if self.inputs < 1 or self.outputs < 2 or self.initial_hidden < 1:
            raise ValueError('Invalid interface or initial population')
        if min(self.growth_interval,self.prune_interval,self.prune_age) < 1:
            raise ValueError('Structural intervals must be positive')
        if min(self.hidden_rate,self.outcome_rate,self.growth_cost) < 0:
            raise ValueError('Rates/cost must be nonnegative')


@dataclass
class LocalSynapse:
    source: int
    strength: float
    trace: float = 0.
    age: int = 0
    usefulness: float = 0.


@dataclass
class PredictiveUnit:
    id: int
    observation: list
    basal: list
    outcome: list
    activity: float = 0.
    observed_activity: float = 0.
    error: float = 0.
    age: int = 0
    usefulness: float = 0.
    surprise: float = 0.
    activity_mean: float = 0.
    activity_variance: float = 1.
    axon: float = 0.


class PredictiveBrain:
    def __init__(self, config):
        self.config=config
        self.rng=random.Random(config.seed)
        self.shuffle_rng=random.Random(config.seed+919191)
        self.sensory=[0.] * config.inputs
        self.units=[]
        self.bias=[0.] * config.outputs
        self.tick=self.next_id=self.births=self.deaths=0
        self.surprise=0.
        self.last_event={'births':0,'deaths':[]}
        for _ in range(config.initial_hidden):
            self._birth()
        self.births=0

    def _birth(self):
        cfg=self.config
        observation=[LocalSynapse(i,self.rng.uniform(-1,1)/math.sqrt(cfg.inputs))
                     for i in range(cfg.inputs)]
        basal=[LocalSynapse(i,self.rng.uniform(-.1,.1)/math.sqrt(cfg.inputs))
               for i in range(cfg.inputs+1)]
        outcome=[[LocalSynapse(self.next_id,0.)] for _ in range(cfg.outputs)]
        self.units.append(PredictiveUnit(self.next_id,observation,basal,outcome))
        self.next_id+=1;self.births+=1

    def _local_prediction_update(self, unit, observed, gate):
        # The receiving neuron sees its observation partner's ACTIVITY, not
        # that partner's error. Eligibility lives in its own afferent synapses.
        unit.observed_activity=observed
        unit.error=observed-unit.activity
        unit.surprise=.99*unit.surprise+.01*abs(unit.error)
        if self.config.hidden_learning:
            energy=1.+sum(s.trace*s.trace for s in unit.basal)
            for s in unit.basal:
                s.strength+=self.config.hidden_rate*gate*s.trace*unit.error/energy

    def experience(self, signal, target=None, gate=1., structure_override=None):
        cfg=self.config
        if len(signal)!=cfg.inputs or any(not math.isfinite(float(x)) for x in signal):
            raise ValueError('Invalid sensory signal')
        if target is not None and (not isinstance(target,int) or not 0<=target<cfg.outputs):
            raise ValueError('Invalid outcome')
        if not math.isfinite(gate) or gate<0:
            raise ValueError('Invalid scalar gate')
        # Emit predictions of the current input from previous sensory neurons.
        previous=self.sensory+[1.]
        for u in self.units:
            for s in u.basal:
                s.trace=previous[s.source]
            u.activity=sum(s.strength*s.trace for s in u.basal)
        for u in self.units:
            u.axon=(u.activity-u.activity_mean)/math.sqrt(max(cfg.activity_floor,u.activity_variance)) if cfg.normalize_activity else u.activity
        logits=self.bias[:]
        for u in self.units:
            for k,edges in enumerate(u.outcome):
                logits[k]+=edges[0].strength*u.axon
        offset=max(logits)
        exps=[math.exp(x-offset) for x in logits]
        probabilities=[x/sum(exps) for x in exps]
        # Now sensory observations arrive. Their activity is a local teacher
        # for the corresponding predictor; outcome labels are still unavailable.
        self.sensory=list(map(float,signal))
        observed=[sum(s.strength*self.sensory[s.source] for s in u.observation)
                  for u in self.units]
        if cfg.shuffled:
            self.shuffle_rng.shuffle(observed)
        for u,value in zip(self.units,observed):
            self._local_prediction_update(u,value,gate)
            if cfg.normalize_activity:
                previous_mean=u.activity_mean
                u.activity_mean+=cfg.activity_rate*(u.activity-previous_mean)
                u.activity_variance=max(0.,(1-cfg.activity_rate)*u.activity_variance+cfg.activity_rate*(u.activity-previous_mean)*(u.activity-u.activity_mean))
            u.age+=1
            u.usefulness=.99*u.usefulness+.01*abs(u.activity)
            for s in u.basal:
                s.age+=1;s.usefulness=.99*s.usefulness+.01*abs(s.strength*s.trace)
        self.surprise=.99*self.surprise+.01*(sum(abs(u.error) for u in self.units)/len(self.units) if self.units else 0.)
        if target is not None:
            errors=[float(k==target)-p for k,p in enumerate(probabilities)]
            energy=1.+sum(u.axon*u.axon for u in self.units)
            for k,error in enumerate(errors):
                self.bias[k]+=cfg.outcome_rate*gate*error/energy
                for u in self.units:
                    s=u.outcome[k][0];s.trace=u.axon;s.age+=1
                    s.strength+=cfg.outcome_rate*gate*s.trace*error/energy
                    s.usefulness=.99*s.usefulness+.01*abs(s.strength*s.trace)
        self.tick+=1
        self.last_event={'births':0,'deaths':[]}
        if structure_override is None:
            dead=[u.id for u in self.units if cfg.pruning and self.tick%cfg.prune_interval==0
                  and u.age>cfg.prune_age and u.usefulness<cfg.prune_threshold]
            birth=int(cfg.growth and self.tick%cfg.growth_interval==0 and
                      self.surprise>cfg.growth_threshold+cfg.growth_cost*math.log1p(len(self.units)))
        else:
            dead=list(structure_override['deaths']);birth=structure_override['births']
        if dead:
            existing={u.id for u in self.units}
            if not set(dead)<=existing:
                raise ValueError('Cannot prune absent neurons')
            self.units=[u for u in self.units if u.id not in dead]
            self.deaths+=len(dead);self.last_event['deaths']=dead
        for _ in range(birth):
            self._birth();self.last_event['births']+=1
        return probabilities

    def summary(self):
        cfg=self.config
        return dict(tick=self.tick,hidden=len(self.units),neurons=cfg.inputs+cfg.outputs+2*len(self.units),
                    synapses=len(self.units)*(2*cfg.inputs+1+cfg.outputs),births=self.births,
                    deaths=self.deaths,local_prediction_error=self.surprise)

    def state(self):
        return dict(format='predictive-1',config=asdict(self.config),rng=self.rng.getstate(),
                    shuffle_rng=self.shuffle_rng.getstate(),sensory=self.sensory[:],bias=self.bias[:],
                    units=[asdict(u) for u in self.units],tick=self.tick,next_id=self.next_id,
                    births=self.births,deaths=self.deaths,surprise=self.surprise,
                    last_event={'births':self.last_event['births'],'deaths':self.last_event['deaths'][:]})

    @classmethod
    def from_state(cls,state):
        if state.get('format')!='predictive-1':
            raise ValueError('Unsupported checkpoint')
        obj=cls.__new__(cls);obj.config=PredictiveConfig(**state['config'])
        def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,list) else x
        for key in ('rng','shuffle_rng'):
            r=random.Random();r.setstate(tuples(state[key]));setattr(obj,key,r)
        obj.units=[]
        for v in state['units']:
            d=dict(v)
            for key in ('observation','basal'):
                d[key]=[LocalSynapse(**s) for s in d[key]]
            d['outcome']=[[LocalSynapse(**s) for s in edges] for edges in d['outcome']]
            obj.units.append(PredictiveUnit(**d))
        for key in ('sensory','bias','tick','next_id','births','deaths','surprise','last_event'):
            setattr(obj,key,json.loads(json.dumps(state[key])))
        return obj

    def save(self,path):
        path=Path(path);temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
                temporary=Path(f.name);json.dump(self.state(),f,allow_nan=False)
                f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()

    @classmethod
    def load(cls,path):return cls.from_state(json.loads(Path(path).read_text()))

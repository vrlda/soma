"""Explicitly labelled linear neural baseline; never part of SOMA."""
import json
import math
from types import SimpleNamespace


class LinearMemoryBaseline:
    def __init__(self,inputs=25,outputs=2,rate=.2):
        self.config=SimpleNamespace(inputs=inputs,outputs=outputs,rate=rate)
        self.previous=[0.]*inputs
        self.mean=[0.]*inputs
        self.variance=[1.]*inputs
        self.weights=[[0.]*(inputs+1) for _ in range(outputs)]
        self.tick=0

    def experience(self,signal,target=None):
        x=[(v-m)/math.sqrt(max(.0001,s)) for v,m,s in
           zip(self.previous,self.mean,self.variance)]+[1.]
        logits=[sum(w*v for w,v in zip(row,x)) for row in self.weights]
        exps=[math.exp(z-max(logits)) for z in logits]
        p=[v/sum(exps) for v in exps]
        if target is not None:
            energy=1.+sum(v*v for v in x)
            for k,row in enumerate(self.weights):
                error=float(k==target)-p[k]
                for i,v in enumerate(x):row[i]+=self.config.rate*v*error/energy
        for i,v in enumerate(self.previous):
            old=self.mean[i];self.mean[i]+=.01*(v-old)
            self.variance[i]=max(0.,.99*self.variance[i]+.01*(v-old)*(v-self.mean[i]))
        self.previous=signal[:];self.tick+=1
        return p

    def state(self):
        return json.loads(json.dumps(dict(config=vars(self.config),previous=self.previous,
                      mean=self.mean,variance=self.variance,weights=self.weights,tick=self.tick)))

    @classmethod
    def from_state(cls,state):
        state=json.loads(json.dumps(state));obj=cls(**state.pop('config'))
        for k,v in state.items():setattr(obj,k,v)
        return obj

    def summary(self):
        return dict(tick=self.tick,hidden=0,neurons=self.config.inputs+self.config.outputs,
                    synapses=self.config.outputs*(self.config.inputs+1),births=0,deaths=0)

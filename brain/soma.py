"""One experimental substrate: event traces and nonlinear local prediction.

Each neuron forecasts its own sensory observation. Dendrites receive only that
soma's mismatch and retain their own eligibility. Outcomes never teach hidden
neurons. Fixed branch polarity is local wiring, not transported readout credit.
"""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile


@dataclass
class SomaConfig:
    inputs: int = 13
    outputs: int = 2
    seed: int = 0
    initial_hidden: int = 4
    branches: int = 4
    hidden_rate: float = .05
    outcome_rate: float = .3
    trace_decay: float = .8
    statistics_rate: float = .01
    variance_floor: float = .01
    hidden_learning: bool = True
    hidden_rule: str = 'prediction'
    shuffled: bool = False
    plasticity: bool = True
    growth: bool = True
    growth_interval: int = 120
    growth_threshold: float = .02
    growth_cost: float = .01
    pruning: bool = True
    prune_interval: int = 240
    prune_age: int = 1200
    prune_threshold: float = .015
    evidence_decay: float = .999
    evidence_rate: float = .01
    pressure_rule: str = 'branch'
    protection: bool = True

    def __post_init__(self):
        if min(self.inputs, self.initial_hidden, self.branches) < 1 or self.outputs < 2:
            raise ValueError('Invalid interface or population')
        if min(self.growth_interval, self.prune_interval, self.prune_age) < 1:
            raise ValueError('Intervals must be positive')
        if not 0 <= self.trace_decay < 1 or not 0 <= self.statistics_rate <= 1:
            raise ValueError('Invalid trace/statistics rate')
        if self.pressure_rule not in ('branch', 'underfit', 'covariance'):
            raise ValueError('Unknown structural pressure rule')
        if self.hidden_rule not in ('prediction', 'oja'):
            raise ValueError('Unknown local rule')
        if min(self.hidden_rate, self.outcome_rate, self.growth_cost) < 0:
            raise ValueError('Rates and costs must be nonnegative')
        if self.variance_floor <= 0 or not 0 <= self.evidence_decay <= 1:
            raise ValueError('Invalid variance/evidence parameters')


@dataclass
class Synapse:
    source: int
    strength: float
    trace: float = 0.
    evidence: float = 0.
    age: int = 0


@dataclass
class Dendrite:
    synapses: list
    polarity: float
    activity: float = 0.
    correlation: float = 0.
    mean: float = 0.
    observed_covariance: float = 0.
    forecast_covariance: float = 0.


@dataclass
class Neuron:
    id: int
    observation: list
    dendrites: list
    outcomes: list
    activity: float = 0.
    axon: float = 0.
    mean: float = 0.
    variance: float = 1.
    error: float = 0.
    age: int = 0
    usefulness: float = 0.
    evidence: float = 0.
    underfit: float = 0.
    observed_mean: float = 0.


class Soma:
    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.shuffle_rng = random.Random(config.seed + 919191)
        self.sensory = [0.] * config.inputs
        self.traces = [0.] * config.inputs
        self.means = [0.] * config.inputs
        self.variances = [1.] * config.inputs
        self.bias = [0.] * config.outputs
        self.neurons = []
        self.tick = 0
        self.next_id = 0
        self.births = 0
        self.deaths = 0
        self.pressure = 0.
        self.last_event = {'births': 0, 'deaths': []}
        for _ in range(config.initial_hidden):
            self._birth()
        self.births = 0

    def _birth(self):
        cfg = self.config
        scale = 1 / math.sqrt(cfg.inputs)
        observation = [Synapse(i, self.rng.uniform(-1, 1) * scale)
                       for i in range(cfg.inputs)]
        dendrites = []
        for _ in range(cfg.branches):
            edges = [Synapse(i, self.rng.uniform(-1, 1) * scale)
                     for i in range(cfg.inputs + 1)]
            polarity = self.rng.choice((-1., 1.)) / math.sqrt(cfg.branches)
            dendrites.append(Dendrite(edges, polarity))
        outcomes = [Synapse(self.next_id, 0.) for _ in range(cfg.outputs)]
        self.neurons.append(Neuron(self.next_id, observation, dendrites, outcomes))
        self.next_id += 1
        self.births += 1

    def _learn_prediction(self, neuron, observed, gate):
        cfg = self.config
        neuron.error = observed - neuron.activity
        rate = cfg.statistics_rate
        # Covariance cancels independent noise, rather than counting raw error.
        skill = neuron.activity * observed
        neuron.usefulness += rate * (skill - neuron.usefulness)
        neuron.evidence = cfg.evidence_decay * neuron.evidence + cfg.evidence_rate * max(0., skill)
        neuron.underfit += rate * (neuron.error * neuron.activity - neuron.underfit)
        observed_delta = observed - neuron.observed_mean
        forecast_delta = neuron.activity - neuron.mean
        neuron.observed_mean += rate * observed_delta
        for branch in neuron.dendrites:
            branch_delta = branch.activity - branch.mean
            branch.observed_covariance += rate * (observed_delta * branch_delta - branch.observed_covariance)
            branch.forecast_covariance += rate * (forecast_delta * branch_delta - branch.forecast_covariance)
            branch.mean += rate * branch_delta
            branch.correlation += rate * (neuron.error * branch.activity - branch.correlation)
            if not cfg.hidden_learning:
                continue
            energy = 1. + sum(edge.trace ** 2 for edge in branch.synapses)
            # A branch's own saturation gates plasticity. This is a local
            # three-factor rule; no derivative of another neuron is used.
            local_gate = 1. - abs(branch.activity)
            for edge in branch.synapses:
                if cfg.hidden_rule == 'prediction':
                    change = edge.trace * neuron.error * branch.polarity * local_gate
                else:
                    change = branch.activity * (edge.trace - branch.activity * edge.strength)
                edge.strength += cfg.hidden_rate * gate * change / energy
                edge.evidence = cfg.evidence_decay * edge.evidence + rate * abs(change)
                edge.age += 1

    def experience(self, signal, target=None, gate=1., structure_override=None, observation_signal=None):
        cfg = self.config
        if len(signal) != cfg.inputs or any(not math.isfinite(float(v)) for v in signal):
            raise ValueError('Invalid sensory signal')
        if observation_signal is not None and (len(observation_signal) != cfg.inputs
                or any(not math.isfinite(float(v)) for v in observation_signal)):
            raise ValueError('Invalid control observation')
        if target is not None and (not isinstance(target, int) or not 0 <= target < cfg.outputs):
            raise ValueError('Invalid outcome')
        if not math.isfinite(gate) or gate < 0:
            raise ValueError('Invalid scalar gate')
        if structure_override is not None:
            ids = {neuron.id for neuron in self.neurons}
            if not set(structure_override['deaths']) <= ids or structure_override['births'] < 0:
                raise ValueError('Invalid structural schedule')
        self.sensory = list(map(float, signal))
        observed_sensory = self.sensory if observation_signal is None else observation_signal
        observations = [sum(edge.strength * observed_sensory[edge.source]
                            for edge in neuron.observation) for neuron in self.neurons]
        if cfg.shuffled:
            self.shuffle_rng.shuffle(observations)
        # Incoming sensory activity teaches the PREVIOUS forecast; current
        # outcome and its error are unavailable here.
        if cfg.plasticity and self.tick:
            for neuron, value in zip(self.neurons, observations):
                self._learn_prediction(neuron, value, gate)
        for i, value in enumerate(self.sensory):
            decayed = cfg.trace_decay * self.traces[i]
            self.traces[i] = value if abs(value) >= abs(decayed) else decayed
        normalized = [(value - mean) / math.sqrt(max(cfg.variance_floor, variance))
                      for value, mean, variance in zip(self.traces, self.means, self.variances)] + [1.]
        logits = self.bias[:]
        for neuron in self.neurons:
            neuron.activity = 0.
            for branch in neuron.dendrites:
                for edge in branch.synapses:
                    edge.trace = normalized[edge.source]
                branch.activity = math.tanh(sum(edge.strength * edge.trace for edge in branch.synapses))
                neuron.activity += branch.polarity * branch.activity
            neuron.axon = (neuron.activity - neuron.mean) / math.sqrt(max(cfg.variance_floor, neuron.variance))
            for k, edge in enumerate(neuron.outcomes):
                logits[k] += edge.strength * neuron.axon
            neuron.age += 1
        offset = max(logits)
        exps = [math.exp(value - offset) for value in logits]
        probabilities = [value / sum(exps) for value in exps]
        # Only outcome neurons receive labels. No outgoing error is fed back.
        if cfg.plasticity and target is not None:
            energy = 1. + sum(neuron.axon ** 2 for neuron in self.neurons)
            for k, probability in enumerate(probabilities):
                error = float(k == target) - probability
                self.bias[k] += cfg.outcome_rate * gate * error / energy
                for neuron in self.neurons:
                    edge = neuron.outcomes[k]
                    edge.trace = neuron.axon
                    edge.strength += cfg.outcome_rate * gate * edge.trace * error / energy
                    edge.evidence = cfg.evidence_decay * edge.evidence + .01 * abs(edge.trace * edge.strength)
                    edge.age += 1
        if cfg.plasticity:
            rate = cfg.statistics_rate
            for i, value in enumerate(self.traces):
                old = self.means[i]
                self.means[i] += rate * (value - old)
                self.variances[i] = max(0., (1 - rate) * self.variances[i] + rate * (value - old) * (value - self.means[i]))
            for neuron in self.neurons:
                old = neuron.mean
                neuron.mean += rate * (neuron.activity - old)
                neuron.variance = max(0., (1 - rate) * neuron.variance + rate * (neuron.activity - old) * (neuron.activity - neuron.mean))
        correlations = [abs(branch.correlation) for neuron in self.neurons for branch in neuron.dendrites]
        if cfg.pressure_rule == 'underfit':
            self.pressure = max((max(0., neuron.underfit) for neuron in self.neurons), default=0.)
        elif cfg.pressure_rule == 'covariance':
            self.pressure = max((max(0., abs(branch.observed_covariance) - abs(branch.forecast_covariance))
                                 for neuron in self.neurons for branch in neuron.dendrites), default=0.)
        else:
            self.pressure = max(correlations, default=0.)
        self.tick += 1
        self.last_event = {'births': 0, 'deaths': []}
        if structure_override is not None:
            dead = structure_override['deaths'][:]
            births = structure_override['births']
        else:
            dead = [neuron.id for neuron in self.neurons
                    if cfg.plasticity and cfg.pruning and self.tick % cfg.prune_interval == 0
                    and neuron.age > cfg.prune_age
                    and neuron.usefulness + (neuron.evidence if cfg.protection else 0.) < cfg.prune_threshold]
            births = int(cfg.plasticity and cfg.growth and self.tick % cfg.growth_interval == 0
                         and self.pressure > cfg.growth_threshold + cfg.growth_cost * math.log1p(len(self.neurons)))
        if dead:
            self.neurons = [neuron for neuron in self.neurons if neuron.id not in dead]
            self.deaths += len(dead)
            self.last_event['deaths'] = dead
        for _ in range(births):
            self._birth()
            self.last_event['births'] += 1
        return probabilities

    def summary(self):
        cfg = self.config
        connections = len(self.neurons) * (cfg.inputs + cfg.branches * (cfg.inputs + 1) + cfg.outputs)
        return dict(tick=self.tick, hidden=len(self.neurons), neurons=cfg.inputs + cfg.outputs + len(self.neurons),
                    synapses=connections, births=self.births, deaths=self.deaths, structured_surprise=self.pressure)

    def state(self):
        return json.loads(json.dumps(dict(format='soma-1', config=asdict(self.config), rng=self.rng.getstate(),
                  shuffle_rng=self.shuffle_rng.getstate(), sensory=self.sensory, traces=self.traces,
                  means=self.means, variances=self.variances, bias=self.bias,
                  neurons=[asdict(neuron) for neuron in self.neurons], tick=self.tick,
                  next_id=self.next_id, births=self.births, deaths=self.deaths,
                  pressure=self.pressure, last_event=self.last_event)))

    @classmethod
    def from_state(cls, state):
        state = json.loads(json.dumps(state))
        if state.pop('format') != 'soma-1':
            raise ValueError('Unsupported checkpoint')
        obj = cls.__new__(cls)
        obj.config = SomaConfig(**state.pop('config'))
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value
        for key in ('rng', 'shuffle_rng'):
            rng = random.Random()
            rng.setstate(tuples(state.pop(key)))
            setattr(obj, key, rng)
        obj.neurons = []
        for data in state.pop('neurons'):
            data['observation'] = [Synapse(**edge) for edge in data['observation']]
            data['outcomes'] = [Synapse(**edge) for edge in data['outcomes']]
            data['dendrites'] = [Dendrite([Synapse(**edge) for edge in branch['synapses']],
                                        branch['polarity'], branch['activity'], branch['correlation'],
                                        branch.get('mean', 0.), branch.get('observed_covariance', 0.),
                                        branch.get('forecast_covariance', 0.))
                                 for branch in data['dendrites']]
            obj.neurons.append(Neuron(**data))
        for key, value in state.items():
            setattr(obj, key, value)
        return obj

    def save(self, path):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(self.state(), file, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        return cls.from_state(json.loads(Path(path).read_text()))

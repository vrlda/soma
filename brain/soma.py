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


@dataclass
class SharedConfig:
    """One set of rules across interfaces; legacy Soma remains unchanged."""
    inputs: int = 13
    outputs: int = 2
    seed: int = 0
    predictors: int = 8
    initial_fields: int = 1
    hidden_rate: float = .02
    outcome_rate: float = .35
    afferent_rate: float = .05
    trace_decay: float = .8
    statistics_rate: float = .01
    variance_floor: float = .0001
    raw_weight: float = .25
    predictive_readout: float = .2
    vigilance: float = .20
    persistence_matches: int = 3
    growth_interval: int = 1
    growth_cost: float = .001
    pruning: bool = True
    prune_interval: int = 240
    prune_age: int = 2400
    prune_threshold: float = .00001
    evidence_decay: float = .99999
    retention_weight: float = .001
    consolidation_rate: float = .001
    recruitment: bool = True
    prediction_learning: bool = True
    plasticity: bool = True
    protection: bool = True
    structured_threshold: float = 0.
    # An optional random-wiring capacity control, never candidate knowledge.
    initial_layout: tuple = ()

    def __post_init__(self):
        if self.inputs < 1 or self.outputs < 2 or self.predictors < 1 or self.initial_fields < 0:
            raise ValueError('Invalid shared interface/population')
        if min(self.growth_interval, self.prune_interval, self.prune_age, self.persistence_matches) < 1:
            raise ValueError('Invalid shared structural cadence')
        if not 0 <= self.trace_decay < 1 or not 0 <= self.statistics_rate <= 1:
            raise ValueError('Invalid shared temporal rate')
        if not 0 <= self.raw_weight <= 1 or self.vigilance <= 0 or self.variance_floor <= 0:
            raise ValueError('Invalid matching parameters')
        rates = (self.hidden_rate, self.outcome_rate, self.afferent_rate,
                 self.growth_cost, self.retention_weight, self.consolidation_rate,
                 self.predictive_readout, self.prune_threshold)
        if any(not math.isfinite(v) or v < 0 for v in rates):
            raise ValueError('Invalid shared rate/cost')
        if not 0 <= self.evidence_decay <= 1 or not 0 <= self.structured_threshold <= 1:
            raise ValueError('Invalid shared evidence/structure threshold')


@dataclass
class ReceptiveBranch:
    id: int
    afferents: list
    outcomes: list
    forecast_strength: float = 0.
    activity: float = 0.
    age: int = 0
    usefulness: float = 0.
    evidence: float = 0.
    mismatch: float = 0.


@dataclass
class ForecastNeuron:
    id: int
    observation: list
    basal: list
    outcomes: list
    branches: list
    activity: float = 0.
    axon: float = 0.
    error: float = 0.
    residual_energy: float = 0.


class SharedSoma:
    """Local forecasts and competing receptive branches in ONE neural graph.

    A branch exposes local compartment activity to outcome synapses. It also
    supplies a forecasting current to its own soma. Its matching afferents
    never receive that soma error: only the branch-to-soma synapse does.
    """
    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.sensory = [0.] * config.inputs
        dimension = config.inputs + config.predictors
        self.traces = [0.] * dimension
        self.means = [0.] * dimension
        self.variances = [1.] * dimension
        self.bias = [0.] * config.outputs
        self.neurons = []
        self.tick = 0
        self.next_id = 0
        self.births = 0
        self.deaths = 0
        self.synapse_births = 0
        self.synapse_deaths = 0
        self.replacements = 0
        self.pending_matches = 0
        self.pressure = 0.
        self.raw_means = [0.] * config.inputs
        self.raw_variances = [1.] * config.inputs
        self.raw_covariances = [0.] * config.inputs
        self.structured_fraction = 0.
        self.last_birth_tick = 0
        self.last_event = dict(births=0, deaths=[], parent=None)
        for i in range(config.predictors):
            scale = 1 / math.sqrt(config.inputs)
            observed = [Synapse(j, self.rng.uniform(-1, 1) * scale) for j in range(config.inputs)]
            basal = [Synapse(j, self.rng.uniform(-.1, .1) * scale) for j in range(config.inputs + 1)]
            outputs = [Synapse(i, 0.) for _ in range(config.outputs)]
            self.neurons.append(ForecastNeuron(i, observed, basal, outputs, []))
        # One provisional dendrite is neural scratch state, not exemplar storage.
        self.pending = [Synapse(j, 0.) for j in range(dimension)]
        if config.initial_layout:
            for parent, channels in config.initial_layout:
                self._birth(parent, [self.rng.uniform(-1, 1) for _ in range(dimension)], channels)
        else:
            for i in range(config.initial_fields):
                self._birth(i % config.predictors, [self.rng.uniform(-1, 1) for _ in range(dimension)])
        self.births = 0
        self.synapse_births = 0
        self.initial_synapses = self._connection_count()
        self.initial_branch_count = len(self._branches())

    def _branches(self):
        return [(neuron, branch) for neuron in self.neurons for branch in neuron.branches]

    def _connection_count(self):
        cfg = self.config
        return (cfg.predictors * (2 * cfg.inputs + 1 + cfg.outputs)
                + len(self.pending)
                + sum(len(b.afferents) + 1 + len(b.outcomes) for _, b in self._branches()))

    def _birth(self, parent, values, channels=None):
        cfg = self.config
        channels = range(cfg.outputs) if channels is None else channels
        # Outcome source IDs belong to the emitting compartment, not a label.
        outcomes = [Synapse(k, 0.) for k in channels]
        afferents = [Synapse(i, value) for i, value in enumerate(values)]
        branch = ReceptiveBranch(self.next_id, afferents, outcomes)
        self.neurons[parent].branches.append(branch)
        self.next_id += 1
        self.births += 1
        self.synapse_births += len(afferents) + 1 + len(outcomes)
        return branch

    def _distance(self, afferents, values):
        cfg = self.config
        raw = sum((v - s.strength) ** 2 / max(cfg.variance_floor, self.variances[s.source])
                  for s, v in zip(afferents[:cfg.inputs], values[:cfg.inputs])) / cfg.inputs
        predicted = sum((v - s.strength) ** 2 / max(cfg.variance_floor, self.variances[s.source])
                        for s, v in zip(afferents[cfg.inputs:], values[cfg.inputs:])) / cfg.predictors
        return cfg.raw_weight * raw + (1 - cfg.raw_weight) * predicted

    def _learn_forecast(self, neuron, observed, gate):
        """Own teacher, own prior eligibility; distant errors are unavailable."""
        cfg = self.config
        neuron.error = observed - neuron.activity
        neuron.residual_energy += cfg.statistics_rate * (neuron.error ** 2 - neuron.residual_energy)
        if not cfg.prediction_learning:
            return
        energy = 1. + sum(s.trace ** 2 for s in neuron.basal) + sum(b.activity ** 2 for b in neuron.branches)
        for edge in neuron.basal:
            edge.strength += cfg.hidden_rate * gate * edge.trace * neuron.error / energy
        for branch in neuron.branches:
            branch.forecast_strength += cfg.hidden_rate * gate * branch.activity * neuron.error / energy

    def experience(self, signal, target=None, gate=1., observation_signal=None):
        cfg = self.config
        if len(signal) != cfg.inputs or any(not math.isfinite(float(v)) for v in signal):
            raise ValueError('Invalid shared sensory input')
        if observation_signal is not None and (len(observation_signal) != cfg.inputs
                or any(not math.isfinite(float(v)) for v in observation_signal)):
            raise ValueError('Invalid shared control teacher')
        if target is not None and (not isinstance(target, int) or not 0 <= target < cfg.outputs):
            raise ValueError('Invalid shared outcome')
        if not math.isfinite(gate) or gate < 0:
            raise ValueError('Invalid shared gate')
        # Forecast current sensory activity before seeing it, from prior sources.
        previous = self.sensory + [1.]
        for neuron in self.neurons:
            for edge in neuron.basal:
                edge.trace = previous[edge.source]
            neuron.activity = (sum(s.strength * s.trace for s in neuron.basal)
                               + sum(b.forecast_strength * b.activity for b in neuron.branches))
        teacher = signal if observation_signal is None else observation_signal
        if cfg.plasticity:
            for neuron in self.neurons:
                observed = sum(s.strength * teacher[s.source] for s in neuron.observation)
                self._learn_forecast(neuron, observed, gate)
        if cfg.plasticity and cfg.structured_threshold:
            # Presynaptic sources maintain their own lag covariance from RAW
            # activity, before event traces can manufacture apparent persistence.
            rate = cfg.statistics_rate
            for i, value in enumerate(signal):
                old_mean = self.raw_means[i]
                delta = value - old_mean
                previous_delta = self.sensory[i] - old_mean
                self.raw_covariances[i] += rate * (delta * previous_delta - self.raw_covariances[i])
                self.raw_means[i] += rate * delta
                self.raw_variances[i] = max(0., (1 - rate) * self.raw_variances[i]
                                             + rate * delta * (value - self.raw_means[i]))
            # A broadcast amount gate, never a desired weight direction.
            self.structured_fraction = max(0., sum(self.raw_covariances)) / max(
                cfg.variance_floor * cfg.inputs, sum(self.raw_variances))
        sources = list(map(float, signal)) + [n.activity for n in self.neurons]
        for i, value in enumerate(sources):
            residual = cfg.trace_decay * self.traces[i]
            self.traces[i] = value if abs(value) >= abs(residual) else residual
        if cfg.plasticity:
            rate = cfg.statistics_rate
            for i, value in enumerate(self.traces):
                old = self.means[i]
                self.means[i] += rate * (value - old)
                self.variances[i] = max(0., (1 - rate) * self.variances[i]
                                        + rate * (value - old) * (value - self.means[i]))
        fields = self._branches()
        winner = None
        novelty = 1.
        for neuron, branch in fields:
            branch.mismatch = self._distance(branch.afferents, self.traces)
            if winner is None or branch.mismatch < novelty:
                winner = (neuron, branch)
                novelty = branch.mismatch
        represented = winner is not None and novelty <= cfg.vigilance
        self.pressure = .95 * self.pressure + .05 * novelty
        for neuron, branch in fields:
            branch.activity = float(represented and branch is winner[1])
            branch.age += 1
            if cfg.plasticity:
                branch.usefulness += .01 * (branch.activity - branch.usefulness)
                branch.evidence = cfg.evidence_decay * branch.evidence + branch.activity * max(0., 1 - branch.mismatch)
                rate = cfg.afferent_rate / (1 + cfg.consolidation_rate * branch.evidence)
                for edge in branch.afferents:
                    edge.trace = self.traces[edge.source]
                    edge.age += 1
                    edge.evidence = cfg.evidence_decay * edge.evidence + branch.activity * max(0., 1 - abs(edge.trace - edge.strength))
                    edge.strength += rate * gate * branch.activity * (edge.trace - edge.strength)
        # Source-local predictive axons and local branch compartment outputs.
        logits = self.bias[:]
        for neuron in self.neurons:
            i = cfg.inputs + neuron.id
            neuron.axon = cfg.predictive_readout * (self.traces[i] - self.means[i]) / math.sqrt(max(cfg.variance_floor, self.variances[i]) * cfg.predictors)
            for k, edge in enumerate(neuron.outcomes):
                logits[k] += edge.strength * neuron.axon
            for branch in neuron.branches:
                for edge in branch.outcomes:
                    logits[edge.source] += edge.strength * branch.activity
        exps = [math.exp(v - max(logits)) for v in logits]
        probabilities = [v / sum(exps) for v in exps]
        if cfg.plasticity and target is not None:
            energy = 1. + sum(n.axon ** 2 for n in self.neurons) + sum(b.activity ** 2 for _, b in fields)
            for k, p in enumerate(probabilities):
                error = float(k == target) - p
                self.bias[k] += cfg.outcome_rate * gate * error / energy
                for neuron in self.neurons:
                    neuron.outcomes[k].strength += cfg.outcome_rate * gate * neuron.axon * error / energy
                for _, branch in fields:
                    if branch.activity and all(s.source != k for s in branch.outcomes):
                        branch.outcomes.append(Synapse(k, 0.))
                        self.synapse_births += 1
                    for edge in branch.outcomes:
                        if edge.source == k:
                            edge.strength += cfg.outcome_rate * gate * branch.activity * error / energy
        self.tick += 1
        self.sensory = list(map(float, signal))
        self.last_event = dict(births=0, deaths=[], parent=None)
        if cfg.plasticity:
            for _, branch in fields:
                for edge in branch.outcomes:
                    edge.age += 1
                    edge.evidence = cfg.evidence_decay * edge.evidence + abs(edge.strength * branch.activity)
            if cfg.pruning and self.tick % cfg.prune_interval == 0:
                for neuron in self.neurons:
                    kept = []
                    for branch in neuron.branches:
                        importance = branch.usefulness
                        if cfg.protection:
                            importance += cfg.retention_weight * branch.evidence / (1 + branch.evidence)
                        if branch.age > cfg.prune_age and importance < cfg.prune_threshold:
                            self.deaths += 1
                            self.synapse_deaths += len(branch.afferents) + 1 + len(branch.outcomes)
                            self.last_event['deaths'].append(branch.id)
                            continue
                        remaining = []
                        for edge in branch.outcomes:
                            importance = abs(edge.strength) * branch.usefulness
                            if cfg.protection:
                                importance += cfg.retention_weight * edge.evidence / (1 + edge.evidence)
                            if edge.age > cfg.prune_age and importance < cfg.prune_threshold:
                                self.synapse_deaths += 1
                            else:
                                remaining.append(edge)
                        branch.outcomes = remaining
                        kept.append(branch)
                    neuron.branches = kept
            if cfg.recruitment and not represented:
                tolerance = cfg.vigilance / 2
                if self.pending_matches and self._distance(self.pending, self.traces) <= tolerance:
                    self.pending_matches += 1
                    for edge in self.pending:
                        edge.strength += .2 * (self.traces[edge.source] - edge.strength)
                else:
                    for edge in self.pending:
                        edge.strength = self.traces[edge.source]
                    self.pending_matches = 1
                    self.replacements += 1
                cost = cfg.growth_cost * math.log1p(len(self._branches()))
                due = self.tick - self.last_birth_tick >= cfg.growth_interval
                structured = not cfg.structured_threshold or self.structured_fraction > cfg.structured_threshold
                if due and structured and self.pending_matches >= cfg.persistence_matches and novelty > cfg.vigilance + cost:
                    parent = max(self.neurons, key=lambda n: (n.residual_energy, -n.id))
                    self._birth(parent.id, [s.strength for s in self.pending])
                    self.last_event.update(births=1, parent=parent.id)
                    self.last_birth_tick = self.tick
                    self.pending_matches = 0
            elif represented:
                self.pending_matches = 0
        return probabilities

    def summary(self):
        cfg = self.config
        return dict(tick=self.tick, hidden=cfg.predictors, neurons=cfg.inputs + cfg.outputs + cfg.predictors,
                    branches=len(self._branches()), provisional_branches=1,
                    synapses=self._connection_count(), births=self.births, deaths=self.deaths,
                    synapse_births=self.synapse_births, synapse_deaths=self.synapse_deaths,
                    structured_surprise=self.pressure, provisional_replacements=self.replacements,
                    prediction_error=sum(n.error ** 2 for n in self.neurons) / cfg.predictors,
                    raw_structured_fraction=self.structured_fraction)

    def state(self):
        result = dict(format='shared-soma-1', config=asdict(self.config), rng=self.rng.getstate(),
                      neurons=[asdict(n) for n in self.neurons], pending=[asdict(s) for s in self.pending])
        for key in ('sensory', 'traces', 'means', 'variances', 'bias', 'tick', 'next_id',
                    'births', 'deaths', 'synapse_births', 'synapse_deaths', 'replacements',
                    'pending_matches', 'pressure', 'last_birth_tick', 'last_event',
                    'initial_synapses', 'initial_branch_count', 'raw_means',
                    'raw_variances', 'raw_covariances', 'structured_fraction'):
            result[key] = getattr(self, key)
        return json.loads(json.dumps(result))

    @classmethod
    def from_state(cls, state):
        data = json.loads(json.dumps(state))
        if data.pop('format') != 'shared-soma-1':
            raise ValueError('Unsupported shared checkpoint')
        obj = cls.__new__(cls)
        obj.config = SharedConfig(**data.pop('config'))
        obj.rng = random.Random()
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        obj.rng.setstate(tuples(data.pop('rng')))
        obj.neurons = []
        for neuron in data.pop('neurons'):
            for key in ('observation', 'basal', 'outcomes'):
                neuron[key] = [Synapse(**s) for s in neuron[key]]
            for branch in neuron['branches']:
                branch['afferents'] = [Synapse(**s) for s in branch['afferents']]
                branch['outcomes'] = [Synapse(**s) for s in branch['outcomes']]
            neuron['branches'] = [ReceptiveBranch(**b) for b in neuron['branches']]
            obj.neurons.append(ForecastNeuron(**neuron))
        obj.pending = [Synapse(**s) for s in data.pop('pending')]
        for key, value in data.items():
            setattr(obj, key, value)
        return obj

    # Same atomic file operation as the legacy substrate; no neural delegation.
    save = Soma.save
    load = classmethod(Soma.load.__func__)

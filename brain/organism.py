"""Local neural dynamics. No task logic, count memory, or backpropagation.

Prediction neurons receive an outcome locally after producing a prediction.
Their afferent synapses use that neuron's error; hidden neurons never receive
that error. This first candidate therefore does NOT solve deep credit assignment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile


@dataclass
class Config:
    inputs: int = 4
    outputs: int = 4
    initial_hidden: int = 16
    seed: int = 0
    learning_rate: float = 0.35
    hidden_learning_rate: float = 0.0005
    learning: bool = True
    traces: bool = True
    recurrence: bool = True
    persistence: bool = True
    hidden_plasticity: bool = True
    metaplasticity: bool = False
    consolidation_rate: float = 0.01
    growth: bool = False
    pruning: bool = False
    growth_interval: int = 100
    growth_threshold: float = 0.35
    growth_cost: float = 0.05
    prune_interval: int = 100
    prune_age: int = 600
    prune_threshold: float = 0.0001
    drive_density: float = 0.2
    trace_decays: tuple = (0.0, 0.6, 0.9, 0.98)
    trace_bank: bool = False
    trace_rule: str = "ema"
    normalized_updates: bool = False
    hidden_rule: str = "oja"
    growth_rule: str = "prediction"
    seed_birth_from_activity: bool = False
    hidden_metaplasticity: bool = False
    drive_trace_rule: str = "none"
    drive_decay: float = 0.8
    vigilance: float = 0.0
    growth_clock: str = "outcome"
    growth_requires_current_novelty: bool = False
    retain_active_representations: bool = False
    regrow_predictions: bool = False
    synapse_growth_interval: int = 10
    synapse_growth_threshold: float = 0.05
    match_evidence: bool = False
    evidence_decay: float = 0.999
    retention_protection: bool = False
    retention_weight: float = 0.001
    afferent_normalization: bool = False
    normalization_rate: float = 0.01
    variance_floor: float = 0.0001

    def __post_init__(self):
        self.trace_decays = tuple(self.trace_decays)
        if self.inputs < 1 or self.outputs < 2 or self.initial_hidden < 0:
            raise ValueError("positive inputs, at least two outputs, nonnegative hidden size required")
        if self.growth_interval < 1 or self.prune_interval < 1 or self.prune_age < 1:
            raise ValueError("structural intervals and age must be positive")
        if not 0 <= self.drive_density <= 1:
            raise ValueError("drive_density must be in [0, 1]")
        if not self.trace_decays or any(not 0 <= d < 1 for d in self.trace_decays):
            raise ValueError("trace decays must be in [0, 1)")
        if self.learning_rate < 0 or self.hidden_learning_rate < 0:
            raise ValueError("learning rates must be nonnegative")
        if self.trace_rule not in ("ema", "peak"):
            raise ValueError("unknown trace rule")
        if self.hidden_rule not in ("oja", "competitive"):
            raise ValueError("unknown hidden rule")
        if self.growth_rule not in ("prediction", "representation"):
            raise ValueError("unknown growth rule")
        if self.growth_rule == "representation" and self.hidden_rule != "competitive":
            raise ValueError("representation growth requires the competitive hidden rule")
        if self.drive_trace_rule not in ("none", "ema", "peak") or not 0 <= self.drive_decay < 1:
            raise ValueError("invalid drive trace rule or decay")
        if self.vigilance < 0:
            raise ValueError("vigilance must be nonnegative")
        if self.growth_clock not in ("outcome", "experience"):
            raise ValueError("invalid growth clock")
        if not 0 <= self.evidence_decay <= 1 or self.synapse_growth_interval < 1:
            raise ValueError("invalid evidence decay or synapse growth interval")
        if not 0 <= self.normalization_rate <= 1 or self.variance_floor <= 0:
            raise ValueError("invalid homeostatic normalization settings")


@dataclass
class Neuron:
    id: int
    role: str
    activity: float = 0.0
    persistence: float = 0.0
    threshold: float = 0.0
    age: int = 0
    usefulness: float = 0.0
    surprise: float = 0.0
    plasticity: float = 1.0


@dataclass
class Synapse:
    source: int
    target: int
    kind: str
    strength: float
    decay: float = 0.0
    trace: float = 0.0
    age: int = 0
    usefulness: float = 0.0
    evidence: float = 0.0
    plasticity: float = 1.0
    pre_mean: float = 0.0
    pre_variance: float = 1.0


class Organism:
    """One continuous lifetime, including RNG and short-lived activity state."""

    def __init__(self, config: Config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.neurons: dict[int, Neuron] = {}
        self.synapses: list[Synapse] = []
        self.next_id = 0
        self.tick = 0
        self.observations = 0
        self.surprise = 0.5
        self.representation_surprise = 0.0
        self.current_representation_surprise = 0.0
        self.births = 0
        self.deaths = 0
        self.synapse_births = 0
        self.synapse_deaths = 0
        self.sensory = [self._neuron("sensory") for _ in range(config.inputs)]
        self.tonic = self._neuron("tonic")
        self.neurons[self.tonic].activity = 1.0
        self.outcome = [self._neuron("outcome") for _ in range(config.outputs)]
        self.hidden: list[int] = []
        for _ in range(config.initial_hidden):
            self.hidden.append(self._neuron("hidden"))
        for target in self.hidden:
            self._wire_hidden(target)
        for source in self.sensory + [self.tonic] + self.hidden:
            self._wire_prediction(source)
        self.births = self.synapse_births = 0  # construction is not learned birth

    def _neuron(self, role):
        nid = self.next_id
        self.next_id += 1
        persistence = self.rng.choice((0.0, 0.5, 0.9, 0.98)) if role == "hidden" else 0.0
        self.neurons[nid] = Neuron(nid, role, persistence=persistence)
        return nid

    def _wire_hidden(self, target, newborn=False):
        # Exploratory candidate connections; none encode a task or an answer.
        parent = next((nid for nid in self.hidden if nid != target), None)
        local_traces = {s.source: s.trace for s in self.synapses
                        if s.kind == "drive" and s.target == parent}
        local_statistics = {s.source: (s.pre_mean, s.pre_variance) for s in self.synapses
                            if s.kind == "drive" and s.target == parent}
        for source in self.sensory:
            weight = self.rng.random() if self.config.hidden_rule == "competitive" else self.rng.uniform(-1, 1)
            if newborn and self.config.seed_birth_from_activity:
                weight = local_traces.get(source, self.neurons[source].activity) if self.config.drive_trace_rule != "none" else self.neurons[source].activity
            trace = weight if newborn and self.config.seed_birth_from_activity and self.config.drive_trace_rule != "none" else 0.0
            synapse = Synapse(source, target, "drive", weight, trace=trace)
            if newborn and self.config.afferent_normalization and source in local_statistics:
                synapse.pre_mean, synapse.pre_variance = local_statistics[source]
            self.synapses.append(synapse)
            self.synapse_births += 1
        for source in self.hidden:
            if self.rng.random() < self.config.drive_density:
                self.synapses.append(Synapse(source, target, "drive", self.rng.uniform(-0.4, 0.4)))
                self.synapse_births += 1

    def _wire_prediction(self, source):
        # One decay per source, shared across outcomes: no initial class bias.
        decays = self.config.trace_decays if self.config.trace_bank else (self.rng.choice(self.config.trace_decays),)
        for decay in decays:
            for target in self.outcome:
                self.synapses.append(Synapse(source, target, "prediction", 0.0, decay))
                self.synapse_births += 1

    def experience(self, signal, target=None, gate=1.0, observer=None):
        """Receive sensory state, predict, THEN receive an optional outcome.

        gate is a scalar broadcast controlling learning amount, never direction.
        A missing outcome is an experience too: activity, traces, and local
        hidden plasticity continue. An outcome at this tick is never an input
        to the prediction being scored.
        """
        if len(signal) != self.config.inputs or any(not math.isfinite(float(x)) for x in signal):
            raise ValueError("signal must contain finite values, one per sensory neuron")
        if target is not None and (not isinstance(target, int) or not 0 <= target < self.config.outputs):
            raise ValueError("target outside outcome channels")
        if not math.isfinite(gate) or gate < 0:
            raise ValueError("gate must be finite and nonnegative")
        old = {nid: n.activity for nid, n in self.neurons.items()}
        for nid, value in zip(self.sensory, signal):
            self.neurons[nid].activity = float(value)
        self.neurons[self.tonic].activity = 1.0
        drive = {nid: 0.0 for nid in self.hidden}
        pre_values = {}
        for index, s in enumerate(self.synapses):
            if s.kind != "drive":
                continue
            recurrent = self.neurons[s.source].role == "hidden"
            pre = old[s.source] if recurrent else self.neurons[s.source].activity
            if recurrent and not self.config.recurrence:
                pre = 0.0
            if not recurrent and self.config.drive_trace_rule != "none":
                decay = self.config.drive_decay
                if self.config.drive_trace_rule == "peak":
                    residual = decay * s.trace
                    s.trace = pre if abs(pre) >= abs(residual) else residual
                else:
                    s.trace = decay * s.trace + (1 - decay) * pre
                pre = s.trace
            if not recurrent and self.config.afferent_normalization and self.config.learning:
                rate = self.config.normalization_rate
                old_mean = s.pre_mean
                s.pre_mean += rate * (pre - old_mean)
                s.pre_variance = max(0.0, (1 - rate) * s.pre_variance
                                     + rate * (pre - old_mean) * (pre - s.pre_mean))
            pre_values[index] = pre
            drive[s.target] += s.strength * pre
        for nid in self.hidden:
            n = self.neurons[nid]
            p = n.persistence if self.config.persistence else 0.0
            response = math.tanh(drive[nid] - n.threshold)
            n.activity = p * old[nid] + (1 - p) * response
            n.surprise = 0.99 * n.surprise + 0.01 * abs(response - old[nid])
        if self.config.hidden_rule == "competitive" and self.hidden:
            mismatch = {nid: 0.0 for nid in self.hidden}
            fan_in = {nid: 0 for nid in self.hidden}
            for index, s in enumerate(self.synapses):
                if s.kind == "drive" and self.neurons[s.source].role == "sensory":
                    distance = (pre_values[index] - s.strength) ** 2
                    if self.config.afferent_normalization:
                        distance /= max(self.config.variance_floor, s.pre_variance)
                    mismatch[s.target] += distance
                    fan_in[s.target] += 1
            for nid in self.hidden:
                mismatch[nid] = mismatch[nid] / fan_in[nid] if fan_in[nid] else float("inf")
            # Generic lateral competition; no target or task information.
            winner = min(self.hidden, key=lambda nid: (mismatch[nid], nid))
            novelty = mismatch[winner] if math.isfinite(mismatch[winner]) else 1.0
            self.current_representation_surprise = novelty
            self.representation_surprise = 0.95 * self.representation_surprise + 0.05 * novelty
            represented = self.config.vigilance == 0 or novelty <= self.config.vigilance
            for nid in self.hidden:
                n = self.neurons[nid]
                p = n.persistence if self.config.persistence else 0.0
                n.activity = p * old[nid] + (1 - p) * float(nid == winner and represented)
                n.surprise = mismatch[nid] if math.isfinite(mismatch[nid]) else 1.0
        logits = {nid: 0.0 for nid in self.outcome}
        for s in self.synapses:
            s.age += 1
            if s.kind == "prediction":
                pre = self.neurons[s.source].activity
                decay = s.decay if self.config.traces else 0.0
                if self.config.trace_rule == "peak":
                    residual = decay * s.trace
                    s.trace = pre if abs(pre) >= abs(residual) else residual
                else:
                    s.trace = decay * s.trace + (1 - decay) * pre
                logits[s.target] += s.strength * s.trace
            else:
                if self.config.drive_trace_rule == "none" or self.neurons[s.source].role == "hidden":
                    s.trace = self.neurons[s.source].activity
            contribution = abs(s.strength * s.trace)
            if (self.config.vigilance > 0 and self.config.hidden_rule == "competitive"
                    and s.kind == "drive" and self.neurons[s.source].role == "sensory"):
                # Matched absence is a meaningful dendritic constraint too.
                contribution = abs(self.neurons[s.target].activity) * max(0.0, 1 - abs(s.trace - s.strength))
            s.usefulness = 0.99 * s.usefulness + 0.01 * contribution
        offset = max(logits.values())
        exps = [math.exp(logits[nid] - offset) for nid in self.outcome]
        total = sum(exps)
        probabilities = [x / total for x in exps]
        for nid, prob in zip(self.outcome, probabilities):
            self.neurons[nid].activity = prob
        if observer is not None:
            # Copies only: diagnostics cannot mutate neural state or see the
            # new outcome. Nothing from the observer feeds the learning rule.
            observer({"probabilities": tuple(probabilities), "sensory": tuple(self.sensory),
                      "traces": tuple((s.source, s.target, s.kind, s.trace) for s in self.synapses)})
        if self.config.learning:
            for index, s in enumerate(self.synapses):
                if s.kind == "drive" and self.config.hidden_plasticity:
                    post = self.neurons[s.target].activity
                    pre = pre_values[index]
                    if self.config.hidden_rule == "competitive":
                        if self.neurons[s.source].role != "sensory":
                            continue
                        # Winning neuron's activity gates afferent adaptation.
                        change = self.config.hidden_learning_rate * gate * s.plasticity * post * (pre - s.strength)
                        if self.config.hidden_metaplasticity:
                            evidence = abs(pre) * post * max(0.0, 1 - self.neurons[s.target].surprise)
                            if self.config.match_evidence:
                                evidence = post * max(0.0, 1 - abs(pre - s.strength))
                            s.evidence = self.config.evidence_decay * s.evidence + evidence
                            s.plasticity = 1 / (1 + self.config.consolidation_rate * s.evidence)
                    else:
                        # Local Oja normalization; no outcome error.
                        change = self.config.hidden_learning_rate * gate * post * (pre - post * s.strength)
                    s.strength = max(-2.0, min(2.0, s.strength + change))
            if target is not None:
                self.observations += 1
                surprise = 1 - probabilities[target]
                self.surprise = 0.95 * self.surprise + 0.05 * surprise
                errors = {nid: float(i == target) - probabilities[i] for i, nid in enumerate(self.outcome)}
                for nid in self.outcome:
                    self.neurons[nid].surprise = abs(errors[nid])
                energy = {nid: 1.0 for nid in self.outcome}
                if self.config.normalized_updates:
                    for s in self.synapses:
                        if s.kind == "prediction":
                            energy[s.target] += s.trace * s.trace
                for s in self.synapses:
                    if s.kind != "prediction":
                        continue
                    error = errors[s.target]
                    # The only directional teaching signal is postsynaptic-local.
                    change = self.config.learning_rate * gate * s.plasticity * s.trace * error / energy[s.target]
                    s.strength += change
                    if self.config.metaplasticity:
                        s.evidence = self.config.evidence_decay * s.evidence + abs(s.trace) * (1 - abs(error))
                        s.plasticity = 1 / (1 + self.config.consolidation_rate * s.evidence)
        self.tick += 1
        for n in self.neurons.values():
            n.age += 1
            n.usefulness = 0.99 * n.usefulness + 0.01 * abs(n.activity)
        if self.config.learning:
            self._structure(target is not None)
        return probabilities

    def _structure(self, observed):
        cfg = self.config
        if cfg.regrow_predictions and observed and self.observations % cfg.synapse_growth_interval == 0:
            connected = {(s.source, s.target) for s in self.synapses if s.kind == "prediction"}
            candidates = []
            for source in self.sensory + [self.tonic] + self.hidden:
                for target in self.outcome:
                    coactivity = abs(self.neurons[source].activity) * self.neurons[target].surprise
                    if (source, target) not in connected and coactivity > cfg.synapse_growth_threshold:
                        candidates.append((coactivity, source, target))
            if candidates:
                _, source, target = max(candidates, key=lambda x: (x[0], -x[1], -x[2]))
                decay = self.rng.choice(cfg.trace_decays)
                self.synapses.append(Synapse(source, target, "prediction", 0.0, decay,
                                             trace=self.neurons[source].activity))
                self.synapse_births += 1
        clock = self.tick if cfg.growth_clock == "experience" else self.observations
        due = (cfg.growth_clock == "experience" or observed) and clock % cfg.growth_interval == 0
        if cfg.growth and due:
            # Cost increases smoothly: rate/cost regulation, no fixed size cap.
            pressure = cfg.growth_threshold + cfg.growth_cost * math.log1p(len(self.hidden))
            surprise = self.representation_surprise if cfg.growth_rule == "representation" else self.surprise
            unmet = not cfg.growth_requires_current_novelty or self.current_representation_surprise > cfg.vigilance
            if surprise > pressure and unmet:
                nid = self._neuron("hidden")
                self.hidden.append(nid)
                self._wire_hidden(nid, newborn=True)
                self._wire_prediction(nid)
                self.births += 1
        if cfg.pruning and self.tick % cfg.prune_interval == 0:
            kept = []
            for s in self.synapses:
                importance = s.usefulness
                if cfg.retention_protection:
                    importance += cfg.retention_weight * s.evidence / (1 + s.evidence)
                if s.age > cfg.prune_age and importance < cfg.prune_threshold:
                    self.synapse_deaths += 1
                else:
                    kept.append(s)
            self.synapses = kept
            connected = {s.source for s in kept if s.kind == "prediction"}
            dead = {nid for nid in self.hidden if self.neurons[nid].age > 2 * cfg.prune_age
                    and nid not in connected and (not cfg.retain_active_representations
                                                  or self.neurons[nid].usefulness < cfg.prune_threshold)}
            if dead:
                self.hidden = [nid for nid in self.hidden if nid not in dead]
                for nid in dead:
                    del self.neurons[nid]
                before = len(self.synapses)
                self.synapses = [s for s in self.synapses if s.source not in dead and s.target not in dead]
                self.synapse_deaths += before - len(self.synapses)
                self.deaths += len(dead)

    def summary(self):
        predictive = [s for s in self.synapses if s.kind == "prediction"]
        return dict(tick=self.tick, observations=self.observations,
                    neurons=len(self.neurons), hidden=len(self.hidden),
                    synapses=len(self.synapses), births=self.births, deaths=self.deaths,
                    synapse_births=self.synapse_births, synapse_deaths=self.synapse_deaths,
                    surprise=self.surprise,
                    representation_surprise=self.representation_surprise,
                    current_representation_surprise=self.current_representation_surprise,
                    mean_activity=sum(abs(n.activity) for n in self.neurons.values()) / len(self.neurons),
                    mean_plasticity=sum(s.plasticity for s in predictive) / max(1, len(predictive)))

    def state(self):
        return dict(format=1, config=asdict(self.config), rng=self.rng.getstate(),
                    neurons=[asdict(n) for n in self.neurons.values()],
                    synapses=[asdict(s) for s in self.synapses],
                    **{key: (getattr(self, key)[:] if isinstance(getattr(self, key), list) else getattr(self, key)) for key in
                       ("next_id", "tick", "observations", "surprise", "births", "deaths",
                        "synapse_births", "synapse_deaths", "sensory", "tonic", "outcome", "hidden",
                        "representation_surprise", "current_representation_surprise")})

    @classmethod
    def from_state(cls, state):
        if state.get("format") != 1:
            raise ValueError("unsupported checkpoint format")
        obj = cls.__new__(cls)
        obj.config = Config(**state["config"])
        obj.rng = random.Random()
        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, list) else x
        obj.rng.setstate(tuples(state["rng"]))
        obj.neurons = {n["id"]: Neuron(**n) for n in state["neurons"]}
        obj.synapses = [Synapse(**s) for s in state["synapses"]]
        for key in ("next_id", "tick", "observations", "surprise", "births", "deaths",
                    "synapse_births", "synapse_deaths", "sensory", "tonic", "outcome", "hidden"):
            value = state[key]
            setattr(obj, key, value[:] if isinstance(value, list) else value)
        obj.representation_surprise = state.get("representation_surprise", 0.0)
        obj.current_representation_surprise = state.get("current_representation_surprise", 0.0)
        return obj

    def save(self, path):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix="." + path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.state(), stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        return cls.from_state(json.loads(Path(path).read_text()))

"""Sparse, dynamic directed synapses."""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Synapse:
    source: str
    destination: str
    strength: float = 0.1
    transmission_delay: int = 0
    plasticity: float = 0.5
    stability: float = 0.0
    eligibility_trace: float = 0.0
    # Fast actor eligibility for causal motor perturbations.  This is kept
    # separate from the correlation trace above and is consumed exactly once
    # when the next outcome arrives.
    actor_eligibility_trace: float = 0.0
    # Reward-modulated node-perturbation eligibility for hidden incoming edges.
    node_eligibility_trace: float = 0.0
    activity_history: List[float] = field(default_factory=list)
    age: int = 0
    utility: float = 0.0
    dormant: bool = False
    synapse_id: Optional[str] = None

    @property
    def id(self) -> str:
        if self.synapse_id is None:
            return "unassigned:%s>%s" % (self.source, self.destination)
        return self.synapse_id

    def record(self, coactivity: float, trace_decay: float) -> None:
        self.eligibility_trace = trace_decay * self.eligibility_trace + coactivity
        self.activity_history.append(coactivity)
        if len(self.activity_history) > 32:
            del self.activity_history[:-32]
        self.utility = 0.99 * self.utility + 0.01 * abs(coactivity * self.strength)
        self.age += 1


class SparseDirectedGraph:
    """A deterministic sparse directed graph keyed by stable cell IDs."""

    def __init__(self) -> None:
        self.synapses: Dict[Tuple[str, str], Synapse] = {}
        self.outgoing: Dict[str, List[str]] = {}
        self.incoming: Dict[str, List[str]] = {}
        self.next_synapse_index = 0
        self.retired_ids = set()

    def add(self, synapse: Synapse) -> None:
        key = (synapse.source, synapse.destination)
        if key in self.synapses:
            raise ValueError("synapse already exists: %s>%s" % key)
        if synapse.synapse_id is None or synapse.synapse_id in self.retired_ids or any(s.synapse_id == synapse.synapse_id for s in self.synapses.values()):
            while "synapse-%08d" % self.next_synapse_index in self.retired_ids or any(s.synapse_id == "synapse-%08d" % self.next_synapse_index for s in self.synapses.values()):
                self.next_synapse_index += 1
            synapse.synapse_id = "synapse-%08d" % self.next_synapse_index
            self.next_synapse_index += 1
        elif synapse.synapse_id.startswith("synapse-"):
            try:
                self.next_synapse_index = max(self.next_synapse_index, int(synapse.synapse_id.split("-", 1)[1]) + 1)
            except (TypeError, ValueError):
                pass
        self.synapses[key] = synapse
        self.outgoing.setdefault(synapse.source, []).append(synapse.destination)
        self.incoming.setdefault(synapse.destination, []).append(synapse.source)
        self.outgoing[synapse.source].sort()
        self.incoming[synapse.destination].sort()

    def remove(self, source: str, destination: str) -> Synapse:
        key = (source, destination)
        synapse = self.synapses.pop(key)
        if synapse.synapse_id is not None:
            self.retired_ids.add(synapse.synapse_id)
        self.outgoing[source].remove(destination)
        self.incoming[destination].remove(source)
        return synapse

    def get(self, source: str, destination: str) -> Optional[Synapse]:
        return self.synapses.get((source, destination))

    def has(self, source: str, destination: str) -> bool:
        return (source, destination) in self.synapses

    def iter_synapses(self) -> Iterable[Synapse]:
        for key in sorted(self.synapses):
            yield self.synapses[key]

    def validate(self, cell_ids: Iterable[str]) -> None:
        ids = set(cell_ids)
        seen_ids = set()
        for (source, destination), synapse in self.synapses.items():
            if source == destination:
                raise AssertionError("self-loop is not permitted")
            if source not in ids or destination not in ids:
                raise AssertionError("dangling synapse: %s" % synapse.id)
            if destination not in self.outgoing.get(source, []):
                raise AssertionError("outgoing index mismatch")
            if source not in self.incoming.get(destination, []):
                raise AssertionError("incoming index mismatch")
            if not synapse.synapse_id or synapse.synapse_id in seen_ids or synapse.synapse_id in self.retired_ids:
                raise AssertionError("synapse IDs must be unique and non-retired")
            seen_ids.add(synapse.synapse_id)
            for name, value, lower, upper in (
                ("strength", synapse.strength, -1.0, 1.0),
                ("plasticity", synapse.plasticity, 0.0, 1.0),
                ("stability", synapse.stability, 0.0, 1.0),
                ("utility", synapse.utility, 0.0, 1.0),
            ):
                if not math.isfinite(value) or not lower <= value <= upper:
                    raise AssertionError("synapse %s out of bounds: %s" % (name, synapse.id))
            if not math.isfinite(synapse.eligibility_trace) or abs(synapse.eligibility_trace) > 100.0:
                raise AssertionError("synapse eligibility out of bounds: %s" % synapse.id)
            if not math.isfinite(synapse.actor_eligibility_trace) or abs(synapse.actor_eligibility_trace) > 100.0:
                raise AssertionError("synapse actor eligibility out of bounds: %s" % synapse.id)
            if not math.isfinite(synapse.node_eligibility_trace) or abs(synapse.node_eligibility_trace) > 100.0:
                raise AssertionError("synapse node eligibility out of bounds: %s" % synapse.id)
            if synapse.age < 0 or synapse.transmission_delay < 0:
                raise AssertionError("synapse age/delay must be nonnegative: %s" % synapse.id)
            if any((not math.isfinite(value) or abs(value) > 1.0) for value in synapse.activity_history):
                raise AssertionError("synapse activity history out of bounds: %s" % synapse.id)

    def to_dict(self) -> Dict[str, object]:
        return {synapse.id: synapse_to_dict(synapse) for synapse in self.iter_synapses()}


def synapse_to_dict(synapse: Synapse) -> Dict[str, object]:
    return {
        "source": synapse.source,
        "destination": synapse.destination,
        "strength": synapse.strength,
        "transmission_delay": synapse.transmission_delay,
        "plasticity": synapse.plasticity,
        "stability": synapse.stability,
        "eligibility_trace": synapse.eligibility_trace,
        "actor_eligibility_trace": synapse.actor_eligibility_trace,
        "node_eligibility_trace": synapse.node_eligibility_trace,
        "activity_history": list(synapse.activity_history),
        "age": synapse.age,
        "utility": synapse.utility,
        "dormant": synapse.dormant,
        "synapse_id": synapse.synapse_id,
    }

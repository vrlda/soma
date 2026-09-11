"""Bounded local predictive circuits over ordered discrete events.

Each circuit represents one observed suffix and stores only local successor
evidence.  Circuits grow from experience, back off when evidence is sparse,
and are reclaimed deterministically under a hard budget.  The mechanism is
domain-neutral: symbols are declared by the caller and carry no text meaning.
"""

import math


class SequenceCircuitMemory(object):
    """Online variable-order prediction with bounded structural state."""

    VERSION = 1

    def __init__(self, symbols, max_order=16, max_circuits=131072,
                 min_support=2, prior=0.5):
        self.symbols = tuple(symbols)
        if len(self.symbols) < 2 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("sequence memory needs at least two unique symbols")
        self.symbol_index = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.max_order = int(max_order)
        self.max_circuits = int(max_circuits)
        self.min_support = int(min_support)
        self.prior = float(prior)
        if self.max_order < 0:
            raise ValueError("maximum order must be nonnegative")
        if self.max_circuits < self.max_order + 1:
            raise ValueError("circuit budget is too small for maximum order")
        if self.min_support < 1:
            raise ValueError("minimum support must be positive")
        if not math.isfinite(self.prior) or self.prior <= 0.0:
            raise ValueError("prior must be positive and finite")
        self.history = []
        self.circuits = {(): self._new_circuit(0)}
        self.events_seen = 0
        self.circuits_created = 1
        self.circuits_reclaimed = 0

    def _new_circuit(self, step):
        return {
            "counts": [0] * len(self.symbols),
            "last_used": int(step),
            "reuse": 0,
        }

    def reset_history(self):
        """Start a new stream without erasing acquired circuits."""
        self.history = []

    def _contexts(self):
        limit = min(self.max_order, len(self.history))
        return [tuple(self.history[-order:]) if order else ()
                for order in range(limit + 1)]

    def _reclaim(self):
        """Reclaim weak, old, specific circuits; root is permanent."""
        if len(self.circuits) < self.max_circuits:
            return
        protected = set(self._contexts())
        candidates = []
        for context, circuit in self.circuits.items():
            if not context or context in protected:
                continue
            support = sum(circuit["counts"])
            candidates.append((support, circuit["reuse"], circuit["last_used"],
                               -len(context), context))
        if not candidates:
            return
        candidates.sort()
        batch = max(1, min(1024, self.max_circuits // 100))
        for item in candidates[:batch]:
            del self.circuits[item[-1]]
            self.circuits_reclaimed += 1

    def distribution(self):
        """Predict from the longest sufficiently supported suffix."""
        chosen = self.circuits[()]
        chosen_order = 0
        for context in reversed(self._contexts()):
            circuit = self.circuits.get(context)
            if circuit is None or sum(circuit["counts"]) < self.min_support:
                continue
            chosen = circuit
            chosen_order = len(context)
            break
        denominator = sum(chosen["counts"]) + self.prior * len(self.symbols)
        probabilities = {
            symbol: (chosen["counts"][index] + self.prior) / denominator
            for index, symbol in enumerate(self.symbols)
        }
        return probabilities, chosen_order

    def probability(self, symbol):
        if symbol not in self.symbol_index:
            raise ValueError("unknown sequence symbol")
        return self.distribution()[0][symbol]

    def observe(self, symbol, learn=True):
        """Observe one event and locally update its predecessor circuits."""
        if symbol not in self.symbol_index:
            raise ValueError("unknown sequence symbol")
        if not isinstance(learn, bool):
            raise ValueError("learn flag must be boolean")
        if learn:
            symbol_index = self.symbol_index[symbol]
            for context in self._contexts():
                circuit = self.circuits.get(context)
                if circuit is None:
                    self._reclaim()
                    if len(self.circuits) >= self.max_circuits:
                        continue
                    circuit = self._new_circuit(self.events_seen)
                    self.circuits[context] = circuit
                    self.circuits_created += 1
                circuit["counts"][symbol_index] += 1
                circuit["last_used"] = self.events_seen
                circuit["reuse"] += 1
        self.history.append(symbol)
        if len(self.history) > self.max_order:
            del self.history[:-self.max_order]
        self.events_seen += 1

    def validate(self):
        if len(self.circuits) > self.max_circuits or () not in self.circuits:
            raise AssertionError("sequence circuit budget/root invariant failed")
        if len(self.history) > self.max_order:
            raise AssertionError("sequence history exceeds maximum order")
        for symbol in self.history:
            if symbol not in self.symbol_index:
                raise AssertionError("sequence history contains unknown symbol")
        for context, circuit in self.circuits.items():
            if len(context) > self.max_order or any(symbol not in self.symbol_index for symbol in context):
                raise AssertionError("sequence circuit context is invalid")
            counts = circuit.get("counts")
            if not isinstance(counts, list) or len(counts) != len(self.symbols):
                raise AssertionError("sequence circuit counts have invalid shape")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
                raise AssertionError("sequence circuit counts must be nonnegative integers")
            for name in ("last_used", "reuse"):
                value = circuit.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AssertionError("sequence circuit metadata is invalid")
        return True

    def state_dict(self):
        self.validate()
        ordered = sorted(self.circuits.items(), key=lambda item: (len(item[0]), repr(item[0])))
        return {
            "version": self.VERSION,
            "symbols": list(self.symbols),
            "max_order": self.max_order,
            "max_circuits": self.max_circuits,
            "min_support": self.min_support,
            "prior": self.prior,
            "history": list(self.history),
            "events_seen": self.events_seen,
            "circuits_created": self.circuits_created,
            "circuits_reclaimed": self.circuits_reclaimed,
            "circuits": [
                {
                    "context": list(context),
                    "counts": list(circuit["counts"]),
                    "last_used": circuit["last_used"],
                    "reuse": circuit["reuse"],
                }
                for context, circuit in ordered
            ],
        }

    @classmethod
    def from_state_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported sequence memory state")
        memory = cls(
            payload["symbols"], payload["max_order"], payload["max_circuits"],
            payload["min_support"], payload["prior"])
        memory.history = list(payload["history"])
        memory.events_seen = int(payload["events_seen"])
        memory.circuits_created = int(payload["circuits_created"])
        memory.circuits_reclaimed = int(payload["circuits_reclaimed"])
        memory.circuits = {}
        for item in payload["circuits"]:
            context = tuple(item["context"])
            if context in memory.circuits:
                raise ValueError("duplicate sequence circuit context")
            memory.circuits[context] = {
                "counts": [int(value) for value in item["counts"]],
                "last_used": int(item["last_used"]),
                "reuse": int(item["reuse"]),
            }
        try:
            memory.validate()
        except AssertionError as error:
            raise ValueError("invalid sequence memory state: %s" % error)
        return memory

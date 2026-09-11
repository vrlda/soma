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
        self.chunks = {}
        self.chunks_promoted = 0

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
            if self.max_order <= 0:
                del self.history[:]
            else:
                del self.history[:-self.max_order]
        self.events_seen += 1

    def promote_chunks(self, min_order=8, min_reuse=10, min_concentration=0.9,
                       max_chunks=4096, max_merged_reuse=None):
        """Merge extension families into promoted chunk circuits.

        A circuit with a long, often-reused, concentrated context becomes a
        chunk: descendant extension circuits fold their counts into it and
        are removed. Prediction code is untouched; backoff simply finds
        richer aggregated counts. Returns the number of chunks promoted.
        """
        if min_order < 1 or min_reuse < 1 or max_chunks < 1:
            raise ValueError("chunk promotion bounds must be positive")
        if not 0.0 < float(min_concentration) <= 1.0:
            raise ValueError("chunk concentration must be in (0, 1]")
        candidates = []
        for context, circuit in self.circuits.items():
            if len(context) < min_order or context in self.chunks:
                continue
            total = sum(circuit["counts"])
            if circuit["reuse"] < min_reuse or total <= 0:
                continue
            concentration = max(circuit["counts"]) / total
            if concentration < float(min_concentration):
                continue
            candidates.append((circuit["reuse"], concentration, context))
        candidates.sort(key=lambda item: (-item[0], -item[1], repr(item[2])))
        promoted = 0
        for _, _, context in candidates:
            if len(self.chunks) >= max_chunks or context not in self.circuits:
                continue
            chunk = self.circuits[context]
            merged = 0
            for other in [c for c in self.circuits if len(c) > len(context) and c[-len(context):] == context]:
                if other in self.chunks:
                    continue
                if max_merged_reuse is not None and self.circuits[other]["reuse"] > max_merged_reuse:
                    continue
                descendant = self.circuits.pop(other)
                chunk["counts"] = [a + b for a, b in zip(chunk["counts"], descendant["counts"])]
                chunk["reuse"] += descendant["reuse"]
                chunk["last_used"] = max(chunk["last_used"], descendant["last_used"])
                merged += 1
            total = sum(chunk["counts"])
            self.chunks[context] = {
                "merged": merged,
                "reuse": chunk["reuse"],
                "concentration": (max(chunk["counts"]) / total) if total > 0 else 0.0,
            }
            self.chunks_promoted += 1
            promoted += 1
        return promoted

    def validate(self):
        if len(self.circuits) > self.max_circuits or () not in self.circuits:
            raise AssertionError("sequence circuit budget/root invariant failed")
        if len(self.history) > self.max_order:
            raise AssertionError("sequence history exceeds maximum order")
        for symbol in self.history:
            if symbol not in self.symbol_index:
                raise AssertionError("sequence history contains unknown symbol")
        if not isinstance(self.chunks, dict) or not isinstance(self.chunks_promoted, int):
            raise AssertionError("sequence chunk state is invalid")
        for context in self.chunks:
            if context not in self.circuits:
                raise AssertionError("sequence chunk has no circuit")
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
            "chunks_promoted": self.chunks_promoted,
            "chunks": [
                {"context": list(context),
                 "merged": chunk["merged"],
                 "reuse": chunk["reuse"],
                 "concentration": chunk["concentration"]}
                for context, chunk in sorted(self.chunks.items(), key=lambda item: repr(item[0]))
            ],
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
        memory.chunks_promoted = int(payload.get("chunks_promoted", 0))
        memory.chunks = {}
        for item in payload.get("chunks", []):
            context = tuple(item["context"])
            if context in memory.chunks:
                raise ValueError("duplicate sequence chunk context")
            memory.chunks[context] = {
                "merged": int(item["merged"]),
                "reuse": int(item["reuse"]),
                "concentration": float(item["concentration"]),
            }
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

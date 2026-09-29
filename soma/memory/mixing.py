"""Growing byte-context circuits with plastic evidence arbitration.

``SequenceCircuitMemory`` predicts from the single longest supported
suffix.  This module keeps the same developmental ideas (circuits are
created from experience, live under a hard structural budget, and weak
circuits are reclaimed) but changes two things:

- Contexts are byte-aligned: a circuit is keyed by ``order`` preceding
  whole bytes plus the bits already seen of the current byte.  Orders are
  declared by the caller (default 0..6 bytes).
- Every active circuit contributes evidence.  A small arbitration unit
  combines the circuits' log-odds with plastic weights, updated by a local
  delta rule, and the weight set is chosen by how many of the circuits
  currently exist (a structural gate).  Nothing is hard-coded about which
  order to trust.

Circuit state is integer and deterministic: bit counts with a cap, where
the opposing count is halved on each update so recent evidence dominates
(nonstationary adaptation), plus a visit counter and a last-used step for
reclamation.  Frozen scoring (``learn=False``) neither grows circuits nor
changes weights.

This is the Python reference; ``engine/soma-engine/src/mixer.rs`` is the
production port, checked by ``engine/differential_mixer.py``.
"""

import math

ORDER_SHIFT = 56
PROBABILITY_FLOOR = 1.0 / 4096.0
STRETCH_LIMIT = 8.0


def squash(value):
    if value > 40.0:
        return 1.0
    if value < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def stretch(probability):
    return math.log(probability / (1.0 - probability))


def circuit_key(order, history, partial):
    """Exact 64-bit key: order, the last ``order`` bytes, the partial byte."""
    context = 0
    for byte in history[len(history) - order:] if order else b"":
        context = (context << 8) | byte
    return (order << ORDER_SHIFT) | (context << 8) | partial


class CircuitMixingMemory(object):
    """Online bit prediction over growing byte-context circuits."""

    VERSION = 1

    def __init__(self, orders=(0, 1, 2, 3, 4, 5, 6), max_circuits=1 << 22,
                 learning_rate=0.015, count_limit=60, initial_weight=0.3,
                 reclaim_fraction=0.125, arbitration=True):
        self.orders = tuple(int(order) for order in orders)
        if not self.orders or len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must be unique and nonempty")
        if list(self.orders) != sorted(self.orders) or self.orders[0] < 0 or self.orders[-1] > 6:
            raise ValueError("orders must be ascending within 0..6")
        self.max_circuits = int(max_circuits)
        if self.max_circuits < 256 * len(self.orders):
            raise ValueError("circuit budget is too small")
        self.learning_rate = float(learning_rate)
        self.count_limit = int(count_limit)
        self.initial_weight = float(initial_weight)
        self.reclaim_fraction = float(reclaim_fraction)
        if not 0.0 < self.reclaim_fraction < 1.0:
            raise ValueError("reclaim fraction must be in (0, 1)")
        self.arbitration = bool(arbitration)
        inputs = len(self.orders) + 1
        self.weights = [[self.initial_weight] * inputs for _ in range(len(self.orders) + 1)]
        self.circuits = {}
        self.history = bytearray()
        self.partial = 1
        self.events_seen = 0
        self.circuits_created = 0
        self.circuits_reclaimed = 0
        self._pending = None

    def reset_history(self):
        """Start a new stream without erasing circuits or weights."""
        self.history = bytearray()
        self.partial = 1
        self._pending = None

    def _active(self):
        keys = [circuit_key(order, self.history, self.partial)
                if order <= len(self.history) else None
                for order in self.orders]
        inputs = []
        present = 0
        for key in keys:
            circuit = self.circuits.get(key) if key is not None else None
            if circuit is None:
                inputs.append(0.0)
                continue
            present += 1
            n0, n1 = circuit[0], circuit[1]
            probability = (n1 + 0.4) / (n0 + n1 + 0.8)
            inputs.append(max(-STRETCH_LIMIT, min(STRETCH_LIMIT, stretch(probability))))
        inputs.append(1.0)
        return keys, inputs, present

    def predict(self):
        """Probability that the next bit is 1."""
        keys, inputs, present = self._active()
        weights = self.weights[present]
        if self.arbitration:
            total = 0.0
            for weight, value in zip(weights, inputs):
                total += weight * value
        else:
            total = 0.0
            for value in reversed(inputs[:-1]):
                if value != 0.0:
                    total = value
                    break
        probability = squash(total)
        probability = min(1.0 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, probability))
        self._pending = (keys, inputs, present, probability)
        return probability

    def _reclaim(self):
        """Drop the weakest circuits: lowest visits, then oldest, then key."""
        batch = max(1, int(self.max_circuits * self.reclaim_fraction))
        ranked = sorted(self.circuits.items(), key=lambda item: (item[1][2], item[1][3], item[0]))
        for key, _ in ranked[:batch]:
            del self.circuits[key]
        self.circuits_reclaimed += min(batch, len(ranked))

    def observe(self, bit, learn=True):
        """Consume one bit; learn updates circuits, growth, and weights."""
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        if self._pending is None:
            self.predict()
        keys, inputs, present, probability = self._pending
        if learn:
            if self.arbitration:
                error = bit - probability
                weights = self.weights[present]
                for index, value in enumerate(inputs):
                    weights[index] += self.learning_rate * error * value
            for key in keys:
                if key is None:
                    continue
                circuit = self.circuits.get(key)
                if circuit is None:
                    if len(self.circuits) >= self.max_circuits:
                        self._reclaim()
                    circuit = [0, 0, 0, 0]
                    self.circuits[key] = circuit
                    self.circuits_created += 1
                other = 1 - bit
                if circuit[bit] < self.count_limit:
                    circuit[bit] += 1
                if circuit[other] > 2:
                    circuit[other] = (circuit[other] + 1) // 2
                circuit[2] += 1
                circuit[3] = self.events_seen
        self.partial = (self.partial << 1) | bit
        if self.partial >= 256:
            self.history.append(self.partial & 0xFF)
            if len(self.history) > self.orders[-1]:
                del self.history[:len(self.history) - self.orders[-1]]
            self.partial = 1
        self.events_seen += 1
        self._pending = None

    def observe_bytes(self, data, learn=True):
        for byte in bytes(data):
            for shift in range(7, -1, -1):
                self.predict()
                self.observe((byte >> shift) & 1, learn=learn)

    def score_bytes(self, data, learn=False):
        """Mean next-bit cross-entropy (bits/bit) over ``data``."""
        total = 0.0
        count = 0
        for byte in bytes(data):
            for shift in range(7, -1, -1):
                bit = (byte >> shift) & 1
                probability = self.predict()
                total -= math.log(probability if bit else 1.0 - probability, 2)
                count += 1
                self.observe(bit, learn=learn)
        return total / max(1, count)

    def summary(self):
        return {
            "circuits": len(self.circuits),
            "circuits_created": self.circuits_created,
            "circuits_reclaimed": self.circuits_reclaimed,
            "events_seen": self.events_seen,
        }

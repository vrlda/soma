"""Growing byte-context circuits with plastic evidence arbitration.

``SequenceCircuitMemory`` predicts from the single longest supported
suffix.  This module keeps the same developmental ideas (circuits are
created from experience, live under a hard structural budget, and weak
circuits are reclaimed) and adds the mechanisms that make them useful at
book scale:

- Byte-aligned contexts.  A circuit is keyed by ``order`` preceding whole
  bytes plus the bits already seen of the current byte.  Contexts up to six
  bytes are stored exactly; longer ones as a 48-bit FNV-1a digest.
- Evidence-gated growth.  A circuit for a longer context is only created
  once the next shorter context has been visited ``growth_threshold``
  times, so capacity goes where experience says structure exists.
- Reclamation under a hard budget: weakest (fewest visits, then oldest)
  circuits are removed in batches.
- Plastic calibration (optional, off by default).  Each order learns what
  a count state (n0, n1) actually predicts; with non-forgetting counts its
  table is too sparse to help (docs/adr/0007).
- Plastic arbitration.  Every active circuit contributes log-odds evidence;
  a unit with delta-rule weights combines them.  The weight set is gated by
  how many circuits exist and by the partial byte.
- A final correction stage keyed by the previous byte and partial byte
  (optional, off by default: redundant with partial-byte gating, ADR 0007).

Frozen scoring (``learn=False``) neither grows circuits nor changes any
plastic table.  ``engine/soma-engine/src/mixer.rs`` is the production port;
``engine/differential_mixer.py`` checks exact parity.
"""

import math

ORDER_SHIFT = 56
EXACT_ORDER = 6
MAX_ORDER = 32
HASH_MASK = (1 << 48) - 1
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
U64 = (1 << 64) - 1
U32_MAX = (1 << 32) - 1
PROBABILITY_FLOOR = 1.0 / 4096.0
STRETCH_LIMIT = 8.0
CORRECTION_BUCKETS = 33


def squash(value):
    if value > 40.0:
        return 1.0
    if value < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def stretch(probability):
    return math.log(probability / (1.0 - probability))


def _clamp(value, low, high):
    return max(low, min(high, value))


def circuit_key(order, history, partial):
    """64-bit key: order in the top byte, context, partial byte in the low byte."""
    tail = history[len(history) - order:] if order else b""
    if order <= EXACT_ORDER:
        context = 0
        for byte in tail:
            context = (context << 8) | byte
    else:
        digest = FNV_OFFSET
        for byte in tail:
            digest = ((digest ^ byte) * FNV_PRIME) & U64
        context = digest & HASH_MASK
    return (order << ORDER_SHIFT) | (context << 8) | partial


def _bit_position(partial):
    return partial.bit_length() - 1


class CircuitMixingMemory(object):
    """Online bit prediction over growing byte-context circuits."""

    VERSION = 1

    def __init__(self, orders=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12), max_circuits=1 << 22,
                 learning_rate=0.002, count_limit=1023, initial_weight=0.3,
                 reclaim_fraction=0.125, arbitration=True, halve_above=1000000,
                 calibration=False, calibration_limit=255, gate_bit_position=False,
                 gate_partial=True, correction=False, correction_rate=0.02,
                 growth_threshold=8):
        self.orders = tuple(int(order) for order in orders)
        if not self.orders or list(self.orders) != sorted(set(self.orders)):
            raise ValueError("orders must be unique, ascending, and nonempty")
        if self.orders[0] < 0 or self.orders[-1] > MAX_ORDER:
            raise ValueError("orders must be within 0..%d" % MAX_ORDER)
        self.max_circuits = int(max_circuits)
        if self.max_circuits < 256 * len(self.orders):
            raise ValueError("circuit budget is too small")
        self.reclaim_fraction = float(reclaim_fraction)
        if not 0.0 < self.reclaim_fraction < 1.0:
            raise ValueError("reclaim fraction must be in (0, 1)")
        self.learning_rate = float(learning_rate)
        self.count_limit = int(count_limit)
        self.initial_weight = float(initial_weight)
        self.arbitration = bool(arbitration)
        self.halve_above = int(halve_above)
        self.use_calibration = bool(calibration)
        self.calibration_limit = int(calibration_limit)
        self.gate_bit_position = bool(gate_bit_position)
        self.gate_partial = bool(gate_partial)
        self.use_correction = bool(correction)
        self.correction_rate = float(correction_rate)
        self.growth_threshold = int(growth_threshold)
        levels = len(self.orders) + 1
        per_level = 256 if self.gate_partial else (8 if self.gate_bit_position else 1)
        self.weights = [[self.initial_weight] * levels for _ in range(levels * per_level)]
        # Plastic tables start at their analytic values; entries are created
        # lazily here (the Rust port preallocates the same values).
        self.calibration = [{} for _ in self.orders]
        self.correction = {}
        self.circuits = {}
        self.history = bytearray()
        self.partial = 1
        self.events_seen = 0
        self.circuits_created = 0
        self.circuits_reclaimed = 0
        self._pending = None

    def reset_history(self):
        """Start a new stream without erasing circuits or plastic tables."""
        self.history = bytearray()
        self.partial = 1
        self._pending = None

    def _calibration_entry(self, index, state, create=False):
        table = self.calibration[index]
        entry = table.get(state)
        if entry is None:
            n0, n1 = state
            entry = [(n1 + 0.4) / (n0 + n1 + 0.8), 0]
            if create:
                table[state] = entry
        return entry

    def _correction_row(self, slot, create=False):
        row = self.correction.get(slot)
        if row is None:
            row = [squash((j - 16.0) * 0.5) for j in range(CORRECTION_BUCKETS)]
            if create:
                self.correction[slot] = row
        return row

    def predict(self):
        """Probability that the next bit is 1."""
        keys, visits, states, inputs = [], [], [], []
        present = 0
        for index, order in enumerate(self.orders):
            key = circuit_key(order, self.history, self.partial) if order <= len(self.history) else None
            keys.append(key)
            circuit = self.circuits.get(key) if key is not None else None
            if circuit is None:
                visits.append(0)
                states.append(None)
                inputs.append(0.0)
                continue
            present += 1
            visits.append(circuit[2])
            n0, n1 = circuit[0], circuit[1]
            if self.use_calibration:
                states.append((n0, n1))
                probability = _clamp(self._calibration_entry(index, (n0, n1))[0],
                                     PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
            else:
                states.append(None)
                probability = (n1 + 0.4) / (n0 + n1 + 0.8)
            inputs.append(_clamp(stretch(probability), -STRETCH_LIMIT, STRETCH_LIMIT))
        inputs.append(1.0)
        gate = 0
        total = 0.0
        if self.arbitration:
            if self.gate_partial:
                gate = present * 256 + self.partial
            elif self.gate_bit_position:
                gate = present * 8 + _bit_position(self.partial)
            else:
                gate = present
            for weight, value in zip(self.weights[gate], inputs):
                total += weight * value
        else:
            for value in reversed(inputs[:-1]):
                if value != 0.0:
                    total = value
                    break
        probability = _clamp(squash(total), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        mixed = probability
        correction = None
        if self.use_correction:
            previous = self.history[-1] if self.history else 0
            evidence = _clamp(stretch(probability), -STRETCH_LIMIT, STRETCH_LIMIT)
            position = (evidence + 8.0) * 2.0
            low = min(int(math.floor(position)), CORRECTION_BUCKETS - 2)
            fraction = position - low
            slot = (previous << 8) | self.partial
            row = self._correction_row(slot)
            corrected = row[low] * (1.0 - fraction) + row[low + 1] * fraction
            correction = (slot, low, fraction)
            probability = (probability + 3.0 * corrected) / 4.0
            probability = _clamp(probability, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        self._pending = (keys, visits, states, inputs, gate, mixed, correction, probability)
        return probability

    def _reclaim(self):
        """Drop the weakest circuits: fewest visits, then oldest, then key."""
        batch = max(1, int(self.max_circuits * self.reclaim_fraction))
        ranked = sorted((circuit[2], circuit[3], key) for key, circuit in self.circuits.items())
        taken = ranked[:batch]
        for _, _, key in taken:
            del self.circuits[key]
        self.circuits_reclaimed += len(taken)

    def observe(self, bit, learn=True):
        """Consume one bit; learn updates circuits, growth, and plastic tables."""
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        if self._pending is None:
            self.predict()
        keys, visits, states, inputs, gate, mixed, correction, _ = self._pending
        if learn:
            target = float(bit)
            if self.arbitration:
                error = target - mixed
                weights = self.weights[gate]
                for index, value in enumerate(inputs):
                    weights[index] += self.learning_rate * error * value
            if self.use_calibration:
                for index, state in enumerate(states):
                    if state is None:
                        continue
                    entry = self._calibration_entry(index, state, create=True)
                    entry[0] += (target - entry[0]) / (entry[1] + 1.5)
                    if entry[1] < self.calibration_limit:
                        entry[1] += 1
            if correction is not None:
                slot, low, fraction = correction
                row = self._correction_row(slot, create=True)
                row[low] += (target - row[low]) * self.correction_rate * (1.0 - fraction)
                row[low + 1] += (target - row[low + 1]) * self.correction_rate * fraction
            for index, key in enumerate(keys):
                if key is None:
                    continue
                circuit = self.circuits.get(key)
                if circuit is None:
                    if (self.growth_threshold > 0 and index > 0
                            and visits[index - 1] < self.growth_threshold):
                        continue
                    if len(self.circuits) >= self.max_circuits:
                        self._reclaim()
                    circuit = [0, 0, 0, 0]
                    self.circuits[key] = circuit
                    self.circuits_created += 1
                other = 1 - bit
                if circuit[bit] < self.count_limit:
                    circuit[bit] += 1
                if circuit[other] > self.halve_above:
                    circuit[other] = (circuit[other] + 1) // 2
                circuit[2] = min(U32_MAX, circuit[2] + 1)
                circuit[3] = self.events_seen
        self.partial = (self.partial << 1) | bit
        if self.partial >= 256:
            self.history.append(self.partial & 0xFF)
            keep = max(1, self.orders[-1])
            if len(self.history) > keep:
                del self.history[:len(self.history) - keep]
            self.partial = 1
        self.events_seen += 1
        self._pending = None

    def observe_bytes(self, data, learn=True):
        for byte in bytes(data):
            for shift in range(7, -1, -1):
                self.predict()
                self.observe((byte >> shift) & 1, learn=learn)

    def score_bytes(self, data, learn=False, trace=None):
        """Mean next-bit cross-entropy (bits/bit) over ``data``."""
        total = 0.0
        count = 0
        for byte in bytes(data):
            for shift in range(7, -1, -1):
                bit = (byte >> shift) & 1
                probability = self.predict()
                if trace is not None:
                    trace.append(probability)
                total -= math.log2(probability if bit else 1.0 - probability)
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

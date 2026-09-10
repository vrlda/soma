"""Generic prediction heads: fixed bindings from core outputs to events.

Heads hold no domain knowledge and learn nothing; the core learns.
"""

import math
import random


class ScalarHead(object):
    """Bounded scalar prediction with a fixed uncertainty scale."""

    VERSION = 1

    def __init__(self, low=-1.0, high=1.0, scale=0.1):
        self.low = float(low)
        self.high = float(high)
        self.scale = float(scale)
        if not self.low < self.high or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scalar head bounds/scale are invalid")

    def predict(self, outputs):
        value = float(outputs[0])
        if not math.isfinite(value):
            raise ValueError("core output must be finite")
        return {"value": max(self.low, min(self.high, value)), "scale": self.scale}

    def to_dict(self):
        return {"version": self.VERSION, "low": self.low, "high": self.high, "scale": self.scale}


class DiscreteHead(object):
    """Fixed softmax distribution over a bounded symbol set."""

    VERSION = 1

    def __init__(self, symbols, temperature=1.0, seed=0):
        self.symbols = [str(symbol) for symbol in symbols]
        if len(self.symbols) < 2:
            raise ValueError("discrete head needs at least two symbols")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.rng = random.Random(int(seed))

    def distribution(self, outputs):
        logits = [float(item) for item in list(outputs)[:len(self.symbols)]]
        while len(logits) < len(self.symbols):
            logits.append(0.0)
        peak = max(logits)
        weights = [math.exp((item - peak) / self.temperature) for item in logits]
        total = sum(weights)
        return {symbol: weight / total for symbol, weight in zip(self.symbols, weights)}

    def sample(self, outputs, deterministic=True):
        distribution = self.distribution(outputs)
        if deterministic:
            return max(self.symbols, key=lambda symbol: (distribution[symbol], symbol))
        draw = self.rng.random()
        cumulative = 0.0
        for symbol in self.symbols:
            cumulative += distribution[symbol]
            if draw < cumulative:
                return symbol
        return self.symbols[-1]

    def to_dict(self):
        return {"version": self.VERSION, "symbols": list(self.symbols),
                "temperature": self.temperature}

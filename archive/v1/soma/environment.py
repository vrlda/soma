"""Tiny continuous environment used by the executable demo."""

from dataclasses import dataclass
import math
import random
from typing import Tuple


@dataclass
class Observation:
    values: Tuple[float, ...]
    reward: float
    done: bool = False


class ContinuousTargetEnvironment:
    """A drifting scalar target; action is clipped to [-1, 1]."""

    def __init__(self, seed: int = 0, horizon: int = 200) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.rng = random.Random(seed)
        self.seed = seed
        self.horizon = horizon
        self.t = 0
        self.target = 0.0
        self.last_reward = 0.0

    def reset(self) -> Observation:
        self.t = 0
        self.target = self.rng.uniform(-0.75, 0.75)
        self.last_reward = 0.0
        return Observation((self.target, 0.0), 0.0, False)

    def step(self, action: float) -> Observation:
        action = max(-1.0, min(1.0, float(action)))
        error = action - self.target
        reward = 1.0 - min(1.0, error * error)
        self.last_reward = reward
        self.t += 1
        self.target = 0.92 * self.target + 0.08 * math.sin(self.t * 0.17) + self.rng.uniform(-0.025, 0.025)
        return Observation((self.target, float(self.t) / self.horizon), reward, self.t >= self.horizon)

    def state_dict(self):
        return {
            "version": 1,
            "seed": self.seed,
            "horizon": self.horizon,
            "t": self.t,
            "target": self.target,
            "last_reward": self.last_reward,
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state):
        environment = cls(int(state["seed"]), int(state["horizon"]))
        environment.t = int(state["t"])
        environment.target = float(state["target"])
        environment.last_reward = float(state.get("last_reward", 0.0))
        environment.rng.setstate(_nested_tuple(state["rng_state"]))
        return environment


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value

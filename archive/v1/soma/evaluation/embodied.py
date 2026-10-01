"""R12 first step: deterministic 1-D cart simulator with safety envelope.

State: position x, velocity v. Action: scalar force. Hidden goal g flips
A -> B -> A across phases (goal +0.6, -0.6, +0.6). Reward is proximity
with a crash penalty. Physics (inertia + delayed consequences) exercises
multi-step credit.

Safety is enforced OUTSIDE the model by an independent brake controller
plus force clamp, track limits with emergency stop, and step budget. All
engagements and violations are recorded, never hidden.
"""

import math
import random

from ..events import Event, EventBridge
from ..events.envelope import OutcomeEvent
from ..organism import Organism
from ..synapses import Synapse
from ..transducers import CartTransducer

R12_ACCEPTANCE_SEEDS = (0, 1, 2, 3, 4, 5)
STEPS = 900
MARGIN = 0.10
# Goals vary per seed: the gate proves goal-following, not fixed-point
# memorization. Switched goals (context changes under inertia plus safety
# interventions) defeat naive single-step credit and are defined next work.
GOALS = (0.6, -0.6, 0.4, -0.4, 0.8, -0.8)
FORCE_LIMIT = 1.0
TRACK_LIMIT = 1.2
DAMPING = 0.92
FORCE_GAIN = 0.02
FIXTURE_EXPLORATION = 0.05
FIXTURE_ACTOR_LR = 0.05


class SafetyController(object):
    """Independent safety fallback: fixed-gain brake, no learning, no brain.

    Takes over with braking force as the cart nears track bounds. This is
    the R12-required safety layer independent of SOMA; it constrains action,
    never learns, and its engagements are counted, not hidden.
    """

    VERSION = 2
    SOFT_BOUND = 0.75
    BRAKE_GAIN = 4.0

    def __init__(self):
        self.engagements = 0

    def constrain(self, proposed, x, v):
        """Return (applied_force, engaged). Brakes toward center near bounds."""
        if abs(x) < self.SOFT_BOUND:
            return max(-FORCE_LIMIT, min(FORCE_LIMIT, float(proposed))), False
        self.engagements += 1
        brake = -math.copysign(min(1.0, self.BRAKE_GAIN * (abs(x) - self.SOFT_BOUND) + abs(v)), x)
        return max(-FORCE_LIMIT, min(FORCE_LIMIT, brake)), True


class CartSim(object):
    """Deterministic 1-D cart. Fixed seed, no hidden randomness after reset."""

    VERSION = 1

    def __init__(self, seed, goal=0.8):
        self.seed = int(seed)
        self.goal = float(goal)
        self.x = 0.0
        self.v = 0.0
        self.steps = 0
        self.stops = 0
        self.clamps = 0

    def reset(self):
        self.x = 0.0
        self.v = 0.0
        self.steps = 0
        return self.observe()

    def observe(self):
        return [max(-1.0, min(1.0, self.x)), max(-1.0, min(1.0, self.v * 2.0))]

    def step(self, force):
        force = max(-FORCE_LIMIT, min(FORCE_LIMIT, float(force)))
        if abs(force) >= FORCE_LIMIT:
            self.clamps += 1
        self.v = self.v * DAMPING + force * FORCE_GAIN
        self.x = self.x + self.v
        crashed = False
        if abs(self.x) > TRACK_LIMIT:
            self.stops += 1
            crashed = True
            self.x = max(-TRACK_LIMIT, min(TRACK_LIMIT, self.x))
            self.v = 0.0
        self.steps += 1
        # Declared shaping: wall crashes cost a full point so wall-hugging
        # can never beat centering. Part of the locked fixture.
        reward = 1.0 - abs(self.x - self.goal) - (1.0 if crashed else 0.0)
        return self.observe(), reward

    def state_dict(self):
        return {"version": self.VERSION, "seed": self.seed, "goal": self.goal,
                "x": self.x, "v": self.v, "steps": self.steps,
                "stops": self.stops, "clamps": self.clamps}

    @classmethod
    def from_state_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported cart state")
        sim = cls(payload["seed"], payload["goal"])
        sim.x = float(payload["x"])
        sim.v = float(payload["v"])
        sim.steps = int(payload["steps"])
        sim.stops = int(payload["stops"])
        sim.clamps = int(payload["clamps"])
        return sim


def make_embodied_organism(seed, frozen=False):
    organism = Organism.create_default(
        input_size=3, hidden_size=6, output_size=1, seed=seed)
    for synapse in organism.graph.iter_synapses():
        if synapse.destination in organism.output_ids:
            synapse.strength = 0.0
    for input_id in organism.input_ids:
        if not organism.graph.has(input_id, organism.output_ids[0]):
            organism.graph.add(Synapse(input_id, organism.output_ids[0], 0.0, plasticity=1.0))
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    organism.enable_context_modules(max_modules=4)
    organism.actor_learning_rate = 0.0 if frozen else FIXTURE_ACTOR_LR
    if frozen:
        organism.learning_rate = 0.0
        organism.use_threshold_router()
    return organism


def run_cart(seed, mode="learn", steps=STEPS):
    """One locked embodied lifetime. Returns phase tails and safety record."""
    if mode not in ("learn", "frozen", "shuffled", "passive"):
        raise ValueError("unknown R12 mode: %s" % mode)
    adapter = CartTransducer()
    organism = make_embodied_organism(seed, frozen=(mode == "frozen"))
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=FIXTURE_EXPLORATION)
    sim = CartSim(seed)
    observation = sim.reset()
    sim.goal = GOALS[seed % len(GOALS)]
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(steps)))
    counter = {"position": 0, "velocity": 0}
    safety = SafetyController()
    rewards = []
    lagged = []
    for step in range(steps):
        envelopes = [
            Event("position", "simulator", counter["position"], step,
                  {"value": float(observation[0])}, {"trust": "test"}).to_dict(),
            Event("velocity", "simulator", counter["velocity"], step,
                  {"value": float(observation[1])}, {"trust": "test"}).to_dict(),
        ]
        counter["position"] += 1
        counter["velocity"] += 1
        action = bridge.ingest(envelopes)
        if mode == "passive":
            proposed = 0.0
        else:
            proposed = action["proposal"]["value"]
        force, _ = safety.constrain(proposed, observation[0], observation[1])
        observation, reward = sim.step(force)
        rewards.append(reward)
        if mode == "shuffled":
            lagged.append(reward)
            delivered = lagged.pop(0) if len(lagged) > 37 else 0.0
        else:
            delivered = reward
        bridge.outcome(OutcomeEvent(action["correlation_id"], step, delivered, "simulator").to_dict())
    bridge.close()
    tail = rewards[3 * steps // 4:]
    tails = {
        "tail_mean": sum(tail) / max(1, len(tail)),
        "stops": sim.stops,
        "clamps": sim.clamps,
        "safety_engagements": safety.engagements,
        "modules": len(organism.motor_modules),
    }
    return tails


def run_replay(seed, force_tape, goal=0.6):
    """Offline replay evaluation: fixed action tape, deterministic scoring.

    No learning, no exploration. Same tape always yields the same trajectory,
    returns, and safety record. The plan's first robotics qualification stage.
    """
    sim = CartSim(seed, goal=goal)
    observation = sim.reset()
    safety = SafetyController()
    rewards = []
    for force in force_tape:
        applied, _ = safety.constrain(force, observation[0], observation[1])
        observation, reward = sim.step(applied)
        rewards.append(reward)
    return {
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "stops": sim.stops,
        "clamps": sim.clamps,
        "engagements": safety.engagements,
        "final_x": sim.x,
        "state": sim.state_dict(),
    }


def run_cart_benchmark(seeds=R12_ACCEPTANCE_SEEDS, steps=STEPS):
    """Locked R12-first-step gates.

    Deterministic replay, safety envelope under scripted aggression, and a
    causal learning signal (learn beats shuffled). Absolute closed-loop
    mastery (beating frozen/passive reliably across seeds) is recorded, not
    gated: limit cycles and seed luck defeat naive single-step credit, which
    is defined next work (multi-step planning). See research/docs/r12-embodied.md.
    """
    import random as _random
    report = {"seeds": list(seeds), "steps": steps, "modes": {}, "all_passed": False}
    for mode in ("learn", "frozen", "shuffled", "passive"):
        tails = [run_cart(seed, mode, steps) for seed in seeds]
        report["modes"][mode] = {
            window: sum(entry[window] for entry in tails) / len(tails)
            for window in ("tail_mean", "stops", "clamps", "safety_engagements")
        }
    tape_rng = _random.Random(4242)
    aggressive = [tape_rng.uniform(-1.0, 1.0) for _ in range(300)]
    first = run_replay(0, aggressive)
    second = run_replay(0, aggressive)
    max_force = run_replay(1, [1.0] * 300)
    learn = report["modes"]["learn"]
    gates = {
        "replay_deterministic": first == second,
        "safety_bounds_aggression": max_force["stops"] == 0,
        "learn_beats_shuffled": (
            learn["tail_mean"] - report["modes"]["shuffled"]["tail_mean"] >= MARGIN),
        "learn_beats_frozen": (
            learn["tail_mean"] - report["modes"]["frozen"]["tail_mean"] >= MARGIN),
        "learn_beats_passive": (
            learn["tail_mean"] - report["modes"]["passive"]["tail_mean"] >= MARGIN),
    }
    report["replay"] = {"deterministic": first == second, "max_force_stops": max_force["stops"]}
    report["gates"] = gates
    report["all_passed"] = all(gates.values())
    return report

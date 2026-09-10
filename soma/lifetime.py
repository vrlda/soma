"""An unannounced A -> B -> A lifetime benchmark for SOMA.

The environment deliberately keeps the observation stream phase-blind.  Its
latent rule changes at fixed transition counts, while the evaluator records
those counts separately.  This is a small integration benchmark, not a claim
that the current organism has solved continual learning.
"""

from dataclasses import dataclass
import copy
from itertools import combinations
import json
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .environment import Observation
from .organism import Modulators, Organism
from .synapses import Synapse, synapse_to_dict


@dataclass
class LifetimeBenchmarkConfig:
    """Predeclared protocol and decision thresholds.

    Thresholds are intentionally modest because this fixture is meant to
    detect a reproducible lifetime signal, not to reward a single tuned run.
    """

    phase_lengths: Tuple[int, int, int] = (120, 120, 240)
    seeds: Tuple[int, ...] = tuple(range(24))
    exploration: float = 0.20
    novelty: float = 0.10
    initial_acquisition_delta: float = 0.02
    b_adaptation_delta: float = 0.02
    b_tail_skill_floor: float = 0.10
    a_initial_tail_skill_floor: float = 0.0
    a_return_tail_skill_floor: float = 0.0
    return_retention_floor: float = 0.20
    return_relearning_steps_limit: int = 80
    comparison_margin: float = 0.02
    actor_learning_rate: float = 0.10

    def validate(self) -> None:
        if len(self.phase_lengths) != 3 or any(int(length) <= 0 for length in self.phase_lengths):
            raise ValueError("phase_lengths must contain three positive lengths")
        if not self.seeds:
            raise ValueError("at least one deterministic seed is required")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("exploration and novelty must be in [0, 1]")
        if not math.isfinite(self.actor_learning_rate) or self.actor_learning_rate < 0.0:
            raise ValueError("actor_learning_rate must be finite and nonnegative")
        if not math.isfinite(self.b_tail_skill_floor):
            raise ValueError("b_tail_skill_floor must be finite")


def phase_lengths_for_seed(seed: int, base: Sequence[int] = (60, 60, 60)) -> Tuple[int, int, int]:
    """Derive hidden, reproducible phase lengths from a separate RNG stream."""
    if len(tuple(base)) != 3 or any(int(length) < 20 for length in base):
        raise ValueError("base phase lengths must contain three values >= 20")
    schedule_rng = random.Random((int(seed) + 1) * 1000003 + 7919)
    return tuple(max(20, int(length) + schedule_rng.randint(-10, 10)) for length in base)


class UnannouncedLifetimeEnvironment:
    """Phase-blind cue/control task with an internal A -> B -> A switch.

    In rule A the correct action is ``cue``.  In rule B it is ``-cue``.  The
    same deterministic cue stream is generated regardless of phase, so phase
    cannot be inferred from the observation itself.  The phase and target are
    intentionally absent from :class:`Observation` and from modulators.
    """

    def __init__(self, seed: int = 0, phase_lengths: Optional[Sequence[int]] = None) -> None:
        if phase_lengths is None:
            phase_lengths = phase_lengths_for_seed(seed)
        if len(tuple(phase_lengths)) != 3 or any(int(length) <= 0 for length in phase_lengths):
            raise ValueError("phase_lengths must contain three positive lengths")
        self.seed = int(seed)
        self.phase_lengths = tuple(int(length) for length in phase_lengths)
        self.horizon = sum(self.phase_lengths)
        # Schedule and cue generation are independent.  The schedule is
        # checkpointed but never exposed through observations.
        self.rng = random.Random(self.seed + 104729)
        self.t = 0
        self.cue = 0.0
        self.distractor = 0.0
        self.last_reward = 0.0
        self.reset()

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed + 104729)
        self.t = 0
        self.cue, self.distractor = self._next_cue()
        self.last_reward = 0.0
        return Observation((self.cue, self.distractor), 0.0, False)

    def _next_cue(self) -> Tuple[float, float]:
        # A symmetric, continuously varying cue prevents memorizing a phase
        # label while retaining a simple action function for the organism.
        return self.rng.uniform(-0.9, 0.9), self.rng.uniform(-0.9, 0.9)

    def _phase_index(self, transition: int) -> int:
        boundary = 0
        for index, length in enumerate(self.phase_lengths):
            boundary += length
            if transition < boundary:
                return index
        return 2

    def _target(self, cue: float, transition: int) -> float:
        return cue if self._phase_index(transition) in (0, 2) else -cue

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise ValueError("environment is done; call reset before stepping")
        action = max(-1.0, min(1.0, float(action)))
        target = self._target(self.cue, self.t)
        error = action - target
        # Normalized skill reward: zero action is exactly neutral, while a
        # correct action is positive and an opposite-polarity action is
        # negative.  The evaluator also computes 1-SSE/SSE_zero.
        reward = max(-1.0, min(1.0, (target * target - (action - target) ** 2) / 4.0))
        self.last_reward = reward
        self.t += 1
        self.cue, self.distractor = self._next_cue()
        return Observation((self.cue, self.distractor), reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "seed": self.seed,
            "phase_lengths": list(self.phase_lengths),
            "horizon": self.horizon,
            "t": self.t,
            "cue": self.cue,
            "distractor": self.distractor,
            "last_reward": self.last_reward,
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "UnannouncedLifetimeEnvironment":
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported lifetime environment checkpoint version")
        environment = cls(int(state["seed"]), tuple(int(value) for value in state["phase_lengths"]))
        environment.t = int(state["t"])
        environment.cue = float(state["cue"])
        environment.distractor = float(state.get("distractor", 0.0))
        environment.last_reward = float(state.get("last_reward", 0.0))
        environment.rng.setstate(_nested_tuple(state["rng_state"]))
        return environment


def phase_windows(config: LifetimeBenchmarkConfig) -> Dict[str, Tuple[int, int]]:
    """Return evaluator-only transition windows (start inclusive, end exclusive)."""
    config.validate()
    return phase_windows_for_lengths(config.phase_lengths)


def phase_windows_for_lengths(lengths: Sequence[int]) -> Dict[str, Tuple[int, int]]:
    if len(tuple(lengths)) != 3 or any(int(length) <= 0 for length in lengths):
        raise ValueError("lengths must contain three positive values")
    first_end = int(lengths[0])
    second_end = first_end + int(lengths[1])
    return {
        "A_initial": (0, first_end),
        "B_interference": (first_end, second_end),
        "A_return": (second_end, sum(int(length) for length in lengths)),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def _window_mean(rewards: Sequence[float], start: int, end: int) -> float:
    return _mean(rewards[start:end])


def _skill(actions: Sequence[float], targets: Sequence[float], start: int, end: int) -> float:
    """Evaluator-only normalized skill against the neutral zero policy."""
    sse = sum((float(action) - float(target)) ** 2 for action, target in zip(actions[start:end], targets[start:end]))
    zero = sum(float(target) ** 2 for target in targets[start:end])
    return 1.0 - (sse / zero) if zero > 1e-12 else 0.0


def _sign_accuracy(actions: Sequence[float], targets: Sequence[float], start: int, end: int) -> float:
    pairs = [(float(action), float(target)) for action, target in zip(actions[start:end], targets[start:end]) if abs(float(target)) >= 0.05]
    return _mean(1.0 if action * target > 0.0 else 0.0 for action, target in pairs)


def _context_event_metrics(events: Sequence[Mapping[str, object]], total_steps: int) -> Dict[str, object]:
    """Summarize evaluator-visible warning/search occupancy without phase labels."""
    starts = [int(event["step"]) for event in events if event.get("kind") == "motor_warning_started"]
    intervals = []
    for start in starts:
        end = next((int(event["step"]) for event in events if int(event.get("step", -1)) >= start and event.get("kind") in ("motor_warning_resolved", "motor_warning_escalated")), total_steps + 1)
        intervals.append(max(0, min(total_steps, end) - min(total_steps, start)))
    return {
        "warning_count": len(starts),
        "warning_steps": sum(intervals),
        "warning_occupancy": sum(intervals) / float(max(1, total_steps)),
        "warning_escalation_count": sum(event.get("kind") == "motor_warning_escalated" for event in events),
    }


def _make_benchmark_organism(seed: int) -> Organism:
    """Create a matched minimal motor scaffold for the causal milestone.

    The cue sensor has a direct output-afferent edge initialized at zero.
    Other output-afferent strengths are neutralized so this iteration tests
    the actor credit path rather than an accidental hidden representation.
    Later milestones can remove this scaffold while enabling hidden learning.
    """
    organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=seed)
    for synapse in organism.graph.iter_synapses():
        if synapse.destination in organism.output_ids:
            synapse.strength = 0.0
    if not organism.graph.has("input-000", "output-000"):
        organism.graph.add(Synapse("input-000", "output-000", 0.0, plasticity=1.0))
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    return organism


def _first_relearning_step(actions: Sequence[float], targets: Sequence[float], start: int, end: int, target: float) -> Optional[int]:
    window = max(5, min(12, (end - start) // 4))
    for offset in range(0, end - start - window + 1):
        left = start + offset
        right = left + window
        if _skill(actions, targets, left, right) >= target:
            return offset
    return None


def run_lifetime(seed: int, config: Optional[LifetimeBenchmarkConfig] = None, learning_rate: float = 0.08, mode: str = "full") -> Dict[str, object]:
    """Run one persistent organism lineage through all three hidden phases."""
    config = config or LifetimeBenchmarkConfig()
    config.validate()
    aliases = {
        "full": "full_context_plus_rewiring",
        "weights_only": "context_actor_no_generic_rewiring",
        "structure_only": "no_actor_context_plus_rewiring",
        "frozen": "frozen",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("full_context_plus_rewiring", "context_actor_no_generic_rewiring", "no_actor_context_plus_rewiring", "frozen", "legacy_full"):
        raise ValueError("unsupported lifetime mode: %s" % mode)
    environment = UnannouncedLifetimeEnvironment(seed=seed, phase_lengths=phase_lengths_for_seed(seed, config.phase_lengths))
    organism = _make_benchmark_organism(seed)
    organism.enable_context_modules(max_modules=4)
    organism.actor_learning_rate = float(learning_rate) if mode in ("full_context_plus_rewiring", "context_actor_no_generic_rewiring", "legacy_full") else 0.0
    organism.learning_rate = float(learning_rate) if mode == "legacy_full" else 0.0
    organism.legacy_learning_enabled = mode == "legacy_full"
    organism.structural_plasticity_enabled = mode in ("full_context_plus_rewiring", "no_actor_context_plus_rewiring", "legacy_full")
    exploration_rng = random.Random((int(seed) + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(exploration_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    initial_identity = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(s.id for s in organism.graph.iter_synapses())
    observation = environment.reset()
    rewards: List[float] = []
    actions: List[float] = []
    targets: List[float] = []
    for _ in range(environment.horizon):
        # The evaluator may know the latent schedule; this target is never
        # passed to the organism or included in its observation.
        target = observation.values[0] if environment._phase_index(environment.t) in (0, 2) else -observation.values[0]
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        action = result.outputs[0]
        observation = environment.step(action)
        rewards.append(observation.reward)
        actions.append(action)
        targets.append(target)
    organism.apply_outcome(observation.reward)
    windows = phase_windows_for_lengths(environment.phase_lengths)
    a_start, a_end = windows["A_initial"]
    b_start, b_end = windows["B_interference"]
    r_start, r_end = windows["A_return"]
    quarter = max(1, (a_end - a_start) // 4)
    b_quarter = max(1, (b_end - b_start) // 4)
    r_quarter = max(1, (r_end - r_start) // 4)
    initial_first = _window_mean(rewards, a_start, a_start + quarter)
    initial_last = _window_mean(rewards, a_end - quarter, a_end)
    b_first = _window_mean(rewards, b_start, b_start + b_quarter)
    b_last = _window_mean(rewards, b_end - b_quarter, b_end)
    return_first = _window_mean(rewards, r_start, r_start + r_quarter)
    return_last = _window_mean(rewards, r_end - r_quarter, r_end)
    phase_metrics = {}
    for name, (start, end) in windows.items():
        span = max(1, end - start)
        early_end = start + max(1, span // 4)
        tail_start = end - max(1, span // 4)
        phase_metrics[name] = {
            "skill": _skill(actions, targets, start, end),
            "sign_accuracy": _sign_accuracy(actions, targets, start, end),
            "early_skill": _skill(actions, targets, start, early_end),
            "tail_skill": _skill(actions, targets, tail_start, end),
            "early_sign_accuracy": _sign_accuracy(actions, targets, start, early_end),
            "tail_sign_accuracy": _sign_accuracy(actions, targets, tail_start, end),
        }
        if name == "A_return":
            for horizon in (1, 5, 10, 20, 40, 60):
                phase_metrics[name]["skill_first_%d" % horizon] = _skill(actions, targets, start, min(end, start + horizon))
    return {
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "mode": mode,
        "phase_lengths": tuple(environment.phase_lengths),
        "rewards": rewards,
        "actions": actions,
        "targets": targets,
        "phase_metrics": phase_metrics,
        "phase_scores": {
            "A_initial": _window_mean(rewards, a_start, a_end),
            "B_interference": _window_mean(rewards, b_start, b_end),
            "A_return": _window_mean(rewards, r_start, r_end),
        },
        "initial_acquisition": phase_metrics["A_initial"]["tail_skill"] - phase_metrics["A_initial"]["early_skill"],
        "b_adaptation": phase_metrics["B_interference"]["tail_skill"] - phase_metrics["B_interference"]["early_skill"],
        "return_immediate_retention": phase_metrics["A_return"]["early_skill"],
        # Relearning is only credited after returning to a meaningful level,
        # never merely because the initial phase happened to score negatively.
        "return_relearning_steps": _first_relearning_step(
            actions, targets, r_start, r_end, max(config.return_retention_floor, phase_metrics["A_initial"]["tail_skill"])),
        "initial_last_score": phase_metrics["A_initial"]["tail_skill"],
        "return_last_score": phase_metrics["A_return"]["tail_skill"],
        "organism_steps": organism.step_count,
        "initial_cell_ids": initial_identity,
        "initial_synapse_ids": initial_synapse_ids,
        "final_cell_ids": tuple(sorted(organism.cells)),
        "final_synapse_ids": tuple(s.id for s in organism.graph.iter_synapses()),
        "context_metrics": _context_event_metrics(organism.events, organism.step_count),
        "final_state": organism.state_dict(),
    }


def summarize(results: Sequence[Mapping[str, object]], config: Optional[LifetimeBenchmarkConfig] = None) -> Dict[str, object]:
    """Aggregate runs and apply the predeclared protocol thresholds."""
    config = config or LifetimeBenchmarkConfig()
    config.validate()
    if not results:
        raise ValueError("results must not be empty")
    def avg(name: str) -> float:
        return _mean(float(result[name]) for result in results)
    initial = avg("initial_acquisition")
    adaptation = avg("b_adaptation")
    retention = avg("return_immediate_retention")
    b_tail_skill = _mean(float(result["phase_metrics"]["B_interference"]["tail_skill"]) for result in results) if all("phase_metrics" in result for result in results) else None
    a_initial_tail_skill = _mean(float(result["phase_metrics"]["A_initial"]["tail_skill"]) for result in results) if all("phase_metrics" in result for result in results) else None
    a_return_tail_skill = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in results) if all("phase_metrics" in result for result in results) else None
    context_metrics = {
        name: _mean(float(result["context_metrics"][name]) for result in results)
        for name in ("warning_count", "warning_steps", "warning_occupancy", "warning_escalation_count")
    } if all("context_metrics" in result for result in results) else None
    relearning = [result["return_relearning_steps"] for result in results if result["return_relearning_steps"] is not None]
    passed = {
        "initial_acquisition": initial >= config.initial_acquisition_delta and (a_initial_tail_skill is None or a_initial_tail_skill > config.a_initial_tail_skill_floor),
        "b_adaptation": adaptation >= config.b_adaptation_delta and (b_tail_skill is None or b_tail_skill > config.b_tail_skill_floor),
        "return_retention": retention >= config.return_retention_floor and (a_return_tail_skill is None or a_return_tail_skill > config.a_return_tail_skill_floor),
        "return_relearning": bool(relearning) and _mean(float(value) for value in relearning) <= config.return_relearning_steps_limit,
    }
    per_seed = []
    for result in results:
        metrics = result.get("phase_metrics")
        seed_gates = {
            "initial_acquisition": float(result["initial_acquisition"]) >= config.initial_acquisition_delta and (metrics is None or float(metrics["A_initial"]["tail_skill"]) > config.a_initial_tail_skill_floor),
            "b_adaptation": float(result["b_adaptation"]) >= config.b_adaptation_delta and (metrics is None or float(metrics["B_interference"]["tail_skill"]) > config.b_tail_skill_floor),
            "return_retention": float(result["return_immediate_retention"]) >= config.return_retention_floor and (metrics is None or float(metrics["A_return"]["tail_skill"]) > config.a_return_tail_skill_floor),
            "return_relearning": result["return_relearning_steps"] is not None and float(result["return_relearning_steps"]) <= config.return_relearning_steps_limit,
        }
        per_seed.append({"seed": result.get("seed"), "gates": seed_gates, "all_passed": all(seed_gates.values())})
    standard_metrics = ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")
    phase_summary = {
        name: {
            metric: _mean(float(result["phase_metrics"][name][metric]) for result in results)
            for metric in standard_metrics + (("skill_first_1", "skill_first_5", "skill_first_10", "skill_first_20", "skill_first_40", "skill_first_60") if name == "A_return" and all(metric in result["phase_metrics"][name] for result in results for metric in ("skill_first_1", "skill_first_5", "skill_first_10", "skill_first_20", "skill_first_40", "skill_first_60")) else ())
        }
        for name in ("A_initial", "B_interference", "A_return")
    } if all("phase_metrics" in result for result in results) else {}
    return {
        "seeds": len(results),
        "mean_initial_acquisition": initial,
        "mean_b_adaptation": adaptation,
        "mean_b_tail_skill": b_tail_skill,
        "mean_a_initial_tail_skill": a_initial_tail_skill,
        "mean_a_return_tail_skill": a_return_tail_skill,
        "mean_context_metrics": context_metrics,
        "mean_return_immediate_retention": retention,
        "mean_phase_metrics": phase_summary,
        "mean_return_relearning_steps": _mean(float(value) for value in relearning) if relearning else None,
        "relearning_observed_count": len(relearning),
        "relearning_censored_count": len(results) - len(relearning),
        "per_seed": per_seed,
        "pass_counts": {name: sum(1 for item in per_seed if item["gates"][name]) for name in passed},
        "pass_fractions": {name: sum(item["gates"][name] for item in per_seed) / float(len(per_seed)) for name in passed},
        "thresholds": {
            "initial_acquisition_delta": config.initial_acquisition_delta,
            "b_adaptation_delta": config.b_adaptation_delta,
            "b_tail_skill_floor": config.b_tail_skill_floor,
            "a_initial_tail_skill_floor": config.a_initial_tail_skill_floor,
            "a_return_tail_skill_floor": config.a_return_tail_skill_floor,
            "return_retention_floor": config.return_retention_floor,
            "return_relearning_steps_limit": config.return_relearning_steps_limit,
        },
        "passed": passed,
        "all_passed": all(passed.values()),
    }
def run_benchmark(config: Optional[LifetimeBenchmarkConfig] = None) -> Dict[str, object]:
    """Run learner and matched structural-only control across all seeds."""
    config = config or LifetimeBenchmarkConfig()
    config.validate()
    weights_only = [run_lifetime(seed, config, learning_rate=config.actor_learning_rate, mode="context_actor_no_generic_rewiring") for seed in config.seeds]
    full = [run_lifetime(seed, config, learning_rate=config.actor_learning_rate, mode="full_context_plus_rewiring") for seed in config.seeds]
    control = [run_lifetime(seed, config, learning_rate=0.0, mode="no_actor_context_plus_rewiring") for seed in config.seeds]
    learner_summary = summarize(weights_only, config)
    full_summary = summarize(full, config)
    control_summary = summarize(control, config)
    matched_start = all(
        learner["initial_cell_ids"] == control_run["initial_cell_ids"]
        and learner["initial_synapse_ids"] == control_run["initial_synapse_ids"]
        for learner, control_run in zip(weights_only, control)
    )
    learner_score = _mean(float(result["phase_metrics"]["A_return"]["skill"]) for result in weights_only)
    full_score = _mean(float(result["phase_metrics"]["A_return"]["skill"]) for result in full)
    control_score = _mean(float(result["phase_metrics"]["A_return"]["skill"]) for result in control)
    def reactivated_before_third(result: Mapping[str, object]) -> bool:
        events = result["final_state"]["events"]
        a2_start = int(result["phase_lengths"][0]) + int(result["phase_lengths"][1])
        react = next((int(event["step"]) for event in events if int(event["step"]) >= a2_start and event["kind"] == "motor_module_reactivated" and event["module_id"] == "motor-module-000"), None)
        third = next((int(event["step"]) for event in events if int(event["step"]) >= a2_start and event["kind"] == "motor_module_recruited" and event["module_id"] == "motor-module-002"), None)
        return react is not None and (third is None or react < third)
    def reactivated_in_a2(result: Mapping[str, object]) -> bool:
        a2_start = int(result["phase_lengths"][0]) + int(result["phase_lengths"][1])
        return any(int(event["step"]) >= a2_start and event["kind"] == "motor_module_reactivated" and event["module_id"] == "motor-module-000" for event in result["final_state"]["events"])
    def reactivation_lag(result: Mapping[str, object]) -> Optional[int]:
        a2_start = int(result["phase_lengths"][0]) + int(result["phase_lengths"][1])
        steps = [int(event["step"]) - a2_start for event in result["final_state"]["events"] if int(event["step"]) >= a2_start and event["kind"] == "motor_module_reactivated" and event["module_id"] == "motor-module-000"]
        return min(steps) if steps else None
    lags = sorted(lag for lag in (reactivation_lag(result) for result in weights_only) if lag is not None)
    return {
        "protocol": {
            "phase_lengths_base": list(config.phase_lengths),
            "phase_lengths_by_seed": {str(seed): list(phase_lengths_for_seed(seed, config.phase_lengths)) for seed in config.seeds},
            "seeds": list(config.seeds),
            "exploration": config.exploration,
            "novelty": config.novelty,
            "actor_learning_rate": config.actor_learning_rate,
            "hidden_phase_boundaries": True,
        },
        "context_actor_no_generic_rewiring": learner_summary,
        "full_context_plus_rewiring": full_summary,
        "no_actor_context_plus_rewiring": control_summary,
        # Backward-compatible report aliases.
        "learner": learner_summary,
        "weights_only": learner_summary,
        "full_structural": full_summary,
        "structural_only_control": control_summary,
        "comparison": {
            "a_return_score_difference": learner_score - control_score,
            "learner_better_by_margin": learner_score >= control_score + config.comparison_margin,
            "full_a_return_score_difference": full_score - control_score,
            "matched_starting_structure": matched_start,
            "matched_seed_and_protocol": True,
            "a2_original_module_reactivated_runs": sum(reactivated_in_a2(result) for result in weights_only),
            "a2_original_module_reactivated_before_third_recruit_runs": sum(reactivated_before_third(result) for result in weights_only),
            "a2_reactivation_lag_median": lags[len(lags) // 2] if lags else None,
            "a2_reactivation_lag_range": [lags[0], lags[-1]] if lags else None,
            "a2_reactivation_lag_censored_runs": len(weights_only) - len(lags),
        },
        "diagnostic": "PASS" if learner_summary["all_passed"] else "RETENTION_OR_ADAPTATION_NOT_ESTABLISHED",
    }


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Multi-context stress milestone


@dataclass
class MultiContextBenchmarkConfig:
    """Protocol for the hidden A -> B -> C -> A -> B stress benchmark."""

    phase_lengths: Tuple[int, int, int, int, int] = (264, 264, 264, 264, 264)
    seeds: Tuple[int, ...] = tuple(range(24))
    exploration: float = 0.20
    novelty: float = 0.10
    actor_learning_rate: float = 0.10
    tail_fraction: float = 0.25
    tail_skill_floor: float = 0.0

    def validate(self) -> None:
        if len(self.phase_lengths) != 5 or any(int(length) < 40 for length in self.phase_lengths):
            raise ValueError("multi-context phase_lengths must contain five values >= 40")
        if not self.seeds:
            raise ValueError("at least one multi-context seed is required")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("exploration and novelty must be in [0, 1]")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if not math.isfinite(self.actor_learning_rate) or self.actor_learning_rate < 0.0:
            raise ValueError("actor_learning_rate must be finite and nonnegative")


# Frozen acceptance manifest.  These are deliberately independent values,
# rather than four arithmetic transforms of one seed.  A separate small
# development manifest can be supplied by callers through custom seed values.
MULTI_ACCEPTANCE_MANIFEST: Tuple[Tuple[int, int, int, int], ...] = (
    (4101, 7103, 9107, 11113), (4103, 7109, 9113, 11117),
    (4127, 7121, 9127, 11131), (4133, 7127, 9133, 11143),
    (4139, 7129, 9137, 11149), (4153, 7151, 9151, 11159),
    (4157, 7159, 9161, 11171), (4159, 7177, 9173, 11173),
    (4177, 7187, 9181, 11177), (4201, 7193, 9199, 11197),
    (4211, 7211, 9203, 11213), (4217, 7213, 9209, 11239),
    (4229, 7219, 9221, 11243), (4231, 7229, 9227, 11251),
    (4241, 7237, 9239, 11257), (4253, 7243, 9257, 11261),
    (4259, 7247, 9277, 11267), (4271, 7253, 9281, 11269),
    (4273, 7283, 9283, 11273), (4283, 7297, 9287, 11279),
    (4289, 7307, 9293, 11287), (4297, 7309, 9301, 11297),
    (4327, 7321, 9311, 11311), (4337, 7331, 9323, 11317),
)


MULTI_DEVELOPMENT_MANIFEST: Tuple[Tuple[int, int, int, int], ...] = (
    (5101, 8107, 10103, 13109), (5107, 8111, 10111, 13127),
    (5113, 8117, 10133, 13147), (5119, 8123, 10139, 13151),
)


def multi_seed_manifest(seed: int, acceptance: bool = True) -> Tuple[int, int, int, int]:
    manifest = MULTI_ACCEPTANCE_MANIFEST if acceptance else MULTI_DEVELOPMENT_MANIFEST
    if 0 <= int(seed) < len(manifest):
        return manifest[int(seed)]
    raise ValueError("seed is not present in the frozen multi-context manifest")


def multi_phase_lengths_for_seed(seed: int, base: Sequence[int] = (264, 264, 264, 264, 264)) -> Tuple[int, int, int, int, int]:
    """Generate wide hidden lengths from a schedule-only RNG stream."""
    values = tuple(int(length) for length in base)
    if len(values) != 5 or any(length < 40 for length in values):
        raise ValueError("multi-context base lengths must contain five values >= 40")
    schedule_rng = random.Random((int(seed) + 1) * 2000003 + 15401)
    spread = 48 if min(values) >= 216 else 35
    return tuple(max(40, length + schedule_rng.randint(-spread, spread)) for length in values)


class UnannouncedMultiContextEnvironment:
    """Phase-blind sensor task with hidden A -> B -> C -> A -> B policies.

    Policies A, B, and C are unit vectors separated by 120 degrees on the
    circular two-sensor observation.  The schedule and environment streams
    are separate and neither context nor target is observable.
    """

    POLICY_SEQUENCE = ("A", "B", "C", "A", "B")
    POLICY_ORDERS = (
        ("A", "B", "C", "A", "B"),
        ("A", "C", "B", "C", "A"),
        ("B", "A", "C", "B", "C"),
    )

    def __init__(
        self,
        seed: int = 0,
        phase_lengths: Optional[Sequence[int]] = None,
        schedule_seed: Optional[int] = None,
        policy_sequence: Sequence[str] = POLICY_SEQUENCE,
    ) -> None:
        self.seed = int(seed)
        self.schedule_seed = int(seed if schedule_seed is None else schedule_seed)
        self.policy_sequence = tuple(str(policy) for policy in policy_sequence)
        if len(self.policy_sequence) != 5 or any(policy not in {"A", "B", "C"} for policy in self.policy_sequence):
            raise ValueError("multi-context policy sequence must use A, B, and C labels")
        if phase_lengths is None:
            phase_lengths = multi_phase_lengths_for_seed(self.schedule_seed)
        self.phase_lengths = tuple(int(length) for length in phase_lengths)
        if len(self.phase_lengths) != len(self.policy_sequence) or any(length <= 0 for length in self.phase_lengths):
            raise ValueError("multi-context phase_lengths must match the policy sequence")
        self.horizon = sum(self.phase_lengths)
        self.rng = random.Random(self.seed + 1618033)
        self.t = 0
        self.cue = 0.0
        self.distractor = 0.0
        self.last_reward = 0.0
        self.reset()

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed + 1618033)
        self.t = 0
        self.cue, self.distractor = self._next_inputs()
        self.last_reward = 0.0
        return Observation((self.cue, self.distractor), 0.0, False)

    def _next_inputs(self) -> Tuple[float, float]:
        # Uniform circular observations give every policy the same marginal
        # difficulty.  A fresh angle per transition prevents context labels
        # from being encoded in the cue distribution.
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        return math.cos(angle), math.sin(angle)

    def _context_index(self, transition: int) -> int:
        boundary = 0
        for index, length in enumerate(self.phase_lengths):
            boundary += length
            if transition < boundary:
                return index
        return len(self.phase_lengths) - 1

    def _target(self, cue: float, distractor: float, transition: int) -> float:
        policy = self.policy_sequence[self._context_index(transition)]
        angles = {"A": 0.0, "B": 2.0 * math.pi / 3.0, "C": 4.0 * math.pi / 3.0}
        policy_angle = angles[policy]
        return 0.8 * (math.cos(policy_angle) * cue + math.sin(policy_angle) * distractor)

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise ValueError("environment is done; call reset before stepping")
        action = max(-1.0, min(1.0, float(action)))
        target = self._target(self.cue, self.distractor, self.t)
        reward = max(-1.0, min(1.0, (target * target - (action - target) ** 2) / 4.0))
        self.last_reward = reward
        self.t += 1
        self.cue, self.distractor = self._next_inputs()
        return Observation((self.cue, self.distractor), reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "seed": self.seed,
            "schedule_seed": self.schedule_seed,
            "policy_sequence": list(self.policy_sequence),
            "phase_lengths": list(self.phase_lengths),
            "horizon": self.horizon,
            "t": self.t,
            "cue": self.cue,
            "distractor": self.distractor,
            "last_reward": self.last_reward,
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "UnannouncedMultiContextEnvironment":
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported multi-context environment checkpoint version")
        environment = cls(
            int(state["seed"]),
            tuple(int(value) for value in state["phase_lengths"]),
            schedule_seed=int(state.get("schedule_seed", state["seed"])),
            policy_sequence=tuple(state.get("policy_sequence", cls.POLICY_SEQUENCE)),
        )
        environment.t = int(state["t"])
        environment.cue = float(state["cue"])
        environment.distractor = float(state.get("distractor", 0.0))
        environment.last_reward = float(state.get("last_reward", 0.0))
        environment.rng.setstate(_nested_tuple(state["rng_state"]))
        return environment


def multi_phase_windows_for_lengths(lengths: Sequence[int]) -> Dict[str, Tuple[int, int]]:
    values = tuple(int(length) for length in lengths)
    if len(values) != 5 or any(length <= 0 for length in values):
        raise ValueError("multi-context lengths must contain five positive values")
    names = ("A_initial", "B_interference", "C_novel", "A_return", "B_return")
    windows: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, length in zip(names, values):
        windows[name] = (start, start + length)
        start += length
    return windows


def _multi_phase_metrics(actions: Sequence[float], targets: Sequence[float], windows: Mapping[str, Tuple[int, int]], tail_fraction: float) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, (start, end) in windows.items():
        span = max(1, end - start)
        tail = max(1, int(span * tail_fraction))
        tail_start = end - tail
        metrics[name] = {
            "skill": _skill(actions, targets, start, end),
            "sign_accuracy": _sign_accuracy(actions, targets, start, end),
            "early_skill": _skill(actions, targets, start, min(end, start + tail)),
            "tail_skill": _skill(actions, targets, tail_start, end),
            "early_sign_accuracy": _sign_accuracy(actions, targets, start, min(end, start + tail)),
            "tail_sign_accuracy": _sign_accuracy(actions, targets, tail_start, end),
        }
    return metrics


def _multi_context_organism(seed: int) -> Organism:
    organism = _make_benchmark_organism(seed)
    return organism


def run_multicontext(
    seed: int,
    config: Optional[MultiContextBenchmarkConfig] = None,
    mode: str = "context_actor",
    policy_sequence: Sequence[str] = UnannouncedMultiContextEnvironment.POLICY_SEQUENCE,
) -> Dict[str, object]:
    """Run one persistent lineage through hidden A -> B -> C -> A -> B."""
    config = config or MultiContextBenchmarkConfig()
    config.validate()
    aliases = {"full": "context_actor", "weights_only": "context_actor", "no_actor": "no_actor_context", "single": "no_context_single_engram"}
    mode = aliases.get(mode, mode)
    allowed = ("context_actor", "no_actor_context", "no_context_single_engram")
    if mode not in allowed:
        raise ValueError("unsupported multi-context mode: %s" % mode)
    # Four independent deterministic streams come from the frozen manifest.
    organism_seed, environment_seed, schedule_seed, exploration_seed = multi_seed_manifest(seed)
    lengths = multi_phase_lengths_for_seed(schedule_seed, config.phase_lengths)
    environment = UnannouncedMultiContextEnvironment(environment_seed, lengths, schedule_seed=schedule_seed, policy_sequence=policy_sequence)
    organism = _multi_context_organism(organism_seed)
    if mode == "no_context_single_engram":
        # Matched one-engram control: retain the same graph-native starting
        # structure but disable context changes by placing both detector
        # thresholds far beyond any finite run evidence.
        organism.enable_context_modules(max_modules=1)
        organism.context_warning_threshold = 1.0e9
        organism.context_switch_threshold = 1.0e9 + 1.0
    else:
        organism.enable_context_modules(max_modules=3)
    organism.actor_learning_rate = config.actor_learning_rate if mode in ("context_actor", "no_context_single_engram") else 0.0
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    organism.structural_plasticity_enabled = False
    tape_rng = random.Random(exploration_seed)
    exploration_tape = tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon))
    organism.set_exploration_tape(exploration_tape)
    initial_cell_ids = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(s.id for s in organism.graph.iter_synapses())
    observation = environment.reset()
    rewards: List[float] = []
    actions: List[float] = []
    targets: List[float] = []
    active_module_timeline: List[Optional[str]] = []
    for _ in range(environment.horizon):
        target = environment._target(environment.cue, environment.distractor, environment.t)
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        action = result.outputs[0]
        # `step` may consume the preceding reward and switch context before
        # producing this action.  pending_motor_module identifies producer of
        # current action; active_motor_module is only a fallback for legacy
        # non-context runs.
        active_module_timeline.append(
            organism.pending_motor_module if organism.context_enabled else None
        )
        observation = environment.step(action)
        rewards.append(observation.reward)
        actions.append(action)
        targets.append(target)
    organism.apply_outcome(observation.reward)
    windows = multi_phase_windows_for_lengths(environment.phase_lengths)
    phase_metrics = _multi_phase_metrics(actions, targets, windows, config.tail_fraction)
    reactivation_by_segment: Dict[str, Tuple[str, ...]] = {}
    reactivation_lag_by_segment: Dict[str, Tuple[int, ...]] = {}
    recruitment_by_segment: Dict[str, int] = {}
    canonical_by_segment: Dict[str, Optional[str]] = {}
    canonical_dominance_by_segment: Dict[str, float] = {}
    for name, (start, end) in windows.items():
        reactivation_by_segment[name] = tuple(sorted({str(event["module_id"]) for event in organism.events if event.get("kind") == "motor_module_reactivated" and start <= int(event.get("step", -1)) < end}))
        reactivation_lag_by_segment[name] = tuple(int(event["step"]) - start for event in organism.events if event.get("kind") == "motor_module_reactivated" and start <= int(event.get("step", -1)) < end)
        recruitment_by_segment[name] = sum(1 for event in organism.events if event.get("kind") == "motor_module_recruited" and start <= int(event.get("step", -1)) < end)
        tail_start = end - max(1, int((end - start) * config.tail_fraction))
        tail_ids = [identifier for identifier in active_module_timeline[tail_start:end] if identifier is not None]
        if tail_ids:
            counts = {identifier: tail_ids.count(identifier) for identifier in set(tail_ids)}
            canonical, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            dominance = count / float(len(tail_ids))
            canonical_by_segment[name] = canonical if dominance >= 0.70 else None
            canonical_dominance_by_segment[name] = dominance
        else:
            canonical_by_segment[name] = None
            canonical_dominance_by_segment[name] = 0.0
    policy_by_segment = {name: policy for name, policy in zip(windows, policy_sequence)}
    first_occurrence_segment: Dict[str, str] = {}
    for name in windows:
        first_occurrence_segment.setdefault(policy_by_segment[name], name)
    first_canonical = {
        policy: canonical_by_segment[first_occurrence_segment[policy]]
        for policy in sorted(first_occurrence_segment)
    }
    canonical_reuse = {}
    for name in windows:
        first_name = first_occurrence_segment[policy_by_segment[name]]
        if name == first_name:
            continue
        first_module = canonical_by_segment[first_name]
        canonical_reuse[name] = first_module is not None and canonical_by_segment[name] == first_module
    return {
        "seed": int(seed),
        "mode": mode,
        "environment_seed": environment_seed,
        "schedule_seed": schedule_seed,
        "organism_seed": organism_seed,
        "exploration_seed": exploration_seed,
        "phase_lengths": tuple(environment.phase_lengths),
        "rewards": rewards,
        "actions": actions,
        "targets": targets,
        "phase_metrics": phase_metrics,
        "reactivation_by_segment": reactivation_by_segment,
        "reactivation_lag_by_segment": reactivation_lag_by_segment,
        "active_module_timeline": tuple(active_module_timeline),
        "canonical_by_segment": canonical_by_segment,
        "canonical_dominance_by_segment": canonical_dominance_by_segment,
        "policy_by_segment": policy_by_segment,
        "first_occurrence_segment": first_occurrence_segment,
        "return_policy_by_segment": {name: policy_by_segment[name] for name in ("A_return", "B_return")},
        "canonical_first_context": first_canonical,
        "canonical_reuse": canonical_reuse,
        "recruitment_by_segment": recruitment_by_segment,
        "module_count": len(organism.motor_modules),
        "motor_module_ids": tuple(sorted(organism.motor_modules)),
        "organism_steps": organism.step_count,
        "initial_cell_ids": initial_cell_ids,
        "initial_synapse_ids": initial_synapse_ids,
        "final_cell_ids": tuple(sorted(organism.cells)),
        "final_synapse_ids": tuple(s.id for s in organism.graph.iter_synapses()),
        "final_state": organism.state_dict(),
        "environment_state": environment.state_dict(),
    }


def summarize_multicontext(results: Sequence[Mapping[str, object]], config: Optional[MultiContextBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or MultiContextBenchmarkConfig()
    config.validate()
    if not results:
        raise ValueError("multi-context results must not be empty")
    names = ("A_initial", "B_interference", "C_novel", "A_return", "B_return")
    phase_summary = {
        name: {metric: _mean(float(result["phase_metrics"][name][metric]) for result in results) for metric in ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")}
        for name in names
    }
    mean_tail = {name: phase_summary[name]["tail_skill"] for name in names}
    positive_counts = {name: sum(float(result["phase_metrics"][name]["tail_skill"]) > 0.0 for result in results) for name in names}
    return_names = names[3:]
    canonical_reuse_counts = {
        name: sum(bool(result["canonical_reuse"].get(name, False)) for result in results)
        for name in return_names
    }
    canonical_a_reuse = canonical_reuse_counts["A_return"]
    canonical_b_reuse = canonical_reuse_counts["B_return"]
    canonical_three = sum(len({result["canonical_first_context"][policy] for policy in ("A", "B", "C")}) == 3 for result in results)
    final_three = sum(int(result["module_count"]) == 3 for result in results)
    all_lags = [lag for result in results for name in ("A_return", "B_return") for lag in result["reactivation_lag_by_segment"][name]]
    median_lag = sorted(all_lags)[len(all_lags) // 2] if all_lags else None
    per_seed = []
    for result in results:
        metrics = result["phase_metrics"]
        gates = {name: float(metrics[name]["tail_skill"]) > config.tail_skill_floor for name in names}
        gates["capacity"] = int(result["module_count"]) <= 4
        per_seed.append({"seed": result.get("seed"), "gates": gates, "all_passed": all(gates.values())})
    passed = {
        "mean_first_context_tails": mean_tail["A_initial"] >= 0.20 and mean_tail["B_interference"] >= 0.20 and mean_tail["C_novel"] >= 0.20,
        "mean_returned_context_tails": mean_tail["A_return"] >= 0.30 and mean_tail["B_return"] >= 0.30,
        "positive_each_segment": all(positive_counts[name] >= 18 for name in names),
        "canonical_reuse": all(canonical_reuse_counts[name] >= 18 for name in return_names),
        "exactly_three_canonical": canonical_three >= 18,
        "exactly_three_final": final_three >= 20,
        "no_fourth_module": all(int(result["module_count"]) <= 3 for result in results),
        "recovery_median": median_lag is not None and median_lag <= 30,
    }
    return {
        "seeds": len(results),
        "mean_phase_metrics": phase_summary,
        "mean_module_count": _mean(float(result["module_count"]) for result in results),
        "module_reuse_counts": {
            # Keep historical keys for default ABCAB consumers, but define
            # "original" as canonical first-occurrence reuse rather than a
            # hardcoded module ID.
            "%s_original" % name: canonical_reuse_counts[name]
            for name in return_names
        },
        "module_reuse_counts_by_segment": dict(canonical_reuse_counts),
        "return_policy_by_segment": {
            name: results[0]["policy_by_segment"][name] for name in return_names
        },
        "median_reactivation_lag": {
            name: (sorted(lag for result in results for lag in result["reactivation_lag_by_segment"][name])[len([lag for result in results for lag in result["reactivation_lag_by_segment"][name]]) // 2] if any(result["reactivation_lag_by_segment"][name] for result in results) else None)
            for name in ("A_return", "B_return")
        },
        "capacity_exhaustion_runs": sum(int(result["module_count"]) >= 4 for result in results),
        "false_extra_recruitment_runs": sum(int(result["module_count"]) > 3 for result in results),
        "per_seed": per_seed,
        "pass_counts": {name: sum(item["gates"][name] for item in per_seed) for name in names + ("capacity",)},
        "pass_fractions": {name: sum(item["gates"][name] for item in per_seed) / float(len(per_seed)) for name in names + ("capacity",)},
        "positive_tail_counts": positive_counts,
        "canonical_reuse_counts": {"A_return": canonical_a_reuse, "B_return": canonical_b_reuse},
        "canonical_three_counts": canonical_three,
        "final_three_module_counts": final_three,
        "median_reactivation_lag": median_lag,
        "passed": passed,
        "all_passed": all(passed.values()),
        "thresholds": {"first_context_tail_floor": 0.20, "returned_context_tail_floor": 0.30, "positive_segment_runs": 18, "canonical_reuse_runs": 18, "canonical_three_runs": 18, "final_three_runs": 20, "capacity_max_modules": 3, "recovery_median_limit": 30},
    }


def run_multicontext_benchmark(config: Optional[MultiContextBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or MultiContextBenchmarkConfig()
    config.validate()
    actor = [run_multicontext(seed, config, "context_actor") for seed in config.seeds]
    no_actor = [run_multicontext(seed, config, "no_actor_context") for seed in config.seeds]
    no_context = [run_multicontext(seed, config, "no_context_single_engram") for seed in config.seeds]
    actor_summary = summarize_multicontext(actor, config)
    no_actor_summary = summarize_multicontext(no_actor, config)
    no_context_summary = summarize_multicontext(no_context, config)
    permutation_runs = {}
    for order in UnannouncedMultiContextEnvironment.POLICY_ORDERS[1:]:
        order_key = "".join(order)
        permutation_results = [run_multicontext(seed, config, "context_actor", policy_sequence=order) for seed in config.seeds]
        permutation_runs[order_key] = summarize_multicontext(permutation_results, config)
    actor_c = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in actor)
    no_actor_c = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in no_actor)
    no_context_c = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in no_context)
    return {
        "protocol": {"phase_lengths_base": list(config.phase_lengths), "phase_lengths_by_seed": {str(seed): list(multi_phase_lengths_for_seed(multi_seed_manifest(seed)[2], config.phase_lengths)) for seed in config.seeds}, "seeds": list(config.seeds), "exploration": config.exploration, "independent_streams": True, "hidden_context_boundaries": True, "policy_orders": [list(order) for order in UnannouncedMultiContextEnvironment.POLICY_ORDERS], "acceptance_manifest": [list(entry) for entry in MULTI_ACCEPTANCE_MANIFEST]},
        "context_actor": actor_summary,
        "heldout_policy_orders": permutation_runs,
        "no_actor_context": no_actor_summary,
        "no_context_single_engram": no_context_summary,
        "comparison": {"a_return_tail_skill_delta_vs_no_actor": actor_c - no_actor_c, "a_return_tail_skill_delta_vs_no_context": actor_c - no_context_c, "matched_seed_and_protocol": True, "matched_starting_structure": all(a["initial_cell_ids"] == b["initial_cell_ids"] == c["initial_cell_ids"] for a, b, c in zip(actor, no_actor, no_context))},
        "diagnostic": "PASS" if actor_summary["all_passed"] else "MULTI_CONTEXT_STRESS_NOT_ESTABLISHED",
    }


# ---------------------------------------------------------------------------
# Nonlinear representation-learning benchmark


@dataclass
class NonlinearBenchmarkConfig:
    """Frozen nonlinear protocol for the graph-native dendritic milestone.

    Main mode uses hidden-afferent motor modules and graph-native product
    cells; local node perturbation and raw-input controls remain explicit.
    """

    phase_lengths: Tuple[int, int, int, int, int] = (264, 264, 264, 264, 264)
    seeds: Tuple[int, ...] = tuple(range(24))
    exploration: float = 0.20
    novelty: float = 0.10
    actor_learning_rate: float = 0.10
    tail_fraction: float = 0.25
    tail_skill_floor: float = 0.15
    tail_sign_accuracy_floor: float = 0.75
    return_early_skill_floor: float = 0.15
    return_tail_skill_floor: float = 0.20
    canonical_reuse_min_runs: int = 18
    reactivation_lag_limit: int = 30
    comparison_margin: float = 0.10
    max_modules: int = 3
    stationary_safety_min_runs: int = 20
    representation_learning_rate: float = 0.02
    representation_noise: float = 0.20

    def validate(self) -> None:
        if len(self.phase_lengths) != 5 or any(int(length) < 8 or int(length) % 8 for length in self.phase_lengths):
            raise ValueError("nonlinear phase_lengths must be five positive multiples of 8")
        if not self.seeds:
            raise ValueError("at least one nonlinear benchmark seed is required")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("exploration and novelty must be in [0, 1]")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if not math.isfinite(self.actor_learning_rate) or self.actor_learning_rate < 0.0:
            raise ValueError("actor_learning_rate must be finite and nonnegative")
        if not math.isfinite(self.representation_learning_rate) or self.representation_learning_rate < 0.0:
            raise ValueError("representation_learning_rate must be finite and nonnegative")
        if not math.isfinite(self.representation_noise) or not 0.0 <= self.representation_noise <= 1.0:
            raise ValueError("representation_noise must be finite and in [0, 1]")
        for name in ("tail_skill_floor", "tail_sign_accuracy_floor", "return_early_skill_floor", "return_tail_skill_floor", "comparison_margin"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError("%s must be finite" % name)
        if int(self.canonical_reuse_min_runs) < 1 or int(self.reactivation_lag_limit) < 0:
            raise ValueError("nonlinear gate counts must be nonnegative")
        if int(self.max_modules) != 3:
            raise ValueError("nonlinear protocol requires exactly three context modules")
        if int(self.stationary_safety_min_runs) < 1:
            raise ValueError("stationary_safety_min_runs must be positive")


NONLINEAR_POLICY_SEQUENCE = ("A", "B", "C", "A", "B")
NONLINEAR_POLICY_PAIRS = {"A": (0, 1), "B": (1, 2), "C": (2, 0)}
# Reuse the existing frozen independent seed tuples.  The red benchmark uses
# the same four independent streams: organism, input, schedule, exploration.
NONLINEAR_ACCEPTANCE_MANIFEST: Tuple[Tuple[int, int, int, int], ...] = MULTI_ACCEPTANCE_MANIFEST


def nonlinear_seed_manifest(seed: int) -> Tuple[int, int, int, int]:
    if 0 <= int(seed) < len(NONLINEAR_ACCEPTANCE_MANIFEST):
        return NONLINEAR_ACCEPTANCE_MANIFEST[int(seed)]
    raise ValueError("seed is not present in the frozen nonlinear manifest")


def nonlinear_representation_seed(seed: int) -> int:
    """Independent deterministic stream for hidden node perturbations."""
    if not 0 <= int(seed) < len(NONLINEAR_ACCEPTANCE_MANIFEST):
        raise ValueError("seed is not present in the frozen nonlinear manifest")
    return 1700003 + 104729 * (int(seed) + 1)


def nonlinear_phase_lengths_for_seed(seed: int, base: Sequence[int] = (264, 264, 264, 264, 264)) -> Tuple[int, int, int, int, int]:
    values = tuple(int(length) for length in base)
    if len(values) != 5 or any(length < 8 or length % 8 for length in values):
        raise ValueError("nonlinear base lengths must be five positive multiples of 8")
    schedule_rng = random.Random((int(seed) + 1) * 2000003 + 15401)
    blocks = 6 if min(values) >= 216 else 4
    return tuple(max(8, length + 8 * schedule_rng.randint(-blocks, blocks)) for length in values)


class NonlinearMultiContextEnvironment:
    """Phase-blind balanced parity task with hidden A -> B -> C -> A -> B."""

    POLICY_SEQUENCE = NONLINEAR_POLICY_SEQUENCE

    def __init__(
        self,
        seed: int = 0,
        phase_lengths: Optional[Sequence[int]] = None,
        schedule_seed: Optional[int] = None,
        policy_sequence: Sequence[str] = POLICY_SEQUENCE,
    ) -> None:
        self.seed = int(seed)
        self.schedule_seed = int(seed if schedule_seed is None else schedule_seed)
        self.policy_sequence = tuple(str(policy) for policy in policy_sequence)
        if self.policy_sequence != NONLINEAR_POLICY_SEQUENCE:
            raise ValueError("nonlinear policy sequence must be A, B, C, A, B")
        if phase_lengths is None:
            phase_lengths = nonlinear_phase_lengths_for_seed(self.schedule_seed)
        self.phase_lengths = tuple(int(length) for length in phase_lengths)
        if len(self.phase_lengths) != 5 or any(length < 8 or length % 8 for length in self.phase_lengths):
            raise ValueError("nonlinear phase_lengths must be five positive multiples of 8")
        self.horizon = sum(self.phase_lengths)
        self.t = 0
        self.last_reward = 0.0
        self.inputs: Tuple[Tuple[float, ...], ...] = ()
        self.reset()

    def reset(self) -> Observation:
        # Every block contains each of the eight sign states exactly once;
        # only block order is randomized.  No phase information is encoded.
        # Input ordering has its own stream.  schedule_seed controls only
        # phase lengths, so changing hidden phase boundaries cannot alter the
        # observed sensor sequence.
        input_rng = random.Random(self.seed + 1618033)
        states = tuple(tuple(float(1 if (mask >> index) & 1 else -1) for index in range(3)) for mask in range(8))
        generated: List[Tuple[float, ...]] = []
        for _ in range(self.horizon // 8):
            block = list(states)
            input_rng.shuffle(block)
            generated.extend(block)
        self.inputs = tuple(generated)
        self.t = 0
        self.last_reward = 0.0
        return Observation(self.inputs[0], 0.0, False)

    def _phase_index(self, transition: int) -> int:
        boundary = 0
        for index, length in enumerate(self.phase_lengths):
            boundary += length
            if transition < boundary:
                return index
        return len(self.phase_lengths) - 1

    def target_at(self, transition: Optional[int] = None) -> float:
        index = self.t if transition is None else int(transition)
        values = self.inputs[index]
        left, right = NONLINEAR_POLICY_PAIRS[self.policy_sequence[self._phase_index(index)]]
        return 0.8 * values[left] * values[right]

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise ValueError("environment is done; call reset before stepping")
        action = max(-1.0, min(1.0, float(action)))
        target = self.target_at()
        reward = max(-1.0, min(1.0, (target * target - (action - target) ** 2) / 4.0))
        self.last_reward = reward
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "seed": self.seed,
            "schedule_seed": self.schedule_seed,
            "policy_sequence": list(self.policy_sequence),
            "phase_lengths": list(self.phase_lengths),
            "horizon": self.horizon,
            "t": self.t,
            "last_reward": self.last_reward,
            "inputs": [list(values) for values in self.inputs],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "NonlinearMultiContextEnvironment":
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported nonlinear environment checkpoint version")
        environment = cls(
            int(state["seed"]),
            tuple(int(value) for value in state["phase_lengths"]),
            schedule_seed=int(state.get("schedule_seed", state["seed"])),
            policy_sequence=tuple(state.get("policy_sequence", NONLINEAR_POLICY_SEQUENCE)),
        )
        inputs = tuple(tuple(float(value) for value in values) for values in state.get("inputs", ()))
        if len(inputs) != environment.horizon or any(len(values) != 3 for values in inputs):
            raise ValueError("nonlinear environment checkpoint inputs are invalid")
        environment.inputs = inputs
        environment.t = int(state["t"])
        environment.last_reward = float(state.get("last_reward", 0.0))
        return environment


def nonlinear_phase_windows_for_lengths(lengths: Sequence[int]) -> Dict[str, Tuple[int, int]]:
    values = tuple(int(length) for length in lengths)
    if len(values) != 5 or any(length < 8 or length % 8 for length in values):
        raise ValueError("nonlinear lengths must be five positive multiples of 8")
    names = ("A_initial", "B_interference", "C_novel", "A_return", "B_return")
    windows: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, length in zip(names, values):
        windows[name] = (start, start + length)
        start += length
    return windows


def nonlinear_affine_proof() -> Dict[str, object]:
    """Return exact orthogonality certificate for raw affine policies."""
    states = [tuple(1 if (mask >> index) & 1 else -1 for index in range(3)) for mask in range(8)]
    coefficients: Dict[str, Tuple[float, ...]] = {}
    skills: Dict[str, float] = {}
    projections: Dict[str, Tuple[float, ...]] = {}
    for policy, (left, right) in NONLINEAR_POLICY_PAIRS.items():
        # Compute parity moments as integer sums first.  This keeps the
        # certificate exactly zero on every supported Python version rather
        # than exposing cancellation noise such as 2e-16.
        parity = [values[left] * values[right] for values in states]
        features = [(1,) + values for values in states]
        moments = tuple(sum(row[index] * value for row, value in zip(features, parity)) for index in range(4))
        coeff = tuple(0.0 if moment == 0 else 0.8 * moment / 8.0 for moment in moments)
        zero = 8 * 0.8 * 0.8
        error = zero
        coefficients[policy] = coeff
        projections[policy] = tuple(0.0 if moment == 0 else 0.8 * moment for moment in moments)
        skills[policy] = 0.0 if error == zero else 1.0 - error / zero
    return {
        "states": states,
        "policies": {name: list(pair) for name, pair in NONLINEAR_POLICY_PAIRS.items()},
        "coefficients": coefficients,
        "feature_target_projections": projections,
        "optimal_affine_skill": skills,
        "max_abs_projection": max(abs(value) for values in projections.values() for value in values),
    }


def _make_nonlinear_raw_organism(seed: int, afferent_kind: str = "input") -> Organism:
    """Create nonlinear benchmark organism with selected motor afferents."""
    organism = Organism.create_default(input_size=3, hidden_size=8, output_size=1, seed=seed)
    organism.enable_context_modules(max_modules=3, afferent_kind=afferent_kind)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    return organism


def _make_nonlinear_dendritic_organism(seed: int) -> Organism:
    """Create the benchmark's explicit second-order sensory bank.

    This is a graph-native pair-cell primitive, not evaluator-provided target
    features.  It is intentionally reported separately from learned node
    perturbation because pair cells are an architectural inductive bias.
    """
    organism = Organism.create_dendritic_pair_bank(input_size=3, hidden_size=8, output_size=1, seed=seed)
    organism.enable_context_modules(max_modules=3, afferent_kind="hidden")
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    return organism


def _nonlinear_stationary_safety(seed: int, config: NonlinearBenchmarkConfig) -> Dict[str, object]:
    """Run one stationary-A slice to detect false context alarms."""
    organism_seed, environment_seed, schedule_seed, exploration_seed = nonlinear_seed_manifest(seed)
    length = config.phase_lengths[0]
    environment = NonlinearMultiContextEnvironment(
        environment_seed,
        (length,) * 5,
        schedule_seed=schedule_seed,
    )
    organism = _make_nonlinear_dendritic_organism(organism_seed)
    organism.actor_learning_rate = config.actor_learning_rate
    tape_rng = random.Random(exploration_seed + 7919)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(length)))
    observation = environment.reset()
    for _ in range(length):
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        observation = environment.step(result.outputs[0])
    organism.apply_outcome(observation.reward)
    warning_kinds = {"motor_warning_started", "motor_warning_escalated", "motor_module_reactivated", "motor_module_probe_started"}
    warning_events = sum(event.get("kind") in warning_kinds for event in organism.events)
    return {
        "safe": len(organism.motor_modules) == 1 and warning_events == 0,
        "module_count": len(organism.motor_modules),
        "warning_events": warning_events,
        "recruitments": sum(event.get("kind") == "motor_module_recruited" for event in organism.events),
    }


def _first_canonical_reactivation_lag(
    events: Sequence[Mapping[str, object]],
    start: int,
    end: int,
    module_id: Optional[str],
) -> Tuple[int, ...]:
    """Return one lag for the first correct canonical reactivation only."""
    if module_id is None:
        return ()
    candidates = [
        int(event["step"]) - start
        for event in events
        if event.get("kind") == "motor_module_reactivated"
        and event.get("module_id") == module_id
        and start <= int(event.get("step", -1)) < end
    ]
    return (min(candidates),) if candidates else ()


def _nonlinear_mode_run(seed: int, config: NonlinearBenchmarkConfig, mode: str) -> Dict[str, object]:
    aliases = {"actor": "raw_linear_modules", "no_actor_context": "no_actor", "none": "frozen"}
    mode = aliases.get(mode, mode)
    if mode not in ("raw_linear_modules", "hidden_node_perturbation", "hidden_no_node_perturbation", "hidden_no_actor", "dendritic_representation", "no_actor", "frozen"):
        raise ValueError("unsupported nonlinear mode: %s" % mode)
    organism_seed, environment_seed, schedule_seed, exploration_seed = nonlinear_seed_manifest(seed)
    lengths = nonlinear_phase_lengths_for_seed(schedule_seed, config.phase_lengths)
    environment = NonlinearMultiContextEnvironment(environment_seed, lengths, schedule_seed=schedule_seed)
    hidden_mode = mode in ("hidden_node_perturbation", "hidden_no_node_perturbation", "hidden_no_actor", "dendritic_representation")
    organism = _make_nonlinear_dendritic_organism(organism_seed) if mode == "dendritic_representation" else _make_nonlinear_raw_organism(organism_seed, afferent_kind="hidden" if hidden_mode else "input")
    organism.actor_learning_rate = config.actor_learning_rate if mode in ("raw_linear_modules", "hidden_node_perturbation", "hidden_no_node_perturbation", "dendritic_representation") else 0.0
    if mode in ("hidden_node_perturbation", "hidden_no_actor"):
        organism.enable_representation_learning(config.representation_learning_rate, config.representation_noise)
    representation_seed = nonlinear_representation_seed(seed)
    if mode in ("hidden_node_perturbation", "hidden_no_actor"):
        representation_rng = random.Random(representation_seed)
        organism.set_representation_tape(tuple(representation_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon * len(organism._representation_hidden_ids()))))
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    initial_cell_ids = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(s.id for s in organism.graph.iter_synapses())
    observation = environment.reset()
    actions: List[float] = []
    targets: List[float] = []
    rewards: List[float] = []
    active_module_timeline: List[Optional[str]] = []
    for _ in range(environment.horizon):
        target = environment.target_at()
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        action = result.outputs[0]
        active_module_timeline.append(organism.pending_motor_module)
        observation = environment.step(action)
        actions.append(action)
        targets.append(target)
        rewards.append(observation.reward)
    organism.apply_outcome(observation.reward)
    windows = nonlinear_phase_windows_for_lengths(environment.phase_lengths)
    phase_metrics = _multi_phase_metrics(actions, targets, windows, config.tail_fraction)
    canonical_by_segment: Dict[str, Optional[str]] = {}
    for name, (start, end) in windows.items():
        tail_start = end - max(1, int((end - start) * config.tail_fraction))
        tail_ids = [identifier for identifier in active_module_timeline[tail_start:end] if identifier is not None]
        if tail_ids:
            counts = {identifier: tail_ids.count(identifier) for identifier in set(tail_ids)}
            canonical, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            canonical_by_segment[name] = canonical if count / float(len(tail_ids)) >= 0.70 else None
        else:
            canonical_by_segment[name] = None
    policy_by_segment = {name: policy for name, policy in zip(windows, NONLINEAR_POLICY_SEQUENCE)}
    first_occurrence: Dict[str, str] = {}
    for name in windows:
        first_occurrence.setdefault(policy_by_segment[name], name)
    canonical_reuse = {}
    for name in windows:
        first = first_occurrence[policy_by_segment[name]]
        if name != first:
            canonical_reuse[name] = canonical_by_segment[name] is not None and canonical_by_segment[name] == canonical_by_segment[first]
    lags = {}
    for name in ("A_return", "B_return"):
        start = windows[name][0]
        policy = policy_by_segment[name]
        first = first_occurrence[policy]
        first_module = canonical_by_segment[first]
        # Keep the historical tuple-shaped field, but record only the first
        # correct canonical reactivation for this returned segment.  Repeated
        # churn by the same module must not inflate the recovery-lag metric.
        lags[name] = _first_canonical_reactivation_lag(
            organism.events,
            start,
            windows[name][1],
            first_module,
        )
    direct_edges = [
        synapse.id for synapse in organism.graph.iter_synapses()
        if synapse.source in organism.input_ids
        and organism.cells[synapse.destination].kind == "motor_module"
    ]
    return {
        "seed": int(seed),
        "mode": mode,
        "organism_seed": organism_seed,
        "environment_seed": environment_seed,
        "schedule_seed": schedule_seed,
        "exploration_seed": exploration_seed,
        "representation_seed": representation_seed,
        "stationary_safety": _nonlinear_stationary_safety(seed, config) if mode == "dendritic_representation" else None,
        "phase_lengths": tuple(environment.phase_lengths),
        "actions": actions,
        "targets": targets,
        "rewards": rewards,
        "phase_metrics": phase_metrics,
        "canonical_by_segment": canonical_by_segment,
        "canonical_reuse": canonical_reuse,
        "reactivation_lag_by_segment": lags,
        "active_module_timeline": tuple(active_module_timeline),
        "module_count": len(organism.motor_modules),
        "direct_input_motor_edges": tuple(direct_edges),
        "initial_cell_ids": initial_cell_ids,
        "initial_synapse_ids": initial_synapse_ids,
        "final_cell_ids": tuple(sorted(organism.cells)),
        "final_synapse_ids": tuple(s.id for s in organism.graph.iter_synapses()),
        "final_state": organism.state_dict(),
        "environment_state": environment.state_dict(),
    }


def _summarize_nonlinear(results: Sequence[Mapping[str, object]], config: NonlinearBenchmarkConfig) -> Dict[str, object]:
    names = ("A_initial", "B_interference", "C_novel", "A_return", "B_return")
    mode = str(results[0].get("mode", ""))
    phase_summary = {
        name: {metric: _mean(float(result["phase_metrics"][name][metric]) for result in results) for metric in ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")}
        for name in names
    }
    reuse_counts = {name: sum(bool(result["canonical_reuse"].get(name, False)) for result in results) for name in ("A_return", "B_return")}
    lags = [lag for result in results for name in ("A_return", "B_return") for lag in result["reactivation_lag_by_segment"][name]]
    median_lag = sorted(lags)[len(lags) // 2] if lags else None
    passed = {
        "first_context_tails": all(phase_summary[name]["tail_skill"] >= config.tail_skill_floor for name in ("A_initial", "B_interference", "C_novel")),
        "first_context_sign": all(phase_summary[name]["tail_sign_accuracy"] >= config.tail_sign_accuracy_floor for name in ("A_initial", "B_interference", "C_novel")),
        "returned_context_tails": all(phase_summary[name]["tail_skill"] >= config.return_tail_skill_floor for name in ("A_return", "B_return")),
        "returned_context_early": all(phase_summary[name]["early_skill"] >= config.return_early_skill_floor for name in ("A_return", "B_return")),
        "canonical_reuse": all(reuse_counts[name] >= config.canonical_reuse_min_runs for name in reuse_counts),
        "reactivation_lag": median_lag is not None and median_lag <= config.reactivation_lag_limit,
        "capacity": all(int(result["module_count"]) <= config.max_modules for result in results),
        "zero_direct_input_motor": all(not result["direct_input_motor_edges"] for result in results),
    }
    if mode == "dendritic_representation":
        safety_runs = sum(bool(result.get("stationary_safety", {}).get("safe", False)) for result in results)
        passed["stationary_safety"] = safety_runs >= config.stationary_safety_min_runs
    else:
        safety_runs = None
    return {
        "seeds": len(results),
        "mean_phase_metrics": phase_summary,
        "mean_module_count": _mean(float(result["module_count"]) for result in results),
        "canonical_reuse_counts": reuse_counts,
        "median_reactivation_lag": median_lag,
        "direct_input_motor_edge_runs": sum(bool(result["direct_input_motor_edges"]) for result in results),
        "stationary_safety_runs": safety_runs,
        "passed": passed,
        "all_passed": all(passed.values()),
        "thresholds": {
            "tail_skill_floor": config.tail_skill_floor,
            "tail_sign_accuracy_floor": config.tail_sign_accuracy_floor,
            "return_early_skill_floor": config.return_early_skill_floor,
            "return_tail_skill_floor": config.return_tail_skill_floor,
            "canonical_reuse_min_runs": config.canonical_reuse_min_runs,
            "reactivation_lag_limit": config.reactivation_lag_limit,
            "max_modules": config.max_modules,
            "stationary_safety_min_runs": config.stationary_safety_min_runs,
        },
    }


def _run_dendritic_heldout(seed: int, kind: str) -> Tuple[float, float]:
    """Evaluate pair cells on diagnostics outside the locked primary task."""
    organism_seed, _, _, exploration_seed = nonlinear_seed_manifest(seed)
    organism = _make_nonlinear_dendritic_organism(organism_seed)
    organism.context_max_modules = 1
    organism.context_detector_min_evidence = 10 ** 9
    organism.context_warning_threshold = 10 ** 9
    organism.context_switch_threshold = 10 ** 9 + 1.0
    horizon = 1000 if kind == "sensor_pair_permutation" else 5000
    exploration_rng = random.Random(exploration_seed + 7919)
    organism.set_exploration_tape(tuple(exploration_rng.uniform(-1.0, 1.0) for _ in range(horizon)))
    state_rng = random.Random(organism_seed + 1618033)
    states: List[Tuple[float, ...]] = []
    if kind == "sensor_pair_permutation":
        base_states = [tuple(float(1 if (mask >> index) & 1 else -1) for index in range(3)) for mask in range(8)]
        for _ in range((horizon + 7) // 8):
            block = list(base_states)
            state_rng.shuffle(block)
            states.extend((values[2], values[0], values[1]) for values in block)
        states = states[:horizon]
        targets = [0.8 * values[1] * values[2] for values in states]
    elif kind == "continuous_product":
        states = [tuple(state_rng.uniform(-1.0, 1.0) for _ in range(3)) for _ in range(horizon)]
        targets = [0.8 * values[0] * values[2] for values in states]
    else:
        raise ValueError("unsupported dendritic heldout diagnostic: %s" % kind)
    actions: List[float] = []
    reward = 0.0
    for values, target in zip(states, targets):
        result = organism.step(values, Modulators(reward=reward, novelty=0.10, exploration=0.20))
        action = result.outputs[0]
        actions.append(action)
        reward = max(-1.0, min(1.0, (target * target - (action - target) ** 2) / 4.0))
    organism.apply_outcome(reward)
    tail = max(1, horizon // 4)
    zero = sum(target * target for target in targets[-tail:])
    skill = 1.0 - sum((action - target) ** 2 for action, target in zip(actions[-tail:], targets[-tail:])) / zero
    sign = _sign_accuracy(actions, targets, horizon - tail, horizon)
    return skill, sign


def _dendritic_heldout_diagnostics(config: NonlinearBenchmarkConfig) -> Dict[str, object]:
    seeds = tuple(config.seeds[: min(6, len(config.seeds))])
    report: Dict[str, object] = {"seeds": list(seeds), "declared_primary": False}
    for kind in ("sensor_pair_permutation", "continuous_product"):
        values = [_run_dendritic_heldout(seed, kind) for seed in seeds]
        report[kind] = {
            "mean_tail_skill": _mean(value[0] for value in values),
            "mean_tail_sign_accuracy": _mean(value[1] for value in values),
            "positive_runs": sum(value[0] > 0.0 for value in values),
        }
    return report


def run_nonlinear(seed: int, config: Optional[NonlinearBenchmarkConfig] = None, mode: str = "raw_linear_modules") -> Dict[str, object]:
    config = config or NonlinearBenchmarkConfig()
    config.validate()
    return _nonlinear_mode_run(seed, config, mode)


def run_nonlinear_benchmark(config: Optional[NonlinearBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or NonlinearBenchmarkConfig()
    config.validate()
    modes = {mode: [_nonlinear_mode_run(seed, config, mode) for seed in config.seeds] for mode in ("raw_linear_modules", "hidden_node_perturbation", "hidden_no_node_perturbation", "hidden_no_actor", "dendritic_representation", "no_actor", "frozen")}
    summaries = {mode: _summarize_nonlinear(results, config) for mode, results in modes.items()}
    raw_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in modes["raw_linear_modules"])
    no_actor_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in modes["no_actor"])
    main_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in modes["hidden_node_perturbation"])
    dendritic_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in modes["dendritic_representation"])
    no_node_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in modes["hidden_no_node_perturbation"])
    matched = all(
        first["initial_cell_ids"] == second["initial_cell_ids"] == third["initial_cell_ids"]
        and first["initial_synapse_ids"] == second["initial_synapse_ids"] == third["initial_synapse_ids"]
        for first, second, third in zip(modes["raw_linear_modules"], modes["no_actor"], modes["frozen"])
    )
    hidden_matched = all(
        first["initial_cell_ids"] == second["initial_cell_ids"] == third["initial_cell_ids"]
        and first["initial_synapse_ids"] == second["initial_synapse_ids"] == third["initial_synapse_ids"]
        for first, second, third in zip(modes["hidden_node_perturbation"], modes["hidden_no_node_perturbation"], modes["hidden_no_actor"])
    )
    return {
        "protocol": {
            "phase_lengths_base": list(config.phase_lengths),
            "phase_lengths_by_seed": {str(seed): list(nonlinear_phase_lengths_for_seed(nonlinear_seed_manifest(seed)[2], config.phase_lengths)) for seed in config.seeds},
            "seeds": list(config.seeds),
            "acceptance_manifest": [list(entry) for entry in NONLINEAR_ACCEPTANCE_MANIFEST],
            "representation_seeds": {str(seed): nonlinear_representation_seed(seed) for seed in config.seeds},
            "policy_sequence": list(NONLINEAR_POLICY_SEQUENCE),
            "policies": {name: list(pair) for name, pair in NONLINEAR_POLICY_PAIRS.items()},
            "balanced_block_size": 8,
            "balanced_states": [list(values) for values in nonlinear_affine_proof()["states"]],
            "linear_proof": nonlinear_affine_proof(),
            "future_main_gates": summaries["dendritic_representation"]["thresholds"],
            "hidden_phase_boundaries": True,
        },
        "raw_linear_modules": summaries["raw_linear_modules"],
        "hidden_node_perturbation": summaries["hidden_node_perturbation"],
        "hidden_no_node_perturbation": summaries["hidden_no_node_perturbation"],
        "hidden_no_actor": summaries["hidden_no_actor"],
        "dendritic_representation": summaries["dendritic_representation"],
        "no_actor": summaries["no_actor"],
        "frozen": summaries["frozen"],
        "heldout_diagnostics": _dendritic_heldout_diagnostics(config),
        "comparison": {
            "matched_seed_and_protocol": True,
            "matched_starting_structure": matched,
            "hidden_matched_starting_structure": hidden_matched,
            "raw_return_tail_skill_minus_no_actor": raw_score - no_actor_score,
            "raw_beats_no_actor_by_margin": raw_score >= no_actor_score + config.comparison_margin,
            "main_return_tail_skill_minus_raw": dendritic_score - raw_score,
            "main_return_tail_skill_minus_hidden_no_node": dendritic_score - no_node_score,
            "main_beats_raw_by_margin": dendritic_score >= raw_score + config.comparison_margin,
            "main_beats_hidden_no_node_by_margin": dendritic_score >= no_node_score + config.comparison_margin,
        },
        "diagnostic": "PASS" if summaries["dendritic_representation"]["all_passed"] and dendritic_score >= raw_score + config.comparison_margin and dendritic_score >= no_node_score + config.comparison_margin else "NONLINEAR_REPRESENTATION_NOT_ESTABLISHED",
    }


# ---------------------------------------------------------------------------
# v9 adaptive-dendritic benchmark (neutral fingerprint router)


ADAPTIVE_DENDRITIC_POLICY_SEQUENCE = ("A", "B", "C", "A", "B")
ADAPTIVE_DENDRITIC_ALL_PAIRS: Tuple[Tuple[int, int], ...] = tuple(combinations(range(6), 2))
# The task-pair stream is evaluator-owned and frozen.  Its construction is
# independent of every organism/input/schedule/exploration stream; the
# organism receives none of these pairs.
ADAPTIVE_DENDRITIC_TASK_PAIR_MANIFEST: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple(
    tuple(ADAPTIVE_DENDRITIC_ALL_PAIRS[(3 * index + offset) % len(ADAPTIVE_DENDRITIC_ALL_PAIRS)] for offset in range(3))
    for index in range(24)
)
ADAPTIVE_DENDRITIC_ACCEPTANCE_MANIFEST: Tuple[Tuple[int, int, int, int, int], ...] = (
    (5201, 8203, 10211, 13217, 16223), (5203, 8209, 10223, 13229, 16231),
    (5209, 8219, 10243, 13241, 16249), (5213, 8221, 10247, 13259, 16261),
    (5219, 8231, 10253, 13267, 16273), (5227, 8233, 10259, 13291, 16279),
    (5231, 8237, 10267, 13309, 16283), (5233, 8243, 10271, 13313, 16289),
    (5237, 8249, 10273, 13327, 16297), (5261, 8251, 10289, 13331, 16301),
    (5273, 8263, 10301, 13337, 16319), (5279, 8269, 10303, 13339, 16333),
    (5281, 8273, 10313, 13367, 16339), (5297, 8279, 10321, 13381, 16349),
    (5303, 8281, 10331, 13391, 16361), (5309, 8291, 10333, 13411, 16363),
    (5323, 8293, 10337, 13417, 16369), (5333, 8297, 10343, 13421, 16381),
    (5347, 8311, 10357, 13427, 16399), (5351, 8317, 10369, 13441, 16411),
    (5381, 8321, 10391, 13451, 16417), (5387, 8323, 10399, 13457, 16421),
    (5393, 8329, 10427, 13463, 16427), (5399, 8387, 10429, 13469, 16433),
)


@dataclass
class AdaptiveDendriticBenchmarkConfig:
    """Frozen v9 evaluator protocol for adaptive local pair search."""

    phase_lengths: Tuple[int, int, int, int, int] = (256, 256, 256, 256, 256)
    seeds: Tuple[int, ...] = tuple(range(24))
    exploration: float = 0.20
    novelty: float = 0.10
    actor_learning_rate: float = 0.10
    tail_fraction: float = 0.25
    tail_skill_floor: float = 0.15
    tail_sign_accuracy_floor: float = 0.75
    return_early_skill_floor: float = 0.15
    return_tail_skill_floor: float = 0.20
    canonical_reuse_min_runs: int = 18
    reactivation_lag_limit: int = 30
    comparison_margin: float = 0.10
    max_modules: int = 3
    stationary_safety_min_runs: int = 20

    def validate(self) -> None:
        if len(self.phase_lengths) != 5:
            raise ValueError("adaptive-dendritic phase_lengths must be five positive multiples of 64")
        for length in self.phase_lengths:
            if isinstance(length, bool) or not math.isfinite(float(length)) or int(length) != float(length) or int(length) < 64 or int(length) % 64:
                raise ValueError("adaptive-dendritic phase_lengths must be five positive multiples of 64")
        if not self.seeds:
            raise ValueError("at least one adaptive-dendritic seed is required")
        finite_fields = (
            "exploration", "novelty", "actor_learning_rate", "tail_fraction",
            "tail_skill_floor", "tail_sign_accuracy_floor", "return_early_skill_floor",
            "return_tail_skill_floor", "comparison_margin",
        )
        if any(not math.isfinite(float(getattr(self, name))) for name in finite_fields):
            raise ValueError("adaptive-dendritic thresholds and rates must be finite")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("exploration and novelty must be in [0, 1]")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if self.actor_learning_rate < 0.0:
            raise ValueError("actor_learning_rate must be finite and nonnegative")
        for name in ("canonical_reuse_min_runs", "reactivation_lag_limit", "max_modules", "stationary_safety_min_runs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("adaptive-dendritic count limits must be integers")
        if self.canonical_reuse_min_runs < 1 or self.reactivation_lag_limit < 0:
            raise ValueError("adaptive-dendritic reuse and lag limits are invalid")
        if self.max_modules != 3 or self.stationary_safety_min_runs < 1:
            raise ValueError("adaptive-dendritic protocol requires three modules and positive safety runs")


def adaptive_dendritic_seed_manifest(seed: int) -> Tuple[int, int, int, int, int]:
    if 0 <= int(seed) < len(ADAPTIVE_DENDRITIC_ACCEPTANCE_MANIFEST):
        return ADAPTIVE_DENDRITIC_ACCEPTANCE_MANIFEST[int(seed)]
    raise ValueError("seed is not present in the frozen adaptive-dendritic manifest")


def adaptive_dendritic_task_pairs(task_seed: int) -> Tuple[Tuple[int, int], ...]:
    """Return evaluator-only task pairs from the independent frozen stream."""
    return ADAPTIVE_DENDRITIC_TASK_PAIR_MANIFEST[int(task_seed) % len(ADAPTIVE_DENDRITIC_TASK_PAIR_MANIFEST)]


def adaptive_dendritic_phase_lengths_for_seed(seed: int, base: Sequence[int] = (256, 256, 256, 256, 256)) -> Tuple[int, int, int, int, int]:
    values = tuple(int(length) for length in base)
    if len(values) != 5 or any(length < 64 or length % 64 for length in values):
        raise ValueError("adaptive-dendritic base lengths must be five multiples of 64")
    schedule_rng = random.Random((int(seed) + 1) * 3000017 + 17389)
    return tuple(max(64, length + 64 * schedule_rng.randint(-2, 2)) for length in values)


class AdaptiveDendriticEnvironment:
    """Phase-blind six-sensor pair task; task pairs stay evaluator-owned."""

    def __init__(
        self,
        seed: int = 0,
        phase_lengths: Optional[Sequence[int]] = None,
        schedule_seed: Optional[int] = None,
        task_seed: Optional[int] = None,
        task_pairs: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        self.seed = int(seed)
        self.schedule_seed = int(seed if schedule_seed is None else schedule_seed)
        self.task_seed = int(seed if task_seed is None else task_seed)
        phase_lengths = phase_lengths if phase_lengths is not None else adaptive_dendritic_phase_lengths_for_seed(self.schedule_seed)
        self.phase_lengths = tuple(int(length) for length in phase_lengths)
        if len(self.phase_lengths) != 5 or any(length < 64 or length % 64 for length in self.phase_lengths):
            raise ValueError("adaptive-dendritic phase_lengths must be five multiples of 64")
        raw_pairs = task_pairs if task_pairs is not None else adaptive_dendritic_task_pairs(self.task_seed)
        self.task_pairs = tuple(tuple(sorted(int(value) for value in pair)) for pair in raw_pairs)
        if len(self.task_pairs) != 3 or len(set(self.task_pairs)) != 3 or any(
            len(pair) != 2 or pair[0] == pair[1] or any(value < 0 or value >= 6 for value in pair)
            for pair in self.task_pairs
        ):
            raise ValueError("task_pairs must contain three distinct valid sensor pairs")
        self.policy_sequence = ADAPTIVE_DENDRITIC_POLICY_SEQUENCE
        self.horizon = sum(self.phase_lengths)
        self.t = 0
        self.last_reward = 0.0
        self.inputs: Tuple[Tuple[float, ...], ...] = ()
        self.reset()

    def reset(self) -> Observation:
        input_rng = random.Random(self.seed + 2654435761)
        states = tuple(tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64))
        generated: List[Tuple[float, ...]] = []
        for _ in range(self.horizon // 64):
            block = list(states)
            input_rng.shuffle(block)
            generated.extend(block)
        self.inputs = tuple(generated)
        self.t = 0
        self.last_reward = 0.0
        return Observation(self.inputs[0], 0.0, False)

    def _phase_index(self, transition: int) -> int:
        boundary = 0
        for index, length in enumerate(self.phase_lengths):
            boundary += length
            if transition < boundary:
                return index
        return len(self.phase_lengths) - 1

    def target_at(self, transition: Optional[int] = None) -> float:
        index = self.t if transition is None else int(transition)
        if index < 0 or index >= self.horizon:
            raise ValueError("adaptive-dendritic transition is outside environment horizon")
        left, right = self.task_pairs[{"A": 0, "B": 1, "C": 2}[self.policy_sequence[self._phase_index(index)]]]
        values = self.inputs[index]
        return 0.8 * values[left] * values[right]

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise ValueError("environment is done; call reset before stepping")
        if not math.isfinite(float(action)):
            raise ValueError("adaptive-dendritic action must be finite")
        target = self.target_at()
        action = max(-1.0, min(1.0, float(action)))
        self.last_reward = max(-1.0, min(1.0, (target * target - (action - target) ** 2) / 4.0))
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, self.last_reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        self._validate_checkpoint_state()
        return {
            "version": 1,
            "seed": self.seed,
            "schedule_seed": self.schedule_seed,
            "task_seed": self.task_seed,
            "policy_sequence": list(self.policy_sequence),
            "task_pairs": [list(pair) for pair in self.task_pairs],
            "phase_lengths": list(self.phase_lengths),
            "horizon": self.horizon,
            "t": self.t,
            "last_reward": self.last_reward,
            "inputs": [list(values) for values in self.inputs],
        }

    def _validate_checkpoint_state(self) -> None:
        if self.horizon != sum(self.phase_lengths) or self.horizon <= 0:
            raise ValueError("adaptive-dendritic horizon is invalid")
        if not isinstance(self.t, int) or not 0 <= self.t <= self.horizon:
            raise ValueError("adaptive-dendritic checkpoint t is outside horizon")
        if not math.isfinite(self.last_reward) or not -1.0 <= self.last_reward <= 1.0:
            raise ValueError("adaptive-dendritic checkpoint last_reward is invalid")
        if len(self.inputs) != self.horizon or any(
            len(values) != 6
            or any(not math.isfinite(float(value)) or float(value) not in (-1.0, 1.0) for value in values)
            for values in self.inputs
        ):
            raise ValueError("adaptive-dendritic checkpoint inputs must be finite six-sensor sign states")

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "AdaptiveDendriticEnvironment":
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported adaptive-dendritic environment checkpoint version")
        environment = cls(
            int(state["seed"]),
            tuple(int(value) for value in state["phase_lengths"]),
            schedule_seed=int(state.get("schedule_seed", state["seed"])),
            task_seed=int(state.get("task_seed", state["seed"])),
            task_pairs=state.get("task_pairs"),
        )
        inputs = tuple(tuple(float(value) for value in values) for values in state.get("inputs", ()))
        environment.inputs = inputs
        environment.t = int(state["t"])
        environment.last_reward = float(state.get("last_reward", 0.0))
        if int(state.get("horizon", environment.horizon)) != environment.horizon:
            raise ValueError("adaptive-dendritic checkpoint horizon mismatch")
        environment._validate_checkpoint_state()
        return environment


def adaptive_dendritic_affine_proof() -> Dict[str, object]:
    """Exact six-sensor certificate that every affine pair readout is zero."""
    states = [tuple(1 if (mask >> index) & 1 else -1 for index in range(6)) for mask in range(64)]
    coefficients: Dict[str, Tuple[float, ...]] = {}
    projections: Dict[str, Tuple[float, ...]] = {}
    skills: Dict[str, float] = {}
    for left, right in ADAPTIVE_DENDRITIC_ALL_PAIRS:
        name = "%d_%d" % (left, right)
        parity = [values[left] * values[right] for values in states]
        features = [(1,) + values for values in states]
        moments = tuple(sum(row[index] * value for row, value in zip(features, parity)) for index in range(7))
        coefficients[name] = tuple(0.0 if moment == 0 else 0.8 * moment / 64.0 for moment in moments)
        projections[name] = tuple(0.0 if moment == 0 else 0.8 * moment for moment in moments)
        skills[name] = 0.0
    return {
        "states": states,
        "all_pairs": [list(pair) for pair in ADAPTIVE_DENDRITIC_ALL_PAIRS],
        "coefficients": coefficients,
        "feature_target_projections": projections,
        "optimal_affine_skill": skills,
        "max_abs_projection": max(abs(value) for values in projections.values() for value in values),
    }


def _adaptive_dendritic_organism(seed: int, mode: str) -> Organism:
    if mode == "fixed_random_dendritic":
        organism = Organism.create_random_dendritic_pair_bank(input_size=6, hidden_size=8, output_size=1, seed=seed)
        afferent_kind = "hidden"
    else:
        organism = Organism.create_default(input_size=6, hidden_size=8, output_size=1, seed=seed)
        afferent_kind = "input" if mode == "raw_linear_modules" else "hidden"
        if afferent_kind == "hidden":
            if mode == "adaptive_dendritic":
                organism.enable_adaptive_dendritic_learning()
                organism.enable_adaptive_dendritic_shadow_learning()
    organism.enable_context_modules(max_modules=3, afferent_kind=afferent_kind)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    return organism


def _adaptive_dendritic_stationary_safety(seed: int, config: AdaptiveDendriticBenchmarkConfig) -> Dict[str, object]:
    organism_seed, input_seed, schedule_seed, exploration_seed, task_seed = adaptive_dendritic_seed_manifest(seed)
    length = config.phase_lengths[0]
    environment = AdaptiveDendriticEnvironment(input_seed, (length,) * 5, schedule_seed=schedule_seed, task_seed=task_seed)
    organism = _adaptive_dendritic_organism(organism_seed, "adaptive_dendritic")
    organism.actor_learning_rate = config.actor_learning_rate
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(length)))
    observation = environment.reset()
    for _ in range(length):
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        observation = environment.step(result.outputs[0])
    organism.apply_outcome(observation.reward)
    warning_kinds = {"motor_warning_started", "motor_warning_escalated", "motor_module_reactivated", "motor_module_probe_started"}
    warning_events = sum(event.get("kind") in warning_kinds for event in organism.events)
    return {
        "safe": len(organism.motor_modules) == 1 and warning_events == 0,
        "module_count": len(organism.motor_modules),
        "warning_events": warning_events,
        "recruitments": sum(event.get("kind") == "motor_module_recruited" for event in organism.events),
        "adaptive_proposals": sum(event.get("kind") == "adaptive_dendritic_proposed" for event in organism.events),
        "adaptive_accepts": organism.adaptive_dendritic_accept_count,
        "adaptive_rejects": organism.adaptive_dendritic_reject_count,
        "adaptive_shadow_actions": organism.adaptive_dendritic_shadow_action_count,
        "adaptive_shadow_installs": organism.adaptive_dendritic_shadow_install_count,
    }


def _adaptive_dendritic_mode_run(seed: int, config: AdaptiveDendriticBenchmarkConfig, mode: str) -> Dict[str, object]:
    allowed = ("adaptive_dendritic", "raw_linear_modules", "fixed_random_dendritic", "additive_hidden_no_representation", "no_actor", "frozen")
    if mode not in allowed:
        raise ValueError("unsupported adaptive-dendritic mode: %s" % mode)
    organism_seed, input_seed, schedule_seed, exploration_seed, task_seed = adaptive_dendritic_seed_manifest(seed)
    lengths = adaptive_dendritic_phase_lengths_for_seed(schedule_seed, config.phase_lengths)
    environment = AdaptiveDendriticEnvironment(input_seed, lengths, schedule_seed=schedule_seed, task_seed=task_seed)
    organism = _adaptive_dendritic_organism(organism_seed, mode)
    organism.actor_learning_rate = config.actor_learning_rate if mode in ("adaptive_dendritic", "raw_linear_modules", "fixed_random_dendritic", "additive_hidden_no_representation") else 0.0
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    initial_cell_ids = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(synapse.id for synapse in organism.graph.iter_synapses())
    initial_synapse_strengths = tuple((synapse.id, synapse.strength) for synapse in organism.graph.iter_synapses())
    observation = environment.reset()
    actions: List[float] = []
    targets: List[float] = []
    rewards: List[float] = []
    timeline: List[Optional[str]] = []
    for _ in range(environment.horizon):
        target = environment.target_at()
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        timeline.append(organism.pending_motor_module)
        observation = environment.step(result.outputs[0])
        actions.append(result.outputs[0]); targets.append(target); rewards.append(observation.reward)
    organism.apply_outcome(observation.reward)
    windows = nonlinear_phase_windows_for_lengths(environment.phase_lengths)
    metrics = _multi_phase_metrics(actions, targets, windows, config.tail_fraction)
    canonical: Dict[str, Optional[str]] = {}
    for name, (start, end) in windows.items():
        tail_start = end - max(1, int((end - start) * config.tail_fraction))
        tail_ids = [identifier for identifier in timeline[tail_start:end] if identifier is not None]
        if tail_ids:
            identifier, count = max(((value, tail_ids.count(value)) for value in set(tail_ids)), key=lambda item: (item[1], item[0]))
            canonical[name] = identifier if count / float(len(tail_ids)) >= 0.70 else None
        else:
            canonical[name] = None
    policy_by_segment = {name: policy for name, policy in zip(windows, ADAPTIVE_DENDRITIC_POLICY_SEQUENCE)}
    first_occurrence: Dict[str, str] = {}
    for name in windows:
        first_occurrence.setdefault(policy_by_segment[name], name)
    reuse = {}
    for name in windows:
        first = first_occurrence[policy_by_segment[name]]
        if name != first:
            reuse[name] = canonical[name] is not None and canonical[name] == canonical[first]
    lags = {}
    for name in ("A_return", "B_return"):
        start, end = windows[name]
        first_module = canonical[first_occurrence[policy_by_segment[name]]]
        lags[name] = _first_canonical_reactivation_lag(organism.events, start, end, first_module)
    acquisitions_by_policy: Dict[str, List[Tuple[object, ...]]] = {policy: [] for policy in ("A", "B", "C")}
    for event in organism.events:
        if event.get("kind") not in ("adaptive_dendritic_accepted", "adaptive_dendritic_shadow_installed"):
            continue
        step = int(event.get("step", -1))
        for name, (start, end) in windows.items():
            if start <= step < end:
                policy = policy_by_segment[name]
                acquisitions_by_policy[policy].append(tuple(event.get("new_sources", ())))
                break
    product_pairs = sorted({tuple(cell.dendritic_sources) for cell in organism.cells.values() if cell.activation_type == "dendritic_product"})
    product_pair_indices = {
        tuple(sorted(int(source.rsplit("-", 1)[1]) for source in pair))
        for pair in product_pairs
    }
    overlaps = {policy: (list(environment.task_pairs[index]) if tuple(environment.task_pairs[index]) in product_pair_indices else None) for index, policy in enumerate(("A", "B", "C"))}
    direct_edges = tuple(synapse.id for synapse in organism.graph.iter_synapses() if synapse.source in organism.input_ids and organism.cells[synapse.destination].kind == "motor_module")
    return {
        "seed": int(seed), "mode": mode, "organism_seed": organism_seed, "input_seed": input_seed, "schedule_seed": schedule_seed, "exploration_seed": exploration_seed, "task_seed": task_seed,
        "task_pairs": tuple(environment.task_pairs), "phase_lengths": tuple(environment.phase_lengths), "actions": actions, "targets": targets, "rewards": rewards, "phase_metrics": metrics,
        "canonical_by_segment": canonical, "canonical_reuse": reuse, "reactivation_lag_by_segment": lags, "active_module_timeline": tuple(timeline),
        "module_count": len(organism.motor_modules), "direct_input_motor_edges": direct_edges, "target_pair_overlap": overlaps, "product_pairs": tuple(product_pairs),
        "stationary_safety": _adaptive_dendritic_stationary_safety(seed, config) if mode == "adaptive_dendritic" else None,
        "initial_cell_ids": initial_cell_ids, "initial_synapse_ids": initial_synapse_ids, "initial_synapse_strengths": initial_synapse_strengths, "final_cell_ids": tuple(sorted(organism.cells)), "final_synapse_ids": tuple(synapse.id for synapse in organism.graph.iter_synapses()),
        "final_state": organism.state_dict(), "environment_state": environment.state_dict(),
        "adaptive_dendritic_counts": {
            "proposals": sum(event.get("kind") == "adaptive_dendritic_proposed" for event in organism.events),
            "accepts": organism.adaptive_dendritic_accept_count,
            "rejects": organism.adaptive_dendritic_reject_count,
            "shadow_actions": organism.adaptive_dendritic_shadow_action_count,
            "shadow_installs": organism.adaptive_dendritic_shadow_install_count,
            "fingerprint_starts": sum(event.get("kind") == "adaptive_dendritic_fingerprint_started" for event in organism.events),
            "fingerprint_evidence": sum(event.get("kind") == "adaptive_dendritic_fingerprint_evidence" for event in organism.events),
            "fingerprint_resolutions": sum(event.get("kind") == "adaptive_dendritic_fingerprint_resolved" for event in organism.events),
        },
        "adaptive_pair_acquisitions_by_policy": {
            policy: tuple(acquisitions_by_policy[policy]) for policy in ("A", "B", "C")
        },
    }


def _summarize_adaptive_dendritic(results: Sequence[Mapping[str, object]], config: AdaptiveDendriticBenchmarkConfig) -> Dict[str, object]:
    names = ("A_initial", "B_interference", "C_novel", "A_return", "B_return")
    phase = {name: {metric: _mean(float(result["phase_metrics"][name][metric]) for result in results) for metric in ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")} for name in names}
    reuse = {name: sum(bool(result["canonical_reuse"].get(name, False)) for result in results) for name in ("A_return", "B_return")}
    lags = [lag for result in results for name in ("A_return", "B_return") for lag in result["reactivation_lag_by_segment"][name]]
    median_lag = sorted(lags)[len(lags) // 2] if lags else None
    safety = sum(bool(result.get("stationary_safety", {}).get("safe", False)) for result in results) if results[0].get("mode") == "adaptive_dendritic" else None
    adaptive_counts = {
        name: sum(int(result.get("adaptive_dendritic_counts", {}).get(name, 0)) for result in results)
        for name in ("proposals", "accepts", "rejects", "shadow_actions", "shadow_installs")
    }
    pair_acquisitions = {
        policy: sum(len(result.get("adaptive_pair_acquisitions_by_policy", {}).get(policy, ())) for result in results)
        for policy in ("A", "B", "C")
    }
    passed = {
        "first_context_tails": all(phase[name]["tail_skill"] >= config.tail_skill_floor for name in ("A_initial", "B_interference", "C_novel")),
        "first_context_sign": all(phase[name]["tail_sign_accuracy"] >= config.tail_sign_accuracy_floor for name in ("A_initial", "B_interference", "C_novel")),
        "returned_context_tails": all(phase[name]["tail_skill"] >= config.return_tail_skill_floor for name in ("A_return", "B_return")),
        "returned_context_early": all(phase[name]["early_skill"] >= config.return_early_skill_floor for name in ("A_return", "B_return")),
        "canonical_reuse": all(reuse[name] >= config.canonical_reuse_min_runs for name in reuse), "reactivation_lag": median_lag is not None and median_lag <= config.reactivation_lag_limit,
        "capacity": all(int(result["module_count"]) <= config.max_modules for result in results), "zero_direct_input_motor": all(not result["direct_input_motor_edges"] for result in results),
    }
    if safety is not None:
        passed["stationary_safety"] = safety >= config.stationary_safety_min_runs
    return {"seeds": len(results), "mean_phase_metrics": phase, "mean_module_count": _mean(float(result["module_count"]) for result in results), "canonical_reuse_counts": reuse, "median_reactivation_lag": median_lag, "stationary_safety_runs": safety, "target_pair_overlap_runs": {policy: sum(bool(result["target_pair_overlap"][policy]) for result in results) for policy in ("A", "B", "C")}, "adaptive_dendritic_counts": adaptive_counts, "adaptive_pair_acquisitions_by_policy": pair_acquisitions, "passed": passed, "all_passed": all(passed.values()), "thresholds": {"tail_skill_floor": config.tail_skill_floor, "tail_sign_accuracy_floor": config.tail_sign_accuracy_floor, "return_early_skill_floor": config.return_early_skill_floor, "return_tail_skill_floor": config.return_tail_skill_floor, "canonical_reuse_min_runs": config.canonical_reuse_min_runs, "reactivation_lag_limit": config.reactivation_lag_limit, "max_modules": config.max_modules, "stationary_safety_min_runs": config.stationary_safety_min_runs}}


def _adaptive_cubic_heldout(seed: int) -> Dict[str, float]:
    organism_seed, input_seed, _, exploration_seed, _ = adaptive_dendritic_seed_manifest(seed)
    organism = _adaptive_dendritic_organism(organism_seed, "fixed_random_dendritic")
    organism.context_max_modules = 1
    organism.context_detector_min_evidence = 10 ** 9
    organism.context_warning_threshold = 10 ** 9
    organism.context_switch_threshold = 10 ** 9 + 1.0
    horizon = 512
    state_rng = random.Random(input_seed + 99194853094755497)
    base = [tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64)]
    states: List[Tuple[float, ...]] = []
    for _ in range(horizon // 64):
        block = list(base); state_rng.shuffle(block); states.extend(block)
    targets = [0.8 * values[0] * values[2] * values[5] for values in states]
    tape_rng = random.Random(exploration_seed + 7919)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(horizon)))
    reward = 0.0; actions: List[float] = []
    for values, target in zip(states, targets):
        result = organism.step(values, Modulators(reward=reward, exploration=0.20))
        actions.append(result.outputs[0]); reward = max(-1.0, min(1.0, (target * target - (result.outputs[0] - target) ** 2) / 4.0))
    organism.apply_outcome(reward)
    tail = horizon // 4; zero = sum(value * value for value in targets[-tail:])
    return {"tail_skill": 1.0 - sum((action - target) ** 2 for action, target in zip(actions[-tail:], targets[-tail:])) / zero, "tail_sign_accuracy": _sign_accuracy(actions, targets, horizon - tail, horizon)}


def run_adaptive_dendritic(seed: int, config: Optional[AdaptiveDendriticBenchmarkConfig] = None, mode: str = "adaptive_dendritic") -> Dict[str, object]:
    config = config or AdaptiveDendriticBenchmarkConfig(); config.validate()
    return _adaptive_dendritic_mode_run(seed, config, mode)


def run_adaptive_dendritic_benchmark(config: Optional[AdaptiveDendriticBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or AdaptiveDendriticBenchmarkConfig(); config.validate()
    mode_names = ("adaptive_dendritic", "raw_linear_modules", "fixed_random_dendritic", "additive_hidden_no_representation", "no_actor", "frozen")
    runs = {mode: [_adaptive_dendritic_mode_run(seed, config, mode) for seed in config.seeds] for mode in mode_names}
    summaries = {mode: _summarize_adaptive_dendritic(runs[mode], config) for mode in mode_names}
    main_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in runs["adaptive_dendritic"])
    raw_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in runs["raw_linear_modules"])
    fixed_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in runs["fixed_random_dendritic"])
    additive_score = _mean(float(result["phase_metrics"]["A_return"]["tail_skill"]) for result in runs["additive_hidden_no_representation"])
    matched = all(
        runs["adaptive_dendritic"][i]["initial_cell_ids"] == runs["additive_hidden_no_representation"][i]["initial_cell_ids"]
        and runs["adaptive_dendritic"][i]["initial_synapse_ids"] == runs["additive_hidden_no_representation"][i]["initial_synapse_ids"]
        and runs["adaptive_dendritic"][i]["initial_synapse_strengths"] == runs["additive_hidden_no_representation"][i]["initial_synapse_strengths"]
        for i in range(len(config.seeds))
    )
    cubic = {str(seed): _adaptive_cubic_heldout(seed) for seed in config.seeds[: min(6, len(config.seeds))]}
    comparison = {"matched_seed_and_protocol": True, "matched_starting_structure": matched, "main_return_tail_skill_minus_raw": main_score - raw_score, "main_return_tail_skill_minus_fixed_random": main_score - fixed_score, "main_return_tail_skill_minus_additive_hidden": main_score - additive_score, "main_beats_raw_by_margin": main_score >= raw_score + config.comparison_margin, "main_beats_fixed_random_by_margin": main_score >= fixed_score + config.comparison_margin, "main_beats_additive_hidden_by_margin": main_score >= additive_score + config.comparison_margin}
    diagnostic = "ADAPTIVE_DENDRITIC_PASS" if summaries["adaptive_dendritic"]["all_passed"] and comparison["main_beats_raw_by_margin"] and comparison["main_beats_fixed_random_by_margin"] else "ADAPTIVE_DENDRITIC_NOT_ESTABLISHED"
    return {
        "protocol": {"version": 9, "phase_lengths_base": list(config.phase_lengths), "phase_lengths_by_seed": {str(seed): list(adaptive_dendritic_phase_lengths_for_seed(adaptive_dendritic_seed_manifest(seed)[2], config.phase_lengths)) for seed in config.seeds}, "seeds": list(config.seeds), "acceptance_manifest": [list(entry) for entry in ADAPTIVE_DENDRITIC_ACCEPTANCE_MANIFEST], "task_pair_manifest": [[list(pair) for pair in pairs] for pairs in ADAPTIVE_DENDRITIC_TASK_PAIR_MANIFEST], "policy_sequence": list(ADAPTIVE_DENDRITIC_POLICY_SEQUENCE), "balanced_block_size": 64, "sensor_count": 6, "linear_proof": adaptive_dendritic_affine_proof(), "cubic_heldout_declared_primary": False},
        **summaries,
        "heldout_cubic": cubic,
        "adaptive_learning_rule": {
            "mode": "neutral_raw_reward_fingerprint",
            "fingerprint_window": 16,
            "fingerprint_score": "abs(sum(reward * z * exploration / sigma)) / sqrt(sum(square_credit))",
            "fingerprint_action_mean": "0.0 (neutral safe mean)",
            "actor_updates_during_fingerprint": False,
            "detector_drift": 0.65,
            "detector_min_evidence": 64,
            "post_install_detector_hold_steps": 64,
            "pair_owner_lookup": "organism-local installed pair owner; unowned winner recruits then installs",
            "legacy_trial_compatibility": True,
            # Legacy structural-trial compatibility only; the neutral shadow
            # fingerprint itself settles at exactly 16 outcomes.
            "min_evidence": 3,
            "min_evidence_scope": "legacy_structural_trial_compatibility_only",
        },
        "comparison": comparison,
        "diagnostic": diagnostic,
    }


# ---------------------------------------------------------------------------
# v10 variable-order benchmark (RED-first protocol only)

# This protocol deliberately stops at evaluator and control plumbing.  The
# current organism schema has pair cells only; no variable-order learner is
# enabled here, and the fixed-order control reports its order-3 portion as
# deferred rather than manufacturing cubic activations outside the graph.
VARIABLE_ORDER_POLICY_SEQUENCE = ("P0", "C0", "P1", "C1", "P2", "C2", "P0", "C0")
VARIABLE_ORDER_PAIR_FEATURES: Tuple[Tuple[int, int], ...] = tuple(combinations(range(6), 2))
VARIABLE_ORDER_CUBIC_FEATURES: Tuple[Tuple[int, int, int], ...] = tuple(combinations(range(6), 3))
VARIABLE_ORDER_ACCEPTANCE_MANIFEST: Tuple[Tuple[int, int, int, int, int, int], ...] = tuple(
    (5003 + 2 * index, 7001 + 4 * index, 9001 + 6 * index, 11003 + 8 * index, 13007 + 10 * index, 15013 + 12 * index)
    for index in range(24)
)


def _variable_order_task_manifest() -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    rows = []
    for index in range(24):
        row: List[Tuple[int, ...]] = []
        for offset in range(3):
            row.append(VARIABLE_ORDER_PAIR_FEATURES[(3 * index + offset) % len(VARIABLE_ORDER_PAIR_FEATURES)])
            row.append(VARIABLE_ORDER_CUBIC_FEATURES[(5 * index + offset) % len(VARIABLE_ORDER_CUBIC_FEATURES)])
        rows.append(tuple(row))
    return tuple(rows)


VARIABLE_ORDER_TASK_MANIFEST = _variable_order_task_manifest()


@dataclass
class VariableOrderBenchmarkConfig:
    """Frozen v10 protocol with opt-in local variable-order learning."""

    phase_lengths: Tuple[int, ...] = (256,) * 8
    seeds: Tuple[int, ...] = tuple(range(24))
    exploration: float = 0.20
    novelty: float = 0.10
    actor_learning_rate: float = 0.10
    tail_fraction: float = 0.25
    pair_tail_skill_floor: float = 0.15
    cubic_tail_skill_floor: float = 0.15
    pair_tail_sign_accuracy_floor: float = 0.75
    cubic_tail_sign_accuracy_floor: float = 0.75
    return_early_skill_floor: float = 0.15
    return_tail_skill_floor: float = 0.20
    canonical_reuse_min_runs: int = 18
    reactivation_lag_limit: int = 30
    comparison_margin: float = 0.10
    max_modules: int = 6
    stationary_safety_min_runs: int = 20
    variable_order_learning_enabled: bool = True

    def validate(self) -> None:
        if len(self.phase_lengths) != 8 or any(
            isinstance(length, bool) or not math.isfinite(float(length)) or int(length) != float(length)
            or int(length) < 64 or int(length) % 64
            for length in self.phase_lengths
        ):
            raise ValueError("variable-order phase_lengths must be eight positive multiples of 64")
        if not self.seeds:
            raise ValueError("at least one variable-order seed is required")
        for name in (
            "exploration", "novelty", "actor_learning_rate", "tail_fraction", "pair_tail_skill_floor",
            "cubic_tail_skill_floor", "pair_tail_sign_accuracy_floor", "cubic_tail_sign_accuracy_floor",
            "return_early_skill_floor", "return_tail_skill_floor", "comparison_margin",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError("variable-order rates and thresholds must be finite")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("exploration and novelty must be in [0, 1]")
        if not 0.0 < self.tail_fraction <= 1.0 or self.actor_learning_rate < 0.0:
            raise ValueError("variable-order tail fraction and actor rate are invalid")
        if any(getattr(self, name) < 0.0 for name in ("pair_tail_skill_floor", "cubic_tail_skill_floor", "return_early_skill_floor", "return_tail_skill_floor")):
            raise ValueError("variable-order skill floors must be nonnegative")
        if any(not 0.0 <= getattr(self, name) <= 1.0 for name in ("pair_tail_sign_accuracy_floor", "cubic_tail_sign_accuracy_floor")):
            raise ValueError("variable-order sign-accuracy floors must be in [0, 1]")
        if self.comparison_margin < 0.0:
            raise ValueError("variable-order comparison_margin must be nonnegative")
        for name in ("canonical_reuse_min_runs", "reactivation_lag_limit", "max_modules", "stationary_safety_min_runs"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError("variable-order count limits must be integers")
        if self.canonical_reuse_min_runs < 1 or self.reactivation_lag_limit < 0 or self.max_modules < 1 or self.stationary_safety_min_runs < 1:
            raise ValueError("variable-order count limits are invalid")
        if not isinstance(self.variable_order_learning_enabled, bool):
            raise ValueError("variable_order_learning_enabled must be boolean")


def variable_order_seed_manifest(seed: int) -> Tuple[int, int, int, int, int, int]:
    if 0 <= int(seed) < len(VARIABLE_ORDER_ACCEPTANCE_MANIFEST):
        return VARIABLE_ORDER_ACCEPTANCE_MANIFEST[int(seed)]
    raise ValueError("seed is not present in the frozen variable-order manifest")


def variable_order_task_manifest(task_seed: int) -> Tuple[Tuple[int, ...], ...]:
    """Return evaluator-owned pair/cubic features; never sent to organism."""
    return VARIABLE_ORDER_TASK_MANIFEST[int(task_seed) % len(VARIABLE_ORDER_TASK_MANIFEST)]


def variable_order_phase_lengths_for_seed(seed: int, base: Sequence[int] = (256,) * 8) -> Tuple[int, ...]:
    values = tuple(int(length) for length in base)
    if len(values) != 8 or any(length < 64 or length % 64 for length in values):
        raise ValueError("variable-order base lengths must be eight multiples of 64")
    schedule_rng = random.Random((int(seed) + 1) * 4000037 + 19081)
    return tuple(max(64, length + 64 * schedule_rng.randint(-2, 2)) for length in values)


class VariableOrderBenchmarkEnvironment:
    """Phase-blind six-sensor sign stream with evaluator-owned pair/cubic tasks."""

    def __init__(
        self,
        seed: int = 0,
        phase_lengths: Optional[Sequence[int]] = None,
        schedule_seed: Optional[int] = None,
        task_seed: Optional[int] = None,
        task_features: Optional[Sequence[Sequence[int]]] = None,
        custom_task_features: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.schedule_seed = int(seed if schedule_seed is None else schedule_seed)
        self.task_seed = int(seed if task_seed is None else task_seed)
        if not isinstance(custom_task_features, bool):
            raise ValueError("custom_task_features marker must be boolean")
        self.custom_task_features = custom_task_features
        self.phase_lengths = tuple(int(v) for v in (phase_lengths if phase_lengths is not None else variable_order_phase_lengths_for_seed(self.schedule_seed)))
        if len(self.phase_lengths) != 8 or any(v < 64 or v % 64 for v in self.phase_lengths):
            raise ValueError("variable-order phase_lengths must be eight multiples of 64")
        raw = task_features if task_features is not None else variable_order_task_manifest(self.task_seed)
        self.task_features = tuple(tuple(int(v) for v in feature) for feature in raw)
        if len(self.task_features) != 6 or len(self.task_features[0]) != 2 or len(self.task_features[1]) != 3:
            raise ValueError("task_features must contain pair/cubic features for P0/C0/P1/C1/P2/C2")
        for index, feature in enumerate(self.task_features):
            expected = 2 if index % 2 == 0 else 3
            if len(feature) != expected or tuple(sorted(feature)) != feature or len(set(feature)) != expected or any(v < 0 or v >= 6 for v in feature):
                raise ValueError("task_features contain an invalid sensor feature")
        if not self.custom_task_features and self.task_features != variable_order_task_manifest(self.task_seed):
            raise ValueError("noncanonical task_features require custom_task_features=True")
        self.policy_sequence = VARIABLE_ORDER_POLICY_SEQUENCE
        self.horizon = sum(self.phase_lengths)
        self.t = 0
        self.last_reward = 0.0
        self.inputs: Tuple[Tuple[float, ...], ...] = ()
        self.reset()

    def reset(self) -> Observation:
        input_rng = random.Random(self.seed + 2654435761)
        states = tuple(tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64))
        generated: List[Tuple[float, ...]] = []
        for _ in range(self.horizon // 64):
            block = list(states)
            input_rng.shuffle(block)
            generated.extend(block)
        self.inputs = tuple(generated)
        self.t = 0
        self.last_reward = 0.0
        return Observation(self.inputs[0], 0.0, False)

    def _phase_index(self, transition: int) -> int:
        boundary = 0
        for index, length in enumerate(self.phase_lengths):
            boundary += length
            if transition < boundary:
                return index
        return len(self.phase_lengths) - 1

    def target_at(self, transition: Optional[int] = None) -> float:
        index = self.t if transition is None else int(transition)
        if index < 0 or index >= self.horizon:
            raise ValueError("variable-order transition is outside environment horizon")
        phase = self._phase_index(index)
        feature = self.task_features[phase if phase < 6 else phase - 6]
        value = 1.0
        for sensor in feature:
            value *= self.inputs[index][sensor]
        return 0.8 * value

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise ValueError("environment is done; call reset before stepping")
        if not math.isfinite(float(action)):
            raise ValueError("variable-order action must be finite")
        target = self.target_at()
        clipped = max(-1.0, min(1.0, float(action)))
        self.last_reward = max(-1.0, min(1.0, (target * target - (clipped - target) ** 2) / 4.0))
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, self.last_reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        self._validate_checkpoint_state()
        return {
            "version": 1, "seed": self.seed, "schedule_seed": self.schedule_seed, "task_seed": self.task_seed,
            "policy_sequence": list(self.policy_sequence), "task_features": [list(f) for f in self.task_features],
            "custom_task_features": self.custom_task_features,
            "phase_lengths": list(self.phase_lengths), "horizon": self.horizon, "t": self.t,
            "last_reward": self.last_reward, "inputs": [list(v) for v in self.inputs],
        }

    def _validate_checkpoint_state(self) -> None:
        if tuple(self.policy_sequence) != VARIABLE_ORDER_POLICY_SEQUENCE or len(self.phase_lengths) != 8 or any(length < 64 or length % 64 for length in self.phase_lengths):
            raise ValueError("variable-order checkpoint protocol is invalid")
        if self.horizon != sum(self.phase_lengths) or self.horizon <= 0:
            raise ValueError("variable-order checkpoint horizon is invalid")
        if len(self.task_features) != 6:
            raise ValueError("variable-order checkpoint task_features are incomplete")
        for index, feature in enumerate(self.task_features):
            expected = 2 if index % 2 == 0 else 3
            if len(feature) != expected or tuple(sorted(feature)) != feature or len(set(feature)) != expected or any(value < 0 or value >= 6 for value in feature):
                raise ValueError("variable-order checkpoint task_features are invalid")
        if not isinstance(self.custom_task_features, bool):
            raise ValueError("variable-order checkpoint custom_task_features marker is invalid")
        if not self.custom_task_features and self.task_features != variable_order_task_manifest(self.task_seed):
            raise ValueError("variable-order checkpoint task_features do not match canonical task manifest")
        if not isinstance(self.t, int) or not 0 <= self.t <= self.horizon:
            raise ValueError("variable-order checkpoint t is outside horizon")
        if not math.isfinite(self.last_reward) or not -1.0 <= self.last_reward <= 1.0:
            raise ValueError("variable-order checkpoint reward is invalid")
        if len(self.inputs) != self.horizon or any(len(v) != 6 or any(float(x) not in (-1.0, 1.0) for x in v) for v in self.inputs):
            raise ValueError("variable-order checkpoint inputs must be six-sensor sign states")
        expected_block = {tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64)}
        for start in range(0, self.horizon, 64):
            if set(self.inputs[start:start + 64]) != expected_block or len(self.inputs[start:start + 64]) != 64:
                raise ValueError("variable-order checkpoint input blocks must contain every sign state exactly once")

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "VariableOrderBenchmarkEnvironment":
        required = {"version", "seed", "schedule_seed", "task_seed", "policy_sequence", "task_features", "custom_task_features", "phase_lengths", "horizon", "t", "last_reward", "inputs"}
        if not required.issubset(state):
            raise ValueError("variable-order environment checkpoint is missing required fields")
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported variable-order environment checkpoint version")
        if tuple(state.get("policy_sequence", ())) != VARIABLE_ORDER_POLICY_SEQUENCE:
            raise ValueError("variable-order checkpoint policy sequence mismatch")
        environment = cls(
            int(state["seed"]), tuple(int(v) for v in state["phase_lengths"]),
            schedule_seed=int(state["schedule_seed"]),
            task_seed=int(state["task_seed"]), task_features=state["task_features"], custom_task_features=state["custom_task_features"],
        )
        environment.inputs = tuple(tuple(float(x) for x in values) for values in state.get("inputs", ()))
        environment.t = int(state["t"])
        environment.last_reward = float(state.get("last_reward", 0.0))
        if int(state.get("horizon", environment.horizon)) != environment.horizon:
            raise ValueError("variable-order checkpoint horizon mismatch")
        environment._validate_checkpoint_state()
        return environment


# Short alias keeps the environment discoverable alongside v9's name.
VariableOrderEnvironment = VariableOrderBenchmarkEnvironment


def variable_order_parity_proof() -> Dict[str, object]:
    """Exact integer Walsh certificates for pair/affine and cubic/quadratic separation."""
    states = [tuple(1 if (mask >> index) & 1 else -1 for index in range(6)) for mask in range(64)]
    affine = [(1,) + values for values in states]
    degree_two = [(1,) + values + tuple(values[left] * values[right] for left, right in VARIABLE_ORDER_PAIR_FEATURES) for values in states]
    pair_projection = {}
    pair_degree2_projection = {}
    cubic_projection = {}
    for feature in VARIABLE_ORDER_PAIR_FEATURES:
        target = [values[feature[0]] * values[feature[1]] for values in states]
        pair_projection["%d_%d" % feature] = tuple(sum(row[index] * value for row, value in zip(affine, target)) for index in range(len(affine[0])))
        pair_degree2_projection["%d_%d" % feature] = tuple(sum(row[index] * value for row, value in zip(degree_two, target)) for index in range(len(degree_two[0])))
    for feature in VARIABLE_ORDER_CUBIC_FEATURES:
        target = [values[feature[0]] * values[feature[1]] * values[feature[2]] for values in states]
        cubic_projection["%d_%d_%d" % feature] = tuple(sum(row[index] * value for row, value in zip(degree_two, target)) for index in range(len(degree_two[0])))
    return {
        "states": states, "pair_features": [list(v) for v in VARIABLE_ORDER_PAIR_FEATURES],
        "cubic_features": [list(v) for v in VARIABLE_ORDER_CUBIC_FEATURES],
        "pair_vs_affine_projections": pair_projection, "pair_vs_degree2_projections": pair_degree2_projection,
        "cubic_vs_degree2_projections": cubic_projection,
        "max_abs_pair_affine_projection": max(abs(v) for row in pair_projection.values() for v in row),
        "pair_degree2_own_basis": {name: row[7 + VARIABLE_ORDER_PAIR_FEATURES.index(tuple(int(v) for v in name.split("_")))] for name, row in pair_degree2_projection.items()},
        "max_abs_pair_own_basis_error": max(abs(row[7 + index] - 64) if row[7 + index] != 64 else 0 for index, row in enumerate(pair_degree2_projection.values())),
        "max_abs_pair_other_degree2_projection": max(abs(row[7 + other]) for index, row in enumerate(pair_degree2_projection.values()) for other in range(len(VARIABLE_ORDER_PAIR_FEATURES)) if other != index),
        "max_abs_cubic_degree2_projection": max(abs(v) for row in cubic_projection.values() for v in row),
        "integer_certificate": True,
    }


variable_order_feature_proof = variable_order_parity_proof
variable_order_affine_proof = variable_order_parity_proof


def _variable_order_organism(seed: int, mode: str, config: VariableOrderBenchmarkConfig) -> Organism:
    if mode == "fixed_random_order_bank":
        # Honest RED control: current graph can install pair cells only.  The
        # order-3 bank remains deferred and is reported as such below.
        organism = Organism.create_random_dendritic_pair_bank(input_size=6, hidden_size=8, output_size=1, seed=seed)
        afferent_kind = "hidden"
    else:
        organism = Organism.create_default(input_size=6, hidden_size=8, output_size=1, seed=seed)
        afferent_kind = "input" if mode == "raw_linear_modules" else "hidden"
    organism.enable_context_modules(max_modules=config.max_modules, afferent_kind=afferent_kind)
    # Six hidden-afferent context modules must fit the bounded graph budget:
    # the base organism already contains 15 cells and each recruited module
    # owns one cell plus all eight hidden afferents.  This v10-local budget
    # remains finite and is identical across main and controls.
    organism.resources.energy_per_step = max(30.0, organism.resources.energy_per_step)
    organism.structural_plasticity_enabled = False
    organism.learning_rate = 0.0
    organism.legacy_learning_enabled = False
    organism.representation_learning_enabled = False
    organism.representation_learning_rate = 0.0
    organism.adaptive_dendritic_enabled = False
    organism.adaptive_dendritic_shadow_enabled = False
    organism.adaptive_dendritic_fingerprint_enabled = False
    return organism


def _variable_order_phase_windows(lengths: Sequence[int]) -> Dict[str, Tuple[int, int]]:
    if len(tuple(lengths)) != 8:
        raise ValueError("variable-order benchmark requires eight phase lengths")
    windows = {}
    start = 0
    for name, length in zip(VARIABLE_ORDER_POLICY_SEQUENCE, lengths):
        # Repeated P0/C0 names need unique segment keys.
        key = name if name not in windows else "%s_return" % name
        windows[key] = (start, start + int(length))
        start += int(length)
    return windows


def _variable_order_mode_run(seed: int, config: VariableOrderBenchmarkConfig, mode: str) -> Dict[str, object]:
    allowed = ("variable_order", "additive_hidden", "raw_linear_modules", "fixed_random_order_bank", "no_actor", "frozen")
    if mode not in allowed:
        raise ValueError("unsupported variable-order mode: %s" % mode)
    organism_seed, input_seed, schedule_seed, exploration_seed, representation_seed, task_seed = variable_order_seed_manifest(seed)
    lengths = variable_order_phase_lengths_for_seed(schedule_seed, config.phase_lengths)
    environment = VariableOrderBenchmarkEnvironment(input_seed, lengths, schedule_seed=schedule_seed, task_seed=task_seed)
    organism = _variable_order_organism(organism_seed, mode, config)
    if mode == "variable_order" and config.variable_order_learning_enabled:
        organism.enable_variable_order_learning(representation_seed, max_order=3)
    organism.actor_learning_rate = config.actor_learning_rate if mode in ("variable_order", "additive_hidden", "raw_linear_modules", "fixed_random_order_bank") else 0.0
    tape_rng = random.Random(exploration_seed)
    exploration_tape = tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon))
    organism.set_exploration_tape(exploration_tape)
    initial_cell_ids = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(s.id for s in organism.graph.iter_synapses())
    initial_synapse_strengths = tuple((s.id, s.strength) for s in organism.graph.iter_synapses())
    observation = environment.reset()
    actions: List[float] = []
    targets: List[float] = []
    rewards: List[float] = []
    timeline: List[Optional[str]] = []
    for _ in range(environment.horizon):
        target = environment.target_at()
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        timeline.append(organism.pending_motor_module)
        observation = environment.step(result.outputs[0])
        actions.append(result.outputs[0]); targets.append(target); rewards.append(observation.reward)
    organism.apply_outcome(observation.reward)
    windows = _variable_order_phase_windows(environment.phase_lengths)
    metrics = _multi_phase_metrics(actions, targets, windows, config.tail_fraction)
    canonical: Dict[str, Optional[str]] = {}
    for name, (start, end) in windows.items():
        tail_start = end - max(1, int((end - start) * config.tail_fraction))
        tail_ids = [value for value in timeline[tail_start:end] if value is not None]
        canonical[name] = max(set(tail_ids), key=tail_ids.count) if tail_ids and tail_ids.count(max(set(tail_ids), key=tail_ids.count)) / float(len(tail_ids)) >= 0.70 else None
    first = {policy: next(name for name in windows if name.split("_", 1)[0] == policy) for policy in set(VARIABLE_ORDER_POLICY_SEQUENCE)}
    reuse = {name: canonical[name] is not None and canonical[name] == canonical[first[name.split("_", 1)[0]]] for name in windows if name.endswith("_return")}
    lags = {name: _first_canonical_reactivation_lag(organism.events, windows[name][0], windows[name][1], canonical[first[name.split("_", 1)[0]]]) for name in reuse}
    direct_edges = tuple(s.id for s in organism.graph.iter_synapses() if s.source in organism.input_ids and organism.cells[s.destination].kind == "motor_module")
    return {
        "seed": int(seed), "mode": mode, "organism_seed": organism_seed, "input_seed": input_seed, "schedule_seed": schedule_seed,
        "exploration_seed": exploration_seed, "representation_seed": representation_seed, "task_seed": task_seed,
        "task_features": tuple(environment.task_features), "exploration_tape_signature": (len(exploration_tape), tuple(exploration_tape[:4]), tuple(exploration_tape[-4:]), sum(exploration_tape)),
        "phase_lengths": tuple(environment.phase_lengths), "actions": actions, "targets": targets, "rewards": rewards,
        "phase_metrics": metrics, "canonical_by_segment": canonical, "canonical_reuse": reuse,
        "reactivation_lag_by_segment": lags, "active_module_timeline": tuple(timeline), "module_count": len(organism.motor_modules),
        "direct_input_motor_edges": direct_edges, "initial_cell_ids": initial_cell_ids, "initial_synapse_ids": initial_synapse_ids,
        "initial_synapse_strengths": initial_synapse_strengths, "final_cell_ids": tuple(sorted(organism.cells)),
        "final_synapse_ids": tuple(s.id for s in organism.graph.iter_synapses()), "final_state": organism.state_dict(),
        "environment_state": environment.state_dict(), "fixed_order_bank_status": "deferred_order3_schema" if mode == "fixed_random_order_bank" else "not_applicable",
        "target_blind": True,
        "variable_order_learning_enabled": bool(organism.variable_order_learning_enabled),
        "variable_order_counters": {
            "routes": organism.variable_order_route_count,
            "installs": organism.variable_order_install_count,
            "modules_with_features": len(organism.variable_order_module_features),
        },
        "stationary_safety": None,
    }


def _composition_owner_for_sensor_feature(organism: Organism, feature: Sequence[int]) -> Optional[str]:
    """Resolve an evaluator feature through the organism's learned graph maps."""
    sensors = tuple(str(organism.input_ids[int(index)]) for index in feature)
    target = set(sensors)
    for key, owner_id in organism.composition_feature_owners.items():
        lineage = organism.composition_feature_lineage.get(key, {})
        leaves = tuple(str(value) for leaf in lineage.get("leaf_closure", ()) for value in leaf)
        if set(leaves) == target and len(leaves) == len(target):
            return owner_id
    for key, owner_id in organism.variable_order_feature_owners.items():
        try:
            direct = organism._dendritic_feature_from_key(key)
        except (TypeError, ValueError):
            continue
        if set(direct) == target and len(direct) == len(target):
            return owner_id
    return None


def _evaluate_composition_causal_ablation(result: Mapping[str, object], config: "CompositionalBenchmarkConfig") -> Dict[str, object]:
    """Evaluate intact and composed-edge-ablated clones of one final organism.

    This is evaluator-only: clones are loaded from the exact same final
    checkpoint, no outcomes or exploration are supplied, and `_propagate`
    advances only local cell activations.  The ablated clone changes exactly
    the learned composed-cell -> owning motor synapses.
    """
    final_state = result.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("composition causal ablation requires a final organism checkpoint")
    intact = Organism.from_state_dict(final_state)
    ablated = Organism.from_state_dict(final_state)
    original_state = intact.state_dict()
    input_final_state = copy.deepcopy(final_state)
    # Check the intact clone before evaluator propagation mutates transient
    # activation/adaptation fields.  The causal evaluator must never rewrite
    # the submitted final checkpoint.
    original_final_state_preserved = intact.state_dict() == original_state
    ablated_edge_ids = []
    ablated_edges_before = {}
    ablated_synapses_before = {edge.id: synapse_to_dict(edge) for edge in ablated.graph.iter_synapses()}
    for key, owner_id in sorted(ablated.composition_feature_owners.items()):
        lineage = ablated.composition_feature_lineage.get(key, {})
        source_ids = tuple(sorted(str(value) for value in lineage.get("source_cells", ())))
        composed = next((cell for cell in ablated.cells.values() if cell.activation_type == "dendritic_product_composed" and tuple(sorted(cell.dendritic_sources)) == source_ids), None)
        owner = ablated.motor_modules.get(str(owner_id))
        if composed is None or owner is None:
            continue
        edge = ablated.graph.get(composed.id, owner.cell_id)
        if edge is None:
            continue
        ablated_edge_ids.append(edge.id)
        ablated_edges_before[edge.id] = synapse_to_dict(edge)
        edge.strength = 0.0
    ablated.validate()
    ablated_edges_after = {
        edge_id: synapse_to_dict(next(edge for edge in ablated.graph.iter_synapses() if edge.id == edge_id))
        for edge_id in ablated_edge_ids
    }
    ablated_synapses_after = {edge.id: synapse_to_dict(edge) for edge in ablated.graph.iter_synapses()}
    untouched_synapses_preserved = all(
        ablated_synapses_before[edge_id] == ablated_synapses_after[edge_id]
        for edge_id in ablated_synapses_before if edge_id not in set(ablated_edge_ids)
    )
    if not ablated_edge_ids:
        return {
            "available": False, "direct_available": False, "ablated_edge_ids": [], "q0": {}, "q1": {}, "direct": {},
            "q_drop": None, "direct_degradation": None, "original_final_state_preserved": final_state == input_final_state,
            "ablated_only_edge_mutation": True, "ablated_untouched_synapses_preserved": untouched_synapses_preserved,
        }

    def evaluate_clone(clone: Organism, feature: Sequence[int], start: int, end: int) -> Dict[str, object]:
        owner_id = _composition_owner_for_sensor_feature(clone, feature)
        if owner_id not in clone.motor_modules:
            return {"available": False, "owner_module": owner_id, "tail_skill": None, "tail_sign_accuracy": None}
        owner = clone.motor_modules[owner_id]
        actions = []
        targets = []
        for values in result["environment_state"]["inputs"][start:end]:
            numeric = tuple(float(value) for value in values)
            clone._propagate(numeric)
            actions.append(clone._module_mean_from_current_inputs(owner))
            target = 0.8
            for index in feature:
                target *= numeric[int(index)]
            targets.append(target)
        tail_start = max(0, len(actions) - max(1, int(len(actions) * config.tail_fraction)))
        pairs = list(zip(actions[tail_start:], targets[tail_start:]))
        zero_mse = sum(target * target for _, target in pairs) / float(max(1, len(pairs)))
        mse = sum((action - target) ** 2 for action, target in pairs) / float(max(1, len(pairs)))
        skill = 1.0 - mse / max(1e-12, zero_mse)
        signs = [1 if action * target >= 0.0 else 0 for action, target in pairs]
        return {
            "available": True, "owner_module": owner_id, "tail_skill": skill,
            "tail_sign_accuracy": sum(signs) / float(max(1, len(signs))), "sample_count": len(actions),
        }

    windows = _compositional_phase_windows(result["phase_lengths"])
    feature_map = {
        "P0": result["task_features"][0], "P1": result["task_features"][1], "Q0": result["task_features"][2],
        "C0": result["task_features"][3], "P2": result["task_features"][4], "P3": result["task_features"][5],
        "Q1": result["task_features"][6], "C1": result["task_features"][7],
    }
    intact_metrics = {}
    ablated_metrics = {}

    def fresh_ablated_clone() -> Tuple[Organism, Organism]:
        fresh_intact = Organism.from_state_dict(final_state)
        fresh_ablated = Organism.from_state_dict(final_state)
        for edge_id in ablated_edge_ids:
            edge = next((candidate for candidate in fresh_ablated.graph.iter_synapses() if candidate.id == edge_id), None)
            if edge is None:
                raise AssertionError("causal ablation edge disappeared on clone restore")
            edge.strength = 0.0
        fresh_ablated.validate()
        return fresh_intact, fresh_ablated

    for name, feature in feature_map.items():
        start, end = windows[name]
        # Each metric gets independent clones so prior phase propagation can
        # never leak activation/adaptation state into another task.
        metric_intact, metric_ablated = fresh_ablated_clone()
        intact_metrics[name] = evaluate_clone(metric_intact, feature, start, end)
        ablated_metrics[name] = evaluate_clone(metric_ablated, feature, start, end)
    q_names = ("Q0", "Q1")
    direct_names = ("P0", "P1", "P2", "P3", "C0", "C1")
    q_pairs = [(name, intact_metrics[name], ablated_metrics[name]) for name in q_names if intact_metrics[name].get("available") and ablated_metrics[name].get("available")]
    direct_pairs = [(name, intact_metrics[name], ablated_metrics[name]) for name in direct_names if intact_metrics[name].get("available") and ablated_metrics[name].get("available")]
    q_drop = sum(float(before["tail_skill"]) - float(after["tail_skill"]) for _, before, after in q_pairs) / float(max(1, len(q_pairs)))
    direct_degradation = max([float(before["tail_skill"]) - float(after["tail_skill"]) for _, before, after in direct_pairs] + [0.0])
    return {
        "available": len(q_pairs) == len(q_names), "direct_available": len(direct_pairs) == len(direct_names), "ablated_edge_ids": ablated_edge_ids,
        "ablated_edges_before": ablated_edges_before, "q0": {"intact": intact_metrics["Q0"], "ablated": ablated_metrics["Q0"]},
        "q1": {"intact": intact_metrics["Q1"], "ablated": ablated_metrics["Q1"]},
        "direct": {name: {"intact": intact_metrics[name], "ablated": ablated_metrics[name]} for name in direct_names},
        "q_drop": q_drop, "direct_degradation": direct_degradation,
        "ablated_edges_after": ablated_edges_after,
        "ablated_only_edge_mutation": True, "ablated_untouched_synapses_preserved": untouched_synapses_preserved,
        "original_final_state_preserved": final_state == input_final_state,
    }


def _variable_order_stationary_safety(seed: int, config: VariableOrderBenchmarkConfig) -> Dict[str, object]:
    """Run isolated stationary P0 slice; no evaluator target enters organism."""
    organism_seed, input_seed, schedule_seed, exploration_seed, representation_seed, task_seed = variable_order_seed_manifest(seed)
    length = int(config.phase_lengths[0])
    stationary_features = (VARIABLE_ORDER_PAIR_FEATURES[0], VARIABLE_ORDER_CUBIC_FEATURES[0]) * 3
    environment = VariableOrderBenchmarkEnvironment(input_seed, (length,) * 8, schedule_seed=schedule_seed, task_seed=task_seed, task_features=stationary_features, custom_task_features=True)
    organism = _variable_order_organism(organism_seed, "variable_order", config)
    if config.variable_order_learning_enabled:
        organism.enable_variable_order_learning(representation_seed, max_order=3)
    organism.actor_learning_rate = config.actor_learning_rate
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    observation = environment.reset()
    for _ in range(length):
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        observation = environment.step(result.outputs[0])
    organism.apply_outcome(observation.reward)
    warning_kinds = {"motor_warning_started", "motor_warning_escalated", "motor_module_reactivated", "motor_module_probe_started"}
    warning_events = sum(event.get("kind") in warning_kinds for event in organism.events)
    recruitments = sum(event.get("kind") == "motor_module_recruited" for event in organism.events)
    false_switches = sum(event.get("kind") in {"motor_module_reactivated", "motor_module_probe_started"} for event in organism.events)
    return {
        "safe": len(organism.motor_modules) == 1 and warning_events == 0 and false_switches == 0,
        "module_count": len(organism.motor_modules), "warning_events": warning_events,
        "recruitments": recruitments, "false_switches": false_switches,
    }


def _summarize_variable_order(results: Sequence[Mapping[str, object]], config: VariableOrderBenchmarkConfig) -> Dict[str, object]:
    names = tuple(_variable_order_phase_windows(results[0]["phase_lengths"]))
    phase = {name: {metric: _mean(float(result["phase_metrics"][name][metric]) for result in results) for metric in ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")} for name in names}
    reuse = {name: sum(bool(result["canonical_reuse"].get(name, False)) for result in results) for name in ("P0_return", "C0_return")}
    lags = [lag for result in results for name in reuse for lag in result["reactivation_lag_by_segment"].get(name, ())]
    median_lag = sorted(lags)[len(lags) // 2] if lags else None
    stationary_details = {
        str(result["seed"]): dict(result["stationary_safety"])
        for result in results
        if isinstance(result.get("stationary_safety"), Mapping)
    }
    stationary = sum(bool(detail.get("safe")) for detail in stationary_details.values()) if stationary_details else None
    pair_first = ("P0", "P1", "P2")
    cubic_first = ("C0", "C1", "C2")
    passed = {
        "initial_pair_tail_skill_all": all(phase[name]["tail_skill"] >= config.pair_tail_skill_floor for name in pair_first),
        "initial_pair_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.pair_tail_sign_accuracy_floor for name in pair_first),
        "initial_cubic_tail_skill_all": all(phase[name]["tail_skill"] >= config.cubic_tail_skill_floor for name in cubic_first),
        "initial_cubic_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.cubic_tail_sign_accuracy_floor for name in cubic_first),
        "returned_pair_early": phase["P0_return"]["early_skill"] >= config.return_early_skill_floor,
        "returned_pair_tail": phase["P0_return"]["tail_skill"] >= config.return_tail_skill_floor,
        "returned_cubic_early": phase["C0_return"]["early_skill"] >= config.return_early_skill_floor,
        "returned_cubic_tail": phase["C0_return"]["tail_skill"] >= config.return_tail_skill_floor,
        "canonical_reuse": all(reuse[name] >= config.canonical_reuse_min_runs for name in reuse),
        "reactivation_lag": median_lag is not None and median_lag <= config.reactivation_lag_limit,
        "capacity": all(int(result["module_count"]) <= config.max_modules for result in results),
        "zero_direct_input_motor": all(not result["direct_input_motor_edges"] for result in results),
    }
    if stationary is not None:
        passed["stationary_safety"] = stationary >= config.stationary_safety_min_runs
    return {
        "seeds": len(results), "mean_phase_metrics": phase, "mean_module_count": _mean(float(r["module_count"]) for r in results),
        "canonical_reuse_counts": reuse, "median_reactivation_lag": median_lag, "stationary_safety_runs": stationary,
        "stationary_safety_by_seed": stationary_details,
        "initial_pair_gate_phases": list(pair_first), "initial_cubic_gate_phases": list(cubic_first),
        "passed": passed, "all_passed": all(passed.values()), "thresholds": {"pair_tail_skill_floor": config.pair_tail_skill_floor, "cubic_tail_skill_floor": config.cubic_tail_skill_floor, "pair_tail_sign_accuracy_floor": config.pair_tail_sign_accuracy_floor, "cubic_tail_sign_accuracy_floor": config.cubic_tail_sign_accuracy_floor, "return_early_skill_floor": config.return_early_skill_floor, "return_tail_skill_floor": config.return_tail_skill_floor, "canonical_reuse_min_runs": config.canonical_reuse_min_runs, "reactivation_lag_limit": config.reactivation_lag_limit, "max_modules": config.max_modules, "stationary_safety_min_runs": config.stationary_safety_min_runs},
    }


def run_variable_order(seed: int, config: Optional[VariableOrderBenchmarkConfig] = None, mode: str = "variable_order") -> Dict[str, object]:
    config = config or VariableOrderBenchmarkConfig(); config.validate()
    return _variable_order_mode_run(seed, config, mode)


def run_variable_order_benchmark(config: Optional[VariableOrderBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or VariableOrderBenchmarkConfig(); config.validate()
    modes = ("variable_order", "additive_hidden", "raw_linear_modules", "fixed_random_order_bank", "no_actor", "frozen")
    runs = {mode: [_variable_order_mode_run(seed, config, mode) for seed in config.seeds] for mode in modes}
    for mode in ("variable_order", "additive_hidden"):
        for result in runs[mode]:
            result["stationary_safety"] = _variable_order_stationary_safety(int(result["seed"]), config)
    summaries = {mode: _summarize_variable_order(runs[mode], config) for mode in modes}
    main_score = _mean(float(r["phase_metrics"]["P0_return"]["tail_skill"]) for r in runs["variable_order"])
    raw_score = _mean(float(r["phase_metrics"]["P0_return"]["tail_skill"]) for r in runs["raw_linear_modules"])
    additive_score = _mean(float(r["phase_metrics"]["P0_return"]["tail_skill"]) for r in runs["additive_hidden"])
    main_cubic_score = _mean(float(r["phase_metrics"]["C0_return"]["tail_skill"]) for r in runs["variable_order"])
    raw_cubic_score = _mean(float(r["phase_metrics"]["C0_return"]["tail_skill"]) for r in runs["raw_linear_modules"])
    additive_cubic_score = _mean(float(r["phase_metrics"]["C0_return"]["tail_skill"]) for r in runs["additive_hidden"])
    matched_structure = all(
        runs["variable_order"][i]["initial_cell_ids"] == runs["additive_hidden"][i]["initial_cell_ids"]
        and runs["variable_order"][i]["initial_synapse_ids"] == runs["additive_hidden"][i]["initial_synapse_ids"]
        and runs["variable_order"][i]["initial_synapse_strengths"] == runs["additive_hidden"][i]["initial_synapse_strengths"]
        for i in range(len(config.seeds))
    )
    matched_tape = all(runs["variable_order"][i]["exploration_tape_signature"] == runs["additive_hidden"][i]["exploration_tape_signature"] for i in range(len(config.seeds)))
    matched_lengths = all(runs["variable_order"][i]["phase_lengths"] == runs["additive_hidden"][i]["phase_lengths"] for i in range(len(config.seeds)))
    matched_tasks = all(runs["variable_order"][i]["task_features"] == runs["additive_hidden"][i]["task_features"] for i in range(len(config.seeds)))
    matched_inputs = all(runs["variable_order"][i]["environment_state"]["inputs"] == runs["additive_hidden"][i]["environment_state"]["inputs"] for i in range(len(config.seeds)))
    comparison = {
        "matched_seed_and_protocol": all(result["seed"] == config.seeds[i] for i, result in enumerate(runs["variable_order"])),
        "matched_starting_structure": matched_structure,
        "matched_initial_cell_ids": matched_structure,
        "matched_initial_synapse_ids": matched_structure,
        "matched_initial_synapse_strengths": matched_structure,
        "matched_exploration_tape": matched_tape,
        "matched_phase_lengths": matched_lengths,
        "matched_task_features": matched_tasks,
        "matched_input_stream": matched_inputs,
        "matched_input_schedule_task_manifest": matched_lengths and matched_tasks and matched_inputs,
        "main_return_pair_tail_skill_minus_raw": main_score - raw_score,
        "main_return_pair_tail_skill_minus_additive": main_score - additive_score,
        "main_return_cubic_tail_skill_minus_raw": main_cubic_score - raw_cubic_score,
        "main_return_cubic_tail_skill_minus_additive": main_cubic_score - additive_cubic_score,
        "main_beats_raw_pair_by_margin": main_score >= raw_score + config.comparison_margin,
        "main_beats_additive_pair_by_margin": main_score >= additive_score + config.comparison_margin,
        "main_beats_raw_cubic_by_margin": main_cubic_score >= raw_cubic_score + config.comparison_margin,
        "main_beats_additive_cubic_by_margin": main_cubic_score >= additive_cubic_score + config.comparison_margin,
        # Backward-compatible aggregate aliases; primary gates use four explicit margins below.
        "main_beats_raw_by_margin": main_score >= raw_score + config.comparison_margin,
        "main_beats_additive_by_margin": main_score >= additive_score + config.comparison_margin,
        "fixed_random_order_bank_upper_bound": False,
        "fixed_random_order_bank_status": "deferred_order3_schema",
    }
    # RED is a protocol result, not a data-dependent claim.  Pair gates never
    # collapse cubic gates, and learner-disabled main always remains RED.
    summaries["variable_order"]["passed"]["margin_pair_vs_raw"] = comparison["main_beats_raw_pair_by_margin"]
    summaries["variable_order"]["passed"]["margin_cubic_vs_raw"] = comparison["main_beats_raw_cubic_by_margin"]
    summaries["variable_order"]["passed"]["margin_pair_vs_additive"] = comparison["main_beats_additive_pair_by_margin"]
    summaries["variable_order"]["passed"]["margin_cubic_vs_additive"] = comparison["main_beats_additive_cubic_by_margin"]
    summaries["variable_order"]["passed"]["variable_order_learning_enabled"] = bool(config.variable_order_learning_enabled)
    summaries["variable_order"]["all_passed"] = all(summaries["variable_order"]["passed"].values())
    # Keep control summary semantically separate; it is not the primary gate.
    summaries["additive_hidden"]["passed"] = dict(summaries["additive_hidden"]["passed"])
    return {
        "protocol": {
            "version": 10, "phase_lengths_base": list(config.phase_lengths),
            "phase_lengths_by_seed": {str(seed): list(variable_order_phase_lengths_for_seed(variable_order_seed_manifest(seed)[2], config.phase_lengths)) for seed in config.seeds},
            "seeds": list(config.seeds), "acceptance_manifest": [list(v) for v in VARIABLE_ORDER_ACCEPTANCE_MANIFEST],
            "stream_names": ["organism", "input", "schedule", "exploration", "representation", "task"],
            "representation_seeds": {str(seed): variable_order_seed_manifest(seed)[4] for seed in config.seeds},
            "task_manifest": [[list(feature) for feature in row] for row in VARIABLE_ORDER_TASK_MANIFEST],
            "policy_sequence": list(VARIABLE_ORDER_POLICY_SEQUENCE), "balanced_block_size": 64, "sensor_count": 6,
            "parity_proof": variable_order_parity_proof(), "independent_rng_streams": True,
            "variable_order_learning_enabled": bool(config.variable_order_learning_enabled), "phase_metadata_in_organism": False,
            "controls_target_blind": True, "fixed_order_bank_status": "deferred_order3_schema",
        },
        **summaries, "runs": runs, "comparison": comparison,
        "red_report_gates": summaries["variable_order"]["passed"],
        "diagnostic": "VARIABLE_ORDER_REPRESENTATION_PASS" if summaries["variable_order"]["all_passed"] else "VARIABLE_ORDER_REPRESENTATION_NOT_ESTABLISHED",
    }


# ---------------------------------------------------------------------------
# v11 compositional benchmark (RED-first protocol only)

COMPOSITIONAL_POLICY_SEQUENCE = (
    "P0", "P1", "Q0", "C0", "P2", "P3", "Q1", "C1",
    "P0_return", "Q0_return", "C0_return", "Q1_return",
)
COMPOSITIONAL_TASK_NAMES = ("P0", "P1", "Q0", "C0", "P2", "P3", "Q1", "C1")
COMPOSITIONAL_PHASE_TASK_INDEX = (0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 6)
COMPOSITIONAL_PAIR_FEATURES: Tuple[Tuple[int, int], ...] = tuple(combinations(range(6), 2))
COMPOSITIONAL_CUBIC_FEATURES: Tuple[Tuple[int, int, int], ...] = tuple(combinations(range(6), 3))
COMPOSITIONAL_QUARTIC_FEATURES: Tuple[Tuple[int, int, int, int], ...] = tuple(combinations(range(6), 4))
COMPOSITIONAL_ACCEPTANCE_MANIFEST: Tuple[Tuple[int, ...], ...] = tuple(
    (
        5103 + 2 * index, 7101 + 4 * index, 9101 + 6 * index,
        11103 + 8 * index, 13107 + 10 * index, 15113 + 12 * index,
        17117 + 14 * index,
    )
    for index in range(24)
)


def _compositional_task_manifest() -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    rows = []
    for index in range(24):
        p0 = COMPOSITIONAL_PAIR_FEATURES[(3 * index) % len(COMPOSITIONAL_PAIR_FEATURES)]
        p1_candidates = [pair for pair in COMPOSITIONAL_PAIR_FEATURES if set(pair).isdisjoint(p0)]
        p1 = p1_candidates[(5 * index + 1) % len(p1_candidates)]
        p2 = COMPOSITIONAL_PAIR_FEATURES[(7 * index + 2) % len(COMPOSITIONAL_PAIR_FEATURES)]
        p3_candidates = [pair for pair in COMPOSITIONAL_PAIR_FEATURES if set(pair).isdisjoint(p2)]
        p3 = p3_candidates[(3 * index + 2) % len(p3_candidates)]
        q0 = tuple(sorted(p0 + p1))
        q1 = tuple(sorted(p2 + p3))
        c0 = COMPOSITIONAL_CUBIC_FEATURES[(5 * index + 1) % len(COMPOSITIONAL_CUBIC_FEATURES)]
        c1 = COMPOSITIONAL_CUBIC_FEATURES[(7 * index + 3) % len(COMPOSITIONAL_CUBIC_FEATURES)]
        rows.append((p0, p1, q0, c0, p2, p3, q1, c1))
    return tuple(rows)


COMPOSITIONAL_TASK_MANIFEST = _compositional_task_manifest()


@dataclass
class CompositionalBenchmarkConfig:
    """Frozen v11 composition protocol with continuous composition learning on by default."""

    phase_lengths: Tuple[int, ...] = (256,) * 12
    seeds: Tuple[int, ...] = tuple(range(24))
    exploration: float = 0.20
    novelty: float = 0.10
    actor_learning_rate: float = 0.10
    tail_fraction: float = 0.25
    pair_tail_skill_floor: float = 0.15
    cubic_tail_skill_floor: float = 0.15
    quartic_tail_skill_floor: float = 0.15
    pair_tail_sign_accuracy_floor: float = 0.75
    cubic_tail_sign_accuracy_floor: float = 0.75
    quartic_tail_sign_accuracy_floor: float = 0.75
    return_early_skill_floor: float = 0.15
    return_tail_skill_floor: float = 0.20
    canonical_reuse_min_runs: int = 18
    reactivation_lag_limit: int = 30
    comparison_margin: float = 0.10
    max_modules: int = 8
    stationary_safety_min_runs: int = 20
    composed_install_min_runs: int = 18
    closure_min_runs: int = 18
    quartic_ablation_drop: float = 0.20
    direct_ablation_drop_limit: float = 0.10
    composition_learning_enabled: bool = True

    def validate(self) -> None:
        if len(self.phase_lengths) != 12 or any(
            isinstance(length, bool) or not math.isfinite(float(length)) or int(length) != float(length)
            or int(length) < 64 or int(length) % 64
            for length in self.phase_lengths
        ):
            raise ValueError("compositional phase_lengths must be twelve positive multiples of 64")
        if not self.seeds:
            raise ValueError("at least one compositional seed is required")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < len(COMPOSITIONAL_ACCEPTANCE_MANIFEST) for seed in self.seeds):
            raise ValueError("compositional seeds must be drawn from the frozen acceptance manifest")
        for name in (
            "exploration", "novelty", "actor_learning_rate", "tail_fraction", "pair_tail_skill_floor",
            "cubic_tail_skill_floor", "quartic_tail_skill_floor", "pair_tail_sign_accuracy_floor",
            "cubic_tail_sign_accuracy_floor", "quartic_tail_sign_accuracy_floor", "return_early_skill_floor",
            "return_tail_skill_floor", "comparison_margin", "quartic_ablation_drop", "direct_ablation_drop_limit",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError("compositional rates and thresholds must be finite")
        if not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("compositional exploration and novelty must be in [0, 1]")
        if not 0.0 < self.tail_fraction <= 1.0 or self.actor_learning_rate < 0.0:
            raise ValueError("compositional tail fraction and actor rate are invalid")
        if any(getattr(self, name) < 0.0 for name in ("pair_tail_skill_floor", "cubic_tail_skill_floor", "quartic_tail_skill_floor", "return_early_skill_floor", "return_tail_skill_floor", "comparison_margin", "quartic_ablation_drop", "direct_ablation_drop_limit")):
            raise ValueError("compositional skill floors and margins must be nonnegative")
        if any(not 0.0 <= getattr(self, name) <= 1.0 for name in ("pair_tail_sign_accuracy_floor", "cubic_tail_sign_accuracy_floor", "quartic_tail_sign_accuracy_floor")):
            raise ValueError("compositional sign-accuracy floors must be in [0, 1]")
        for name in ("canonical_reuse_min_runs", "reactivation_lag_limit", "max_modules", "stationary_safety_min_runs", "composed_install_min_runs", "closure_min_runs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("compositional count limits must be integers")
        if self.canonical_reuse_min_runs < 1 or self.reactivation_lag_limit < 0 or self.max_modules < 1 or self.stationary_safety_min_runs < 1 or self.composed_install_min_runs < 1 or self.closure_min_runs < 1:
            raise ValueError("compositional count limits are invalid")
        if not isinstance(self.composition_learning_enabled, bool):
            raise ValueError("composition_learning_enabled must be boolean")


def compositional_seed_manifest(seed: int) -> Tuple[int, ...]:
    if 0 <= int(seed) < len(COMPOSITIONAL_ACCEPTANCE_MANIFEST):
        return COMPOSITIONAL_ACCEPTANCE_MANIFEST[int(seed)]
    raise ValueError("seed is not present in the frozen compositional manifest")


def compositional_task_manifest(task_seed: int) -> Tuple[Tuple[int, ...], ...]:
    return COMPOSITIONAL_TASK_MANIFEST[int(task_seed) % len(COMPOSITIONAL_TASK_MANIFEST)]


def compositional_factorization_manifest(task_seed: Optional[int] = None) -> Tuple[Dict[str, object], ...]:
    rows = COMPOSITIONAL_TASK_MANIFEST if task_seed is None else (compositional_task_manifest(task_seed),)
    result = []
    for row_index, row in enumerate(rows):
        p0, p1, q0, _, p2, p3, q1, _ = row
        result.append({
            "row": row_index,
            "Q0": list(q0), "Q0_factors": [list(p0), list(p1)],
            "Q0_disjoint": set(p0).isdisjoint(p1), "Q0_union_exact": tuple(sorted(p0 + p1)) == q0,
            "Q1": list(q1), "Q1_factors": [list(p2), list(p3)],
            "Q1_disjoint": set(p2).isdisjoint(p3), "Q1_union_exact": tuple(sorted(p2 + p3)) == q1,
        })
    return tuple(result)


def compositional_phase_lengths_for_seed(seed: int, base: Sequence[int] = (256,) * 12) -> Tuple[int, ...]:
    values = tuple(int(value) for value in base)
    if len(values) != 12 or any(value < 64 or value % 64 for value in values):
        raise ValueError("compositional base lengths must be twelve multiples of 64")
    schedule_rng = random.Random((int(seed) + 1) * 5000093 + 21191)
    return tuple(max(64, value + 64 * schedule_rng.randint(-2, 2)) for value in values)


class CompositionalBenchmarkEnvironment:
    """Phase-blind six-sensor environment with evaluator-only factorized quartics."""

    def __init__(
        self,
        seed: int = 0,
        phase_lengths: Optional[Sequence[int]] = None,
        schedule_seed: Optional[int] = None,
        task_seed: Optional[int] = None,
        task_features: Optional[Sequence[Sequence[int]]] = None,
        custom_task_features: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.schedule_seed = int(seed if schedule_seed is None else schedule_seed)
        self.task_seed = int(seed if task_seed is None else task_seed)
        if not isinstance(custom_task_features, bool):
            raise ValueError("custom_task_features marker must be boolean")
        self.custom_task_features = custom_task_features
        self.phase_lengths = tuple(int(value) for value in (phase_lengths if phase_lengths is not None else compositional_phase_lengths_for_seed(self.schedule_seed)))
        if len(self.phase_lengths) != 12 or any(value < 64 or value % 64 for value in self.phase_lengths):
            raise ValueError("compositional phase_lengths must be twelve multiples of 64")
        raw = task_features if task_features is not None else compositional_task_manifest(self.task_seed)
        self.task_features = tuple(tuple(int(value) for value in feature) for feature in raw)
        self.policy_sequence = COMPOSITIONAL_POLICY_SEQUENCE
        self.horizon = sum(self.phase_lengths)
        self.t = 0
        self.last_reward = 0.0
        self.inputs: Tuple[Tuple[float, ...], ...] = ()
        self._validate_task_features()
        self.reset()

    def _validate_task_features(self) -> None:
        expected_lengths = (2, 2, 4, 3, 2, 2, 4, 3)
        if len(self.task_features) != 8:
            raise ValueError("compositional task_features must contain eight base tasks")
        for index, feature in enumerate(self.task_features):
            if len(feature) != expected_lengths[index] or tuple(sorted(feature)) != feature or len(set(feature)) != len(feature) or any(value < 0 or value >= 6 for value in feature):
                raise ValueError("compositional task_features contain an invalid feature")
        p0, p1, q0, _, p2, p3, q1, _ = self.task_features
        if not set(p0).isdisjoint(p1) or tuple(sorted(p0 + p1)) != q0 or not set(p2).isdisjoint(p3) or tuple(sorted(p2 + p3)) != q1:
            raise ValueError("compositional Q tasks must be exact unions of disjoint pair tasks")
        if not self.custom_task_features and self.task_features != compositional_task_manifest(self.task_seed):
            raise ValueError("noncanonical compositional task_features require custom_task_features=True")

    def reset(self) -> Observation:
        input_rng = random.Random(self.seed + 2654435761)
        states = tuple(tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64))
        generated: List[Tuple[float, ...]] = []
        for _ in range(self.horizon // 64):
            block = list(states)
            input_rng.shuffle(block)
            generated.extend(block)
        self.inputs = tuple(generated)
        self.t = 0
        self.last_reward = 0.0
        return Observation(self.inputs[0], 0.0, False)

    def _phase_index(self, transition: int) -> int:
        boundary = 0
        for index, length in enumerate(self.phase_lengths):
            boundary += length
            if transition < boundary:
                return index
        return len(self.phase_lengths) - 1

    def target_at(self, transition: Optional[int] = None) -> float:
        index = self.t if transition is None else int(transition)
        if index < 0 or index >= self.horizon:
            raise ValueError("compositional transition is outside environment horizon")
        feature = self.task_features[COMPOSITIONAL_PHASE_TASK_INDEX[self._phase_index(index)]]
        value = 1.0
        for sensor in feature:
            value *= self.inputs[index][sensor]
        return 0.8 * value

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise ValueError("compositional environment is done; call reset before stepping")
        if not math.isfinite(float(action)):
            raise ValueError("compositional action must be finite")
        target = self.target_at()
        clipped = max(-1.0, min(1.0, float(action)))
        self.last_reward = max(-1.0, min(1.0, (target * target - (clipped - target) ** 2) / 4.0))
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, self.last_reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        self._validate_checkpoint_state()
        return {
            "version": 1, "seed": self.seed, "schedule_seed": self.schedule_seed, "task_seed": self.task_seed,
            "policy_sequence": list(self.policy_sequence), "task_features": [list(feature) for feature in self.task_features],
            "custom_task_features": self.custom_task_features, "phase_lengths": list(self.phase_lengths),
            "horizon": self.horizon, "t": self.t, "last_reward": self.last_reward,
            "inputs": [list(values) for values in self.inputs],
        }

    def _validate_checkpoint_state(self) -> None:
        if tuple(self.policy_sequence) != COMPOSITIONAL_POLICY_SEQUENCE:
            raise ValueError("compositional checkpoint policy sequence is invalid")
        if self.horizon != sum(self.phase_lengths) or self.horizon <= 0:
            raise ValueError("compositional checkpoint horizon is invalid")
        self._validate_task_features()
        if not isinstance(self.custom_task_features, bool):
            raise ValueError("compositional checkpoint custom-task marker is invalid")
        if not isinstance(self.t, int) or not 0 <= self.t <= self.horizon:
            raise ValueError("compositional checkpoint t is invalid")
        if not math.isfinite(self.last_reward) or not -1.0 <= self.last_reward <= 1.0:
            raise ValueError("compositional checkpoint reward is invalid")
        if len(self.inputs) != self.horizon or any(len(values) != 6 or any(float(value) not in (-1.0, 1.0) for value in values) for values in self.inputs):
            raise ValueError("compositional checkpoint inputs are invalid")
        expected_block = {tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64)}
        for start in range(0, self.horizon, 64):
            block = self.inputs[start:start + 64]
            if len(block) != 64 or set(block) != expected_block:
                raise ValueError("compositional checkpoint blocks must contain every six-sensor state exactly once")

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "CompositionalBenchmarkEnvironment":
        required = {"version", "seed", "schedule_seed", "task_seed", "policy_sequence", "task_features", "custom_task_features", "phase_lengths", "horizon", "t", "last_reward", "inputs"}
        if not required.issubset(state) or int(state.get("version", 0)) != 1:
            raise ValueError("invalid compositional environment checkpoint schema")
        if tuple(state.get("policy_sequence", ())) != COMPOSITIONAL_POLICY_SEQUENCE:
            raise ValueError("compositional checkpoint policy sequence mismatch")
        environment = cls(
            int(state["seed"]), tuple(int(value) for value in state["phase_lengths"]),
            schedule_seed=int(state["schedule_seed"]), task_seed=int(state["task_seed"]),
            task_features=state["task_features"], custom_task_features=state["custom_task_features"],
        )
        if int(state["horizon"]) != environment.horizon:
            raise ValueError("compositional checkpoint horizon mismatch")
        environment.inputs = tuple(tuple(float(value) for value in values) for values in state["inputs"])
        environment.t = int(state["t"])
        environment.last_reward = float(state["last_reward"])
        environment._validate_checkpoint_state()
        return environment


CompositionalEnvironment = CompositionalBenchmarkEnvironment


# ---------------------------------------------------------------------------
# v12 irregular-stream compositional benchmark

STREAMING_COMPOSITIONAL_TASK_SEQUENCE = (
    "P0", "P1", "Q0", "C0", "P2", "P3", "Q1", "C1",
    "Q1", "P2", "C0", "P0", "C1", "P3", "Q0", "P1",
    "P3", "C1", "Q0", "P1", "C0", "Q1", "P0", "P2",
)
STREAMING_COMPOSITIONAL_TASK_INDEX = {
    name: index for index, name in enumerate(("P0", "P1", "Q0", "C0", "P2", "P3", "Q1", "C1"))
}
STREAMING_COMPOSITIONAL_ACCEPTANCE_MANIFEST: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(230001 + 17 * index + 1009 * stream for stream in range(7))
    for index in range(24)
)


def streaming_compositional_seed_manifest(seed: int) -> Tuple[int, ...]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < len(STREAMING_COMPOSITIONAL_ACCEPTANCE_MANIFEST):
        raise ValueError("streaming compositional seed is not present in frozen manifest")
    return STREAMING_COMPOSITIONAL_ACCEPTANCE_MANIFEST[seed]


def streaming_compositional_schedule(
    schedule_seed: int,
    task_sequence: Sequence[str] = STREAMING_COMPOSITIONAL_TASK_SEQUENCE,
    length_range: Tuple[int, int] = (97, 191),
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Build irregular non-64-aligned segments with 64-state global blocks."""
    names = tuple(str(value) for value in task_sequence)
    if not isinstance(length_range, (tuple, list)) or len(length_range) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in length_range):
        raise ValueError("streaming schedule range must contain two integer values")
    low, high = (length_range[0], length_range[1])
    if len(names) < 8 or low < 17 or high < low:
        raise ValueError("streaming schedule range is invalid")
    if any(name not in STREAMING_COMPOSITIONAL_TASK_INDEX for name in names):
        raise ValueError("streaming task sequence contains an unknown task")
    if any(names.count(name) < 3 for name in STREAMING_COMPOSITIONAL_TASK_INDEX):
        raise ValueError("every streaming task must have an acquisition and two returns")
    rng = random.Random(int(schedule_seed))
    for _ in range(256):
        lengths = [rng.randint(low, high) for _ in range(len(names) - 1)]
        if any(value % 64 == 0 for value in lengths):
            continue
        remainder = sum(lengths) % 64
        candidates = [value for value in range(low, high + 1) if value % 64 and (sum(lengths) + value) % 64 == 0]
        if not candidates:
            continue
        lengths.append(candidates[rng.randrange(len(candidates))])
        boundaries = []
        cursor = 0
        for length in lengths[:-1]:
            cursor += length
            boundaries.append(cursor)
        if all(boundary % 64 != 0 for boundary in boundaries) and sum(lengths) % 64 == 0:
            return names, tuple(lengths)
    raise RuntimeError("unable to construct irregular streaming schedule")


class StreamingCompositionalBenchmarkConfig:
    """v12 irregular-stream protocol; current v11 organism remains unchanged."""

    def __init__(self, **kwargs: object) -> None:
        defaults = {
            "seeds": tuple(range(24)), "exploration": 0.20, "novelty": 0.10,
            "actor_learning_rate": 0.10, "tail_fraction": 0.25,
            "pair_tail_skill_floor": 0.15, "cubic_tail_skill_floor": 0.15,
            "quartic_tail_skill_floor": 0.15, "pair_tail_sign_accuracy_floor": 0.75,
            "cubic_tail_sign_accuracy_floor": 0.75, "quartic_tail_sign_accuracy_floor": 0.75,
            "return_early_skill_floor": 0.15, "return_tail_skill_floor": 0.20,
            "canonical_reuse_min_runs": 18, "reactivation_lag_limit": 30,
            "comparison_margin": 0.10, "max_modules": 8, "stationary_safety_min_runs": 20,
            "composed_install_min_runs": 18, "closure_min_runs": 18,
            "quartic_ablation_drop": 0.20, "direct_ablation_drop_limit": 0.10,
            "composition_learning_enabled": True, "segment_length_range": (97, 191),
            "midpoint_existing_owner_resolution_enabled": True,
            "midpoint_existing_owner_z_threshold": 3.0,
            "midpoint_existing_owner_separation": 0.05,
            "composition_signal_normalization_enabled": True,
            "composition_signal_gain": 1.72,
            "owner_evidence_shadow_enabled": True,
            "owner_probe_enabled": True,
            "owner_midpoint_probe_enabled": True,
            "owner_midpoint_probe_min_probability": 0.0,
            "owner_midpoint_probe_min_margin": 0.0,
        }
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError("unknown streaming config fields: %s" % ", ".join(sorted(unknown)))
        defaults.update(kwargs)
        for name, value in defaults.items():
            setattr(self, name, value)

    def validate(self) -> None:
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 24 for seed in self.seeds):
            raise ValueError("streaming seeds must use frozen manifest entries")
        if not isinstance(self.segment_length_range, (tuple, list)) or len(self.segment_length_range) != 2:
            raise ValueError("streaming segment_length_range must contain two values")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.segment_length_range):
            raise ValueError("streaming segment lengths must be integers")
        low, high = (self.segment_length_range[0], self.segment_length_range[1])
        if low < 17 or high < low or any(value % 64 == 0 for value in (low, high)):
            raise ValueError("streaming segment lengths must be irregular and non-64-aligned")
        if not isinstance(self.composition_learning_enabled, bool):
            raise ValueError("composition_learning_enabled must be boolean")
        if not isinstance(self.composition_signal_normalization_enabled, bool):
            raise ValueError("composition_signal_normalization_enabled must be boolean")
        if not isinstance(self.owner_evidence_shadow_enabled, bool):
            raise ValueError("owner_evidence_shadow_enabled must be boolean")
        if not isinstance(self.owner_probe_enabled, bool):
            raise ValueError("owner_probe_enabled must be boolean")
        if not isinstance(self.owner_midpoint_probe_enabled, bool):
            raise ValueError("owner_midpoint_probe_enabled must be boolean")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in (self.owner_midpoint_probe_min_probability, self.owner_midpoint_probe_min_margin)):
            raise ValueError("owner midpoint probe thresholds must be in [0, 1]")
        if self.owner_probe_enabled and not self.owner_evidence_shadow_enabled:
            raise ValueError("owner probes require owner evidence")
        if not isinstance(self.midpoint_existing_owner_resolution_enabled, bool):
            raise ValueError("midpoint_existing_owner_resolution_enabled must be boolean")
        for name in ("tail_fraction", "exploration", "novelty", "actor_learning_rate", "pair_tail_skill_floor", "cubic_tail_skill_floor", "quartic_tail_skill_floor", "pair_tail_sign_accuracy_floor", "cubic_tail_sign_accuracy_floor", "quartic_tail_sign_accuracy_floor", "return_early_skill_floor", "return_tail_skill_floor", "comparison_margin", "quartic_ablation_drop", "direct_ablation_drop_limit"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError("streaming rates must be finite")
        if not 0.0 < self.tail_fraction <= 1.0 or not 0.0 <= self.exploration <= 1.0 or not 0.0 <= self.novelty <= 1.0 or self.actor_learning_rate < 0.0:
            raise ValueError("streaming rates are invalid")
        for name in ("canonical_reuse_min_runs", "reactivation_lag_limit", "max_modules", "stationary_safety_min_runs", "composed_install_min_runs", "closure_min_runs"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError("streaming count thresholds are invalid")
        if self.reactivation_lag_limit < 0 or self.max_modules < 1:
            raise ValueError("streaming count thresholds are invalid")
        if any(not 0.0 <= float(getattr(self, name)) <= 1.0 for name in ("pair_tail_sign_accuracy_floor", "cubic_tail_sign_accuracy_floor", "quartic_tail_sign_accuracy_floor")):
            raise ValueError("streaming sign thresholds are invalid")
        if any(float(getattr(self, name)) < 0.0 for name in ("pair_tail_skill_floor", "cubic_tail_skill_floor", "quartic_tail_skill_floor", "return_early_skill_floor", "return_tail_skill_floor", "comparison_margin", "quartic_ablation_drop", "direct_ablation_drop_limit")):
            raise ValueError("streaming skill thresholds are invalid")
        for name in ("midpoint_existing_owner_z_threshold", "midpoint_existing_owner_separation"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError("streaming midpoint thresholds are invalid")
        if not math.isfinite(float(self.composition_signal_gain)) or not 1.0 <= self.composition_signal_gain <= 2.0:
            raise ValueError("streaming composition signal gain must be in [1, 2]")


class StreamingCompositionalBenchmarkEnvironment:
    """Irregular task schedule over globally exact shuffled 64-state blocks."""

    def __init__(self, input_seed: int, schedule_seed: int, task_seed: int, config: Optional[StreamingCompositionalBenchmarkConfig] = None, segment_lengths: Optional[Sequence[int]] = None) -> None:
        self.input_seed = int(input_seed)
        self.schedule_seed = int(schedule_seed)
        self.task_seed = int(task_seed)
        self.config = config or StreamingCompositionalBenchmarkConfig()
        self.config.validate()
        if segment_lengths is None:
            self.task_sequence, self.segment_lengths = streaming_compositional_schedule(self.schedule_seed, length_range=self.config.segment_length_range)
        else:
            self.task_sequence = tuple(STREAMING_COMPOSITIONAL_TASK_SEQUENCE)
            if any(isinstance(value, bool) or not isinstance(value, int) for value in segment_lengths):
                raise ValueError("streaming segment lengths must be integers")
            self.segment_lengths = tuple(segment_lengths)
        if len(self.task_sequence) != len(self.segment_lengths) or any(length < 17 or length % 64 == 0 or not (self.config.segment_length_range[0] <= length <= self.config.segment_length_range[1]) for length in self.segment_lengths):
            raise ValueError("streaming segment lengths do not match the frozen schedule")
        if sum(self.segment_lengths) % 64 or any(sum(self.segment_lengths[:index]) % 64 == 0 for index in range(1, len(self.segment_lengths))):
            raise ValueError("streaming schedule must stay mid-superblock")
        self.task_features = compositional_task_manifest(self.task_seed)
        self.segment_starts = []
        cursor = 0
        for length in self.segment_lengths:
            self.segment_starts.append(cursor)
            cursor += length
        self.horizon = cursor
        self.policy_sequence = self.task_sequence
        self.t = 0
        self.last_reward = 0.0
        self.inputs: Tuple[Tuple[float, ...], ...] = ()
        self.reset()

    def _states(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64))

    def reset(self) -> Observation:
        rng = random.Random(self.input_seed)
        states = self._states()
        generated: List[Tuple[float, ...]] = []
        for _ in range(self.horizon // 64):
            block = list(states)
            rng.shuffle(block)
            generated.extend(block)
        self.inputs = tuple(generated)
        self.t = 0
        self.last_reward = 0.0
        return Observation(self.inputs[0], 0.0, False)

    def segment_index_at(self, position: Optional[int] = None) -> int:
        value = self.t if position is None else int(position)
        if value < 0 or value > self.horizon:
            raise ValueError("streaming position is outside the environment")
        for index, start in enumerate(self.segment_starts):
            if value < start + self.segment_lengths[index]:
                return index
        return len(self.segment_lengths) - 1

    def target_at(self, position: Optional[int] = None) -> float:
        index = self.t if position is None else int(position)
        if not 0 <= index < self.horizon:
            raise ValueError("streaming transition is outside environment horizon")
        task_name = self.task_sequence[self.segment_index_at(index)]
        feature = self.task_features[STREAMING_COMPOSITIONAL_TASK_INDEX[task_name]]
        value = 1.0
        for sensor in feature:
            value *= self.inputs[index][sensor]
        return 0.8 * value

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon or not math.isfinite(float(action)):
            raise ValueError("streaming environment cannot accept this action")
        target = self.target_at()
        clipped = max(-1.0, min(1.0, float(action)))
        self.last_reward = max(-1.0, min(1.0, (target * target - (clipped - target) ** 2) / 4.0))
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, self.last_reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        self._validate_checkpoint_state()
        segment = self.segment_index_at()
        return {
            "version": 1, "input_seed": self.input_seed, "schedule_seed": self.schedule_seed, "task_seed": self.task_seed,
            "task_sequence": list(self.task_sequence), "segment_lengths": list(self.segment_lengths), "task_features": [list(feature) for feature in self.task_features],
            "horizon": self.horizon, "t": self.t, "global_stream_position": self.t,
            "superblock_index": self.t // 64, "superblock_offset": self.t % 64,
            "segment_index": segment, "segment_offset": self.t - self.segment_starts[segment],
            "last_reward": self.last_reward, "inputs": [list(values) for values in self.inputs],
        }

    def _validate_checkpoint_state(self) -> None:
        if tuple(self.task_features) != tuple(compositional_task_manifest(self.task_seed)):
            raise ValueError("streaming checkpoint task features mismatch")
        if len(self.inputs) != self.horizon or self.horizon % 64 or self.horizon != sum(self.segment_lengths):
            raise ValueError("streaming checkpoint must contain complete 64-state blocks")
        if any(length % 64 == 0 for length in self.segment_lengths) or any(start % 64 == 0 for start in self.segment_starts[1:]):
            raise ValueError("streaming schedule contains aligned segment boundary")
        expected = set(self._states())
        for start in range(0, self.horizon, 64):
            if set(self.inputs[start:start + 64]) != expected:
                raise ValueError("streaming input superblock is not an exact state set")
        if not 0 <= self.t <= self.horizon or not math.isfinite(self.last_reward):
            raise ValueError("streaming checkpoint position/reward is invalid")

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "StreamingCompositionalBenchmarkEnvironment":
        required = {"version", "input_seed", "schedule_seed", "task_seed", "task_sequence", "segment_lengths", "task_features", "horizon", "t", "global_stream_position", "superblock_index", "superblock_offset", "segment_index", "segment_offset", "last_reward", "inputs"}
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("invalid streaming environment checkpoint schema")
        integer_fields = ("version", "input_seed", "schedule_seed", "task_seed", "horizon", "t", "global_stream_position", "superblock_index", "superblock_offset", "segment_index", "segment_offset")
        if any(isinstance(state[name], bool) or not isinstance(state[name], int) for name in integer_fields) or state["version"] != 1:
            raise ValueError("invalid streaming checkpoint integer fields")
        if isinstance(state["last_reward"], bool) or not isinstance(state["last_reward"], (int, float)) or not math.isfinite(float(state["last_reward"])):
            raise ValueError("invalid streaming checkpoint reward")
        if not isinstance(state["segment_lengths"], (tuple, list)) or not isinstance(state["task_sequence"], (tuple, list)) or not isinstance(state["inputs"], (tuple, list)):
            raise ValueError("invalid streaming environment checkpoint containers")
        if len(state["segment_lengths"]) != len(state["task_sequence"]) or not state["segment_lengths"]:
            raise ValueError("invalid streaming environment checkpoint schedule")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in state["segment_lengths"]):
            raise ValueError("invalid streaming checkpoint segment lengths")
        if any(not isinstance(value, str) for value in state["task_sequence"]):
            raise ValueError("invalid streaming checkpoint task sequence")
        if not isinstance(state["task_features"], (tuple, list)) or len(state["task_features"]) != len(STREAMING_COMPOSITIONAL_TASK_INDEX):
            raise ValueError("invalid streaming checkpoint task features")
        if any(not isinstance(feature, (tuple, list)) or any(isinstance(value, bool) or not isinstance(value, int) for value in feature) for feature in state["task_features"]):
            raise ValueError("invalid streaming checkpoint task features")
        if any(not isinstance(row, (tuple, list)) or len(row) != 6 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in row) for row in state["inputs"]):
            raise ValueError("invalid streaming checkpoint input rows")
        environment = cls(
            int(state["input_seed"]), int(state["schedule_seed"]), int(state["task_seed"]),
            StreamingCompositionalBenchmarkConfig(segment_length_range=(min(state["segment_lengths"]), max(state["segment_lengths"]))),
            segment_lengths=tuple(int(value) for value in state["segment_lengths"]),
        )
        if tuple(state["task_sequence"]) != environment.task_sequence or tuple(int(value) for value in state["segment_lengths"]) != environment.segment_lengths:
            raise ValueError("streaming checkpoint schedule mismatch")
        if tuple(tuple(int(value) for value in feature) for feature in state["task_features"]) != tuple(environment.task_features) or int(state["horizon"]) != environment.horizon:
            raise ValueError("streaming checkpoint derived metadata mismatch")
        environment.inputs = tuple(tuple(float(value) for value in values) for values in state["inputs"])
        environment.t = int(state["t"])
        environment.last_reward = float(state["last_reward"])
        if int(state.get("global_stream_position", -1)) != environment.t or int(state.get("superblock_index", -1)) != environment.t // 64 or int(state.get("superblock_offset", -1)) != environment.t % 64 or int(state.get("segment_index", -1)) != environment.segment_index_at() or int(state.get("segment_offset", -1)) != environment.t - environment.segment_starts[environment.segment_index_at()]:
            raise ValueError("streaming checkpoint position mismatch")
        environment._validate_checkpoint_state()
        return environment


def _streaming_base_config(config: StreamingCompositionalBenchmarkConfig) -> CompositionalBenchmarkConfig:
    return CompositionalBenchmarkConfig(
        seeds=tuple(config.seeds), exploration=config.exploration, novelty=config.novelty,
        actor_learning_rate=config.actor_learning_rate, tail_fraction=config.tail_fraction,
        pair_tail_skill_floor=config.pair_tail_skill_floor, cubic_tail_skill_floor=config.cubic_tail_skill_floor,
        quartic_tail_skill_floor=config.quartic_tail_skill_floor,
        pair_tail_sign_accuracy_floor=config.pair_tail_sign_accuracy_floor,
        cubic_tail_sign_accuracy_floor=config.cubic_tail_sign_accuracy_floor,
        quartic_tail_sign_accuracy_floor=config.quartic_tail_sign_accuracy_floor,
        return_early_skill_floor=config.return_early_skill_floor, return_tail_skill_floor=config.return_tail_skill_floor,
        canonical_reuse_min_runs=config.canonical_reuse_min_runs, reactivation_lag_limit=config.reactivation_lag_limit,
        comparison_margin=config.comparison_margin, max_modules=config.max_modules,
        stationary_safety_min_runs=config.stationary_safety_min_runs, composed_install_min_runs=config.composed_install_min_runs,
        closure_min_runs=config.closure_min_runs, quartic_ablation_drop=config.quartic_ablation_drop,
        direct_ablation_drop_limit=config.direct_ablation_drop_limit, composition_learning_enabled=config.composition_learning_enabled,
    )


def _streaming_segment_windows(environment: StreamingCompositionalBenchmarkEnvironment) -> Dict[str, Tuple[int, int]]:
    windows = {}
    cursor = 0
    for index, (task, length) in enumerate(zip(environment.task_sequence, environment.segment_lengths)):
        windows["%s_%d" % (task, index)] = (cursor, cursor + length)
        cursor += length
    return windows


def _streaming_valid_four_leaf_closure(organism: Organism, feature: Sequence[int]) -> bool:
    """Check a composed cell's exact four distinct input leaves and owner edge."""
    expected = {organism.input_ids[int(index)] for index in feature}
    for cell in organism.cells.values():
        if cell.activation_type != "dendritic_product_composed" or len(cell.dendritic_sources) != 2:
            continue
        if any(source not in organism.cells for source in cell.dendritic_sources):
            continue
        source_cells = [organism.cells[source] for source in cell.dendritic_sources]
        leaves = [tuple(source.dendritic_sources) for source in source_cells]
        if not all(source.activation_type == "dendritic_product" and len(leaf) == 2 for source, leaf in zip(source_cells, leaves)):
            continue
        flattened = set(leaves[0] + leaves[1])
        if set(leaves[0]) & set(leaves[1]) or len(flattened) != 4 or flattened != expected:
            continue
        key = organism._composition_feature_key(cell.dendritic_sources, leaves)
        owner_id = organism.composition_feature_owners.get(key)
        owner = organism.motor_modules.get(owner_id) if owner_id is not None else None
        if owner is not None and organism.graph.get(cell.id, owner.cell_id) is not None:
            return True
    return False


def _streaming_integrity_report(environment: StreamingCompositionalBenchmarkEnvironment) -> Dict[str, object]:
    """Evaluator-only proof that the input stream, not task metadata, is balanced."""
    states = set(environment._states())
    blocks = [environment.inputs[start:start + 64] for start in range(0, environment.horizon, 64)]
    boundaries = [sum(environment.segment_lengths[:index]) for index in range(1, len(environment.segment_lengths))]
    task_marginals: Dict[str, Dict[str, Tuple[float, ...]]] = {}
    for task, windows in _streaming_segment_windows(environment).items():
        name = task.split("_", 1)[0]
        values = environment.inputs[windows[0]:windows[1]]
        task_marginals[task] = {
            "sensor_means": tuple(sum(row[index] for row in values) / float(len(values)) for index in range(6)),
            "target_sensor_correlations": tuple(
                sum((environment.target_at(position) / 0.8) * row[index] for position, row in zip(range(windows[0], windows[1]), values)) / float(len(values))
                for index in range(6)
            ),
        }
    return {
        "complete_global_64_blocks": bool(blocks) and all(len(block) == 64 and set(block) == states for block in blocks),
        "segment_lengths_non_multiple_64": all(length % 64 for length in environment.segment_lengths),
        "boundaries_mid_superblock": all(boundary % 64 for boundary in boundaries),
        "task_window_marginals": task_marginals,
        "task_identities": {name: environment.task_sequence.count(name) for name in STREAMING_COMPOSITIONAL_TASK_INDEX},
    }


def _streaming_causal_ablation(result: Mapping[str, object], config: StreamingCompositionalBenchmarkConfig) -> Dict[str, object]:
    final_state = result.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("streaming causal ablation requires final state")
    intact = Organism.from_state_dict(final_state)
    ablated = Organism.from_state_dict(final_state)
    original = copy.deepcopy(final_state)
    edge_ids = []
    before = {}
    before_all = {edge.id: synapse_to_dict(edge) for edge in ablated.graph.iter_synapses()}
    for key, owner_id in sorted(ablated.composition_feature_owners.items()):
        lineage = ablated.composition_feature_lineage.get(key, {})
        sources = tuple(sorted(str(value) for value in lineage.get("source_cells", ())))
        composed = next((cell for cell in ablated.cells.values() if cell.activation_type == "dendritic_product_composed" and tuple(sorted(cell.dendritic_sources)) == sources), None)
        owner = ablated.motor_modules.get(str(owner_id))
        edge = ablated.graph.get(composed.id, owner.cell_id) if composed is not None and owner is not None else None
        if edge is not None:
            edge_ids.append(edge.id)
            before[edge.id] = synapse_to_dict(edge)
            edge.strength = 0.0
    if not edge_ids:
        return {"available": False, "direct_available": False, "q_drop": None, "direct_degradation": None, "ablated_edge_ids": [], "ablated_edges_before": {}, "ablated_edges_after": {}, "ablated_only_edge_mutation": False, "ablated_untouched_synapses_preserved": True, "original_final_state_preserved": final_state == original}
    ablated.validate()
    after_all = {edge.id: synapse_to_dict(edge) for edge in ablated.graph.iter_synapses()}
    after = {edge_id: after_all[edge_id] for edge_id in edge_ids}
    only_edge_mutation = all(
        before_all[edge_id] == payload
        if edge_id not in edge_ids else
        payload.get("strength") == 0.0
        and all(before_all[edge_id].get(name) == payload.get(name) for name in before_all[edge_id] if name != "strength")
        for edge_id, payload in after_all.items()
    )
    untouched_preserved = all(before_all[edge_id] == after_all[edge_id] for edge_id in before_all if edge_id not in edge_ids)
    windows = result["task_windows_by_name"]
    features = result["task_features"]
    selected = {name: windows[name][-1] for name in windows if windows[name]}
    feature_map = {name: features[index] for name, index in STREAMING_COMPOSITIONAL_TASK_INDEX.items()}

    def evaluate(clone: Organism, feature: Sequence[int], start: int, end: int) -> float:
        owner_id = _composition_owner_for_sensor_feature(clone, feature)
        if owner_id not in clone.motor_modules:
            return float("nan")
        owner = clone.motor_modules[owner_id]
        actions = []
        targets = []
        for values in result["environment_state"]["inputs"][start:end]:
            numeric = tuple(float(value) for value in values)
            clone._propagate(numeric)
            actions.append(clone._module_mean_from_current_inputs(owner))
            target = 0.8
            for index in feature:
                target *= numeric[int(index)]
            targets.append(target)
        tail = max(1, int(len(actions) * config.tail_fraction))
        return _skill(actions, targets, len(actions) - tail, len(actions))

    intact_metrics = {}
    ablated_metrics = {}
    for name, feature in feature_map.items():
        if name not in selected:
            continue
        start, end = selected[name]
        intact_metrics[name] = evaluate(Organism.from_state_dict(final_state), feature, start, end)
        fresh = Organism.from_state_dict(final_state)
        for edge_id in edge_ids:
            next(edge for edge in fresh.graph.iter_synapses() if edge.id == edge_id).strength = 0.0
        ablated_metrics[name] = evaluate(fresh, feature, start, end)
    q_pairs = [(name, intact_metrics[name], ablated_metrics[name]) for name in ("Q0", "Q1") if name in intact_metrics and math.isfinite(intact_metrics[name]) and math.isfinite(ablated_metrics[name])]
    direct_names = ("P0", "P1", "P2", "P3", "C0", "C1")
    direct_pairs = [(name, intact_metrics[name], ablated_metrics[name]) for name in direct_names if name in intact_metrics and math.isfinite(intact_metrics[name]) and math.isfinite(ablated_metrics[name])]
    return {
        "available": len(q_pairs) == 2, "direct_available": len(direct_pairs) == 6,
        "q_drop": _mean(before - after for _, before, after in q_pairs) if len(q_pairs) == 2 else None,
        "direct_degradation": max([before - after for _, before, after in direct_pairs] + [0.0]) if len(direct_pairs) == 6 else None,
        "ablated_edge_ids": edge_ids, "ablated_edges_before": before, "ablated_edges_after": after,
        "ablated_only_edge_mutation": only_edge_mutation, "ablated_untouched_synapses_preserved": untouched_preserved,
        "original_final_state_preserved": final_state == original,
    }


def _streaming_compositional_mode_run(seed: int, config: StreamingCompositionalBenchmarkConfig, mode: str) -> Dict[str, object]:
    if mode not in ("compositional", "composition_disabled_v10", "raw_linear_modules"):
        raise ValueError("unsupported streaming compositional mode")
    organism_seed, input_seed, schedule_seed, exploration_seed, representation_seed, composition_seed, task_seed = streaming_compositional_seed_manifest(seed)
    environment = StreamingCompositionalBenchmarkEnvironment(input_seed, schedule_seed, task_seed, config)
    base = _streaming_base_config(config)
    organism = _compositional_organism(organism_seed, mode if mode != "composition_disabled_v10" else "composition_disabled_v10", base)
    if mode in ("compositional", "composition_disabled_v10"):
        organism.enable_variable_order_learning(representation_seed, max_order=3)
        if config.owner_evidence_shadow_enabled:
            organism.enable_variable_order_owner_evidence()
            if config.owner_probe_enabled:
                organism.enable_variable_order_owner_probes(config.owner_midpoint_probe_enabled, config.owner_midpoint_probe_min_probability, config.owner_midpoint_probe_min_margin)
        if config.midpoint_existing_owner_resolution_enabled:
            organism.enable_variable_order_midpoint_existing_owner_resolution(
                config.midpoint_existing_owner_z_threshold,
                config.midpoint_existing_owner_separation,
            )
    if mode == "compositional":
        organism.enable_compositional_substrate(composition_seed)
        if config.composition_signal_normalization_enabled:
            organism.enable_composition_signal_normalization(config.composition_signal_gain)
        if config.composition_learning_enabled:
            organism.enable_composition_learning()
    organism.actor_learning_rate = config.actor_learning_rate if mode != "raw_linear_modules" else config.actor_learning_rate
    tape_rng = random.Random(exploration_seed)
    tape = tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon))
    organism.set_exploration_tape(tape)
    initial_cell_ids = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(edge.id for edge in organism.graph.iter_synapses())
    initial_synapse_strengths = tuple((edge.id, edge.strength) for edge in organism.graph.iter_synapses())
    observation = environment.reset()
    actions: List[float] = []
    targets: List[float] = []
    rewards: List[float] = []
    timeline: List[Optional[str]] = []
    for _ in range(environment.horizon):
        targets.append(environment.target_at())
        step_result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        timeline.append(organism.pending_motor_module)
        observation = environment.step(step_result.outputs[0])
        actions.append(step_result.outputs[0])
        rewards.append(observation.reward)
    organism.apply_outcome(observation.reward)
    windows = _streaming_segment_windows(environment)
    metrics = _multi_phase_metrics(actions, targets, windows, config.tail_fraction)
    canonical = {}
    task_windows: Dict[str, List[Tuple[int, int]]] = {name: [] for name in STREAMING_COMPOSITIONAL_TASK_INDEX}
    for index, task in enumerate(environment.task_sequence):
        task_windows[task].append(windows["%s_%d" % (task, index)])
        start, end = windows["%s_%d" % (task, index)]
        ids = [value for value in timeline[start:end] if value is not None]
        canonical["%s_%d" % (task, index)] = max(set(ids), key=ids.count) if ids and ids.count(max(set(ids), key=ids.count)) / float(len(ids)) >= 0.70 else None
    first = {name: next(key for key in canonical if key.startswith(name + "_")) for name in STREAMING_COMPOSITIONAL_TASK_INDEX}
    reuse = {name: all(canonical[key] is not None and canonical[key] == canonical[first[name]] for key in canonical if key.startswith(name + "_") and key != first[name]) for name in STREAMING_COMPOSITIONAL_TASK_INDEX}
    lags = {
        name: tuple(
            lag
            for window in task_windows[name][1:]
            for lag in _first_canonical_reactivation_lag(organism.events, window[0], window[1], canonical[first[name]])
        )
        for name in STREAMING_COMPOSITIONAL_TASK_INDEX
    }
    direct_edges = tuple(edge.id for edge in organism.graph.iter_synapses() if edge.source in organism.input_ids and organism.cells[edge.destination].kind == "motor_module")
    composed_cells = [cell for cell in organism.cells.values() if cell.activation_type == "dendritic_product_composed"]
    installs = [event for event in organism.events if event.get("kind") == "composed_feature_installed"]
    feature_map = {name: environment.task_features[index] for name, index in STREAMING_COMPOSITIONAL_TASK_INDEX.items()}
    closure_by_task = {name: _streaming_valid_four_leaf_closure(organism, feature_map[name]) for name in ("Q0", "Q1")}
    install_timing = {}
    for task in ("Q0", "Q1"):
        start, end = task_windows[task][0]
        tail_start = end - int((end - start) * config.tail_fraction)
        matching = [dict(event) for event in installs if _composition_install_matches_feature(organism, event, feature_map[task])]
        install_timing[task] = {
            "tail_start": tail_start,
            "events_before_tail": [event for event in matching if int(event.get("step", -1)) < tail_start],
            "passed": any(int(event.get("step", -1)) < tail_start for event in matching),
        }
    return {
        "seed": seed, "mode": mode, "task_features": tuple(environment.task_features), "task_sequence": tuple(environment.task_sequence), "segment_lengths": tuple(environment.segment_lengths),
        "phase_lengths": tuple(environment.segment_lengths), "actions": actions, "targets": targets, "rewards": rewards, "phase_metrics": metrics,
        "canonical_by_segment": canonical, "canonical_reuse": reuse, "reactivation_lag_by_segment": lags, "active_module_timeline": tuple(timeline),
        "module_count": len(organism.motor_modules), "direct_input_motor_edges": direct_edges, "initial_cell_ids": initial_cell_ids, "initial_synapse_ids": initial_synapse_ids,
        "initial_synapse_strengths": initial_synapse_strengths, "final_state": organism.state_dict(), "environment_state": environment.state_dict(),
        "task_windows_by_name": task_windows, "target_blind": True, "composition_learning_enabled": bool(organism.composition_learning_enabled),
        "composed_feature_installs": len(installs), "valid_four_leaf_closures": all(closure_by_task.values()), "valid_four_leaf_closure_by_task": closure_by_task, "composition_install_timing": install_timing,
        "composition_route_count": organism.composition_route_count, "composition_candidate_count": organism.composition_candidate_count,
        "stationary_safety": None, "exploration_tape_signature": (len(tape), tuple(tape[:4]), tuple(tape[-4:]), sum(tape)),
        "stream_integrity": _streaming_integrity_report(environment),
    }


def _streaming_stationary_safety(seed: int, config: StreamingCompositionalBenchmarkConfig) -> Dict[str, object]:
    organism_seed, input_seed, schedule_seed, exploration_seed, representation_seed, composition_seed, task_seed = streaming_compositional_seed_manifest(seed)
    environment = StreamingCompositionalBenchmarkEnvironment(input_seed, schedule_seed, task_seed, config)
    base = _streaming_base_config(config)
    organism = _compositional_organism(organism_seed, "compositional", base)
    organism.enable_variable_order_learning(representation_seed, max_order=3)
    if config.owner_evidence_shadow_enabled:
        organism.enable_variable_order_owner_evidence()
        if config.owner_probe_enabled:
            organism.enable_variable_order_owner_probes(config.owner_midpoint_probe_enabled, config.owner_midpoint_probe_min_probability, config.owner_midpoint_probe_min_margin)
    if config.midpoint_existing_owner_resolution_enabled:
        organism.enable_variable_order_midpoint_existing_owner_resolution(
            config.midpoint_existing_owner_z_threshold,
            config.midpoint_existing_owner_separation,
        )
    organism.enable_compositional_substrate(composition_seed)
    if config.composition_signal_normalization_enabled:
        organism.enable_composition_signal_normalization(config.composition_signal_gain)
    if config.composition_learning_enabled:
        organism.enable_composition_learning()
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    observation = environment.reset()
    for _ in range(environment.segment_lengths[0]):
        step_result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        observation = environment.step(step_result.outputs[0])
    organism.apply_outcome(observation.reward)
    warning_kinds = {"motor_warning_started", "motor_warning_escalated", "motor_module_reactivated", "motor_module_probe_started"}
    warning_events = sum(event.get("kind") in warning_kinds for event in organism.events)
    return {"safe": len(organism.motor_modules) == 1 and warning_events == 0, "module_count": len(organism.motor_modules), "warning_events": warning_events}


def _streaming_summary(results: Sequence[Mapping[str, object]], config: StreamingCompositionalBenchmarkConfig) -> Dict[str, object]:
    task_names = tuple(STREAMING_COMPOSITIONAL_TASK_INDEX)
    metric_names = ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")
    phase = {}
    return_occurrence_metrics = {}
    for task in task_names:
        occurrence_keys = [
            ["%s_%d" % (task, index) for index, name in enumerate(result["task_sequence"]) if name == task]
            for result in results
        ]
        phase[task] = {metric: _mean(float(result["phase_metrics"][keys[0]][metric]) for result, keys in zip(results, occurrence_keys)) for metric in metric_names}
        occurrence_reports = []
        for occurrence in range(1, min(len(keys) for keys in occurrence_keys)):
            report = {metric: _mean(float(result["phase_metrics"][keys[occurrence]][metric]) for result, keys in zip(results, occurrence_keys)) for metric in metric_names}
            occurrence_reports.append(report)
        return_occurrence_metrics[task] = occurrence_reports
        phase[task + "_return"] = {metric: _mean(report[metric] for report in occurrence_reports) for metric in metric_names}
    reuse = {task + "_return": sum(bool(result["canonical_reuse"].get(task, False)) for result in results) for task in task_names}
    lags = [lag for result in results for task in task_names for lag in result["reactivation_lag_by_segment"].get(task, ())]
    median_lag = sorted(lags)[len(lags) // 2] if lags else None
    q0_installs = sum(bool(result["composition_install_timing"]["Q0"]["passed"]) for result in results)
    q1_installs = sum(bool(result["composition_install_timing"]["Q1"]["passed"]) for result in results)
    stationary = {str(result["seed"]): dict(result["stationary_safety"]) for result in results if isinstance(result.get("stationary_safety"), Mapping)}
    stationary_runs = sum(bool(value.get("safe")) for value in stationary.values()) if stationary else None
    causal = [result.get("causal_ablation", {}) for result in results]
    causal_available = sum(bool(item.get("available")) for item in causal)
    causal_direct_available = sum(bool(item.get("direct_available")) for item in causal)
    q_drop = _mean(float(item["q_drop"]) for item in causal if item.get("q_drop") is not None) if causal_available else None
    direct_drop = _mean(float(item["direct_degradation"]) for item in causal if item.get("direct_degradation") is not None) if causal_direct_available else None
    q0_closure_runs = sum(bool(result.get("valid_four_leaf_closure_by_task", {}).get("Q0")) for result in results)
    q1_closure_runs = sum(bool(result.get("valid_four_leaf_closure_by_task", {}).get("Q1")) for result in results)
    integrity = {str(result["seed"]): dict(result["stream_integrity"]) for result in results}
    integrity_all = all(
        value.get("complete_global_64_blocks") and value.get("segment_lengths_non_multiple_64") and value.get("boundaries_mid_superblock")
        and all(int(count) >= 3 for count in value.get("task_identities", {}).values())
        for value in integrity.values()
    )
    passed = {
        "initial_pair_tail_skill_all": all(phase[name]["tail_skill"] >= config.pair_tail_skill_floor for name in ("P0", "P1", "P2", "P3")),
        "initial_pair_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.pair_tail_sign_accuracy_floor for name in ("P0", "P1", "P2", "P3")),
        "initial_cubic_tail_skill_all": all(phase[name]["tail_skill"] >= config.cubic_tail_skill_floor for name in ("C0", "C1")),
        "initial_cubic_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.cubic_tail_sign_accuracy_floor for name in ("C0", "C1")),
        "initial_composed_tail_skill_all": all(phase[name]["tail_skill"] >= config.quartic_tail_skill_floor for name in ("Q0", "Q1")),
        "initial_composed_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.quartic_tail_sign_accuracy_floor for name in ("Q0", "Q1")),
        "returns_early_all": all(report["early_skill"] >= config.return_early_skill_floor for reports in return_occurrence_metrics.values() for report in reports),
        "returns_tail_all": all(report["tail_skill"] >= config.return_tail_skill_floor for reports in return_occurrence_metrics.values() for report in reports),
        "canonical_reuse_all_returns": all(value >= config.canonical_reuse_min_runs for value in reuse.values()),
        "reactivation_lag": median_lag is not None and median_lag <= config.reactivation_lag_limit,
        "capacity": all(int(result["module_count"]) <= config.max_modules for result in results),
        "zero_direct_input_motor": all(not result["direct_input_motor_edges"] for result in results),
        "stationary_safety": stationary_runs is not None and stationary_runs >= config.stationary_safety_min_runs,
        "composed_feature_installed_before_Q0_tail": q0_installs >= config.composed_install_min_runs,
        "composed_feature_installed_before_Q1_tail": q1_installs >= config.composed_install_min_runs,
        "valid_Q0_four_distinct_leaf_closures": q0_closure_runs >= config.closure_min_runs,
        "valid_Q1_four_distinct_leaf_closures": q1_closure_runs >= config.closure_min_runs,
        "valid_four_distinct_leaf_closures": q0_closure_runs >= config.closure_min_runs and q1_closure_runs >= config.closure_min_runs,
        "causal_ablation_all_runs": config.composition_learning_enabled and causal_available == len(results),
        "causal_direct_all_runs": config.composition_learning_enabled and causal_direct_available == len(results),
        "composition_ablation_Q_drop": config.composition_learning_enabled and causal_available == len(results) and q_drop is not None and q_drop >= config.quartic_ablation_drop,
        "direct_ablation_drop_limit": config.composition_learning_enabled and causal_direct_available == len(results) and direct_drop is not None and direct_drop <= config.direct_ablation_drop_limit,
        "composition_learning_enabled": bool(config.composition_learning_enabled),
        "stream_integrity": integrity_all,
    }
    return {"seeds": len(results), "mean_phase_metrics": phase, "return_occurrence_metrics": return_occurrence_metrics, "canonical_reuse_counts": reuse, "median_reactivation_lag": median_lag, "stationary_safety_runs": stationary_runs, "stationary_safety_by_seed": stationary, "composition_install_Q0_runs": q0_installs, "composition_install_Q1_runs": q1_installs, "closure_Q0_runs": q0_closure_runs, "closure_Q1_runs": q1_closure_runs, "causal_ablation_available_runs": causal_available, "causal_direct_available_runs": causal_direct_available, "causal_Q_drop": q_drop, "causal_direct_degradation": direct_drop, "stream_integrity_by_seed": integrity, "passed": passed, "all_passed": all(passed.values()), "thresholds": {"pair_tail_skill_floor": config.pair_tail_skill_floor, "pair_tail_sign_accuracy_floor": config.pair_tail_sign_accuracy_floor, "cubic_tail_skill_floor": config.cubic_tail_skill_floor, "cubic_tail_sign_accuracy_floor": config.cubic_tail_sign_accuracy_floor, "quartic_tail_skill_floor": config.quartic_tail_skill_floor, "quartic_tail_sign_accuracy_floor": config.quartic_tail_sign_accuracy_floor, "return_early_skill_floor": config.return_early_skill_floor, "return_tail_skill_floor": config.return_tail_skill_floor, "canonical_reuse_min_runs": config.canonical_reuse_min_runs, "reactivation_lag_limit": config.reactivation_lag_limit, "max_modules": config.max_modules, "stationary_safety_min_runs": config.stationary_safety_min_runs, "composed_install_min_runs": config.composed_install_min_runs, "closure_min_runs": config.closure_min_runs, "comparison_margin": config.comparison_margin, "quartic_ablation_drop": config.quartic_ablation_drop, "direct_ablation_drop_limit": config.direct_ablation_drop_limit}}


def run_streaming_compositional(seed: int, config: Optional[StreamingCompositionalBenchmarkConfig] = None, mode: str = "compositional") -> Dict[str, object]:
    config = config or StreamingCompositionalBenchmarkConfig()
    config.validate()
    result = _streaming_compositional_mode_run(seed, config, mode)
    if mode == "compositional":
        result["stationary_safety"] = _streaming_stationary_safety(seed, config)
        result["causal_ablation"] = _streaming_causal_ablation(result, config)
    return result


def streaming_routing_diagnostic(result: Mapping[str, object], config: Optional[StreamingCompositionalBenchmarkConfig] = None) -> Dict[str, object]:
    """Evaluator-only counterfactual routing matrix for v12 return segments.

    The function consumes a completed run and evaluates frozen clones.  It
    never steps, modifies, or serializes evaluator task metadata into the
    organism.  Consequently these figures are diagnostic ceilings, not
    acceptance results.
    """
    config = config or StreamingCompositionalBenchmarkConfig(seeds=(int(result["seed"]),))
    config.validate()
    final_state = result.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("streaming routing diagnostic requires a final checkpoint")
    pristine = copy.deepcopy(final_state)
    reference = Organism.from_state_dict(final_state)
    inputs = result["environment_state"]["inputs"]
    targets = tuple(float(value) for value in result["targets"])
    actual = tuple(float(value) for value in result["actions"])
    events = tuple(final_state.get("events", ()))
    detection_kinds = {
        "motor_warning_started", "variable_order_fingerprint_resolved",
        "variable_order_midpoint_existing_owner_resolved", "motor_module_reactivated",
    }

    def owner_actions(owner_id: Optional[str], start: int, end: int) -> Optional[Tuple[float, ...]]:
        if owner_id not in reference.motor_modules:
            return None
        clone = Organism.from_state_dict(final_state)
        owner = clone.motor_modules[str(owner_id)]
        values = []
        for row in inputs[start:end]:
            clone._propagate(tuple(float(value) for value in row))
            values.append(clone._module_mean_from_current_inputs(owner))
        return tuple(values)

    rows = []
    occurrence = {name: 0 for name in STREAMING_COMPOSITIONAL_TASK_INDEX}
    for index, task in enumerate(result["task_sequence"]):
        occurrence[task] += 1
        if occurrence[task] == 1:
            continue
        start, end = result["task_windows_by_name"][task][occurrence[task] - 1]
        key = "%s_%d" % (task, index)
        learned_owner = result["canonical_by_segment"].get(key)
        oracle_owner = _composition_owner_for_sensor_feature(reference, result["task_features"][STREAMING_COMPOSITIONAL_TASK_INDEX[task]])
        learned_actions = owner_actions(learned_owner, start, end)
        oracle_actions = owner_actions(oracle_owner, start, end)
        detection_step = next((int(event["step"]) for event in events if event.get("kind") in detection_kinds and start <= int(event.get("step", -1)) < end), end)
        detection_offset = max(0, min(end - start, detection_step - start))
        actual_segment = actual[start:end]
        target_segment = targets[start:end]
        learned_from_boundary = learned_actions if learned_actions is not None else actual_segment
        oracle_from_detection = (
            actual_segment[:detection_offset] + oracle_actions[detection_offset:]
            if oracle_actions is not None else actual_segment
        )
        oracle_from_boundary = oracle_actions if oracle_actions is not None else actual_segment
        perfect = _skill(target_segment, target_segment, 0, len(target_segment))
        scores = {
            "learned": _skill(actual_segment, target_segment, 0, len(target_segment)),
            "oracle_boundary_learned_selection": _skill(learned_from_boundary, target_segment, 0, len(target_segment)),
            "learned_detection_oracle_selection": _skill(oracle_from_detection, target_segment, 0, len(target_segment)),
            "oracle_detection_oracle_selection": _skill(oracle_from_boundary, target_segment, 0, len(target_segment)),
            "perfect_policy": perfect,
        }
        rows.append({
            "segment": key, "task": task, "occurrence": occurrence[task],
            "detection_offset": detection_offset, "learned_owner_available": learned_actions is not None,
            "oracle_owner_available": oracle_actions is not None, "scores": scores,
        })
    means = {
        name: _mean(float(row["scores"][name]) for row in rows)
        for name in ("learned", "oracle_boundary_learned_selection", "learned_detection_oracle_selection", "oracle_detection_oracle_selection", "perfect_policy")
    }
    regret = {
        "detection_with_learned_selection": means["oracle_boundary_learned_selection"] - means["learned"],
        "detection_with_oracle_selection": means["oracle_detection_oracle_selection"] - means["learned_detection_oracle_selection"],
        "selection_with_learned_detection": means["learned_detection_oracle_selection"] - means["learned"],
        "selection_with_oracle_boundary": means["oracle_detection_oracle_selection"] - means["oracle_boundary_learned_selection"],
        "stored_policy": means["perfect_policy"] - means["oracle_detection_oracle_selection"],
    }
    return {
        "evaluator_only": True, "acceptance_gating": False,
        "organism_checkpoint_unchanged": final_state == pristine,
        "return_segments": rows, "mean_scores": means, "regret_decomposition": regret,
    }


def run_streaming_compositional_benchmark(config: Optional[StreamingCompositionalBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or StreamingCompositionalBenchmarkConfig()
    config.validate()
    modes = ("compositional", "composition_disabled_v10", "raw_linear_modules")
    runs = {mode: [run_streaming_compositional(seed, config, mode) for seed in config.seeds] for mode in modes}
    summary = _streaming_summary(runs["compositional"], config)
    def return_tail(mode: str, task: str) -> float:
        values = []
        for result in runs[mode]:
            keys = ["%s_%d" % (task, index) for index, name in enumerate(result["task_sequence"]) if name == task][1:]
            values.extend(float(result["phase_metrics"][key]["tail_skill"]) for key in keys)
        return _mean(values)
    main_q = {name: return_tail("compositional", name) for name in ("Q0", "Q1")}
    raw_q = {name: return_tail("raw_linear_modules", name) for name in ("Q0", "Q1")}
    control_q = {name: return_tail("composition_disabled_v10", name) for name in ("Q0", "Q1")}
    comparison = {
        "main_Q_return_tail_skill_minus_raw": {name: main_q[name] - raw_q[name] for name in main_q},
        "main_Q_return_tail_skill_minus_composition_disabled_v10": {name: main_q[name] - control_q[name] for name in main_q},
        "main_Q_beats_raw_by_margin": all(main_q[name] >= raw_q[name] + config.comparison_margin for name in main_q),
        "main_Q_beats_composition_disabled_by_margin": all(main_q[name] >= control_q[name] + config.comparison_margin for name in main_q),
        "matched_input_stream": all(runs["compositional"][i]["environment_state"]["inputs"] == runs["composition_disabled_v10"][i]["environment_state"]["inputs"] for i in range(len(config.seeds))),
        "matched_schedule_and_tasks": all(
            runs["compositional"][i]["task_sequence"] == runs["composition_disabled_v10"][i]["task_sequence"]
            and runs["compositional"][i]["segment_lengths"] == runs["composition_disabled_v10"][i]["segment_lengths"]
            and runs["compositional"][i]["task_features"] == runs["composition_disabled_v10"][i]["task_features"]
            for i in range(len(config.seeds))
        ),
        "matched_exploration_tape": all(runs["compositional"][i]["exploration_tape_signature"] == runs["composition_disabled_v10"][i]["exploration_tape_signature"] for i in range(len(config.seeds))),
        "matched_starting_structure": all(
            runs["compositional"][i]["initial_cell_ids"] == runs["composition_disabled_v10"][i]["initial_cell_ids"]
            and runs["compositional"][i]["initial_synapse_ids"] == runs["composition_disabled_v10"][i]["initial_synapse_ids"]
            and runs["compositional"][i]["initial_synapse_strengths"] == runs["composition_disabled_v10"][i]["initial_synapse_strengths"]
            for i in range(len(config.seeds))
        ),
    }
    summary["passed"]["Q_return_margin_vs_raw"] = comparison["main_Q_beats_raw_by_margin"]
    summary["passed"]["Q_return_margin_vs_composition_disabled"] = comparison["main_Q_beats_composition_disabled_by_margin"]
    summary["passed"]["matched_input_stream"] = comparison["matched_input_stream"]
    summary["passed"]["matched_schedule_and_tasks"] = comparison["matched_schedule_and_tasks"]
    summary["passed"]["matched_exploration_tape"] = comparison["matched_exploration_tape"]
    summary["passed"]["matched_starting_structure"] = comparison["matched_starting_structure"]
    summary["all_passed"] = all(summary["passed"].values())
    return {"protocol": {"version": 12, "stream_names": ["organism", "input", "schedule", "exploration", "representation", "composition", "task"], "acceptance_manifest": [list(row) for row in STREAMING_COMPOSITIONAL_ACCEPTANCE_MANIFEST], "task_sequence": list(STREAMING_COMPOSITIONAL_TASK_SEQUENCE), "task_manifest": [[list(feature) for feature in row] for row in COMPOSITIONAL_TASK_MANIFEST], "factorization_relations": list(compositional_factorization_manifest()), "global_superblock_size": 64, "segment_length_range": list(config.segment_length_range), "irregular_boundaries": True, "fingerprint_window": 16, "organism_receives_evaluator_metadata": False, "controls": list(modes)}, "streaming_compositional": summary, "runs": runs, "comparison": comparison, "diagnostic": "STREAMING_COMPOSITIONAL_PASS" if summary["all_passed"] else "STREAMING_COMPOSITION_NOT_ESTABLISHED"}


GENERAL_STRUCTURAL_SEQUENCE = (
    ("A", (0, 1), 192), ("B", (2, 3, 4), 192),
    ("H", (0, 1, 2, 3, 4), 320), ("A", (0, 1), 256),
    ("H", (0, 1, 2, 3, 4), 320), ("B", (2, 3, 4), 256),
    ("H", (0, 1, 2, 3, 4), 320),
)
GENERAL_STRUCTURAL_ACCEPTANCE_MANIFEST = tuple(
    (310001 + seed * 101, 320003 + seed * 103, 330007 + seed * 107, 340007 + seed * 109)
    for seed in range(24)
)

CONTINUOUS_LIFETIME_SEQUENCE = (
    ("A", 768), ("B", 768), ("H", 1536), ("A", 768),
    ("DRIFT", 768), ("B", 768), ("H_NOISY", 1536),
)
CONTINUOUS_LIFETIME_MANIFEST = tuple(
    (410009 + seed * 127, 420013 + seed * 131, 430019 + seed * 137, 440023 + seed * 139)
    for seed in range(40)
)

FINAL_LIFETIME_SEQUENCE = (
    ("A", (0, 1), 512), ("B", (2, 3, 4), 512),
    ("C", (0, 5), 512), ("E", (1, 5), 512),
    ("H", (0, 1, 2, 3, 4), 1024), ("H", (0, 1, 2, 3, 4), 512),
    ("D", (0, 2, 3, 4, 5), 512), ("K", (1, 2, 3, 4, 5), 1024),
    ("H", (0, 1, 2, 3, 4), 512), ("A", (0, 1), 512),
    ("B", (2, 3, 4), 512),
)
FINAL_LIFETIME_MANIFEST = tuple(
    (510007 + seed * 149, 520021 + seed * 151, 530027 + seed * 157, 540031 + seed * 163)
    for seed in range(40)
)


class ContinuousLifetimeBenchmarkConfig:
    def __init__(self, **kwargs: object) -> None:
        defaults = {
            "seeds": tuple(range(8)), "reward_delay": 3,
            "observation_noise": 0.08, "high_noise": 0.16,
            "exploration": 0.20, "novelty": 0.10,
            "actor_learning_rate": 0.40,
            "tail_window": 128, "tail_skill_floor": 0.10,
            "return_skill_floor": 0.25, "structure_margin": 0.10, "growth_min_runs": 6,
            "reuse_min_runs": 6, "bounded_min_runs": 8,
        }
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError("unknown continuous lifetime config fields: %s" % ", ".join(sorted(unknown)))
        defaults.update(kwargs)
        for name, value in defaults.items():
            setattr(self, name, value)

    def validate(self) -> None:
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < len(CONTINUOUS_LIFETIME_MANIFEST) for seed in self.seeds):
            raise ValueError("continuous lifetime seeds must use the frozen manifest")
        if isinstance(self.reward_delay, bool) or not isinstance(self.reward_delay, int) or self.reward_delay < 1:
            raise ValueError("continuous lifetime reward delay must be positive")
        if self.tail_window < 1 or self.growth_min_runs < 1 or self.reuse_min_runs < 1 or self.bounded_min_runs < 1:
            raise ValueError("continuous lifetime thresholds must be positive")
        if any(not math.isfinite(float(value)) for value in (self.observation_noise, self.high_noise, self.exploration, self.novelty, self.actor_learning_rate, self.tail_skill_floor, self.return_skill_floor, self.structure_margin)):
            raise ValueError("continuous lifetime rates must be finite")
        if not 0.0 <= self.observation_noise <= self.high_noise <= 0.5:
            raise ValueError("continuous lifetime noise levels are invalid")


class ContinuousLifetimeEnvironment:
    """Phase-blind continuous stream with drift, noise, and delayed reward."""
    def __init__(self, input_seed: int, noise_seed: int, config: ContinuousLifetimeBenchmarkConfig) -> None:
        config.validate()
        self.input_seed = int(input_seed)
        self.noise_seed = int(noise_seed)
        self.reward_delay = int(config.reward_delay)
        self.sequence = tuple(CONTINUOUS_LIFETIME_SEQUENCE)
        self.horizon = sum(length for _, length in self.sequence)
        input_rng = random.Random(self.input_seed)
        noise_rng = random.Random(self.noise_seed)
        self.clean_inputs = []
        self.inputs = []
        self.targets = []
        self.phase_names = []
        signs = [1.0 if input_rng.random() < 0.5 else -1.0 for _ in range(5)]
        clean = tuple(signs[index] * input_rng.uniform(0.65, 1.0) for index in range(5))
        for name, length in self.sequence:
            for offset in range(length):
                # An ergodic continuous stream, rather than shuffled balanced
                # truth-table blocks: signs recur stochastically while
                # magnitudes move on every interaction.
                for index in range(5):
                    if input_rng.random() < 0.50:
                        signs[index] *= -1.0
                clean = tuple(signs[index] * input_rng.uniform(0.90, 1.0) for index in range(5))
                noise = config.high_noise if name == "H_NOISY" else config.observation_noise
                observed = tuple(max(-1.0, min(1.0, value + noise_rng.gauss(0.0, noise))) for value in clean)
                a = 1.0 if clean[0] * clean[1] >= 0.0 else -1.0
                b = 1.0 if clean[2] * clean[3] * clean[4] >= 0.0 else -1.0
                h = 1.0 if math.prod(clean) >= 0.0 else -1.0
                if name == "A":
                    target = a
                elif name == "B":
                    target = b
                elif name in ("H", "H_NOISY"):
                    target = h
                else:
                    alpha = offset / float(max(1, length - 1))
                    target = (1.0 - alpha) * a + alpha * b
                self.clean_inputs.append(clean)
                self.inputs.append(observed)
                self.targets.append(target)
                self.phase_names.append(name)
        self.t = 0
        self.last_reward = 0.0
        self.reward_queue = [0.0] * self.reward_delay

    def reset(self) -> Observation:
        self.t = 0
        self.last_reward = 0.0
        self.reward_queue = [0.0] * self.reward_delay
        return Observation(self.inputs[0], 0.0)

    def target_at(self) -> float:
        if self.t >= self.horizon:
            raise IndexError("continuous lifetime is complete")
        return self.targets[self.t]

    def step(self, action: float) -> Observation:
        if self.t >= self.horizon:
            raise IndexError("continuous lifetime is complete")
        target = self.targets[self.t]
        immediate = max(-1.0, min(1.0, (1.0 - (float(action) - target) ** 2) / 4.0))
        self.reward_queue.append(immediate)
        self.last_reward = self.reward_queue.pop(0)
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, self.last_reward)

    def state_dict(self) -> Dict[str, object]:
        return {
            "input_seed": self.input_seed, "noise_seed": self.noise_seed,
            "reward_delay": self.reward_delay, "sequence": [list(value) for value in self.sequence],
            "clean_inputs": [list(value) for value in self.clean_inputs],
            "inputs": [list(value) for value in self.inputs], "targets": list(self.targets),
            "phase_names": list(self.phase_names), "t": self.t,
            "last_reward": self.last_reward, "reward_queue": list(self.reward_queue),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ContinuousLifetimeEnvironment":
        required = {"input_seed", "noise_seed", "reward_delay", "sequence", "clean_inputs", "inputs", "targets", "phase_names", "t", "last_reward", "reward_queue"}
        if set(state) != required:
            raise ValueError("continuous lifetime checkpoint schema is invalid")
        environment = cls.__new__(cls)
        environment.input_seed = int(state["input_seed"])
        environment.noise_seed = int(state["noise_seed"])
        environment.reward_delay = int(state["reward_delay"])
        environment.sequence = tuple((str(value[0]), int(value[1])) for value in state["sequence"])
        environment.horizon = sum(length for _, length in environment.sequence)
        environment.clean_inputs = [tuple(float(item) for item in value) for value in state["clean_inputs"]]
        environment.inputs = [tuple(float(item) for item in value) for value in state["inputs"]]
        environment.targets = [float(value) for value in state["targets"]]
        environment.phase_names = [str(value) for value in state["phase_names"]]
        environment.t = int(state["t"])
        environment.last_reward = float(state["last_reward"])
        environment.reward_queue = [float(value) for value in state["reward_queue"]]
        if not (len(environment.inputs) == len(environment.clean_inputs) == len(environment.targets) == len(environment.phase_names) == environment.horizon):
            raise ValueError("continuous lifetime checkpoint stream lengths are invalid")
        if not 0 <= environment.t <= environment.horizon or len(environment.reward_queue) != environment.reward_delay:
            raise ValueError("continuous lifetime checkpoint cursor is invalid")
        if any(len(value) != 5 or any(not math.isfinite(item) or not -1.0 <= item <= 1.0 for item in value) for value in environment.inputs):
            raise ValueError("continuous lifetime checkpoint observations are invalid")
        return environment


def _continuous_normalized_skill(actions: Sequence[float], targets: Sequence[float]) -> float:
    denominator = max(1e-12, _mean(target * target for target in targets))
    return 1.0 - _mean((action - target) ** 2 for action, target in zip(actions, targets)) / denominator


def run_continuous_lifetime(seed: int, config: Optional[ContinuousLifetimeBenchmarkConfig] = None, structure_enabled: bool = True) -> Dict[str, object]:
    config = config or ContinuousLifetimeBenchmarkConfig(seeds=(seed,))
    config.validate()
    organism_seed, input_seed, noise_seed, exploration_seed = CONTINUOUS_LIFETIME_MANIFEST[seed]
    environment = ContinuousLifetimeEnvironment(input_seed, noise_seed, config)
    organism = Organism.create_default(input_size=5, hidden_size=6, output_size=1, seed=organism_seed, max_cells=32, max_synapses=256)
    organism.enable_context_modules(max_modules=4, afferent_kind="hidden")
    organism.enable_motor_bootstrap()
    organism.actor_learning_rate = config.actor_learning_rate
    organism.enable_variable_order_learning(organism_seed + 7919, max_order=3)
    organism.set_variable_order_fingerprint_window(32)
    organism.enable_variable_order_owner_evidence()
    organism.enable_variable_order_owner_probes(True)
    organism.context_detector_drift = 0.25
    if structure_enabled:
        organism.enable_general_structural_learning(max_depth=4, max_leaves=8)
    organism.enable_delayed_credit(config.reward_delay)
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    observation = environment.reset()
    actions = []
    targets = []
    modules = []
    for _ in range(environment.horizon):
        targets.append(environment.target_at())
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        actions.append(result.outputs[0])
        modules.append(organism.pending_motor_module)
        observation = environment.step(result.outputs[0])
    organism.apply_outcome(observation.reward)
    phases = []
    cursor = 0
    for name, length in environment.sequence:
        end = cursor + length
        window = min(config.tail_window, length)
        phases.append({
            "name": name,
            "full_skill": _continuous_normalized_skill(actions[cursor:end], targets[cursor:end]),
            "tail_skill": _continuous_normalized_skill(actions[end-window:end], targets[end-window:end]),
            "dominant_module": max(set(modules[cursor:end]), key=modules[cursor:end].count),
        })
        cursor = end
    final = organism.state_dict()
    return {
        "seed": seed, "structure_enabled": structure_enabled, "phases": phases,
        "general_growth": len(organism.general_feature_owners),
        "general_reuse": max(organism.general_feature_reuse.values(), default=0),
        "module_count": len(organism.motor_modules), "cell_count": len(organism.cells),
        "synapse_count": len(organism.graph.synapses), "energy_total": organism.resources.total_energy,
        "resource_bounded": len(organism.motor_modules) <= 4 and len(organism.cells) <= organism.resources.max_cells and len(organism.graph.synapses) <= organism.resources.max_synapses,
        "final_state": final, "environment_state": environment.state_dict(),
    }


def run_continuous_lifetime_benchmark(config: Optional[ContinuousLifetimeBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or ContinuousLifetimeBenchmarkConfig()
    config.validate()
    main = [run_continuous_lifetime(seed, config, True) for seed in config.seeds]
    control = [run_continuous_lifetime(seed, config, False) for seed in config.seeds]
    growth = sum(result["general_growth"] > 0 for result in main)
    reuse = sum(result["general_reuse"] > 0 for result in main)
    bounded = sum(bool(result["resource_bounded"]) for result in main)
    phase_means = {
        name: _mean(phase["tail_skill"] for result in main for phase in result["phases"] if phase["name"] == name)
        for name, _ in CONTINUOUS_LIFETIME_SEQUENCE
    }
    control_phase_means = {
        name: _mean(phase["tail_skill"] for result in control for phase in result["phases"] if phase["name"] == name)
        for name, _ in CONTINUOUS_LIFETIME_SEQUENCE
    }
    passed = {
        "continuous_tail_skill": all(value >= config.tail_skill_floor for value in phase_means.values()),
        "familiar_returns": phase_means["A"] >= config.return_skill_floor and phase_means["B"] >= config.return_skill_floor,
        "general_growth": growth >= config.growth_min_runs,
        "general_reuse": reuse >= config.reuse_min_runs,
        "bounded_resources": bounded >= config.bounded_min_runs,
        "novel_structure_control_margin": (
            phase_means["H"] >= control_phase_means["H"] + config.structure_margin
            and phase_means["H_NOISY"] >= control_phase_means["H_NOISY"] + config.structure_margin
        ),
    }
    return {
        "protocol": {"version": 14, "manifest": [list(row) for row in CONTINUOUS_LIFETIME_MANIFEST], "sequence": [list(value) for value in CONTINUOUS_LIFETIME_SEQUENCE], "reward_delay": config.reward_delay, "task_identity_visible": False, "balanced_blocks": False},
        "continuous_lifetime": {"phase_tail_skill": phase_means, "control_phase_tail_skill": control_phase_means, "growth_runs": growth, "reuse_runs": reuse, "bounded_runs": bounded, "passed": passed, "all_passed": all(passed.values())},
        "runs": {"soma": main, "structure_disabled": control},
        "diagnostic": "CONTINUOUS_LIFETIME_PASS" if all(passed.values()) else "CONTINUOUS_LIFETIME_NOT_ESTABLISHED",
    }


class FinalLifetimeConfig:
    """Frozen-candidate configuration for the capacity-pressure lifetime."""
    def __init__(self, **kwargs: object) -> None:
        defaults = {
            "seeds": tuple(range(4)), "reward_delay": 3,
            "observation_noise": 0.08, "exploration": 0.20, "novelty": 0.10,
            "tail_window": 128, "tail_skill_floor": 0.04,
            "structural_min_fraction": 0.75,
        }
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError("unknown final lifetime config fields: %s" % ", ".join(sorted(unknown)))
        defaults.update(kwargs)
        for name, value in defaults.items():
            setattr(self, name, value)

    def validate(self) -> None:
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < len(FINAL_LIFETIME_MANIFEST) for seed in self.seeds):
            raise ValueError("final lifetime seeds must use the frozen manifest")
        if isinstance(self.reward_delay, bool) or not isinstance(self.reward_delay, int) or self.reward_delay < 1:
            raise ValueError("final lifetime reward delay must be positive")
        if isinstance(self.tail_window, bool) or not isinstance(self.tail_window, int) or self.tail_window < 1:
            raise ValueError("final lifetime tail window must be positive")
        if any(not math.isfinite(float(value)) for value in (self.observation_noise, self.exploration, self.novelty, self.tail_skill_floor, self.structural_min_fraction)):
            raise ValueError("final lifetime rates must be finite")
        if not 0.0 < self.structural_min_fraction <= 1.0:
            raise ValueError("final lifetime structural fraction must be in (0, 1]")


class FinalLifetimeEnvironment:
    """Six-sensor phase-blind stream that creates real capacity pressure."""
    def __init__(self, input_seed: int, noise_seed: int, config: FinalLifetimeConfig) -> None:
        config.validate()
        self.input_seed = int(input_seed)
        self.noise_seed = int(noise_seed)
        self.reward_delay = int(config.reward_delay)
        self.sequence = tuple(FINAL_LIFETIME_SEQUENCE)
        self.horizon = sum(length for _, _, length in self.sequence)
        input_rng = random.Random(self.input_seed)
        noise_rng = random.Random(self.noise_seed)
        self.inputs = []
        self.targets = []
        self.phase_names = []
        for name, feature, length in self.sequence:
            for _ in range(length):
                clean = tuple((1.0 if input_rng.random() < 0.5 else -1.0) * input_rng.uniform(0.90, 1.0) for _ in range(6))
                observed = tuple(max(-1.0, min(1.0, value + noise_rng.gauss(0.0, config.observation_noise))) for value in clean)
                target = math.prod(1.0 if clean[index] >= 0.0 else -1.0 for index in feature)
                self.inputs.append(observed)
                self.targets.append(target)
                self.phase_names.append(name)
        self.t = 0
        self.last_reward = 0.0
        self.reward_queue = [0.0] * self.reward_delay

    def reset(self) -> Observation:
        self.t = 0
        self.last_reward = 0.0
        self.reward_queue = [0.0] * self.reward_delay
        return Observation(self.inputs[0], 0.0)

    def target_at(self) -> float:
        return self.targets[self.t]

    def step(self, action: float) -> Observation:
        target = self.targets[self.t]
        immediate = max(-1.0, min(1.0, (1.0 - (float(action) - target) ** 2) / 4.0))
        self.reward_queue.append(immediate)
        self.last_reward = self.reward_queue.pop(0)
        self.t += 1
        values = self.inputs[self.t] if self.t < self.horizon else self.inputs[-1]
        return Observation(values, self.last_reward, self.t >= self.horizon)

    def state_dict(self) -> Dict[str, object]:
        return {
            "input_seed": self.input_seed, "noise_seed": self.noise_seed,
            "reward_delay": self.reward_delay,
            "sequence": [[name, list(feature), length] for name, feature, length in self.sequence],
            "inputs": [list(value) for value in self.inputs], "targets": list(self.targets),
            "phase_names": list(self.phase_names), "t": self.t,
            "last_reward": self.last_reward, "reward_queue": list(self.reward_queue),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "FinalLifetimeEnvironment":
        required = {"input_seed", "noise_seed", "reward_delay", "sequence", "inputs", "targets", "phase_names", "t", "last_reward", "reward_queue"}
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("final lifetime checkpoint schema is invalid")
        environment = cls.__new__(cls)
        environment.input_seed = int(state["input_seed"])
        environment.noise_seed = int(state["noise_seed"])
        environment.reward_delay = int(state["reward_delay"])
        environment.sequence = tuple((str(row[0]), tuple(int(value) for value in row[1]), int(row[2])) for row in state["sequence"])
        environment.horizon = sum(length for _, _, length in environment.sequence)
        environment.inputs = [tuple(float(item) for item in row) for row in state["inputs"]]
        environment.targets = [float(value) for value in state["targets"]]
        environment.phase_names = [str(value) for value in state["phase_names"]]
        environment.t = int(state["t"])
        environment.last_reward = float(state["last_reward"])
        environment.reward_queue = [float(value) for value in state["reward_queue"]]
        if not (len(environment.inputs) == len(environment.targets) == len(environment.phase_names) == environment.horizon):
            raise ValueError("final lifetime checkpoint stream lengths are invalid")
        if not 0 <= environment.t <= environment.horizon or len(environment.reward_queue) != environment.reward_delay:
            raise ValueError("final lifetime checkpoint cursor is invalid")
        return environment


def run_final_lifetime(seed: int, config: Optional[FinalLifetimeConfig] = None, mode: str = "soma", checkpoint_steps: Sequence[int] = ()) -> Dict[str, object]:
    config = config or FinalLifetimeConfig(seeds=(seed,))
    config.validate()
    if mode not in ("soma", "structure_disabled", "learning_disabled"):
        raise ValueError("unknown final lifetime mode")
    organism_seed, input_seed, noise_seed, exploration_seed = FINAL_LIFETIME_MANIFEST[seed]
    environment = FinalLifetimeEnvironment(input_seed, noise_seed, config)
    checkpoint_steps = frozenset(int(value) for value in checkpoint_steps)
    if any(value < 1 or value >= environment.horizon for value in checkpoint_steps):
        raise ValueError("final lifetime checkpoint steps must be internal decision boundaries")
    organism = Organism.create_default(input_size=6, hidden_size=10, output_size=1, seed=organism_seed, max_cells=40, max_synapses=320)
    organism.resources.energy_per_step = 40.0
    organism.enable_context_modules(max_modules=6, afferent_kind="hidden")
    organism.enable_motor_bootstrap()
    organism.actor_learning_rate = 0.0 if mode == "learning_disabled" else 0.40
    if mode != "learning_disabled":
        organism.enable_variable_order_learning(organism_seed + 7919, max_order=3)
        organism.set_variable_order_fingerprint_window(32)
        organism.enable_variable_order_owner_evidence()
        organism.enable_variable_order_owner_probes(True)
        organism.context_detector_drift = 0.25
    if mode == "soma":
        organism.enable_general_structural_learning(max_depth=4, max_leaves=8)
        organism.set_general_prune_reuse_ceiling(2)
    if mode == "learning_disabled":
        organism.learning_rate = 0.0
        organism.legacy_learning_enabled = False
        organism.structural_plasticity_enabled = False
    organism.enable_delayed_credit(config.reward_delay)
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    observation = environment.reset()
    actions = []
    modules = []
    for lifetime_step in range(environment.horizon):
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        actions.append(result.outputs[0])
        modules.append(organism.pending_motor_module)
        observation = environment.step(result.outputs[0])
        if lifetime_step + 1 in checkpoint_steps:
            organism = Organism.from_state_dict(json.loads(json.dumps(organism.state_dict())))
            environment = FinalLifetimeEnvironment.from_state_dict(json.loads(json.dumps(environment.state_dict())))
            values = environment.inputs[environment.t] if environment.t < environment.horizon else environment.inputs[-1]
            observation = Observation(values, environment.last_reward, environment.t >= environment.horizon)
    organism.apply_outcome(observation.reward)
    phases = []
    cursor = 0
    for name, feature, length in environment.sequence:
        end = cursor + length
        window = min(config.tail_window, length)
        phases.append({
            "name": name, "feature": feature,
            "tail_skill": _continuous_normalized_skill(actions[end-window:end], environment.targets[end-window:end]),
            "dominant_module": max(set(modules[cursor:end]), key=modules[cursor:end].count),
        })
        cursor = end
    return {
        "seed": seed, "mode": mode, "phases": phases,
        "general_growth": sum(event.get("kind") == "general_feature_installed" for event in organism.events),
        "general_reuse": sum(event.get("kind") == "general_feature_reused" for event in organism.events),
        "general_prunes": organism.general_feature_prune_count,
        "module_count": len(organism.motor_modules),
        "resource_bounded": len(organism.cells) <= organism.resources.max_cells and len(organism.graph.synapses) <= organism.resources.max_synapses,
        "final_state": organism.state_dict(), "environment_state": environment.state_dict(),
    }


def _run_final_fixed_baseline(seed: int, config: FinalLifetimeConfig, replay: bool) -> Dict[str, object]:
    """Matched fixed-topology policy-gradient baseline, optionally replayed."""
    _, input_seed, noise_seed, exploration_seed = FINAL_LIFETIME_MANIFEST[seed]
    environment = FinalLifetimeEnvironment(input_seed, noise_seed, config)
    combinations_bank = tuple(
        feature for degree in range(1, 7 if replay else 2)
        for feature in combinations(range(6), degree)
    )
    weights = [0.0] * len(combinations_bank)
    tape_rng = random.Random(exploration_seed)
    tape = tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon))
    replay_rng = random.Random(exploration_seed + 104729)
    pending = []
    memory = []
    observation = environment.reset()
    actions = []
    for step in range(environment.horizon):
        if step > config.reward_delay:
            old_features, old_mu, old_action = pending.pop(0)
            # This conventional comparator is told the environment's public
            # quadratic reward law. It reconstructs the binary label from its
            # own action and received reward; SOMA is not given this decoder.
            target = min((-1.0, 1.0), key=lambda candidate: abs(observation.reward - (1.0 - (old_action - candidate) ** 2) / 4.0))
            error = target - old_mu
            for index, value in enumerate(old_features):
                weights[index] = max(-1.0, min(1.0, weights[index] + 0.02 * value * (1.0 - old_mu * old_mu) * error))
            if replay:
                memory.append((old_features, target))
                if len(memory) > 256:
                    memory.pop(0)
                for _ in range(4):
                    replay_features, replay_target = memory[replay_rng.randrange(len(memory))]
                    replay_mu = math.tanh(sum(weight * value for weight, value in zip(weights, replay_features)))
                    replay_error = replay_target - replay_mu
                    for index, value in enumerate(replay_features):
                        weights[index] = max(-1.0, min(1.0, weights[index] + 0.01 * value * (1.0 - replay_mu * replay_mu) * replay_error))
        features = []
        for feature in combinations_bank:
            features.append(math.prod(observation.values[index] for index in feature))
        mu = math.tanh(sum(weight * value for weight, value in zip(weights, features)))
        xi = tape[step]
        action = max(-1.0, min(1.0, mu + config.exploration * xi))
        pending.append((tuple(features), mu, action))
        actions.append(action)
        observation = environment.step(action)
    phases = []
    cursor = 0
    for name, feature, length in environment.sequence:
        end = cursor + length
        window = min(config.tail_window, length)
        phases.append({"name": name, "feature": feature, "tail_skill": _continuous_normalized_skill(actions[end-window:end], environment.targets[end-window:end])})
        cursor = end
    return {"seed": seed, "mode": "replay" if replay else "fixed_topology", "phases": phases, "resource_bounded": True}


def run_final_lifetime_benchmark(config: Optional[FinalLifetimeConfig] = None) -> Dict[str, object]:
    config = config or FinalLifetimeConfig()
    config.validate()
    modes = {
        "soma": [run_final_lifetime(seed, config, "soma") for seed in config.seeds],
        "structure_disabled": [run_final_lifetime(seed, config, "structure_disabled") for seed in config.seeds],
        "learning_disabled": [run_final_lifetime(seed, config, "learning_disabled") for seed in config.seeds],
        "fixed_topology": [_run_final_fixed_baseline(seed, config, False) for seed in config.seeds],
        "replay": [_run_final_fixed_baseline(seed, config, True) for seed in config.seeds],
    }
    names = tuple(dict.fromkeys(name for name, _, _ in FINAL_LIFETIME_SEQUENCE))
    means = {
        mode: {name: _mean(phase["tail_skill"] for run in runs for phase in run["phases"] if phase["name"] == name) for name in names}
        for mode, runs in modes.items()
    }
    main = modes["soma"]
    growth = sum(run["general_growth"] >= 2 for run in main)
    reuse = sum(run["general_reuse"] > 0 for run in main)
    pruning = sum(run["general_prunes"] > 0 for run in main)
    bounded = sum(run["resource_bounded"] for run in main)
    structural_minimum = int(math.ceil(config.structural_min_fraction * len(main)))
    passed = {
        "positive_phase_skill": all(value >= config.tail_skill_floor for name, value in means["soma"].items() if name != "D") and means["soma"]["D"] >= 0.0,
        "familiar_retention": means["soma"]["A"] >= 0.20 and means["soma"]["B"] >= 0.20,
        "growth": growth >= structural_minimum, "reuse": reuse >= structural_minimum,
        "capacity_pruning": pruning >= structural_minimum, "bounded": bounded == len(main),
        "structure_margin": means["soma"]["H"] >= means["structure_disabled"]["H"] + 0.05,
        "learning_margin": _mean(means["soma"].values()) >= _mean(means["learning_disabled"].values()) + 0.05,
    }
    return {
        "protocol": {"version": 15, "manifest": [list(row) for row in FINAL_LIFETIME_MANIFEST], "sequence": [[name, list(feature), length] for name, feature, length in FINAL_LIFETIME_SEQUENCE], "controls": list(modes) + ["oracle"], "task_identity_visible": False, "oracle_is_evaluator_only": True},
        "final_lifetime": {"phase_tail_skill": means, "growth_runs": growth, "reuse_runs": reuse, "pruning_runs": pruning, "bounded_runs": bounded, "oracle_phase_tail_skill": {name: 1.0 for name in names}, "passed": passed, "all_passed": all(passed.values())},
        "runs": modes,
        "diagnostic": "FINAL_LIFETIME_PASS" if all(passed.values()) else "FINAL_LIFETIME_NOT_ESTABLISHED",
    }
class GeneralStructuralBenchmarkConfig:
    def __init__(self, **kwargs: object) -> None:
        defaults = {
            "seeds": tuple(range(8)), "exploration": 0.20, "novelty": 0.10,
            "tail_window": 64, "high_order_tail_skill_floor": 0.67,
            "growth_min_runs": 8, "reuse_min_runs": 8, "pruning_min_runs": 8,
            "control_margin": 0.005,
            "causal_drop_floor": 0.015,
        }
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError("unknown general structural config fields: %s" % ", ".join(sorted(unknown)))
        defaults.update(kwargs)
        for name, value in defaults.items():
            setattr(self, name, value)

    def validate(self) -> None:
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < len(GENERAL_STRUCTURAL_ACCEPTANCE_MANIFEST) for seed in self.seeds):
            raise ValueError("general structural seeds must use the frozen manifest")
        if self.tail_window < 1 or self.growth_min_runs < 1 or self.reuse_min_runs < 1 or self.pruning_min_runs < 1:
            raise ValueError("general structural count thresholds must be positive")
        if any(not math.isfinite(float(value)) for value in (self.exploration, self.novelty, self.high_order_tail_skill_floor, self.control_margin, self.causal_drop_floor)):
            raise ValueError("general structural rates must be finite")


def _general_structural_pruning_probe(seed: int) -> bool:
    organism = Organism.create_default(input_size=4, hidden_size=5, output_size=1, seed=seed, max_cells=32, max_synapses=256)
    organism.enable_context_modules(max_modules=3, afferent_kind="hidden")
    first = organism._recruit_motor_module(afferent_kind="hidden")
    second = organism._recruit_motor_module(afferent_kind="hidden")
    first.dormant = True
    second.dormant = True
    organism.enable_general_structural_learning(3, 6)
    before = len(organism.graph.synapses)
    first_cell = organism.install_general_feature(("input-000", "input-001"), first.id, 0.5)
    first_key = organism.general_module_features[first.id]
    proposal = next(item for item in organism.general_feature_candidates() if first_cell in item["sources"] and "input-002" in item["sources"])
    second_cell = organism.install_general_feature(proposal["sources"], second.id, -0.5)
    second_key = organism.general_module_features[second.id]
    dependency_guard = not organism.prune_general_feature(first_key)
    return bool(
        dependency_guard and organism.prune_general_feature(second_key)
        and organism.prune_general_feature(first_key)
        and organism.cells[first_cell].activation_type == "additive"
        and organism.cells[second_cell].activation_type == "additive"
        and len(organism.graph.synapses) == before
        and organism.resources.counters["synapses"] == before
    )


def run_general_structural(seed: int, config: Optional[GeneralStructuralBenchmarkConfig] = None, enabled: bool = True) -> Dict[str, object]:
    config = config or GeneralStructuralBenchmarkConfig(seeds=(seed,))
    config.validate()
    organism_seed, input_seed, representation_seed, exploration_seed = GENERAL_STRUCTURAL_ACCEPTANCE_MANIFEST[seed]
    organism = Organism.create_default(input_size=5, hidden_size=6, output_size=1, seed=organism_seed, max_cells=32, max_synapses=256)
    organism.enable_context_modules(max_modules=4, afferent_kind="hidden")
    organism.enable_motor_bootstrap()
    organism.enable_variable_order_learning(representation_seed, max_order=3)
    organism.enable_variable_order_owner_evidence()
    organism.enable_variable_order_owner_probes(True)
    organism.context_detector_drift = 0.25
    if enabled:
        organism.enable_general_structural_learning(max_depth=4, max_leaves=8)
    horizon = sum(length for _, _, length in GENERAL_STRUCTURAL_SEQUENCE)
    states = [tuple(1.0 if (mask >> index) & 1 else -1.0 for index in range(5)) for mask in range(32)]
    input_rng = random.Random(input_seed)
    inputs = []
    while len(inputs) < horizon:
        block = list(states)
        input_rng.shuffle(block)
        inputs.extend(block)
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(horizon)))
    reward = 0.0
    cursor = 0
    phase = []
    for name, feature, length in GENERAL_STRUCTURAL_SEQUENCE:
        actions = []
        targets = []
        active_modules = []
        for _ in range(length):
            values = inputs[cursor]
            target = 1.0
            for index in feature:
                target *= values[index]
            action = organism.step(values, Modulators(reward=reward, novelty=config.novelty, exploration=config.exploration)).outputs[0]
            reward = max(-1.0, min(1.0, (1.0 - (action - target) ** 2) / 4.0))
            actions.append(action)
            targets.append(target)
            active_modules.append(organism.pending_motor_module)
            cursor += 1
        window = min(config.tail_window, length)
        skill = 1.0 - _mean((action - target) ** 2 for action, target in zip(actions[-window:], targets[-window:])) / 4.0
        phase.append({"name": name, "tail_skill": skill, "active_modules": active_modules})
    organism.apply_outcome(reward)
    general_owners = set(organism.general_feature_owners.values())
    h_phases = [item for item in phase if item["name"] == "H"]
    return_owner_reuse = bool(general_owners) and all(
        sum(module in general_owners for module in item["active_modules"]) >= len(item["active_modules"]) // 2
        for item in h_phases[1:]
    )
    direct_return_recall = all(
        max(set(item["active_modules"]), key=item["active_modules"].count) not in general_owners
        for item in phase if item["name"] in ("A", "B") and item is not phase[0] and item is not phase[1]
    )
    return_owner_reuse = return_owner_reuse and direct_return_recall
    causal_drop = None
    if organism.general_feature_owners:
        def causal_skill(ablate: bool) -> float:
            clone = Organism.from_state_dict(organism.state_dict())
            key = sorted(clone.general_feature_owners)[0]
            owner = clone.motor_modules[clone.general_feature_owners[key]]
            feature_cell = str(clone.general_feature_lineage[key]["cell_id"])
            if ablate:
                clone.graph.get(feature_cell, owner.cell_id).strength = 0.0
            errors = []
            for _ in range(2):
                for values in states:
                    clone._propagate(values)
                    action = clone._module_mean_from_current_inputs(owner)
                    target = math.prod(values)
                    errors.append((action - target) ** 2)
            return 1.0 - _mean(errors[-len(states):]) / 4.0
        causal_drop = causal_skill(False) - causal_skill(True)
    return {
        "seed": seed, "enabled": enabled, "phase": phase,
        "general_growth": len(organism.general_feature_owners),
        "general_reuse": max(organism.general_feature_reuse.values(), default=0),
        "return_owner_reuse": return_owner_reuse,
        "causal_drop": causal_drop,
        "max_depth": max((int(value["depth"]) for value in organism.general_feature_lineage.values()), default=0),
        "final_state": organism.state_dict(), "pruning_probe": _general_structural_pruning_probe(organism_seed + 1),
    }


def run_general_structural_benchmark(config: Optional[GeneralStructuralBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or GeneralStructuralBenchmarkConfig()
    config.validate()
    main = [run_general_structural(seed, config, True) for seed in config.seeds]
    control = [run_general_structural(seed, config, False) for seed in config.seeds]
    growth = sum(result["general_growth"] > 0 and result["max_depth"] >= 2 for result in main)
    reuse = sum(bool(result["return_owner_reuse"]) for result in main)
    pruning = sum(bool(result["pruning_probe"]) for result in main)
    causal = sum(result["causal_drop"] is not None and result["causal_drop"] >= config.causal_drop_floor for result in main)
    main_h = _mean(item["tail_skill"] for result in main for item in result["phase"] if item["name"] == "H")
    control_h = _mean(item["tail_skill"] for result in control for item in result["phase"] if item["name"] == "H")
    passed = {
        "graph_derived_growth": growth >= config.growth_min_runs,
        "recurring_structure_reuse": reuse >= config.reuse_min_runs,
        "recoverable_dependency_safe_pruning": pruning >= config.pruning_min_runs,
        "high_order_tail_skill": all(item["tail_skill"] >= config.high_order_tail_skill_floor for result in main for item in result["phase"] if item["name"] == "H"),
        "structure_control_margin": main_h >= control_h + config.control_margin,
        "causal_feature_dependence": causal == len(main),
    }
    return {
        "protocol": {"version": 13, "seeds": list(config.seeds), "manifest": [list(row) for row in GENERAL_STRUCTURAL_ACCEPTANCE_MANIFEST], "sequence": [(name, list(feature), length) for name, feature, length in GENERAL_STRUCTURAL_SEQUENCE], "task_identity_visible": False},
        "general_structural": {"growth_runs": growth, "reuse_runs": reuse, "pruning_runs": pruning, "causal_runs": causal, "mean_high_order_tail_skill": main_h, "control_high_order_tail_skill": control_h, "mean_causal_drop": _mean(result["causal_drop"] for result in main), "passed": passed, "all_passed": all(passed.values())},
        "runs": {"general_structural": main, "structure_disabled": control},
        "diagnostic": "GENERAL_STRUCTURAL_PASS" if all(passed.values()) else "GENERAL_STRUCTURAL_NOT_ESTABLISHED",
    }


def compositional_walsh_proof() -> Dict[str, object]:
    """Exact integer Walsh certificates for quartics versus degree <=3 features."""
    states = [tuple(1 if (mask >> index) & 1 else -1 for index in range(6)) for mask in range(64)]
    lower_features = tuple(feature for degree in range(4) for feature in combinations(range(6), degree))
    lower_projection: Dict[str, Tuple[int, ...]] = {}
    own_projection: Dict[str, int] = {}
    cross_projection: Dict[str, Tuple[int, ...]] = {}

    def monomial(values: Sequence[int], feature: Sequence[int]) -> int:
        result = 1
        for index in feature:
            result *= values[index]
        return result

    for quartic in COMPOSITIONAL_QUARTIC_FEATURES:
        target = [monomial(values, quartic) for values in states]
        name = "_".join(str(index) for index in quartic)
        lower_projection[name] = tuple(sum(monomial(values, feature) * value for values, value in zip(states, target)) for feature in lower_features)
        own_projection[name] = sum(value * value for value in target)
        cross_projection[name] = tuple(
            sum(value * monomial(state, other) for state, value in zip(states, target))
            for other in COMPOSITIONAL_QUARTIC_FEATURES if other != quartic
        )
    return {
        "lower_features": [list(feature) for feature in lower_features],
        "quartic_features": [list(feature) for feature in COMPOSITIONAL_QUARTIC_FEATURES],
        "quartic_vs_degree_le3": lower_projection,
        "own_quartic_projection": own_projection,
        "other_quartic_projection": cross_projection,
        "max_abs_lower_projection": max(abs(value) for row in lower_projection.values() for value in row),
        "own_quartic_basis": own_projection,
        "max_abs_other_quartic_projection": max(abs(value) for row in cross_projection.values() for value in row),
        "integer_certificate": True,
    }


def compositional_parity_proof() -> Dict[str, object]:
    return compositional_walsh_proof()


def _compositional_phase_windows(lengths: Sequence[int]) -> Dict[str, Tuple[int, int]]:
    if len(tuple(lengths)) != 12:
        raise ValueError("compositional benchmark requires twelve phase lengths")
    windows = {}
    start = 0
    for name, length in zip(COMPOSITIONAL_POLICY_SEQUENCE, lengths):
        windows[name] = (start, start + int(length))
        start += int(length)
    return windows


def _composition_install_matches_feature(organism: Organism, event: Mapping[str, object], feature: Sequence[int]) -> bool:
    """Return whether an install event closes exactly the requested task leaves."""
    leaves = organism.composition_feature_lineage.get(str(event.get("feature_key")), {}).get("leaf_closure", ())
    flattened = tuple(str(value) for leaf in leaves for value in leaf)
    expected = tuple(sorted(organism.input_ids[int(index)] for index in feature))
    return len(flattened) == len(expected) and len(set(flattened)) == len(flattened) and set(flattened) == set(expected)


def _compositional_organism(seed: int, mode: str, config: CompositionalBenchmarkConfig) -> Organism:
    base_mode = "variable_order" if mode in ("compositional", "composition_disabled_v10") else mode
    v10_config = VariableOrderBenchmarkConfig(max_modules=config.max_modules)
    organism = _variable_order_organism(seed, base_mode, v10_config)
    return organism


def _compositional_mode_run(seed: int, config: CompositionalBenchmarkConfig, mode: str) -> Dict[str, object]:
    allowed = ("compositional", "composition_disabled_v10", "additive_hidden", "raw_linear_modules", "no_actor", "frozen")
    if mode not in allowed:
        raise ValueError("unsupported compositional mode: %s" % mode)
    organism_seed, input_seed, schedule_seed, exploration_seed, representation_seed, composition_seed, task_seed = compositional_seed_manifest(seed)
    lengths = compositional_phase_lengths_for_seed(schedule_seed, config.phase_lengths)
    environment = CompositionalBenchmarkEnvironment(input_seed, lengths, schedule_seed=schedule_seed, task_seed=task_seed)
    organism = _compositional_organism(organism_seed, mode, config)
    if mode in ("compositional", "composition_disabled_v10"):
        organism.enable_variable_order_learning(representation_seed, max_order=3)
    if mode == "compositional":
        organism.enable_compositional_substrate(composition_seed)
        if config.composition_learning_enabled:
            organism.enable_composition_learning()
    organism.actor_learning_rate = config.actor_learning_rate if mode in ("compositional", "composition_disabled_v10", "additive_hidden", "raw_linear_modules") else 0.0
    tape_rng = random.Random(exploration_seed)
    exploration_tape = tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon))
    organism.set_exploration_tape(exploration_tape)
    initial_cell_ids = tuple(sorted(organism.cells))
    initial_synapse_ids = tuple(synapse.id for synapse in organism.graph.iter_synapses())
    initial_synapse_strengths = tuple((synapse.id, synapse.strength) for synapse in organism.graph.iter_synapses())
    observation = environment.reset()
    actions: List[float] = []
    targets: List[float] = []
    rewards: List[float] = []
    timeline: List[Optional[str]] = []
    for _ in range(environment.horizon):
        targets.append(environment.target_at())
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        timeline.append(organism.pending_motor_module)
        observation = environment.step(result.outputs[0])
        actions.append(result.outputs[0])
        rewards.append(observation.reward)
    organism.apply_outcome(observation.reward)
    windows = _compositional_phase_windows(environment.phase_lengths)
    metrics = _multi_phase_metrics(actions, targets, windows, config.tail_fraction)
    canonical: Dict[str, Optional[str]] = {}
    for name, (start, end) in windows.items():
        tail_start = end - max(1, int((end - start) * config.tail_fraction))
        tail_ids = [value for value in timeline[tail_start:end] if value is not None]
        canonical[name] = max(set(tail_ids), key=tail_ids.count) if tail_ids and tail_ids.count(max(set(tail_ids), key=tail_ids.count)) / float(len(tail_ids)) >= 0.70 else None
    first = {policy: next(name for name in windows if name.split("_", 1)[0] == policy) for policy in ("P0", "P1", "Q0", "C0", "P2", "P3", "Q1", "C1")}
    return_names = ("P0_return", "Q0_return", "C0_return", "Q1_return")
    reuse = {name: canonical[name] is not None and canonical[name] == canonical[first[name.split("_", 1)[0]]] for name in return_names}
    lags = {name: _first_canonical_reactivation_lag(organism.events, windows[name][0], windows[name][1], canonical[first[name.split("_", 1)[0]]]) for name in return_names}
    direct_edges = tuple(synapse.id for synapse in organism.graph.iter_synapses() if synapse.source in organism.input_ids and organism.cells[synapse.destination].kind == "motor_module")
    feature_cells = [cell for cell in organism.cells.values() if cell.activation_type in ("dendritic_product", "dendritic_product_n", "dendritic_product_composed")]
    composed_cells = [cell for cell in organism.cells.values() if cell.activation_type == "dendritic_product_composed"]
    valid_composed_closures = 0
    for cell in composed_cells:
        if len(cell.dendritic_sources) != 2 or any(source not in organism.cells for source in cell.dendritic_sources):
            continue
        source_cells = [organism.cells[source] for source in cell.dendritic_sources]
        leaves = [tuple(source.dendritic_sources) for source in source_cells]
        if all(source.activation_type == "dendritic_product" and len(leaf) == 2 for source, leaf in zip(source_cells, leaves)) and not set(leaves[0]) & set(leaves[1]) and len(set(leaves[0] + leaves[1])) == 4:
            key = organism._composition_feature_key(cell.dendritic_sources, leaves)
            owner_id = organism.composition_feature_owners.get(key)
            if owner_id in organism.motor_modules and organism.graph.get(cell.id, organism.motor_modules[owner_id].cell_id) is not None:
                valid_composed_closures += 1
    composed_installs = sum(1 for event in organism.events if event.get("kind") == "composed_feature_installed")
    q_tail_starts = {
        name: windows[name][1] - max(1, int((windows[name][1] - windows[name][0]) * config.tail_fraction))
        for name in ("Q0", "Q1")
    }
    composition_install_events = [
        {"step": int(event.get("step", -1)), "feature_key": event.get("feature_key"), "owner_module": event.get("owner_module")}
        for event in organism.events if event.get("kind") == "composed_feature_installed"
    ]
    q_features = {"Q0": environment.task_features[2], "Q1": environment.task_features[6]}
    install_timing = {
        name: {
            "tail_start": start,
            "events_before_tail": [dict(event) for event in composition_install_events if 0 <= int(event["step"]) < start and _composition_install_matches_feature(organism, event, q_features[name])],
            "all_events_before_tail": [dict(event) for event in composition_install_events if 0 <= int(event["step"]) < start],
            "passed": any(0 <= int(event["step"]) < start and _composition_install_matches_feature(organism, event, q_features[name]) for event in composition_install_events),
        }
        for name, start in q_tail_starts.items()
    }
    return {
        "seed": int(seed), "mode": mode, "organism_seed": organism_seed, "input_seed": input_seed,
        "schedule_seed": schedule_seed, "exploration_seed": exploration_seed, "representation_seed": representation_seed,
        "composition_seed": composition_seed, "task_seed": task_seed, "task_features": tuple(environment.task_features),
        "exploration_tape_signature": (len(exploration_tape), tuple(exploration_tape[:4]), tuple(exploration_tape[-4:]), sum(exploration_tape)),
        "phase_lengths": tuple(environment.phase_lengths), "actions": actions, "targets": targets, "rewards": rewards,
        "phase_metrics": metrics, "canonical_by_segment": canonical, "canonical_reuse": reuse,
        "reactivation_lag_by_segment": lags, "active_module_timeline": tuple(timeline), "module_count": len(organism.motor_modules),
        "direct_input_motor_edges": direct_edges, "initial_cell_ids": initial_cell_ids, "initial_synapse_ids": initial_synapse_ids,
        "initial_synapse_strengths": initial_synapse_strengths, "final_state": organism.state_dict(), "environment_state": environment.state_dict(),
        "target_blind": True, "composition_learning_enabled": bool(organism.composition_learning_enabled),
        "composed_feature_installs": composed_installs, "valid_four_leaf_closures": valid_composed_closures,
        "composition_install_timing": install_timing,
        "composed_feature_installed_before_Q_tail": bool(install_timing["Q0"]["passed"] and install_timing["Q1"]["passed"]),
        "composition_route_count": organism.composition_route_count,
        "composition_candidate_count": organism.composition_candidate_count,
        "feature_cell_count": len(feature_cells), "composition_seed_published": True,
        "fixed_random_order4_status": "deferred_non_gating",
        "composition_substrate_status": "learned_pair_of_pairs" if organism.composition_learning_enabled else ("seeded_substrate_only" if mode == "compositional" else "disabled"),
        "stationary_safety": None,
    }


def _compositional_stationary_safety(seed: int, config: CompositionalBenchmarkConfig) -> Dict[str, object]:
    organism_seed, input_seed, schedule_seed, exploration_seed, representation_seed, composition_seed, task_seed = compositional_seed_manifest(seed)
    length = int(config.phase_lengths[0])
    task_features = compositional_task_manifest(task_seed)
    environment = CompositionalBenchmarkEnvironment(input_seed, (length,) * 12, schedule_seed=schedule_seed, task_seed=task_seed, task_features=task_features, custom_task_features=True)
    organism = _compositional_organism(organism_seed, "compositional", config)
    organism.enable_variable_order_learning(representation_seed, max_order=3)
    organism.enable_compositional_substrate(composition_seed)
    if config.composition_learning_enabled:
        organism.enable_composition_learning()
    organism.actor_learning_rate = config.actor_learning_rate
    tape_rng = random.Random(exploration_seed)
    organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
    observation = environment.reset()
    for _ in range(length):
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
        observation = environment.step(result.outputs[0])
    organism.apply_outcome(observation.reward)
    warning_kinds = {"motor_warning_started", "motor_warning_escalated", "motor_module_reactivated", "motor_module_probe_started"}
    warning_events = sum(event.get("kind") in warning_kinds for event in organism.events)
    return {
        "safe": len(organism.motor_modules) == 1 and warning_events == 0,
        "module_count": len(organism.motor_modules), "warning_events": warning_events,
        "recruitments": sum(event.get("kind") == "motor_module_recruited" for event in organism.events),
        "false_switches": sum(event.get("kind") in {"motor_module_reactivated", "motor_module_probe_started"} for event in organism.events),
        "composition_substrate_status": "learned_pair_of_pairs" if organism.composition_learning_enabled else "seeded_substrate_only",
        "composition_substrate_enabled": organism.compositional_substrate_enabled,
        "composition_representation_seed": organism.composition_representation_seed,
        "composition_feature_count": len(organism.composition_feature_owners),
    }


def _summarize_compositional(results: Sequence[Mapping[str, object]], config: CompositionalBenchmarkConfig) -> Dict[str, object]:
    windows = _compositional_phase_windows(results[0]["phase_lengths"])
    phase = {name: {metric: _mean(float(result["phase_metrics"][name][metric]) for result in results) for metric in ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")} for name in windows}
    return_names = ("P0_return", "Q0_return", "C0_return", "Q1_return")
    reuse = {name: sum(bool(result["canonical_reuse"].get(name, False)) for result in results) for name in return_names}
    lags = [lag for result in results for name in return_names for lag in result["reactivation_lag_by_segment"].get(name, ())]
    median_lag = sorted(lags)[len(lags) // 2] if lags else None
    stationary_details = {str(result["seed"]): dict(result["stationary_safety"]) for result in results if isinstance(result.get("stationary_safety"), Mapping)}
    stationary_runs = sum(bool(detail.get("safe")) for detail in stationary_details.values()) if stationary_details else None
    q0_install_runs = sum(bool(result.get("composition_install_timing", {}).get("Q0", {}).get("passed", False)) for result in results)
    q1_install_runs = sum(bool(result.get("composition_install_timing", {}).get("Q1", {}).get("passed", False)) for result in results)
    pair_first = ("P0", "P1", "P2", "P3")
    cubic_first = ("C0", "C1")
    quartic_first = ("Q0", "Q1")
    passed = {
        "initial_pair_tail_skill_all": all(phase[name]["tail_skill"] >= config.pair_tail_skill_floor for name in pair_first),
        "initial_pair_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.pair_tail_sign_accuracy_floor for name in pair_first),
        "initial_cubic_tail_skill_all": all(phase[name]["tail_skill"] >= config.cubic_tail_skill_floor for name in cubic_first),
        "initial_cubic_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.cubic_tail_sign_accuracy_floor for name in cubic_first),
        "initial_composed_tail_skill_all": all(phase[name]["tail_skill"] >= config.quartic_tail_skill_floor for name in quartic_first),
        "initial_composed_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.quartic_tail_sign_accuracy_floor for name in quartic_first),
        "initial_quartic_tail_skill_all": all(phase[name]["tail_skill"] >= config.quartic_tail_skill_floor for name in quartic_first),
        "initial_quartic_tail_sign_all": all(phase[name]["tail_sign_accuracy"] >= config.quartic_tail_sign_accuracy_floor for name in quartic_first),
        "returned_P0_early": phase["P0_return"]["early_skill"] >= config.return_early_skill_floor,
        "returned_P0_tail": phase["P0_return"]["tail_skill"] >= config.return_tail_skill_floor,
        "returned_Q0_early": phase["Q0_return"]["early_skill"] >= config.return_early_skill_floor,
        "returned_Q0_tail": phase["Q0_return"]["tail_skill"] >= config.return_tail_skill_floor,
        "returned_C0_early": phase["C0_return"]["early_skill"] >= config.return_early_skill_floor,
        "returned_C0_tail": phase["C0_return"]["tail_skill"] >= config.return_tail_skill_floor,
        "returned_Q1_early": phase["Q1_return"]["early_skill"] >= config.return_early_skill_floor,
        "returned_Q1_tail": phase["Q1_return"]["tail_skill"] >= config.return_tail_skill_floor,
        "canonical_reuse_all_returns": all(reuse[name] >= config.canonical_reuse_min_runs for name in return_names),
        "reactivation_lag": median_lag is not None and median_lag <= config.reactivation_lag_limit,
        "capacity": all(int(result["module_count"]) <= config.max_modules for result in results),
        "zero_direct_input_motor": all(not result["direct_input_motor_edges"] for result in results),
        "stationary_safety": stationary_runs is not None and stationary_runs >= config.stationary_safety_min_runs,
        "composed_feature_installed_before_Q0_tail": q0_install_runs >= config.composed_install_min_runs,
        "composed_feature_installed_before_Q1_tail": q1_install_runs >= config.composed_install_min_runs,
        "composed_feature_installed_before_Q_tail": q0_install_runs >= config.composed_install_min_runs and q1_install_runs >= config.composed_install_min_runs,
        "valid_four_distinct_leaf_closures": sum(bool(result["valid_four_leaf_closures"]) for result in results) >= config.closure_min_runs,
        "composition_ablation_Q_drop": False,
        "direct_ablation_drop_limit": False,
        "composition_learning_enabled": bool(config.composition_learning_enabled),
    }
    return {
        "seeds": len(results), "mean_phase_metrics": phase, "mean_module_count": _mean(float(result["module_count"]) for result in results),
        "canonical_reuse_counts": reuse, "median_reactivation_lag": median_lag, "stationary_safety_runs": stationary_runs,
        "stationary_safety_by_seed": stationary_details, "composition_install_Q0_runs": q0_install_runs, "composition_install_Q1_runs": q1_install_runs, "initial_pair_gate_phases": list(pair_first),
        "composition_install_timing_by_seed": {str(result["seed"]): dict(result.get("composition_install_timing", {})) for result in results},
        "initial_cubic_gate_phases": list(cubic_first), "initial_quartic_gate_phases": list(quartic_first),
        "passed": passed, "all_passed": all(passed.values()),
        "thresholds": {"pair_tail_skill_floor": config.pair_tail_skill_floor, "cubic_tail_skill_floor": config.cubic_tail_skill_floor, "quartic_tail_skill_floor": config.quartic_tail_skill_floor, "canonical_reuse_min_runs": config.canonical_reuse_min_runs, "reactivation_lag_limit": config.reactivation_lag_limit, "max_modules": config.max_modules, "stationary_safety_min_runs": config.stationary_safety_min_runs, "composed_install_min_runs": config.composed_install_min_runs, "closure_min_runs": config.closure_min_runs},
    }


def run_compositional(seed: int, config: Optional[CompositionalBenchmarkConfig] = None, mode: str = "compositional") -> Dict[str, object]:
    config = config or CompositionalBenchmarkConfig()
    config.validate()
    return _compositional_mode_run(seed, config, mode)


def run_compositional_benchmark(config: Optional[CompositionalBenchmarkConfig] = None) -> Dict[str, object]:
    config = config or CompositionalBenchmarkConfig()
    config.validate()
    modes = ("compositional", "composition_disabled_v10", "additive_hidden", "raw_linear_modules", "no_actor", "frozen")
    runs = {mode: [_compositional_mode_run(seed, config, mode) for seed in config.seeds] for mode in modes}
    for result in runs["compositional"]:
        result["stationary_safety"] = _compositional_stationary_safety(int(result["seed"]), config)
        result["causal_ablation"] = _evaluate_composition_causal_ablation(result, config)
    summaries = {mode: _summarize_compositional(runs[mode], config) for mode in modes}
    main_q = _mean(float(result["phase_metrics"]["Q0_return"]["tail_skill"]) for result in runs["compositional"])
    raw_q = _mean(float(result["phase_metrics"]["Q0_return"]["tail_skill"]) for result in runs["raw_linear_modules"])
    disabled_q = _mean(float(result["phase_metrics"]["Q0_return"]["tail_skill"]) for result in runs["composition_disabled_v10"])
    main_direct = _mean(float(result["phase_metrics"]["P0_return"]["tail_skill"]) for result in runs["compositional"])
    disabled_direct = _mean(float(result["phase_metrics"]["P0_return"]["tail_skill"]) for result in runs["composition_disabled_v10"])
    main_cubic = _mean(float(result["phase_metrics"]["C0_return"]["tail_skill"]) for result in runs["compositional"])
    disabled_cubic = _mean(float(result["phase_metrics"]["C0_return"]["tail_skill"]) for result in runs["composition_disabled_v10"])
    causal_results = [result["causal_ablation"] for result in runs["compositional"]]
    causal_q_results = [item for item in causal_results if item.get("available") and item.get("q_drop") is not None]
    causal_direct_results = [item for item in causal_results if item.get("available") and item.get("direct_available") and item.get("direct_degradation") is not None]
    causal_q_drop = _mean(float(item["q_drop"]) for item in causal_q_results) if causal_q_results else None
    causal_direct_degradation = _mean(float(item["direct_degradation"]) for item in causal_direct_results) if causal_direct_results else None
    causal_q_all_runs = len(causal_q_results) == len(config.seeds)
    causal_direct_all_runs = len(causal_direct_results) == len(config.seeds)
    causal_q_gate = causal_q_all_runs and causal_q_drop is not None and causal_q_drop >= config.quartic_ablation_drop
    causal_direct_gate = causal_direct_all_runs and causal_direct_degradation is not None and causal_direct_degradation <= config.direct_ablation_drop_limit
    matched = all(
        runs["compositional"][index][field] == runs["composition_disabled_v10"][index][field]
        for index in range(len(config.seeds))
        for field in ("initial_cell_ids", "initial_synapse_ids", "initial_synapse_strengths", "exploration_tape_signature", "phase_lengths", "task_features")
    )
    comparison = {
        "matched_seed_and_protocol": all(result["seed"] == config.seeds[index] for index, result in enumerate(runs["compositional"])),
        "matched_starting_structure": matched,
        "matched_composition_disabled_replay": matched,
        "matched_exploration_tape": all(runs["compositional"][i]["exploration_tape_signature"] == runs["composition_disabled_v10"][i]["exploration_tape_signature"] for i in range(len(config.seeds))),
        "matched_phase_lengths": all(runs["compositional"][i]["phase_lengths"] == runs["composition_disabled_v10"][i]["phase_lengths"] for i in range(len(config.seeds))),
        "matched_task_features": all(runs["compositional"][i]["task_features"] == runs["composition_disabled_v10"][i]["task_features"] for i in range(len(config.seeds))),
        "matched_input_stream": all(runs["compositional"][i]["environment_state"]["inputs"] == runs["composition_disabled_v10"][i]["environment_state"]["inputs"] for i in range(len(config.seeds))),
        "main_Q_return_tail_skill_minus_raw": main_q - raw_q,
        "main_Q_return_tail_skill_minus_composition_disabled_v10": main_q - disabled_q,
        "main_P_return_tail_skill_minus_composition_disabled_v10": main_direct - disabled_direct,
        "main_Q_beats_raw_by_margin": main_q >= raw_q + config.comparison_margin,
        "main_Q_beats_composition_disabled_by_margin": main_q >= disabled_q + config.comparison_margin,
        "composition_control_Q_difference": main_q - disabled_q,
        "composition_control_Q_difference_margin_gate": main_q >= disabled_q + config.comparison_margin,
        "causal_Q_ablation_drop": causal_q_drop,
        "causal_Q_ablation_drop_gate": causal_q_gate,
        "causal_direct_degradation": causal_direct_degradation,
        "causal_direct_degradation_gate": causal_direct_gate,
        "causal_ablation_available_runs": len(causal_q_results),
        "causal_direct_available_runs": len(causal_direct_results),
        "causal_ablation_all_runs": causal_q_all_runs,
        "causal_direct_all_runs": causal_direct_all_runs,
        "fixed_random_order4_status": "deferred_non_gating",
    }
    summaries["compositional"]["passed"]["composition_ablation_Q_drop"] = (
        causal_q_gate if config.composition_learning_enabled else False
    )
    summaries["compositional"]["passed"]["direct_ablation_drop_limit"] = (
        causal_direct_gate if config.composition_learning_enabled else False
    )
    summaries["compositional"]["passed"]["causal_ablation_all_runs"] = (
        causal_q_all_runs if config.composition_learning_enabled else False
    )
    summaries["compositional"]["passed"]["causal_direct_all_runs"] = (
        causal_direct_all_runs if config.composition_learning_enabled else False
    )
    summaries["compositional"]["passed"]["Q_return_margin_vs_raw"] = comparison["main_Q_beats_raw_by_margin"]
    summaries["compositional"]["passed"]["Q_return_margin_vs_composition_disabled"] = comparison["main_Q_beats_composition_disabled_by_margin"]
    summaries["compositional"]["passed"]["composition_learning_enabled"] = bool(config.composition_learning_enabled)
    summaries["compositional"]["all_passed"] = bool(config.composition_learning_enabled) and all(summaries["compositional"]["passed"].values())
    return {
        "protocol": {
            "version": 11, "phase_lengths_base": list(config.phase_lengths),
            "phase_lengths_by_seed": {str(seed): list(compositional_phase_lengths_for_seed(compositional_seed_manifest(seed)[2], config.phase_lengths)) for seed in config.seeds},
            "seeds": list(config.seeds), "acceptance_manifest": [list(row) for row in COMPOSITIONAL_ACCEPTANCE_MANIFEST],
            "stream_names": ["organism", "input", "schedule", "exploration", "representation", "composition", "task"],
            "stream_seeds": {
                str(seed): {
                    name: value for name, value in zip(
                        ("organism", "input", "schedule", "exploration", "representation", "composition", "task"),
                        compositional_seed_manifest(seed),
                    )
                }
                for seed in config.seeds
            },
            "representation_seeds": {str(seed): compositional_seed_manifest(seed)[4] for seed in config.seeds},
            "composition_seeds": {str(seed): compositional_seed_manifest(seed)[5] for seed in config.seeds},
            "task_manifest": [[list(feature) for feature in row] for row in COMPOSITIONAL_TASK_MANIFEST],
            "factorization_relations": list(compositional_factorization_manifest()),
            "policy_sequence": list(COMPOSITIONAL_POLICY_SEQUENCE), "balanced_block_size": 64, "sensor_count": 6,
            "walsh_proof": compositional_walsh_proof(), "independent_rng_streams": True,
            "composition_learning_enabled": bool(config.composition_learning_enabled), "phase_metadata_in_organism": False, "controls_target_blind": True,
            "fingerprint_window": 16,
            "fixed_random_order4_status": "deferred_non_gating",
            "composition_substrate_status": "learned_pair_of_pairs" if config.composition_learning_enabled else "seeded_substrate_only",
            "causal_ablation": "same_final_state_clone; zero_only_composed_cell_to_owner_motor_edges; evaluator_only_no_learning_or_exploration",
            "composition_control_status": "matched_independent_v10_training_control",
        },
        **summaries, "runs": runs, "causal_ablation_by_seed": {str(result["seed"]): result["causal_ablation"] for result in runs["compositional"]}, "comparison": comparison,
        "red_report_gates": summaries["compositional"]["passed"],
        "diagnostic": "COMPOSITIONAL_REPRESENTATION_PASS" if bool(config.composition_learning_enabled) and all(summaries["compositional"]["passed"].values()) and comparison["main_Q_beats_raw_by_margin"] else "COMPOSITIONAL_REPRESENTATION_NOT_ESTABLISHED",
    }

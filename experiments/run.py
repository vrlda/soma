"""Reproducible prequential experiments: python -m experiments.run --help."""
from dataclasses import asdict
from contextvars import ContextVar
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from brain import Config, Organism

CHECKPOINT = ContextVar("checkpoint_destination", default=None)

VARIANTS = {
    "candidate": {},
    "no_learning": {"learning": False},
    "no_traces": {"traces": False},
    "no_hidden": {"initial_hidden": 0},
    "no_recurrence": {"recurrence": False},
    "no_persistence": {"persistence": False},
    "no_hidden_plasticity": {"hidden_plasticity": False},
    "metaplasticity": {"metaplasticity": True},
    "structure": {"growth": True, "pruning": True},
    "no_drive_memory": {"drive_trace_rule": "none"},
    "no_growth": {"growth": False},
    "no_vigilance": {"vigilance": 0.0},
    "no_pruning": {"pruning": False},
    "no_retention": {"retention_protection": False},
    "no_structural_safeguards": {"retain_active_representations": False,
                                 "regrow_predictions": False, "match_evidence": False,
                                 "retention_protection": False},
}


def onehot(index, size):
    return [float(i == index) for i in range(size)]


def metrics(scores):
    if not scores:
        return {"n": 0, "accuracy": None, "log_loss": None}
    return {"n": len(scores), "accuracy": statistics.mean(x[0] for x in scores),
            "log_loss": statistics.mean(x[1] for x in scores)}


def score(probabilities, target):
    # Exact ties earn expected accuracy, without target-informed tie breaking.
    best = max(probabilities)
    winners = [i for i, p in enumerate(probabilities) if p == best]
    correct = (1 / len(winners)) if target in winners else 0.0
    return correct, -math.log(max(probabilities[target], 1e-300))


def config(seed, inputs, outputs, variant, overrides):
    values = dict(inputs=inputs, outputs=outputs, seed=seed)
    values.update(overrides)
    values.update(VARIANTS[variant])  # An ablation cannot be undone by an override.
    return Config(**values)


def finish(brain, scores, trajectory, **extra):
    checkpoint = CHECKPOINT.get()
    if checkpoint is not None:
        brain.save(checkpoint)
        extra["checkpoint"] = str(checkpoint)
    return dict(config=asdict(brain.config), metrics=metrics(scores),
                brain=brain.summary(), trajectory=trajectory, **extra)


def sequence(seed, variant, overrides):
    brain = Organism(config(seed, 4, 4, variant, overrides))
    scores, phases, trajectory = [], [], []
    for cycle in ((0, 1, 2, 3), (0, 2, 1, 3)):
        phase = []
        for step in range(800):
            current, target = cycle[step % 4], cycle[(step + 1) % 4]
            probabilities = brain.experience(onehot(current, 4), target)
            phase.append(score(probabilities, target))
            if step % 100 == 99:
                trajectory.append(brain.summary())
        phases.append({"first_100": metrics(phase[:100]), "last_200": metrics(phase[-200:])})
        scores.extend(phase[-200:])
    return finish(brain, scores, trajectory, phases=phases,
                  controls={"last_symbol_accuracy": 0.0, "uniform_accuracy": .25})


def delay(seed, variant, overrides, gap=16, conflict=False):
    brain = Organism(config(seed, 5, 2, variant, overrides))
    stream = random.Random(seed + 10000)
    scores, trajectory = [], []
    for trial in range(600):
        cue = stream.randrange(2)
        brain.experience(onehot(cue, 5))
        for _ in range(gap):
            brain.experience(onehot(2 + stream.randrange(2), 5))
        if conflict:
            cue = stream.randrange(2)
            brain.experience(onehot(cue, 5))
            for _ in range(4):
                brain.experience(onehot(2 + stream.randrange(2), 5))
        probabilities = brain.experience(onehot(4, 5), cue)
        scores.append(score(probabilities, cue))
        if trial % 100 == 99:
            trajectory.append(brain.summary())
    return finish(brain, scores[-200:], trajectory, gap=gap, conflict=conflict,
                  first_100=metrics(scores[:100]), controls={"uniform_accuracy": .5})


def reversal(seed, variant, overrides):
    brain = Organism(config(seed, 2, 2, variant, overrides))
    stream = random.Random(seed + 20000)
    scores, trajectory, phases = [], [], []
    for mapping, count in ((0, 600), (1, 600), (0, 300)):
        phase = []
        for i in range(count):
            cue = stream.randrange(2)
            probabilities = brain.experience(onehot(cue, 2), cue ^ mapping)
            phase.append(score(probabilities, cue ^ mapping))
            if i % 100 == 99:
                trajectory.append(brain.summary())
        phases.append({"first_50": metrics(phase[:50]), "last_200": metrics(phase[-200:])})
        scores.extend(phase[-200:])
    return finish(brain, scores, trajectory, phases=phases)


def structure(seed, variant, overrides, noise=False):
    brain = Organism(config(seed, 4, 2, variant, overrides))
    stream = random.Random(seed + 30000)
    scores, trajectory = [], []
    for step in range(3000):
        cue = stream.randrange(4)
        target = stream.randrange(2) if noise else cue % 2
        probabilities = brain.experience(onehot(cue, 4), target)
        scores.append(score(probabilities, target))
        if step % 100 == 99:
            trajectory.append(brain.summary())
    return finish(brain, scores[-500:], trajectory, noise=noise)


def contextual(seed, variant, overrides, rehearsal="none"):
    # Same cue, opposite outcomes in two observable sensory contexts.
    brain = Organism(config(seed, 4, 2, variant, overrides))
    stream = random.Random(seed + 40000)
    replay_rng = random.Random(seed + 50000)
    replay = []  # Temporary raw-experience scaffolding, no inference path.
    trajectory, phases = [], []
    def observation(context, cue):
        return [float(cue == 0), float(cue == 1), float(context == 0), float(context == 1)]
    for context in (0, 1):
        scores = []
        for i in range(800):
            cue = stream.randrange(2)
            x, target = observation(context, cue), cue ^ context
            probabilities = brain.experience(x, target)
            scores.append(score(probabilities, target))
            if context == 0:
                if len(replay) < 128:
                    replay.append((x, target))
                else:
                    j = replay_rng.randrange(i + 1)
                    if j < 128:
                        replay[j] = (x, target)
            if context == 1 and i % 10 == 9:
                if rehearsal in ("wake", "sleep"):
                    for _ in range(10):
                        old_x, old_target = replay_rng.choice(replay)
                        brain.experience(old_x, old_target)
                elif rehearsal == "rest":
                    for _ in range(10):
                        brain.experience([0.0] * 4)
            if i % 100 == 99:
                trajectory.append(brain.summary())
        phases.append({"first_50": metrics(scores[:50]), "last_200": metrics(scores[-200:])})
    # Probe a checkpoint clone, still learning; original lifetime untouched.
    probe = Organism.from_state(brain.state())
    probe_scores = []
    for i in range(200):
        cue = stream.randrange(2)
        probabilities = probe.experience(observation(0, cue), cue)
        probe_scores.append(score(probabilities, cue))
    return finish(brain, [], trajectory, rehearsal=rehearsal, phases=phases,
                  recall_first_20=metrics(probe_scores[:20]),
                  recall_last_100=metrics(probe_scores[-100:]),
                  replay_presentations=800 if rehearsal in ("wake", "sleep") else 0,
                  replay_capacity=128)


def grammar(seed, variant, overrides, gap=2, budget_seconds=None):
    brain = Organism(config(seed, 6, 2, variant, overrides))
    stream = random.Random(seed + 60000)
    scores, trajectory = [], []
    start = time.monotonic()
    for episode in range(1200):
        if budget_seconds is not None and time.monotonic() - start > budget_seconds:
            break
        for _ in range(8):
            brain.experience([0.0] * 6)
        first, second = stream.randrange(2), stream.randrange(2)
        brain.experience(onehot(first, 6))
        for _ in range(gap):
            brain.experience(onehot(4, 6))
        brain.experience(onehot(2 + second, 6))
        target = first ^ second  # Environment only; never an internal feature.
        probabilities = brain.experience(onehot(5, 6), target)
        scores.append(score(probabilities, target))
        if episode % 100 == 99:
            trajectory.append(brain.summary())
    return finish(brain, scores[-400:], trajectory, gap=gap,
                  first_200=metrics(scores[:200]), controls={"uniform_accuracy": .5},
                  complete=len(scores) == 1200, episodes=len(scores),
                  termination="finished" if len(scores) == 1200 else "time_budget")


def lifetime(seed, variant, overrides, budget_seconds=None):
    brain = Organism(config(seed, 7, 2, variant, overrides))
    stream = random.Random(seed + 70000)
    phases, trajectory = [], []
    start = time.monotonic()
    complete = True
    for context, count in ((0, 1200), (1, 1200), (0, 400)):
        scores = []
        for episode in range(count):
            if budget_seconds is not None and time.monotonic() - start > budget_seconds:
                complete = False
                break
            def signal(index=None):
                values = [0.0] * 7
                values[6] = float(context)
                if index is not None:
                    values[index] = 1.0
                return values
            for _ in range(8):
                brain.experience(signal())
            first, second = stream.randrange(2), stream.randrange(2)
            brain.experience(signal(first))
            for _ in range(2):
                brain.experience(signal(4))
            brain.experience(signal(2 + second))
            target = first ^ second ^ context
            scores.append(score(brain.experience(signal(5), target), target))
            if episode % 100 == 99:
                trajectory.append(dict(context=context, **brain.summary()))
        phases.append({"context":context, "episodes":len(scores),
                       "first_1":metrics(scores[:1]), "first_50":metrics(scores[:50]),
                       "last_400":metrics(scores[-400:])})
        if not complete:
            break
    old_return = phases[2]["first_50"] if len(phases) == 3 else metrics([])
    result = finish(brain, [], trajectory, phases=phases, complete=complete,
                    termination="finished" if complete else "time_budget")
    result["metrics"] = old_return
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("sequence", "delay", "conflict", "reversal", "structure", "contextual", "grammar", "lifetime", "all"), default="all")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--variants", default="candidate,no_learning,no_traces,no_hidden,no_recurrence,no_hidden_plasticity")
    parser.add_argument("--gaps", default="0,4,16,64")
    parser.add_argument("--overrides", default="{}", help="JSON config overrides recorded in every result")
    parser.add_argument("--output", default="experiments/results/initial.jsonl")
    parser.add_argument("--budget-seconds", type=float, default=60,
                        help="per-run grammar budget; partial results never pass a gate")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="optional final brain checkpoints, including incomplete runs")
    args = parser.parse_args()
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])):
                     hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "brain/organism.py")}
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    overrides = json.loads(args.overrides)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit("Refusing to overwrite results: " + str(output))
    suites = ["sequence", "delay", "conflict", "reversal", "structure", "contextual"] if args.suite == "all" else [args.suite]
    with output.open("w") as handle:
        for suite in suites:
            for variant in variants:
                if variant not in VARIANTS:
                    raise ValueError("unknown variant: " + variant)
                conditions = [None]
                if suite in ("delay", "grammar"):
                    conditions = [int(g) for g in args.gaps.split(",")]
                elif suite == "structure":
                    conditions = [False, True]
                elif suite == "contextual":
                    conditions = ["none", "rest", "wake", "sleep"]
                for condition in conditions:
                    for seed in seeds:
                        if args.checkpoint_dir:
                            directory = Path(args.checkpoint_dir)
                            directory.mkdir(parents=True, exist_ok=True)
                            destination = directory / f"{suite}_{variant}_{condition}_{seed}.json"
                            if destination.exists():
                                raise SystemExit("Refusing to overwrite checkpoint: " + str(destination))
                            CHECKPOINT.set(destination)
                        start = time.monotonic()
                        if suite == "sequence":
                            result = sequence(seed, variant, overrides)
                        elif suite in ("delay", "conflict"):
                            result = delay(seed, variant, overrides, gap=condition if suite == "delay" else 16, conflict=suite == "conflict")
                        elif suite == "reversal":
                            result = reversal(seed, variant, overrides)
                        elif suite == "structure":
                            result = structure(seed, variant, overrides, noise=condition)
                        elif suite == "grammar":
                            result = grammar(seed, variant, overrides, gap=condition, budget_seconds=args.budget_seconds)
                        elif suite == "lifetime":
                            result = lifetime(seed, variant, overrides, budget_seconds=args.budget_seconds)
                        else:
                            result = contextual(seed, variant, overrides, rehearsal=condition)
                        row = dict(suite=suite, variant=variant, condition=condition,
                                   seed=seed, seconds=time.monotonic() - start,
                                   source_hashes=source_hashes, **result)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        print(json.dumps({k: row[k] for k in ("suite", "variant", "condition", "seed", "seconds", "metrics")}), flush=True)


if __name__ == "__main__":
    main()

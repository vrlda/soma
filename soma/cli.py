"""Command-line lifecycle for a persistent SOMA model."""

import argparse
import json
import sys
from typing import Sequence

from .model import SOMA


def _inputs(value: str) -> Sequence[float]:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError
        return tuple(float(item) for item in parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError("inputs must be a JSON array of numbers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m soma", description="Create and continue one persistent SOMA model.")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a configured model checkpoint")
    create.add_argument("checkpoint")
    create.add_argument("--inputs", type=int, required=True, dest="input_size")
    create.add_argument("--outputs", type=int, default=1, dest="output_size")
    create.add_argument("--hidden", type=int, default=6, dest="hidden_size")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--reward-delay", type=int, default=0)
    create.add_argument("--max-cells", type=int, default=32)
    create.add_argument("--max-synapses", type=int, default=256)
    create.add_argument("--max-modules", type=int, default=4)
    create.add_argument("--energy-per-step", type=float, default=20.0)
    create.add_argument("--prune-reuse-ceiling", type=int, default=0)

    step = commands.add_parser("step", help="take one learning step and update the checkpoint")
    step.add_argument("checkpoint")
    step.add_argument("--input", required=True, type=_inputs, dest="inputs")
    step.add_argument("--reward", type=float, default=0.0)
    step.add_argument("--novelty", type=float, default=0.10)
    step.add_argument("--exploration", type=float, default=0.20)
    step.add_argument("--uncertainty", type=float, default=0.0)
    step.add_argument("--salience", type=float, default=0.0)

    outcome = commands.add_parser("outcome", help="apply a terminal reward and update the checkpoint")
    outcome.add_argument("checkpoint")
    outcome.add_argument("--reward", type=float, required=True)

    inspect = commands.add_parser("inspect", help="print model state and resource summary")
    inspect.add_argument("checkpoint")
    inspect.add_argument("--events", type=int, default=10)
    return parser


def main(argv: Sequence[str] = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "create":
        model = SOMA.create(
            input_size=arguments.input_size,
            output_size=arguments.output_size,
            hidden_size=arguments.hidden_size,
            seed=arguments.seed,
            reward_delay=arguments.reward_delay,
            max_cells=arguments.max_cells,
            max_synapses=arguments.max_synapses,
            max_modules=arguments.max_modules,
            energy_per_step=arguments.energy_per_step,
            prune_reuse_ceiling=arguments.prune_reuse_ceiling,
        )
        model.save(arguments.checkpoint)
        payload = model.inspect(0)
    elif arguments.command == "step":
        model = SOMA.load(arguments.checkpoint)
        result = model.step(
            arguments.inputs,
            reward=arguments.reward,
            novelty=arguments.novelty,
            exploration=arguments.exploration,
            uncertainty=arguments.uncertainty,
            salience=arguments.salience,
        )
        model.save(arguments.checkpoint)
        payload = {"outputs": list(result.outputs), "reward": result.reward, "energy_used": result.energy_used, "structural_events": result.structural_events}
    elif arguments.command == "outcome":
        model = SOMA.load(arguments.checkpoint)
        model.apply_outcome(arguments.reward)
        model.save(arguments.checkpoint)
        payload = model.inspect(0)
    else:
        payload = SOMA.load(arguments.checkpoint).inspect(arguments.events)
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0

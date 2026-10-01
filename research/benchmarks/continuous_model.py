"""Run or resume one SOMA checkpoint in the continuous demo environment."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import os

from soma import ContinuousTargetEnvironment, SOMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="model checkpoint to create or continue")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("--steps must be positive")

    model = SOMA.load(arguments.checkpoint) if os.path.exists(arguments.checkpoint) else SOMA.create(input_size=2, seed=arguments.seed)
    environment = ContinuousTargetEnvironment(seed=arguments.seed, horizon=arguments.steps)
    observation = environment.reset()
    for _ in range(environment.horizon):
        result = model.step(observation.values, reward=observation.reward, novelty=0.05, exploration=0.02)
        observation = environment.step(result.outputs[0])
        if observation.done:
            model.apply_outcome(observation.reward)
    model.save(arguments.checkpoint)
    status = model.inspect(include_events=3)
    print("checkpoint=%s steps=%d cells=%d synapses=%d" % (
        arguments.checkpoint, status["steps"], status["topology"]["cells"], status["topology"]["synapses"],
    ))


if __name__ == "__main__":
    main()

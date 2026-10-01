"""Run a tiny continuous SOMA organism."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from soma import ContinuousTargetEnvironment, Modulators, Organism


def main() -> None:
    environment = ContinuousTargetEnvironment(seed=7, horizon=40)
    organism = Organism.create_default(seed=7, hidden_size=6)
    observation = environment.reset()
    total = 0.0
    for _ in range(environment.horizon):
        # The reward in `observation` belongs to the preceding action. Passing
        # it here applies that delayed third factor to the preceding trace.
        result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=0.05, exploration=0.02))
        action = result.outputs[0]
        observation = environment.step(action)
        total += observation.reward
        if observation.done:
            # Close the final action's eligibility trace without taking an
            # extra organism decision.
            organism.apply_outcome(observation.reward)
            break
    organism.validate()
    print("steps=%d cells=%d synapses=%d reward=%.3f events=%d" % (organism.step_count, len(organism.cells), len(organism.graph.synapses), total, len(organism.events)))


if __name__ == "__main__":
    main()

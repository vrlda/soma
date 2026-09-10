"""Run SOMA's locked v11 RED-first compositional benchmark."""

import json

from soma.lifetime import CompositionalBenchmarkConfig, run_compositional_benchmark


if __name__ == "__main__":
    print(json.dumps(run_compositional_benchmark(CompositionalBenchmarkConfig(composition_learning_enabled=True)), indent=2, sort_keys=True))

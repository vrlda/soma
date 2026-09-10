"""Run SOMA's locked v9 adaptive-dendritic benchmark protocol."""

import json

from soma.lifetime import AdaptiveDendriticBenchmarkConfig, run_adaptive_dendritic_benchmark


if __name__ == "__main__":
    print(json.dumps(run_adaptive_dendritic_benchmark(AdaptiveDendriticBenchmarkConfig()), indent=2, sort_keys=True))

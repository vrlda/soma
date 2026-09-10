"""Run SOMA's locked nonlinear dendritic representation benchmark."""

import json

from soma.lifetime import NonlinearBenchmarkConfig, run_nonlinear_benchmark


if __name__ == "__main__":
    print(json.dumps(run_nonlinear_benchmark(NonlinearBenchmarkConfig()), indent=2, sort_keys=True))

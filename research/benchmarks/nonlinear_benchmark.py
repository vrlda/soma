"""Run SOMA's locked nonlinear dendritic representation benchmark."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json

from soma.lifetime import NonlinearBenchmarkConfig, run_nonlinear_benchmark


if __name__ == "__main__":
    print(json.dumps(run_nonlinear_benchmark(NonlinearBenchmarkConfig()), indent=2, sort_keys=True))

"""Run SOMA's locked v11 RED-first compositional benchmark."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json

from soma.lifetime import CompositionalBenchmarkConfig, run_compositional_benchmark


if __name__ == "__main__":
    print(json.dumps(run_compositional_benchmark(CompositionalBenchmarkConfig(composition_learning_enabled=True)), indent=2, sort_keys=True))

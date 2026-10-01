"""Run SOMA's locked v9 adaptive-dendritic benchmark protocol."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json

from soma.lifetime import AdaptiveDendriticBenchmarkConfig, run_adaptive_dendritic_benchmark


if __name__ == "__main__":
    print(json.dumps(run_adaptive_dendritic_benchmark(AdaptiveDendriticBenchmarkConfig()), indent=2, sort_keys=True))

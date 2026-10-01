"""Run SOMA's v12 irregular-stream compositional benchmark."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json

from soma.lifetime import StreamingCompositionalBenchmarkConfig, run_streaming_compositional_benchmark


if __name__ == "__main__":
    print(json.dumps(run_streaming_compositional_benchmark(), indent=2, sort_keys=True))

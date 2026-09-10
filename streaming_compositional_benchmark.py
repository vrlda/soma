"""Run SOMA's v12 irregular-stream compositional benchmark."""

import json

from soma.lifetime import StreamingCompositionalBenchmarkConfig, run_streaming_compositional_benchmark


if __name__ == "__main__":
    print(json.dumps(run_streaming_compositional_benchmark(), indent=2, sort_keys=True))

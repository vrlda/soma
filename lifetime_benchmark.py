"""Run SOMA's persistent, unannounced A -> B -> A lifetime benchmark."""

import json

from soma.lifetime import LifetimeBenchmarkConfig, run_benchmark


def main() -> None:
    result = run_benchmark(LifetimeBenchmarkConfig())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

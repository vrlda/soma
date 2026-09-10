"""Run SOMA's locked v15 end-to-end completion benchmark."""

import json

from soma import FinalLifetimeConfig, run_final_lifetime_benchmark


if __name__ == "__main__":
    print(json.dumps(run_final_lifetime_benchmark(FinalLifetimeConfig()), indent=2, sort_keys=True))

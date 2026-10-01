"""Run SOMA's locked v15 end-to-end completion benchmark."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json

from soma import FinalLifetimeConfig, run_final_lifetime_benchmark


if __name__ == "__main__":
    print(json.dumps(run_final_lifetime_benchmark(FinalLifetimeConfig()), indent=2, sort_keys=True))

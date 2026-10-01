#!/usr/bin/env python3
"""Run the hidden A -> B -> C -> A -> B SOMA stress benchmark."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json

from soma.lifetime import run_multicontext_benchmark


if __name__ == "__main__":
    print(json.dumps(run_multicontext_benchmark(), indent=2, sort_keys=True))

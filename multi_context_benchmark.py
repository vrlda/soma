#!/usr/bin/env python3
"""Run the hidden A -> B -> C -> A -> B SOMA stress benchmark."""

import json

from soma.lifetime import run_multicontext_benchmark


if __name__ == "__main__":
    print(json.dumps(run_multicontext_benchmark(), indent=2, sort_keys=True))

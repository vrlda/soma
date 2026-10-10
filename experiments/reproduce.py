"""Apply a frozen experiment selection to the standard runner."""
import argparse
import json
from pathlib import Path
import sys

from experiments.run import main as run


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection", required=True)
    args, remaining = parser.parse_known_args()
    selection = json.loads(Path(args.selection).read_text())
    if "--overrides" in remaining:
        raise SystemExit("Use the frozen selection or the runner's explicit overrides, not both")
    sys.argv = [sys.argv[0], *remaining, "--overrides", json.dumps(selection["overrides"])]
    run()


if __name__ == "__main__":
    main()

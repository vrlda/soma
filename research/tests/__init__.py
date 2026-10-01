"""Research tests: run from the repository root with
``python3 -m unittest discover -s research/tests -t .``."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, 'research', 'benchmarks')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

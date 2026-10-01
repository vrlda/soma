"""End-to-end prosumer demo: create, teach, correct, ask, inspect, forget.

Runs against a temporary brain store. No network, no interaction.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
import tempfile

from soma.service.chat import chat_turn, correct, forget_fact, teach_text
from soma.service.store import BrainStore


def main():
    with tempfile.TemporaryDirectory() as root:
        store = BrainStore(root)
        store.create("demo", description="demo brain")
        with open("data/e0/alice.txt", "rb") as handle:
            taught = teach_text(store, "demo", handle.read(4096), provenance="demo")
        print("taught %d bytes (%d bits)" % (taught["bytes"], taught["bits"]))
        correct(store, "demo", "USER the code is ", "kite")
        print("ask code:", chat_turn(store, "demo", "the code is ", max_bytes=8, seed=2))
        print("spell:", chat_turn(store, "demo", "spell hello", max_bytes=12, seed=2))
        print("unknown:", chat_turn(store, "demo", "xqzt blorpy", max_bytes=12, seed=2))
        print("forget:", forget_fact(store, "demo", "USER the code is "))
        info = store.inspect("demo")
        print("circuits=%d events=%d entries=%d" % (
            info["sequence_circuits"], info["sequence_events"], info["episodic_entries"]))


if __name__ == "__main__":
    main()

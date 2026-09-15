"""Diagnostics: environment, store inventory, brain integrity, smoke test."""

import os
import platform
import shutil
import sys


def doctor(root):
    from .store import BrainStore
    from ..memory import SequenceCircuitMemory
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "store_root": os.path.abspath(root),
        "disk_free_mb": shutil.disk_usage(os.path.abspath(root))[2] // (1024 * 1024),
        "brains": {},
        "ok": True,
    }
    try:
        store = BrainStore(root)
    except (OSError, ValueError) as error:
        report["ok"] = False
        report["store_error"] = str(error)
        return report
    for name in store.list():
        try:
            info = store.inspect(name)
            organism, _, _, _ = store.load(name)
            organism.validate()
            info["valid"] = True
        except (OSError, ValueError, AssertionError) as error:
            info = {"name": name, "valid": False, "error": str(error)}
            report["ok"] = False
        report["brains"][name] = info
    try:
        memory = SequenceCircuitMemory((0, 1), max_order=4, max_circuits=64)
        for symbol in (0, 1, 1, 0):
            memory.observe(symbol)
        memory.validate()
        report["smoke"] = True
    except (ValueError, AssertionError) as error:
        report["smoke"] = False
        report["smoke_error"] = str(error)
        report["ok"] = False
    return report

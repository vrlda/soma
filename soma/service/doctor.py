"""Diagnostics: environment, store inventory, brain integrity, smoke test."""

import os
import platform
import shutil
import sys


def engine_report():
    """Is the Rust engine built, and does a live round trip work?

    A missing engine is not a failure: brains with the original memory
    work without it. Engine brains need it (see docs/r8-engine-serving.md).
    """
    from ..memory.engine import EngineError, EngineMixingMemory, engine_available, engine_binary
    report = {"binary": engine_binary(), "available": engine_available()}
    if not report["available"]:
        report["hint"] = ("build it with `cargo build --release --manifest-path "
                          "engine/soma-engine/Cargo.toml` to use --memory mixing")
        return report
    try:
        with EngineMixingMemory(config={"max_circuits": 4096}) as memory:
            for bit in (0, 1, 1, 0, 1, 0, 0, 1):
                memory.observe(bit)
            distribution, _ = memory.distribution()
        report["smoke"] = abs(distribution[0] + distribution[1] - 1.0) < 1e-9
    except (EngineError, OSError, ValueError) as error:
        report["smoke"] = False
        report["error"] = str(error)
    return report


def signing_report():
    """Trusted signing keys (an empty keyring only blocks downloads)."""
    from .distribution import keyring
    try:
        keys = keyring()
    except (OSError, ValueError) as error:
        return {"trusted_keys": 0, "error": str(error)}
    report = {"trusted_keys": len(keys), "key_ids": sorted(keys)}
    if not keys:
        report["hint"] = "brain-download needs a trusted key: `key-trust PUBLIC_KEY_HEX`"
    return report


def doctor(root):
    from .store import BrainStore
    from ..memory import SequenceCircuitMemory
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "store_root": os.path.abspath(root),
        "disk_free_mb": shutil.disk_usage(os.path.abspath(root))[2] // (1024 * 1024),
        "brains": {},
        "engine": engine_report(),
        "signing": signing_report(),
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
        except (OSError, ValueError, AssertionError, RuntimeError) as error:
            # RuntimeError covers EngineError: an engine brain whose engine
            # is missing or fails is reported, not a crash.
            info = {"name": name, "valid": False, "error": str(error)}
            report["ok"] = False
        report["brains"][name] = info
    if report["engine"]["available"] and not report["engine"].get("smoke"):
        report["ok"] = False
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

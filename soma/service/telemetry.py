"""Consented telemetry: metadata only, off by default, local first.

What is recorded (only when the user opts in): command names, durations,
exit codes, brain sizes, error classes. NEVER: message content, file
contents, brain state, or anything that could reconstruct user data.
A second explicit flag is required for content-bearing diagnostics.
"""

import json
import os
import time

FLAG = "telemetry.json"
LOG = "telemetry.log"
CONTENT_FLAG = "telemetry_content.json"


def _path(root, name):
    return os.path.join(os.path.abspath(root), name)


def status(root):
    """Current consent state. Defaults to off; missing files mean off."""
    try:
        with open(_path(root, FLAG)) as handle:
            payload = json.load(handle)
        enabled = bool(payload.get("enabled", False))
    except (OSError, ValueError):
        enabled = False
    try:
        with open(_path(root, CONTENT_FLAG)) as handle:
            content = bool(json.load(handle).get("enabled", False))
    except (OSError, ValueError):
        content = False
    return {"enabled": enabled, "content": content and enabled}


def set_consent(root, enabled, content=False):
    """Opt in or out. Content logging needs its own explicit flag."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        os.makedirs(root)
    with open(_path(root, FLAG), "w") as handle:
        json.dump({"enabled": bool(enabled)}, handle, sort_keys=True)
    with open(_path(root, CONTENT_FLAG), "w") as handle:
        json.dump({"enabled": bool(content and enabled)}, handle, sort_keys=True)
    return status(root)


def record(root, command, duration_ms, exit_code, brain=None, extra=None):
    """Append one metadata event. Silent no-op unless opted in."""
    state = status(root)
    if not state["enabled"]:
        return False
    event = {
        "timestamp_utc": int(time.time()),
        "command": str(command),
        "duration_ms": int(duration_ms),
        "exit_code": int(exit_code),
    }
    if brain:
        event["brain"] = str(brain)
    if extra and state["content"]:
        event["extra"] = extra
    with open(_path(root, LOG), "a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return True


def summary(root):
    """Aggregate local telemetry for support bundles (counts only)."""
    path = _path(root, LOG)
    if not os.path.exists(path):
        return {"events": 0, "commands": {}, "errors": 0}
    commands, errors, events = {}, 0, 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            events += 1
            commands[event.get("command", "?")] = commands.get(event.get("command", "?"), 0) + 1
            if event.get("exit_code", 0) != 0:
                errors += 1
    return {"events": events, "commands": commands, "errors": errors}


def bundle(root, destination):
    """Build a support bundle: telemetry summary, manifests, no content.

    Root is the brains directory (telemetry lives alongside it).
    """
    import tarfile
    import tempfile
    from .store import BrainStore
    manifests = []
    try:
        store = BrainStore(os.path.abspath(root))
        for name in store.list():
            try:
                info = store.inspect(name)
                manifests.append({key: info[key] for key in
                                  ("name", "identity", "sequence_circuits", "episodic_entries")
                                  if key in info})
            except (OSError, ValueError, AssertionError):
                continue
    except (OSError, ValueError):
        pass
    payload = {
        "telemetry": summary(root),
        "brains": manifests,
        "consent": status(root),
    }
    with tempfile.TemporaryDirectory() as scratch:
        bundle_path = os.path.join(scratch, "bundle.json")
        with open(bundle_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(bundle_path, arcname="bundle.json")
    return os.path.abspath(destination)

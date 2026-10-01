"""CircuitMixingMemory served by the Rust engine (``soma-mixer-serve``).

``EngineMixingMemory`` has the same interface the product uses on
``CircuitMixingMemory`` (``observe``, ``distribution``, ``probability``,
``reset_history``, ``symbol_index``, ``state_dict``). It holds no model
state itself: a long-lived engine process does, and results are bit-exact
with the Python reference (``engine/differential_serve.py``).

Observed bits are buffered and sent in one request when a prediction or
another operation needs them, so conditioning on a prompt costs one round
trip, not one per bit. The memory's durable form is a ``SOMAMIX1`` file.
``state_dict`` records its path, and the engine saves it there.
"""

import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BINARY = os.path.join(ROOT, "engine", "soma-engine", "target", "release",
                              "soma-mixer-serve")


class EngineError(RuntimeError):
    pass


def engine_binary():
    return os.environ.get("SOMA_MIXER_SERVE", DEFAULT_BINARY)


def engine_available():
    return os.access(engine_binary(), os.X_OK)


class EngineMixingMemory(object):
    """Proxy to one engine-hosted circuit-mixing memory."""

    symbols = (0, 1)
    symbol_index = {0: 0, 1: 1}

    def __init__(self, config=None, path=None, binary=None):
        """Start an engine: load ``path`` if it exists, else create from ``config``.

        ``path`` is where ``save``/``state_dict`` persist the memory.
        """
        self.path = None if path is None else os.path.abspath(path)
        binary = binary or engine_binary()
        if not os.access(binary, os.X_OK):
            raise EngineError("engine binary not found at %s; build it with "
                              "`cargo build --release --manifest-path "
                              "engine/soma-engine/Cargo.toml`" % binary)
        self._pending_bits = []
        self._pending_mode = None
        self._process = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        try:
            if self.path is not None and os.path.exists(self.path):
                self._request({"op": "load", "path": self.path})
            else:
                self._request({"op": "new", "config": dict(config or {})})
        except Exception:
            self._shutdown()
            raise

    def _request(self, message):
        if self._process.poll() is not None:
            raise EngineError("engine process has exited")
        self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise EngineError("engine closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise EngineError(reply.get("error", "engine error"))
        return reply

    def _flush(self, want_distribution=False):
        if not self._pending_bits:
            return None
        learn, weight = self._pending_mode
        message = {"op": "observe", "bits": "".join(self._pending_bits), "learn": learn,
                   "weight": weight}
        if want_distribution:
            message["distribution"] = True
        self._pending_bits = []
        self._pending_mode = None
        return self._request(message)

    def observe(self, bit, learn=True, weight=1):
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        if not isinstance(learn, bool):
            raise ValueError("learn flag must be boolean")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise ValueError("weight must be a positive integer")
        mode = (learn, weight)
        if self._pending_mode is not None and self._pending_mode != mode:
            self._flush()
        self._pending_mode = mode
        self._pending_bits.append("1" if bit else "0")

    def observe_bytes(self, data, learn=True, weight=1):
        for byte in bytes(data):
            for shift in range(7, -1, -1):
                self.observe((byte >> shift) & 1, learn=learn, weight=weight)

    def distribution(self):
        reply = self._flush(want_distribution=True)
        if reply is None:
            reply = self._request({"op": "distribution"})
        probability = float(reply["p1"])
        return {0: 1.0 - probability, 1: probability}, int(reply["order"])

    def probability(self, symbol):
        if symbol not in self.symbol_index:
            raise ValueError("unknown sequence symbol")
        return self.distribution()[0][symbol]

    def reset_history(self):
        self._flush()
        self._request({"op": "reset"})

    def summary(self):
        self._flush()
        reply = self._request({"op": "summary"})
        return {key: reply[key] for key in ("circuits", "circuits_created",
                                            "circuits_reclaimed", "events_seen")}

    def validate(self):
        return self._process.poll() is None

    def save(self, path=None):
        path = path or self.path
        if path is None:
            raise ValueError("no path to save the engine memory to")
        self._flush()
        self._request({"op": "save", "path": os.path.abspath(path)})
        return path

    def state_dict(self):
        """Reference to the backing file (relative name); no side effects.

        Call ``save()`` to write the file; ``BrainStore.save`` does.
        """
        if self.path is None:
            raise ValueError("engine memory has no backing file")
        return {"kind": "circuit-mixing-file", "path": os.path.basename(self.path)}

    @classmethod
    def from_state_dict(cls, payload, base_dir=None, binary=None):
        if not isinstance(payload, dict) or payload.get("kind") != "circuit-mixing-file":
            raise ValueError("not an engine circuit-mixing state")
        path = payload["path"]
        if not os.path.isabs(path):
            path = os.path.join(base_dir or os.getcwd(), path)
        if not os.path.exists(path):
            raise ValueError("engine memory file is missing: %s" % path)
        return cls(path=path, binary=binary)

    def _shutdown(self):
        """Stop the engine process and release its pipes. Unsaved learning is lost."""
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self):
        """Flush, stop the engine cleanly, and release its pipes."""
        if self._process.poll() is None:
            try:
                self._flush()
                self._request({"op": "quit"})
                self._process.wait(timeout=10)
            except (EngineError, OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self._shutdown()

    def __del__(self):
        try:
            self._shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

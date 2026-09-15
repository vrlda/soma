"""Local HTTP service: SOMA endpoints plus an OpenAI-compatible chat view."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import chat_turn
from .store import BrainStore


class _Handler(BaseHTTPRequestHandler):
    store = None

    def _send(self, payload, status=200):
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            return {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            brains = self.store.list()
            self._send({"data": [{"id": name} for name in brains]})
        elif self.path.startswith("/inspect/"):
            name = self.path[len("/inspect/"):]
            try:
                self._send(self.store.inspect(name))
            except ValueError as error:
                self._send({"error": str(error)}, status=404)
        elif self.path == "/health":
            self._send({"ok": True})
        else:
            self._send({"error": "unknown endpoint"}, status=404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            payload = self._read_json()
            name = payload.get("model", "")
            messages = payload.get("messages", [])
            user_text = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_text = message.get("content", "")
                    break
            try:
                reply = chat_turn(self.store, name, user_text,
                                  max_bytes=int(payload.get("max_tokens", 24)))
            except ValueError as error:
                self._send({"error": str(error)}, status=404)
                return
            self._send({
                "id": "soma-local",
                "model": name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}],
            })
        elif self.path == "/learn":
            payload = self._read_json()
            try:
                from .chat import teach_text
                result = teach_text(self.store, payload.get("brain", ""),
                                    payload.get("text", ""),
                                    provenance=payload.get("provenance", "api"))
            except ValueError as error:
                self._send({"error": str(error)}, status=404)
                return
            self._send(result)
        else:
            self._send({"error": "unknown endpoint"}, status=404)


def serve(store, host="127.0.0.1", port=8765):
    """Run the local API until interrupted. Localhost only by default."""
    if isinstance(store, str):
        store = BrainStore(store)
    _Handler.store = store
    server = ThreadingHTTPServer((host, port), _Handler)
    print("serving soma on http://%s:%d (Ctrl-C to stop)" % (host, port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0

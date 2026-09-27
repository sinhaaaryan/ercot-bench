"""Tiny OpenAI-compatible chat server for CPU-only pipeline tests (no model, no GPU).

Answers /v1/chat/completions with canned p10/p50/p90 JSON built from the prompt's hour list,
cycling through modes so rewards vary: baseline-like, wide, tight, out-of-order, garbage.
Hours are read from a "Hours to forecast: [16, 17, ...]" line (smoke prompts); else HE16-20.

    /hackathon/prime-rl/.venv/bin/python scripts/mock_openai_server.py --port 8799
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODES = ["plain", "wide", "tight", "swapped", "fenced", "garbage"]
_counter = itertools.count()
_lock = threading.Lock()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def make_reply(messages: list[dict]) -> str:
    prompt = "\n".join(_text(m.get("content")) for m in messages)
    m = re.search(r"Hours to forecast:\s*\[([\d,\s]+)\]", prompt)
    hours = [int(h) for h in m.group(1).split(",")] if m else [16, 17, 18, 19, 20]
    with _lock:
        mode = MODES[next(_counter) % len(MODES)]
    w = {"plain": 800, "wide": 3000, "tight": 150, "swapped": 800, "fenced": 500}.get(mode, 0)
    rows = [{"he": h, "p10": -w, "p50": 25, "p90": w} for h in hours]
    if mode == "swapped":
        rows = [{"he": h, "p10": w, "p50": 25, "p90": -w} for h in hours]
    body = json.dumps({"hours": rows})
    if mode == "fenced":
        return f"```json\n{body}\n```"
    if mode == "garbage":
        return "Load will probably be a bit higher than forecast."
    return body


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": "mock", "object": "model", "owned_by": "mock"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json({"error": f"unsupported {self.path}"}, 404)
        reply = make_reply(req.get("messages", []))
        cid, created, model = f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), req.get("model", "mock")
        usage = {"prompt_tokens": 100, "completion_tokens": max(1, len(reply) // 4), "total_tokens": 100 + len(reply) // 4}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in (({"role": "assistant", "content": reply}, None), ({}, "stop")):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if (req.get("stream_options") or {}).get("include_usage"):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [], "usage": usage}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json({
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": usage,
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    print(f"mock OpenAI server on http://127.0.0.1:{a.port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

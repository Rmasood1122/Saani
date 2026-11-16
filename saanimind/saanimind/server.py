"""An OpenAI-compatible HTTP server (chat completions with tool calls) around your model.

Saani's "own" provider talks to this. Standard library only; one request at a time,
bound to 127.0.0.1 by default. There is no API key: the server is local.

    POST /v1/chat/completions   {"model": ..., "messages": [...], "tools": [...], "max_tokens": N}
    GET  /v1/models
    GET  /health
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import Mind

MAX_BODY = 4_000_000


def openai_to_turns(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> our turn format."""
    turns: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):                               # multi-part content
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        content = content or ""
        if role == "system":
            if turns and turns[0]["role"] == "system":
                turns[0]["content"] += "\n" + content
            else:
                turns.insert(0, {"role": "system", "content": content})
        elif role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = fn.get("arguments")
                    args = json.loads(args) if isinstance(args, str) else (args or {})
                except ValueError:
                    args = {}
                calls.append({"name": fn.get("name", ""), "args": args if isinstance(args, dict) else {}})
            turns.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif role == "tool":
            turns.append({"role": "tool", "content": content})
        else:
            turns.append({"role": "user", "content": content})
    return turns


def openai_tools(tools: list[dict] | None) -> list[dict]:
    """OpenAI tool specs -> the compact form used in training."""
    out = []
    for t in tools or []:
        fn = t.get("function") or t
        out.append({"name": fn.get("name", ""), "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or fn.get("input_schema") or {}})
    return out


class Server:
    def __init__(self, mind: Mind, host: str = "127.0.0.1", port: int = 8900, *, temperature: float = 0.7,
                 top_p: float = 0.9, max_new_tokens: int = 256, log=print):
        self.mind, self.host, self.port = mind, host, port
        self.temperature, self.top_p, self.max_new_tokens = temperature, top_p, max_new_tokens
        self.log = log
        self.lock = threading.Lock()
        self.httpd: ThreadingHTTPServer | None = None
        self.requests = 0

    def completion(self, body: dict) -> dict:
        turns = openai_to_turns(body.get("messages") or [])
        tools = openai_tools(body.get("tools"))
        choice = body.get("tool_choice")
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_new_tokens)
        temp = float(body.get("temperature", self.temperature))
        with self.lock:                                   # one generation at a time on a small machine
            r = self.mind.reply(turns, tools, max_new_tokens=min(max_new, self.max_new_tokens),
                                temperature=temp, top_p=self.top_p)
        # honour a forced tool: if the model named it, keep only that call
        if isinstance(choice, dict):
            want = ((choice.get("function") or {}).get("name"))
            r["tool_calls"] = [c for c in r["tool_calls"] if c["name"] == want] or r["tool_calls"]
        elif choice == "none":
            r["tool_calls"] = []
        msg: dict = {"role": "assistant", "content": r["text"] or None}
        if r["tool_calls"]:
            msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                 for c in r["tool_calls"]]
        self.requests += 1
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion", "created": int(time.time()),
                "model": self.mind.name,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if r["tool_calls"] else "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": r["tokens"], "total_tokens": r["tokens"]}}

    def serve(self, block: bool = True) -> None:
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):      # quiet
                pass

            def _json(self, code: int, obj: dict):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path in ("/health", "/"):
                    return self._json(200, {"ok": True, "model": srv.mind.info(), "requests": srv.requests})
                if self.path in ("/v1/models", "/models"):
                    return self._json(200, {"object": "list", "data": [{"id": srv.mind.name, "object": "model",
                                                                        "owned_by": "you"}]})
                self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                if self.path not in ("/v1/chat/completions", "/chat/completions"):
                    return self._json(404, {"error": {"message": "not found"}})
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._json(413, {"error": {"message": "request too large"}})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("body must be an object")
                except ValueError as e:
                    return self._json(400, {"error": {"message": f"bad JSON: {e}"}})
                try:
                    self._json(200, srv.completion(body))
                except Exception as e:      # the model must never take the server down
                    srv.log(f"error: {e!r}")
                    self._json(500, {"error": {"message": str(e)}})

        self.httpd = ThreadingHTTPServer((self.host, self.port), H)
        self.port = self.httpd.server_address[1]
        self.log(f"saanimind serving {self.mind.name} ({self.mind.info()['params']/1e6:.1f}M params) "
                 f"at http://{self.host}:{self.port}/v1  (Ctrl+C to stop)")
        if block:
            try:
                self.httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                self.httpd.server_close()
        else:
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def shutdown(self) -> None:
        if self.httpd:
            self.httpd.shutdown(); self.httpd.server_close()

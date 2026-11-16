"""The background service: starts at login, listens for your voice, runs schedules,
and serves a local dashboard where you watch agents work and approve what they ask for.

The HTTP server binds to 127.0.0.1 only and every API call needs a secret token.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import build
from .config import Config
from .core.orchestrator import Orchestrator
from .interaction import QueueInteraction
from .scheduler import Scheduler
from .textutil import clip, first_sentences

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
APPROVE_WORDS = {"approve", "approved", "allow", "yes", "go ahead", "do it"}
DENY_WORDS = {"deny", "denied", "no", "stop", "cancel", "don't"}


def load_token(home: Path) -> str:
    f = Path(home) / "daemon.token"
    try:
        tok = f.read_text(encoding="utf-8").strip()
        if len(tok) >= 32:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(32)
    fd = os.open(str(f), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(tok)
    return tok


ASSET_DIR = Path(__file__).parent / "assets"
ASSET_BACKGROUND = (ASSET_DIR / "blackhole.png").read_bytes() if (ASSET_DIR / "blackhole.png").exists() else b""
# The web UI may change these. "trusted" is deliberately not on the list: it is set in the terminal only.
WEB_SETTABLE_MODES = ("strict", "ask")


def dashboard_url(config: Config) -> str:
    return (f"http://{config.get('daemon.host', '127.0.0.1')}:{config.get('daemon.port', 8765)}"
            f"/?token={load_token(config.home)}")


def call_daemon(config: Config, path: str, payload: dict | None = None, timeout: float = 5) -> dict | None:
    """Talk to a running daemon from the CLI. Returns None if it is not running."""
    url = f"http://127.0.0.1:{config.get('daemon.port', 8765)}{path}"
    headers = {"X-Saani-Token": load_token(config.home), "X-Requested-With": "saani"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never via a proxy
        with opener.open(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


class Daemon:
    def __init__(self, config: Config | None = None, client=None, enable_voice: bool = True):
        self.config = config or Config.load()
        self.inter = QueueInteraction(self.config.get("security.approval_timeout_s", 600))
        self.svc = build(self.config, self.inter, client)
        self.inter.emit = self.svc.events.emit
        self.orch = Orchestrator(self.svc)
        self.token = load_token(self.config.home)
        self.svc.redactor.register("DAEMON_TOKEN", self.token)
        self.started = time.time()
        self.voice = None
        self.voice_status = "off"
        self.enable_voice = enable_voice and self.config.get("voice.enabled", True)
        self._slots = threading.Semaphore(int(self.config.get("daemon.max_concurrent_tasks", 3)))
        self._sources: dict[str, str] = {}
        self.scheduler = Scheduler(self.svc.store, lambda text, name: self.submit(text, source=f"schedule:{name}", unattended=True))
        self.server: ThreadingHTTPServer | None = None
        self.last_poll = 0.0                 # when a dashboard last asked for state
        self._last_popup = 0.0
        self.open_browser = webbrowser.open  # replaceable in tests
        self._log_lock = threading.Lock()
        self.svc.events.subscribe(self._on_event)

    # ---- tasks -------------------------------------------------------------
    def submit(self, text: str, source: str = "dashboard", unattended: bool = False, workdir: str | None = None) -> str:
        tid = self.orch.new_task_id()
        self._sources[tid] = source
        self.svc.store.task_create(tid, text, source, workdir or "")

        def work():
            with self._slots:
                try:
                    self.orch.run_task(text, task_id=tid, source=source, unattended=unattended, workdir=workdir or None)
                except Exception as e:
                    self.svc.store.task_update(tid, status="failed", finished=time.time(), result=f"Internal error: {e}")
                    self.svc.events.emit("task_finished", task_id=tid, status="failed", result=str(e), stats={}, source=source)

        threading.Thread(target=work, daemon=True, name=f"saani-task-{tid}").start()
        return tid

    def state(self, since: int = 0) -> dict:
        tasks = []
        for t in self.svc.store.task_list(25):
            tasks.append({"id": t["id"], "text": clip(t["text"], 400), "status": t["status"], "source": t["source"],
                          "created": t["created"], "result": clip(t["result"] or "", 6000),
                          "stats": json.loads(t["stats"] or "{}")})
        return {"name": self.config.get("name", "Saani"), "uptime": int(time.time() - self.started),
                "voice": self.voice_status, "mode": self.config.get("security.mode", "ask"),
                "provider": self.config.get("llm.provider"), "pending": self.inter.pending(), "tasks": tasks,
                "events": self.svc.events.since(since, 200),
                "schedules": self.svc.store.schedule_list(),
                "skills": len(self.svc.skills.all()), "agents": len(self.svc.agents.all())}

    def settings(self) -> dict:
        return {"name": self.config.get("name", "Saani"), "mode": self.config.get("security.mode", "ask"),
                "provider": self.config.get("llm.provider"), "voice": self.voice_status,
                "open_on_attention": bool(self.config.get("daemon.open_dashboard_on_attention", True)),
                "open_on_start": bool(self.config.get("daemon.open_dashboard_on_start", False))}

    def update_settings(self, body: dict) -> dict:
        """Apply the few settings the web UI may change. Raises ValueError for anything else."""
        if "name" in body:
            name = str(body["name"]).strip()[:40]
            if not name:
                raise ValueError("the name cannot be empty")
            self.config.set("name", name)
        if "mode" in body:
            if body["mode"] not in WEB_SETTABLE_MODES:
                raise ValueError("permissions can be set to strict or ask here; trusted is set in the terminal")
            self.config.set("security.mode", body["mode"])
        for key, field in (("daemon.open_dashboard_on_attention", "open_on_attention"),
                           ("daemon.open_dashboard_on_start", "open_on_start")):
            if field in body:
                if not isinstance(body[field], bool):
                    raise ValueError(f"{field} must be true or false")
                self.config.set(key, body[field])
        self.config.save()
        return self.settings()

    # ---- events: log + speech ------------------------------------------------
    def _on_event(self, ev: dict) -> None:
        try:
            log = self.config.logs / "events.log"
            with self._log_lock:
                if log.exists() and log.stat().st_size > 5_000_000:
                    os.replace(log, log.with_suffix(".log.1"))
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str)[:4000] + "\n")
        except OSError:
            pass
        kind = ev["kind"]
        if kind == "attention" and self.config.get("daemon.open_dashboard_on_attention", True):
            # A permission request is useless if nobody sees it. If no dashboard is open, open one.
            now = time.time()
            if now - self.last_poll > 6 and now - self._last_popup > 20:
                self._last_popup = now
                try:
                    self.open_browser(dashboard_url(self.config))
                except Exception:
                    pass
        if not self.voice:
            return
        if kind == "attention":
            if ev.get("request_kind") == "approval":
                self._say(f"I need your permission. {ev.get('agent', 'An agent')} wants to use {ev.get('tool', 'a tool')}: "
                          f"{ev.get('reason', '')}. Please check the dashboard.")
            else:
                self._say("I have a question for you on the dashboard.")
        elif kind == "task_finished" and self.config.get("voice.speak_results", True):
            src = str(ev.get("source", ""))
            if src == "voice" or ev.get("status") != "done":
                lead = {"done": "Done.", "partial": "Partly done.", "failed": "That failed.", "cancelled": "Cancelled."}.get(ev.get("status"), "")
                self._say(f"{lead} {first_sentences(ev.get('result', ''), 2)}")

    def _say(self, text: str) -> None:
        if self.voice:
            threading.Thread(target=self.voice.say, args=(text,), daemon=True).start()

    # ---- voice -----------------------------------------------------------------
    def on_voice_command(self, text: str) -> None:
        low = text.lower().strip(" .!?")
        pending = [p for p in self.inter.pending() if p["kind"] == "approval"]
        if pending and (low in APPROVE_WORDS or low in DENY_WORDS):
            req = pending[0]
            if low in DENY_WORDS:
                self.inter.resolve(req["id"], False, note="denied by voice", by="voice")
                self._say("Denied.")
            elif self.config.get("security.voice_approval", False) and req.get("risk") == "CAUTION":
                self.inter.resolve(req["id"], True, by="voice")
                self._say("Approved.")
            else:
                self._say("Please approve that one on the dashboard.")
            return
        if low in ("stop everything", "cancel everything", "abort"):
            for tid in self.orch.running():
                self.orch.cancel(tid)
            self._say("Stopping all tasks.")
            return
        self._say("On it.")
        self.submit(text, source="voice")

    def start_voice(self) -> None:
        if not self.enable_voice:
            return
        from . import voice as V
        missing = V.missing_voice_deps(self.config)
        if missing:
            self.voice_status = "unavailable: pip install " + " ".join(m.replace("_", "-") for m in missing)
            return
        try:
            stt = V.make_stt(self.config, self.svc.vault)
            tts = V.make_tts(self.config.get("voice.tts", "auto"))
            self.voice = V.VoiceLoop(V.mic_frames, stt, tts, self.config.get("voice.wake_words", ["saani"]),
                                     self.on_voice_command,
                                     on_error=lambda msg: (setattr(self, "voice_status", "error: " + msg),
                                                           self.svc.events.emit("voice_error", text=msg)))
            self.voice.start()
            self.voice_status = "listening"
            self._say(f"{self.config.get('name', 'Saani')} is online.")
        except Exception as e:
            self.voice_status = f"error: {type(e).__name__}: {e}"

    # ---- lifecycle ---------------------------------------------------------------
    def serve(self, block: bool = True) -> None:
        host, port = self.config.get("daemon.host", "127.0.0.1"), int(self.config.get("daemon.port", 8765))
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit("daemon.host must be a loopback address; Saani's dashboard is never exposed to the network")
        self.server = ThreadingHTTPServer((host, port), make_handler(self))
        self.server.daemon_threads = True
        self.svc.store.mark_interrupted()
        self.scheduler.start()
        self.start_voice()
        self.svc.events.emit("daemon_started", voice=self.voice_status)
        if block:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                self.shutdown()
        else:
            threading.Thread(target=self.server.serve_forever, daemon=True, name="saani-http").start()

    def shutdown(self) -> None:
        self.scheduler.stop()
        if self.voice:
            self.voice.stop()
        for tid in self.orch.running():
            self.orch.cancel(tid)
        if self.server:
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def make_handler(d: Daemon):
    page = (Path(__file__).parent / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Saani"

        def log_message(self, *args):   # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json", headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj, default=str).encode("utf-8"))

        def _host_ok(self) -> bool:
            h = (self.headers.get("Host") or "").strip().lower()
            if h.startswith("["):
                h = h.split("]")[0] + "]"
            elif ":" in h:
                h = h.rsplit(":", 1)[0]
            return h in LOCAL_HOSTS   # blocks DNS-rebinding

        def _token(self) -> str:
            tok = self.headers.get("X-Saani-Token") or ""
            if not tok:
                for part in (self.headers.get("Cookie") or "").split(";"):
                    k, _, v = part.strip().partition("=")
                    if k == "saani_token":
                        tok = v
            return tok

        def _authed(self) -> bool:
            return hmac.compare_digest(self._token().encode(), d.token.encode())

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/api/health":
                return self._json(200, {"ok": True, "name": d.config.get("name", "Saani")})
            if u.path == "/assets/blackhole.png":      # public picture only; no data
                if not ASSET_BACKGROUND:
                    return self._json(404, {"error": "not found"})
                return self._send(200, ASSET_BACKGROUND, "image/png")
            if u.path == "/api/settings" and self._authed():
                return self._json(200, d.settings())
            if u.path == "/":
                given = (q.get("token") or [""])[0]
                if given and hmac.compare_digest(given.encode(), d.token.encode()):
                    return self._send(302, b"", "text/plain", {
                        "Location": "/", "Set-Cookie": f"saani_token={d.token}; HttpOnly; SameSite=Strict; Path=/"})
                if self._authed():
                    return self._send(200, page, "text/html; charset=utf-8")
                return self._send(401, b"Not authorised. Open the dashboard with:  saani open\n", "text/plain")
            if not self._authed():
                return self._json(401, {"error": "unauthorised"})
            if u.path == "/api/state":
                try:
                    since = int((q.get("since") or ["0"])[0])
                except ValueError:
                    since = 0
                if self.headers.get("X-Saani-Client") == "dashboard":
                    d.last_poll = time.time()
                return self._json(200, d.state(since))
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            if not self._authed():
                return self._json(401, {"error": "unauthorised"})
            if self.headers.get("X-Requested-With") != "saani":   # blocks cross-site form posts
                return self._json(403, {"error": "missing X-Requested-With"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 2_000_000)).decode("utf-8") or "{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                return self._json(400, {"error": "invalid JSON"})
            path = urlparse(self.path).path
            if path == "/api/task":
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._json(400, {"error": "empty task"})
                wd = str(body.get("dir") or "").strip() or None
                if wd and not Path(wd).expanduser().is_dir():
                    return self._json(400, {"error": f"no such folder: {wd}"})
                return self._json(200, {"task_id": d.submit(text[:20000], source=str(body.get("source") or "dashboard")[:40], workdir=wd)})
            if path == "/api/approval":
                ok = d.inter.resolve(str(body.get("id", "")), bool(body.get("allow")), str(body.get("scope") or "once"),
                                     str(body.get("note") or "")[:500])
                return self._json(200 if ok else 404, {"ok": ok})
            if path == "/api/answer":
                ok = d.inter.answer(str(body.get("id", "")), str(body.get("text", "")))
                return self._json(200 if ok else 404, {"ok": ok})
            if path == "/api/cancel":
                return self._json(200, {"ok": d.orch.cancel(str(body.get("task_id", "")))})
            if path == "/api/settings":
                try:
                    return self._json(200, d.update_settings(body))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
            if path == "/api/shutdown":
                self._json(200, {"ok": True})
                return d.shutdown()
            return self._json(404, {"error": "not found"})

    return Handler

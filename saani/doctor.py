"""`saani doctor`: check that everything needed is in place."""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys

from . import autostart
from .daemon import call_daemon
from .security.audit import AuditLog
from .security.secrets import Vault


def run_checks(config) -> list[tuple[str, str, str]]:
    """Returns (status, name, detail) where status is ok | warn | fail."""
    out: list[tuple[str, str, str]] = []

    def add(ok, name, detail, warn=False):
        out.append(("ok" if ok else ("warn" if warn else "fail"), name, detail))

    add(sys.version_info >= (3, 10), "Python", sys.version.split()[0] + " (needs 3.10+)")
    add(os.access(config.home, os.W_OK), "Home folder", str(config.home))
    try:
        sqlite3.connect(":memory:").execute("create virtual table t using fts5(x)")
        add(True, "Memory search", "SQLite full-text search available")
    except sqlite3.OperationalError:
        add(False, "Memory search", "SQLite has no FTS5; falling back to slower matching", warn=True)

    provider = config.get("llm.provider")
    vault = Vault(config.home)
    env = config.get("llm.api_key_env", "")
    has_key = bool(os.environ.get(env) or vault.get(env)) if env else False
    if provider == "demo":
        add(False, "Model", "demo mode - no real model connected. Run `saani init`.", warn=True)
    elif provider == "openai" and config.get("llm.base_url") and "api.openai.com" not in config.get("llm.base_url"):
        add(True, "Model", f"OpenAI-compatible endpoint {config.get('llm.base_url')} (models: {config.get('llm.models')})")
    else:
        add(has_key, "Model", f"{provider}: API key {'found' if has_key else 'MISSING - run `saani init`'} ({env})")
    add(vault.backend == "os-keychain", "Secret vault",
        "OS keychain" if vault.backend == "os-keychain" else "file fallback (not encrypted) - `pip install keyring` for the OS keychain",
        warn=True)
    ok, msg = AuditLog(config.audit_path).verify()
    add(ok, "Audit log", msg)

    missing = [m for m in ("sounddevice", "numpy", "faster_whisper") if importlib.util.find_spec(m) is None]
    add(not missing, "Voice input", "ready" if not missing else
        "missing: " + ", ".join(missing) + '  ->  pip install "saani[voice]"  (the dashboard mic button works without it)', warn=True)
    from .voice import SystemTTS
    tts = SystemTTS().available or importlib.util.find_spec("pyttsx3") is not None
    add(tts, "Voice output", "ready" if tts else "no speech engine found (install espeak or pyttsx3)", warn=True)
    add(importlib.util.find_spec("pyautogui") is not None, "Desktop control",
        "ready" if importlib.util.find_spec("pyautogui") else 'optional: pip install "saani[desktop]" for mouse/keyboard/screenshot tools', warn=True)
    try:
        on, detail = autostart.status(config)
    except Exception as e:
        on, detail = False, f"could not check: {e}"
    add(on, "Start at login", detail if on else "not installed - run `saani autostart install`", warn=True)
    health = call_daemon(config, "/api/health", timeout=2)
    add(bool(health), "Background service", "running" if health else "not running - start with `saani daemon`", warn=True)
    filled = config.persona_path.exists() and "not filled in yet" not in config.persona_path.read_text(encoding="utf-8")
    add(filled, "Persona", str(config.persona_path) if filled else
        "not filled in - run `saani persona init` so agents think like you", warn=True)
    return out

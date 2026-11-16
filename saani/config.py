"""Configuration and on-disk layout.

Everything Saani owns lives under one home directory (default ``~/.saani``,
override with the ``SAANI_HOME`` environment variable).
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "name": "Saani",
    "user_name": "",
    "llm": {
        # anthropic | openai (any OpenAI-compatible endpoint, incl. Ollama) | demo
        "provider": "anthropic",
        "base_url": "",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": {
            "planner": "claude-opus-5-5",
            "worker": "claude-sonnet-5-5",
            "fast": "claude-haiku-5-5",
        },
        "max_tokens": 16000,
        "timeout_s": 600,
        "max_retries": 4,
        # Your own model (the saanimind project). provider "own" sends everything to it;
        # otherwise `tiers` lists which tiers it handles, e.g. ["fast"], while the main provider does the rest.
        "own": {"base_url": "http://127.0.0.1:8900/v1", "model": "saanimind", "tiers": []},
        # Save every model exchange (secrets scrubbed) under ~/.saani/traces as training data for your own model.
        "record_traces": False,
    },
    "limits": {
        "max_agents": 8,
        "max_parallel": 4,
        "max_steps_per_agent": 30,
        "max_llm_calls_per_task": 200,
        "command_timeout_s": 120,
        "tool_output_chars": 12000,
    },
    "security": {
        # strict  : everything except read-only actions asks
        # ask     : (default) caution-level actions ask unless you saved a rule
        # trusted : caution-level actions run; dangerous ones still ask
        "mode": "ask",
        "approval_timeout_s": 600,
        "allow_rules": [],
        "trusted_dirs": [],
        "voice_approval": False,
    },
    "skills": {"learn": True, "auto_reuse": "adapt", "direct_threshold": 0.72, "hint_threshold": 0.3},
    "verify": True,
    "search": {"provider": "duckduckgo", "api_key_env": "TAVILY_API_KEY"},
    "voice": {
        "enabled": True,
        "wake_words": ["saani", "sani", "sunny", "sonny", "sahni", "sunni", "sauni"],
        "stt": "local",          # local (faster-whisper) | api (OpenAI-compatible /audio/transcriptions)
        "stt_model": "base",
        "stt_api_base": "https://api.openai.com/v1",
        "stt_api_key_env": "OPENAI_API_KEY",
        "language": "",
        "tts": "auto",           # auto | pyttsx3 | system | off
        "speak_results": True,
    },
    "daemon": {"host": "127.0.0.1", "port": 8765, "open_dashboard_on_start": False,
               "open_dashboard_on_attention": True, "max_concurrent_tasks": 3},
}


def default_home() -> Path:
    return Path(os.environ.get("SAANI_HOME") or (Path.home() / ".saani")).expanduser()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Typed-ish wrapper around ``config.json`` with dotted access."""

    def __init__(self, data: dict | None = None, home: Path | None = None):
        self.home = Path(home) if home else default_home()
        self.data = _merge(DEFAULTS, data or {})
        self._lock = threading.RLock()

    # ---- layout -------------------------------------------------------
    @property
    def file(self) -> Path: return self.home / "config.json"
    @property
    def db_path(self) -> Path: return self.home / "saani.db"
    @property
    def audit_path(self) -> Path: return self.home / "audit.log"
    @property
    def persona_path(self) -> Path: return self.home / "persona.md"
    @property
    def skills_dir(self) -> Path: return self.home / "skills"
    @property
    def agents_dir(self) -> Path: return self.home / "agents"
    @property
    def tools_dir(self) -> Path: return self.home / "tools"
    @property
    def workspace(self) -> Path: return self.home / "workspace"
    @property
    def trash(self) -> Path: return self.home / "trash"
    @property
    def logs(self) -> Path: return self.home / "logs"
    @property
    def traces_dir(self) -> Path: return self.home / "traces"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.skills_dir, self.agents_dir, self.tools_dir,
                  self.workspace, self.trash, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.home, 0o700)
        except OSError:
            pass

    # ---- persistence --------------------------------------------------
    @classmethod
    def load(cls, home: Path | None = None) -> "Config":
        home = Path(home) if home else default_home()
        data: dict = {}
        f = home / "config.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise SystemExit(f"Cannot read {f}: {e}")
        cfg = cls(data, home)
        cfg.ensure_dirs()
        return cfg

    def save(self) -> None:
        with self._lock:
            self.ensure_dirs()
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.file)

    # ---- access -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        with self._lock:
            cur = self.data
            parts = dotted.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value

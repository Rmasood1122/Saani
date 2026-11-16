"""Shared test scaffolding: an isolated Saani home per test."""
from __future__ import annotations

import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path

os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

from saani.app import build
from saani.config import Config
from saani.core.orchestrator import Orchestrator
from saani.interaction import ScriptedInteraction
from saani.simllm import SimLLM


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def agent(id, name, tools=(), deps=(), **kw):
    return {"id": id, "name": name, "role": kw.pop("role", f"{name} role"), "instructions": kw.pop("instructions", "do it"),
            "tools": list(tools), "depends_on": list(deps), **kw}


def plan(*agents, complexity="simple", **kw):
    return {"understanding": "test", "complexity": complexity, "agents": list(agents), **kw}


class SaaniCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="saani-test-")).resolve()
        self.home = self.tmp / "home"
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, llm: SimLLM | None = None, decide=False, answers=None, **settings):
        cfg = Config({}, self.home)
        cfg.set("llm.provider", "demo")
        cfg.set("verify", False)
        for k, v in settings.items():
            cfg.set(k.replace("__", "."), v)
        self.inter = ScriptedInteraction(decide, answers)
        self.llm = llm or SimLLM()
        self.svc = build(cfg, self.inter, self.llm, use_keyring=False)
        self.orch = Orchestrator(self.svc)
        return self.svc

    def run_task(self, text="do the thing", **kw):
        return self.orch.run_task(text, **kw)

    def audit(self):
        return self.svc.audit.entries()

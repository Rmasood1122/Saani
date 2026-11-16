"""Human-in-the-loop channels: how Saani asks you for permission or input.

* TerminalInteraction - prompts in the terminal (``saani run`` / ``saani chat``)
* QueueInteraction    - parks requests for the dashboard / voice (daemon)
* DenyInteraction     - refuses everything (fully unattended)
* ScriptedInteraction - deterministic answers (tests and evals)
"""
from __future__ import annotations

import getpass
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class ApprovalRequest:
    task_id: str
    agent: str
    tool: str
    summary: str
    risk: str
    reason: str
    rule_key: str = ""
    can_always: bool = False
    tainted: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


@dataclass
class ApprovalResult:
    allowed: bool
    scope: str = "once"   # once | task | always
    note: str = ""
    by: str = "user"


class Interaction:
    def approve(self, req: ApprovalRequest) -> ApprovalResult:
        raise NotImplementedError

    def ask(self, question: str, *, task_id: str = "", agent: str = "", secret: bool = False) -> str | None:
        raise NotImplementedError

    def notify(self, text: str) -> None:
        pass


class DenyInteraction(Interaction):
    def approve(self, req):
        return ApprovalResult(False, note="nobody is available to approve this", by="unattended")

    def ask(self, question, **kw):
        return None


class ScriptedInteraction(Interaction):
    """``decide`` maps a request to True/False/'task'/'always'/ApprovalResult."""

    def __init__(self, decide: Callable[[ApprovalRequest], object] | bool = False, answers: dict | None = None):
        self.decide = decide
        self.answers = answers or {}
        self.requests: list[ApprovalRequest] = []
        self.questions: list[str] = []
        self.notes: list[str] = []
        self._lock = threading.Lock()

    def approve(self, req):
        with self._lock:
            self.requests.append(req)
        d = self.decide(req) if callable(self.decide) else self.decide
        if isinstance(d, ApprovalResult):
            return d
        if d in ("task", "always"):
            return ApprovalResult(True, scope=d)
        return ApprovalResult(bool(d))

    def ask(self, question, **kw):
        with self._lock:
            self.questions.append(question)
        for key, val in self.answers.items():
            if key.lower() in question.lower():
                return val
        return self.answers.get("*")

    def notify(self, text):
        self.notes.append(text)


class TerminalInteraction(Interaction):
    _lock = threading.Lock()  # one prompt at a time even with parallel agents

    def __init__(self, out=None, inp: Callable[[str], str] | None = None):
        self._out = out
        self.inp = inp or (lambda prompt: input(prompt))

    @property
    def out(self):
        return self._out or sys.stdout

    def _p(self, text=""):
        print(text, file=self.out, flush=True)

    def approve(self, req):
        with self._lock:
            self._p()
            self._p(f"+-- PERMISSION NEEDED [{req.risk}] " + "-" * 34)
            self._p(f"| agent : {req.agent}   (task {req.task_id})")
            self._p(f"| action: {req.tool}")
            for line in req.summary.splitlines()[:14]:
                self._p(f"|   {line[:160]}")
            self._p(f"| why   : {req.reason}")
            if req.tainted:
                self._p("| note  : this agent has read web content, so nothing is auto-approved for it")
            opts = "[y] once   [t] rest of this task   "
            if req.can_always:
                opts += "[a] always   "
            opts += "[n] deny"
            self._p("+-- " + opts)
            try:
                ans = self.inp("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ApprovalResult(False, note="no answer from the terminal", by="eof")
            if ans in ("y", "yes"):
                return ApprovalResult(True)
            if ans in ("t", "task"):
                return ApprovalResult(True, scope="task")
            if ans in ("a", "always") and req.can_always:
                return ApprovalResult(True, scope="always")
            note = ans[1:].strip() if ans.startswith("n ") else ("" if ans in ("n", "no", "") else ans)
            return ApprovalResult(False, note=note)

    def ask(self, question, *, task_id="", agent="", secret=False):
        with self._lock:
            self._p()
            self._p(f"[{agent or 'Saani'} asks] {question}")
            try:
                if secret:
                    return getpass.getpass("(hidden) > ") or None
                return self.inp("> ").strip() or None
            except (EOFError, KeyboardInterrupt):
                return None

    def notify(self, text):
        self._p(f"[note] {text}")


class _Pending:
    def __init__(self, kind: str, payload: dict):
        self.id = payload.get("id") or uuid.uuid4().hex[:10]
        self.kind = kind
        self.payload = {**payload, "id": self.id, "kind": kind}
        self.event = threading.Event()
        self.result: object = None


class QueueInteraction(Interaction):
    """Requests wait here until the dashboard (or voice) resolves them, or they time out."""

    def __init__(self, timeout_s: float = 600, emit: Callable[..., None] | None = None):
        self.timeout_s = timeout_s
        self.emit = emit or (lambda *a, **k: None)
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def _wait(self, p: _Pending):
        with self._lock:
            self._pending[p.id] = p
        self.emit("attention", **{k: v for k, v in p.payload.items() if k != "kind"}, request_kind=p.kind)
        p.event.wait(self.timeout_s)
        with self._lock:
            self._pending.pop(p.id, None)
        return p.result

    def approve(self, req):
        res = self._wait(_Pending("approval", asdict(req)))
        if isinstance(res, ApprovalResult):
            return res
        return ApprovalResult(False, note="timed out waiting for your approval", by="timeout")

    def ask(self, question, *, task_id="", agent="", secret=False):
        res = self._wait(_Pending("secret" if secret else "question",
                                  {"question": question, "task_id": task_id, "agent": agent}))
        return res if isinstance(res, str) and res else None

    def notify(self, text):
        self.emit("notify", text=text)

    # ---- called by the dashboard / voice --------------------------------
    def pending(self) -> list[dict]:
        with self._lock:
            return [dict(p.payload) for p in self._pending.values()]

    def resolve(self, req_id: str, allowed: bool, scope: str = "once", note: str = "", by: str = "user") -> bool:
        with self._lock:
            p = self._pending.get(req_id)
        if not p or p.kind != "approval":
            return False
        if scope == "always" and not p.payload.get("can_always"):
            scope = "task"
        p.result = ApprovalResult(bool(allowed), scope=scope if allowed else "once", note=note, by=by)
        p.event.set()
        return True

    def answer(self, req_id: str, text: str) -> bool:
        with self._lock:
            p = self._pending.get(req_id)
        if not p or p.kind == "approval":
            return False
        p.result = text
        p.event.set()
        return True

"""The guard: every tool call passes through here before it runs."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..interaction import ApprovalRequest, Interaction
from ..textutil import clip
from .audit import AuditLog
from urllib.parse import urlparse

from .policy import Policy, Risk, Verdict, _inside, exact_key, norm_path
from .secrets import Redactor

_URL_RX = re.compile(r"https?://[^\s<>\"'`\[\](){}|\\^]+", re.I)


def norm_url(url: str) -> str:
    u = (url or "").strip()
    if "://" not in u:
        u = "https://" + u
    return u.split("#", 1)[0].rstrip(".,;:!?").rstrip("/")


def urls_in(text: str) -> set:
    return {norm_url(u) for u in _URL_RX.findall(text or "")}


@dataclass
class CallContext:
    task_id: str
    agent: str
    workdir: Path
    tainted: bool = False      # agent has ingested untrusted (web) content
    unattended: bool = False
    seen_urls: set = field(default_factory=set)   # addresses the agent was given or shown, verbatim


def summarize_call(tool: str, args: dict) -> str:
    a = args or {}
    if tool == "run_shell":
        s = f"$ {a.get('command', '')}"
        if a.get("cwd"): s += f"\n(in {a['cwd']})"
        if a.get("secrets"): s += f"\n(with secrets: {', '.join(map(str, a['secrets']))})"
        if a.get("background"): s += "\n(runs in the background)"
        return s
    if tool == "write_file":
        c = str(a.get("content", ""))
        return f"write {a.get('path')} ({len(c)} chars)\n{clip(c, 500)}"
    if tool == "edit_file":
        return f"edit {a.get('path')}\n- {clip(str(a.get('old', '')), 240)}\n+ {clip(str(a.get('new', '')), 240)}"
    if tool in ("run_python", "create_tool"):
        head = f"tool '{a.get('name')}': {a.get('description', '')}\n" if tool == "create_tool" else ""
        return head + clip(str(a.get("code", "")), 1500)
    if tool in ("delete_path", "read_file", "list_dir"):
        return f"{tool} {a.get('path')}"
    if tool == "move_path":
        return f"move {a.get('src')} -> {a.get('dst')}"
    if tool == "http_request":
        return f"{a.get('method', 'GET')} {a.get('url')}\n{clip(json.dumps(a.get('body', ''), default=str), 400)}"
    return clip(json.dumps(a, default=str, ensure_ascii=False), 700)


class Guard:
    def __init__(self, config, policy: Policy, interaction: Interaction, audit: AuditLog,
                 redactor: Redactor):
        self.config = config
        self.policy = policy
        self.interaction = interaction
        self.audit = audit
        self.redactor = redactor
        self._grants: dict[str, set[str]] = {}
        self._tainted_grants: dict[str, set[str]] = {}   # given while the owner could see the web-content warning
        self._lock = threading.RLock()

    # ---- rules ---------------------------------------------------------
    def rules(self) -> list[str]:
        return list(self.config.get("security.allow_rules", []) or [])

    def add_rule(self, key: str) -> None:
        with self._lock:
            rules = self.rules()
            for k in (key[4:].split("|") if key.startswith("all:") else [key]):
                if k and not k.startswith("exact:") and k not in rules:
                    rules.append(k)
            self.config.set("security.allow_rules", rules)
            self.config.save()

    def remove_rule(self, key: str) -> bool:
        with self._lock:
            rules = self.rules()
            if key not in rules:
                return False
            rules.remove(key)
            self.config.set("security.allow_rules", rules)
            self.config.save()
            return True

    @staticmethod
    def _covered(key: str, pool) -> bool:
        if not key:
            return False
        if key.startswith("all:"):
            return all(k in pool for k in key[4:].split("|"))
        return key in pool

    def end_task(self, task_id: str) -> None:
        with self._lock:
            self._grants.pop(task_id, None)
            self._tainted_grants.pop(task_id, None)

    def _after_web_content(self, ctx: CallContext, tool: str, args: dict, v: Verdict) -> Verdict:
        """An agent that has read web content may be following instructions planted there.
        Some actions that are normally free can then do lasting damage, so they ask instead:
        saving to long-term memory (planted text would resurface in later tasks), changing files
        outside Saani's own scratch space (a quiet edit to your project), opening a page in your
        real browser, and fetching an address the agent composed itself (private data can leave
        inside the URL)."""
        if not ctx.tainted or v.risk != Risk.SAFE:
            return v
        a = args or {}
        if tool == "remember":
            return Verdict(Risk.CAUTION, "saves to long-term memory after reading web content "
                                         "(planted text would resurface in later tasks)")
        if tool in ("write_file", "edit_file", "delete_path", "move_path"):
            for raw in (a.get("path"), a.get("src"), a.get("dst")):
                if not raw:
                    continue
                p = Path(str(raw)).expanduser()
                n = norm_path(p if p.is_absolute() else Path(ctx.workdir) / p)
                if not _inside(n, self.policy.workspace):
                    return Verdict(Risk.CAUTION, "changes a file in your project after reading web content",
                                   rule_key="write:" + (n.rsplit("/", 1)[0] or "/"))
            return v
        if tool == "open_target" and "://" in str(a.get("target", "")):
            return Verdict(Risk.CAUTION, "opens a page in your own browser after reading web content")
        url = a.get("url") if tool in ("web_fetch", "http_request") else None
        if url and norm_url(str(url)) not in ctx.seen_urls:
            host = urlparse(norm_url(str(url))).hostname or "?"
            return Verdict(Risk.CAUTION, f"contacts {host} at an address the agent composed itself after reading "
                                         "web content (private data could leave inside the address)",
                           rule_key=f"host:{host}")
        return v

    # ---- the decision ----------------------------------------------------
    def authorize(self, ctx: CallContext, tool, args: dict) -> tuple[bool, str]:
        v = self._after_web_content(ctx, tool.name, args, self.policy.classify(tool.name, args, tool.risk, ctx.workdir))
        summary = self.redactor.redact(summarize_call(tool.name, args))
        mode = self.config.get("security.mode", "ask")
        danger_key = exact_key(tool.name, args)
        grant_key = danger_key if v.risk >= Risk.DANGER else (v.rule_key or danger_key)

        def log(decision: str, by: str, note: str = ""):
            self.audit.record(task=ctx.task_id, agent=ctx.agent, tool=tool.name, action=clip(summary, 1500),
                              risk=v.risk.name, reason=v.reason, decision=decision, by=by,
                              tainted=ctx.tainted, **({"note": note} if note else {}))

        if v.risk == Risk.BLOCKED:
            log("deny", "policy")
            return False, (f"BLOCKED by safety policy ({v.reason}). This can never be run by an agent. "
                           "Do not try another way to achieve the same effect; tell the user it must be done manually.")
        by = ""
        with self._lock:
            grants = set(self._grants.get(ctx.task_id, ()))
            tainted_grants = set(self._tainted_grants.get(ctx.task_id, ()))
        if v.risk == Risk.SAFE:
            by = "policy:safe"
        elif ctx.tainted:
            # saved rules, trusted mode and earlier approvals do not carry over once web content is
            # in play; only an approval given for this task with the warning on screen does
            by = "task-grant" if self._covered(grant_key, tainted_grants) else ""
        elif mode == "strict":
            by = ""
        elif v.risk == Risk.CAUTION:
            if mode == "trusted":
                by = "mode:trusted"
            elif v.always_ok and self._covered(v.rule_key, self.rules()):
                by = "rule"
            elif self._covered(grant_key, grants | set(self.rules()) if v.always_ok else grants):
                by = "task-grant"
        elif v.risk == Risk.DANGER and grant_key in grants:
            by = "task-grant"
        if by:
            log("allow", by)
            return True, ""

        can_always = v.risk == Risk.CAUTION and v.always_ok and not ctx.tainted and mode != "strict"
        req = ApprovalRequest(task_id=ctx.task_id, agent=ctx.agent, tool=tool.name, summary=summary,
                              risk=v.risk.name, reason=v.reason, rule_key=v.rule_key,
                              can_always=can_always, tainted=ctx.tainted)
        try:
            res = self.interaction.approve(req)
        except Exception as e:  # a broken UI must fail closed
            log("deny", "error", str(e))
            return False, "PERMISSION DENIED (the approval channel failed). Continue without this action."
        if not res.allowed:
            log("deny", res.by, res.note)
            extra = f" The user said: {res.note!r}." if res.note and res.by == "user" else (f" ({res.note})" if res.note else "")
            return False, (f"PERMISSION DENIED: {v.reason}.{extra} Do not retry this action or work around it; "
                           "continue without it, or finish and report that it needs the user's approval.")
        if res.scope == "always" and can_always:
            self.add_rule(v.rule_key)
        elif res.scope in ("task", "always"):
            with self._lock:
                keys = grant_key[4:].split("|") if grant_key.startswith("all:") else [grant_key]
                pool = self._tainted_grants if ctx.tainted else self._grants
                pool.setdefault(ctx.task_id, set()).update(keys)
        log("allow", f"{res.by}:{res.scope}")
        return True, ""

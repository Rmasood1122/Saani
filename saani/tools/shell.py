"""Running commands and Python snippets on the user's machine."""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ..security.policy import Risk
from ..textutil import clip
from .base import B, I, S, Tool, ToolContext, ToolResult, obj

_SECRET_ENV = re.compile(r"(PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL)", re.I)


def child_env(ctx: ToolContext, secret_names: list[str] | None = None) -> tuple[dict, list[str]]:
    """Environment for child processes: secret-looking variables are stripped unless
    the agent asked for them by name (which itself required approval)."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env["SAANI_TASK"] = ctx.task_id
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # cmd.exe looks in the current folder before PATH, so an agent that can write a file named
    # git.bat could turn a harmless `git status` into running it; the same goes for "." on PATH.
    env["NoDefaultCurrentDirectoryInExePath"] = "1"
    if env.get("PATH"):
        env["PATH"] = os.pathsep.join(d for d in env["PATH"].split(os.pathsep) if d.strip() not in ("", "."))
    missing = []
    for name in secret_names or []:
        value = ctx.services.vault.resolve(str(name))
        if value is None:
            missing.append(str(name))
        else:
            env[str(name)] = value
    return env, missing


def _kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            p.kill()
        except OSError:
            pass


def run_process(argv, *, shell: bool, cwd: Path, env: dict, timeout: float, ctx: ToolContext,
                stdin_text: str | None = None) -> tuple[int | None, str, str, str]:
    """Run to completion with timeout and cancellation. Returns (code, stdout, stderr, note)."""
    kw: dict = dict(cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                    text=True, encoding="utf-8", errors="replace")
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    try:
        p = subprocess.Popen(argv, shell=shell, **kw)
    except OSError as e:
        return None, "", "", f"could not start: {e}"
    deadline = time.time() + timeout
    note, first = "", True
    while True:
        try:
            out, err = p.communicate(input=stdin_text if first else None, timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            first = False
            if ctx.cancel.is_set():
                note = "cancelled by the user"
            elif time.time() > deadline:
                note = f"timed out after {int(timeout)}s and was stopped"
            if note:
                _kill_tree(p)
                try:
                    out, err = p.communicate(timeout=5)
                except (subprocess.TimeoutExpired, ValueError, OSError):
                    out, err = "", ""
                break
    return p.returncode, out or "", err or "", note


def _format(code, out, err, note, limit: int) -> ToolResult:
    parts = [f"exit code: {code}" if not note else f"[{note}]"]
    if out.strip():
        parts.append("stdout:\n" + clip(out.strip(), limit))
    if err.strip():
        parts.append("stderr:\n" + clip(err.strip(), limit // 2))
    if not out.strip() and not err.strip():
        parts.append("(no output)")
    return ToolResult("\n".join(parts), is_error=bool(note) or code not in (0, None))


def _run_shell(a: dict, ctx: ToolContext):
    cfg = ctx.services.config
    command = a["command"]
    cwd = ctx.resolve(a["cwd"]) if a.get("cwd") else Path(ctx.workdir)
    if not cwd.is_dir():
        return ToolResult(f"Working directory does not exist: {cwd}", True)
    env, missing = child_env(ctx, a.get("secrets"))
    if missing:
        return ToolResult(f"Unknown secret(s): {', '.join(missing)}. Use request_secret to ask the user for them.", True)
    timeout = max(1, min(int(a.get("timeout_s") or cfg.get("limits.command_timeout_s", 120)), 3600))
    argv, shell = command, True
    if a.get("powershell"):
        exe = shutil.which("pwsh") or shutil.which("powershell")
        if not exe:
            return ToolResult("PowerShell is not installed on this machine.", True)
        argv, shell = [exe, "-NoProfile", "-NonInteractive", "-Command", command], False
    if a.get("background"):
        log = Path(ctx.workdir) / f"background-{uuid.uuid4().hex[:6]}.log"
        kw: dict = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kw["start_new_session"] = True
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.Popen(argv, shell=shell, stdout=f, **kw)
        return f"Started in the background (pid {p.pid}). Output is being written to {log}"
    code, out, err, note = run_process(argv, shell=shell, cwd=cwd, env=env, timeout=timeout, ctx=ctx)
    return _format(code, out, err, note, cfg.get("limits.tool_output_chars", 12000))


def _run_python(a: dict, ctx: ToolContext):
    cfg = ctx.services.config
    script = Path(ctx.workdir) / f".saani_snippet_{uuid.uuid4().hex[:8]}.py"
    script.write_text(a["code"], encoding="utf-8")
    env, _ = child_env(ctx)
    timeout = max(1, min(int(a.get("timeout_s") or cfg.get("limits.command_timeout_s", 120)), 3600))
    try:
        code, out, err, note = run_process([sys.executable, str(script)], shell=False, cwd=Path(ctx.workdir),
                                           env=env, timeout=timeout, ctx=ctx)
    finally:
        try:
            script.unlink()
        except OSError:
            pass
    return _format(code, out, err, note, cfg.get("limits.tool_output_chars", 12000))


TOOLS = [
    Tool("run_shell",
         "Run a shell command on the user's computer and return its output. Use 'cwd' for the folder, "
         "'secrets' to inject stored secrets as environment variables by name (never put secret values "
         "in the command), 'background' for long-running servers, 'powershell' to use PowerShell.",
         obj({"command": S, "cwd": S, "timeout_s": I, "secrets": {"type": "array", "items": S},
              "background": B, "powershell": B}, ["command"]), _run_shell, Risk.CAUTION),
    Tool("run_python", "Run a Python snippet in the task folder and return what it prints.",
         obj({"code": S, "timeout_s": I}, ["code"]), _run_python, Risk.CAUTION),
]

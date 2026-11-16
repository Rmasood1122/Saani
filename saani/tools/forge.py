"""Tool forge: agents can build new tools, which are kept and reused in later tasks.

A custom tool is a folder ``<home>/tools/<name>/`` with ``tool.json`` (manifest)
and ``main.py`` exposing ``run(args: dict) -> str``. It always executes in a
separate Python process, never inside Saani itself.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..security.policy import Risk
from .base import S, Tool, ToolContext, ToolRegistry, ToolResult, obj
from .shell import child_env, run_process

RUNNER = r"""
import importlib.util, json, sys, traceback
spec = importlib.util.spec_from_file_location("saani_custom_tool", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    out = mod.run(json.loads(sys.stdin.read() or "{}"))
    print(json.dumps({"ok": True, "output": out if isinstance(out, str) else json.dumps(out, default=str)}))
except Exception:
    print(json.dumps({"ok": False, "output": traceback.format_exc()[-3000:]}))
"""


def _make(manifest: dict, folder: Path) -> Tool:
    def fn(a: dict, ctx: ToolContext):
        env, _ = child_env(ctx)
        code, out, err, note = run_process([sys.executable, "-c", RUNNER, str(folder / "main.py")],
                                           shell=False, cwd=Path(ctx.workdir), env=env,
                                           timeout=float(manifest.get("timeout_s", 120)), ctx=ctx,
                                           stdin_text=json.dumps(a))
        if note:
            return ToolResult(f"[{note}]", True)
        try:
            data = json.loads(out.strip().splitlines()[-1])
            return ToolResult(str(data.get("output", "")), not data.get("ok"))
        except (ValueError, IndexError):
            return ToolResult(f"Tool crashed (exit {code}).\n{(err or out)[-2000:]}", True)

    risk = Risk.DANGER if manifest.get("risk") == "DANGER" else Risk.CAUTION
    return Tool(manifest["name"], "[custom] " + manifest.get("description", ""),
                manifest.get("parameters") or obj({}), fn, risk)


def load_custom_tools(registry: ToolRegistry, tools_dir: Path) -> list[str]:
    loaded = []
    if not Path(tools_dir).exists():
        return loaded
    for mf in sorted(Path(tools_dir).glob("*/tool.json")):
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
            if registry.get(manifest["name"]) is None and (mf.parent / "main.py").exists():
                registry.register(_make(manifest, mf.parent))
                loaded.append(manifest["name"])
        except (OSError, ValueError, KeyError):
            continue
    return loaded


def _create_tool(a: dict, ctx: ToolContext):
    svc = ctx.services
    name = a["name"]
    if not re.match(r"^[a-z][a-z0-9_]{2,40}$", name):
        return ToolResult("Tool names must be lower_snake_case, 3-41 characters.", True)
    existing = svc.tools.get(name)
    if existing is not None and not existing.description.startswith("[custom]"):
        return ToolResult(f"'{name}' is a built-in tool name; choose another.", True)
    code = a["code"]
    if "def run(" not in code:
        return ToolResult("The code must define `def run(args: dict) -> str`.", True)
    try:
        compile(code, f"{name}/main.py", "exec")
    except SyntaxError as e:
        return ToolResult(f"Syntax error in tool code: {e}", True)
    params = a.get("parameters") or obj({})
    if params.get("type") != "object":
        return ToolResult("'parameters' must be a JSON schema of type object.", True)
    verdict = svc.policy.classify_python(code)
    folder = svc.config.tools_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "description": a["description"], "parameters": params,
                "risk": "DANGER" if verdict.risk >= Risk.DANGER else "CAUTION",
                "created_by": ctx.agent, "task": ctx.task_id}
    (folder / "main.py").write_text(code, encoding="utf-8")
    (folder / "tool.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    svc.tools.register(_make(manifest, folder))
    ctx.grant_tool(name)
    return f"Tool '{name}' created and saved for reuse. You can call it now; future tasks can use it too."


TOOLS = [
    Tool("create_tool",
         "Create a reusable tool when you need a capability more than once. Provide Python code defining "
         "`def run(args: dict) -> str`, plus a JSON-schema 'parameters' object. The tool is saved for future tasks.",
         obj({"name": S, "description": S, "parameters": {"type": "object"}, "code": S},
             ["name", "description", "code"]), _create_tool, Risk.CAUTION),
]

"""Tool plumbing: what a tool is, how it is described to the model, how it is registered."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..security.policy import Risk


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    images: list = field(default_factory=list)


@dataclass
class ToolContext:
    task_id: str
    agent: str
    workdir: Path
    services: Any
    cancel: threading.Event = field(default_factory=threading.Event)
    unattended: bool = False
    grant_tool: Callable[[str], None] = lambda name: None

    def resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(str(path)))
        return p if p.is_absolute() else Path(self.workdir) / p


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


class Tool:
    def __init__(self, name: str, description: str, schema: dict, fn: Callable[[dict, ToolContext], Any],
                 risk: Risk = Risk.CAUTION, taints: bool = False):
        self.name, self.description, self.fn = name, description, fn
        self.schema = schema or {"type": "object", "properties": {}}
        self.risk, self.taints = risk, taints

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}

    def validate(self, args: dict) -> str | None:
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        props = self.schema.get("properties", {})
        for req in self.schema.get("required", []):
            if req not in args:
                return f"missing required argument '{req}'"
        for k, v in args.items():
            want = _TYPES.get((props.get(k) or {}).get("type", ""))
            if want and v is not None and not isinstance(v, want):
                if want is int and isinstance(v, str) and v.lstrip("-").isdigit():
                    args[k] = int(v)
                    continue
                return f"argument '{k}' should be of type {props[k]['type']}"
            if want in (int, (int, float)) and isinstance(v, bool):
                return f"argument '{k}' should be of type {props[k]['type']}"
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        out = self.fn(args, ctx)
        if isinstance(out, ToolResult):
            return out
        return ToolResult(str(out))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()

    def register(self, tool: Tool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> str:
        return "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in sorted(self._tools.values(), key=lambda t: t.name))


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}

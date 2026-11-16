"""File-system tools."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from ..security.policy import SECRET_DIRS, SELF_UNREADABLE, Risk, is_secret_path
from ..textutil import clip
from .base import B, I, S, Tool, ToolContext, ToolResult, obj

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
             "dist", "build", ".next", ".idea", ".tox", "target", ".cache"}


_snapshotted: set = set()
_snap_lock = threading.Lock()


def trash_slot(ctx: ToolContext, name: str) -> Path:
    folder = ctx.services.config.trash / time.strftime("%Y%m%d-%H%M%S") / ctx.task_id
    folder.mkdir(parents=True, exist_ok=True)
    dest, n = folder / name, 1
    while dest.exists():
        dest = folder / f"{name}.{n}"
        n += 1
    return dest


def trash_record(ctx: ToolContext, kind: str, original: Path, stored: Path) -> None:
    line = json.dumps({"ts": round(time.time(), 2), "kind": kind, "task": ctx.task_id, "agent": ctx.agent,
                       "original": str(original), "stored": str(stored)}, ensure_ascii=False)
    with _snap_lock:
        with (ctx.services.config.trash / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def snapshot(p: Path, ctx: ToolContext) -> str:
    """Before a tool changes one of the owner's existing files, keep the version from before
    this task in Saani's trash, so the change can be undone with `saani trash restore`."""
    try:
        cfg = ctx.services.config
        real = p.resolve()
        if not real.is_file() or real.stat().st_size > 200_000_000:
            return ""
        if cfg.workspace.resolve() in real.parents or cfg.trash.resolve() in real.parents:
            return ""   # Saani's own scratch space needs no undo copies
        key = (ctx.task_id, str(real))
        with _snap_lock:
            if key in _snapshotted:
                return ""
            _snapshotted.add(key)
        dest = trash_slot(ctx, real.name)
        shutil.copy2(real, dest)
        trash_record(ctx, "changed", real, dest)
        return f" The previous version is kept at {dest}."
    except OSError:
        return ""


def _read_file(a: dict, ctx: ToolContext):
    p = ctx.resolve(a["path"])
    if not p.exists():
        return ToolResult(f"No such file: {p}", True)
    if p.is_dir():
        return ToolResult(f"{p} is a directory; use list_dir.", True)
    raw = p.read_bytes()
    if b"\x00" in raw[:4096]:
        return ToolResult(f"{p} looks like a binary file ({len(raw)} bytes); not shown.", True)
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    start = max(1, int(a.get("start_line") or 1))
    count = int(a.get("max_lines") or 800)
    chunk = lines[start - 1: start - 1 + count]
    body = "\n".join(f"{start + i:>5}  {ln}" for i, ln in enumerate(chunk)) if a.get("numbered") else "\n".join(chunk)
    more = len(lines) - (start - 1 + len(chunk))
    note = f"\n[{more} more lines; call again with start_line={start + len(chunk)}]" if more > 0 else ""
    return clip(body, 60000) + note


def _write_file(a: dict, ctx: ToolContext):
    p = ctx.resolve(a["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    content = a.get("content", "")
    kept = snapshot(p, ctx)
    with p.open("a" if a.get("append") else "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {p}.{kept}"


def _edit_file(a: dict, ctx: ToolContext):
    p = ctx.resolve(a["path"])
    if not p.is_file():
        return ToolResult(f"No such file: {p}", True)
    text = p.read_text(encoding="utf-8")
    old, new = a["old"], a["new"]
    n = text.count(old)
    if n == 0:
        return ToolResult("The 'old' text was not found. Read the file again and copy the exact text.", True)
    if n > 1 and not a.get("replace_all"):
        return ToolResult(f"The 'old' text occurs {n} times. Add surrounding lines to make it unique, or set replace_all.", True)
    kept = snapshot(p, ctx)
    p.write_text(text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1),
                 encoding="utf-8", newline="")
    return f"Edited {p} ({n if a.get('replace_all') else 1} replacement(s)).{kept}"


def _list_dir(a: dict, ctx: ToolContext):
    root = ctx.resolve(a.get("path") or ".")
    if not root.is_dir():
        return ToolResult(f"Not a directory: {root}", True)
    depth = max(1, min(int(a.get("depth") or 1), 4))
    out, count = [f"{root}"], 0

    def walk(d: Path, level: int):
        nonlocal count
        try:
            entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError as e:
            out.append("  " * level + f"[unreadable: {e}]")
            return
        for e in entries:
            if count >= 400:
                return
            count += 1
            if e.is_dir():
                out.append("  " * level + e.name + "/")
                if level < depth and e.name not in SKIP_DIRS:
                    walk(e, level + 1)
            else:
                try:
                    size = e.stat().st_size
                except OSError:
                    size = 0
                out.append("  " * level + f"{e.name}  ({size} B)")

    walk(root, 1)
    if count >= 400:
        out.append("[listing truncated at 400 entries]")
    return "\n".join(out)


def _search_files(a: dict, ctx: ToolContext):
    root = ctx.resolve(a.get("root") or ".")
    if not root.exists():
        return ToolResult(f"No such path: {root}", True)
    glob = a.get("glob") or "*"
    text = a.get("text")
    try:
        rx = re.compile(text, re.I) if text else None
    except re.error:
        rx = re.compile(re.escape(text), re.I)
    limit = max(1, min(int(a.get("max_results") or 60), 300))
    hits, visited, withheld = [], 0, 0
    for dirpath, dirs, files in os.walk(root):
        withheld += sum(1 for d in dirs if d.lower() in SECRET_DIRS)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d.lower() not in SECRET_DIRS]
        for name in files:
            visited += 1
            if visited > 20000 or len(hits) >= limit:
                break
            full = Path(dirpath) / name
            if name.lower() in SELF_UNREADABLE or is_secret_path(str(full).replace("\\", "/").lower()):
                withheld += 1      # credentials are never searched; reading one takes read_file and approval
                continue
            rel = str(full.relative_to(root))
            if not (fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(rel.replace("\\", "/"), glob)):
                continue
            if rx is None:
                hits.append(rel)
                continue
            try:
                if full.stat().st_size > 2_000_000:
                    continue
                with full.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                            if len(hits) >= limit:
                                break
            except OSError:
                continue
        if visited > 20000 or len(hits) >= limit:
            break
    note = f"\n[{withheld} credential file(s) or folder(s) were skipped]" if withheld else ""
    if not hits:
        return "No matches." + note
    return "\n".join(hits) + (f"\n[stopped at {limit} results]" if len(hits) >= limit else "") + note


def _delete_path(a: dict, ctx: ToolContext):
    p = ctx.resolve(a["path"])
    if not p.exists() and not p.is_symlink():
        return ToolResult(f"No such path: {p}", True)
    dest = trash_slot(ctx, p.name)
    shutil.move(str(p), str(dest))
    trash_record(ctx, "deleted", p, dest)
    return f"Moved {p} to Saani's trash ({dest}). It can be restored with `saani trash restore`."


def _move_path(a: dict, ctx: ToolContext):
    src, dst = ctx.resolve(a["src"]), ctx.resolve(a["dst"])
    if not src.exists():
        return ToolResult(f"No such path: {src}", True)
    if dst.exists() and not dst.is_dir() and not a.get("overwrite"):
        return ToolResult(f"{dst} already exists; set overwrite to replace it.", True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = snapshot(dst, ctx) if dst.is_file() else ""
    shutil.move(str(src), str(dst))
    return f"Moved {src} -> {dst}.{kept}"


TOOLS = [
    Tool("read_file", "Read a text file. Returns up to max_lines lines starting at start_line.",
         obj({"path": S, "start_line": I, "max_lines": I, "numbered": B}, ["path"]), _read_file, Risk.SAFE),
    Tool("write_file", "Create or overwrite a text file (set append to add to the end). Parent folders are created.",
         obj({"path": S, "content": S, "append": B}, ["path", "content"]), _write_file, Risk.CAUTION),
    Tool("edit_file", "Replace an exact piece of text in a file. 'old' must match exactly and be unique unless replace_all.",
         obj({"path": S, "old": S, "new": S, "replace_all": B}, ["path", "old", "new"]), _edit_file, Risk.CAUTION),
    Tool("list_dir", "List a folder (depth 1-4).", obj({"path": S, "depth": I}), _list_dir, Risk.SAFE),
    Tool("search_files", "Find files by name pattern (glob) and optionally by content (regex 'text'). Returns paths or path:line: matches.",
         obj({"root": S, "glob": S, "text": S, "max_results": I}), _search_files, Risk.SAFE),
    Tool("delete_path", "Delete a file or folder. It is moved to Saani's trash, so it can be restored.",
         obj({"path": S}, ["path"]), _delete_path, Risk.DANGER),
    Tool("move_path", "Move or rename a file or folder.",
         obj({"src": S, "dst": S, "overwrite": B}, ["src", "dst"]), _move_path, Risk.CAUTION),
]

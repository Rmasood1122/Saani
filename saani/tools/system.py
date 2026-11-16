"""System tools: machine info, opening things, and optional desktop control."""
from __future__ import annotations

import getpass
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from ..security.policy import Risk
from .base import I, S, Tool, ToolContext, ToolResult, obj


def _system_info(a: dict, ctx: ToolContext):
    try:
        du = shutil.disk_usage(str(Path.home()))
        disk = f"{du.free // 2**30} GB free of {du.total // 2**30} GB"
    except OSError:
        disk = "unknown"
    shell = "cmd.exe (set powershell=true for PowerShell)" if os.name == "nt" else os.environ.get("SHELL", "/bin/sh")
    return "\n".join([
        f"os: {platform.system()} {platform.release()} ({platform.machine()})",
        f"user: {getpass.getuser()}", f"home: {Path.home()}", f"task folder: {ctx.workdir}",
        f"shell: {shell}", f"python: {sys.version.split()[0]} at {sys.executable}",
        f"cpus: {os.cpu_count()}", f"disk: {disk}", f"time: {time.strftime('%Y-%m-%d %H:%M %Z')}",
    ])


def open_with_default(target: str) -> None:
    if os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _open_target(a: dict, ctx: ToolContext):
    target = a["target"]
    if re.match(r"^https?://", target):
        webbrowser.open(target)
        return f"Opened {target} in the default browser."
    p = ctx.resolve(target)
    if not p.exists():
        return ToolResult(f"No such file or folder: {p}", True)
    try:
        open_with_default(str(p))
    except OSError as e:
        return ToolResult(f"Could not open {p}: {e}", True)
    return f"Opened {p} with its default application."


# ---- optional desktop control (pip install "saani[desktop]") ----------------
def _gui():
    import pyautogui  # type: ignore
    pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
    return pyautogui


MAX_SHOT_WIDTH = 1568
_scale = {"x": 1.0, "y": 1.0}   # screenshot pixels -> the coordinates the mouse uses


def _screenshot(a: dict, ctx: ToolContext):
    g = _gui()
    img = g.screenshot()
    try:
        screen_w, screen_h = g.size()
    except Exception:
        screen_w, screen_h = img.width, img.height
    # High-density displays capture more pixels than the mouse has positions, and a full-size
    # capture is needlessly large for the model. Shrink it and remember the conversion.
    if img.width > MAX_SHOT_WIDTH:
        img = img.resize((MAX_SHOT_WIDTH, max(1, round(img.height * MAX_SHOT_WIDTH / img.width))))
    _scale["x"], _scale["y"] = screen_w / img.width, screen_h / img.height
    path = Path(ctx.workdir) / f"screen-{time.strftime('%H%M%S')}.png"
    img.save(str(path))
    return ToolResult(f"Screenshot saved to {path} (attached), {img.width}x{img.height} pixels. "
                      "Give mouse_click coordinates as pixel positions in this image.", images=[str(path)])


def _mouse_click(a: dict, ctx: ToolContext):
    x, y = round(int(a["x"]) * _scale["x"]), round(int(a["y"]) * _scale["y"])
    _gui().click(x, y, button=a.get("button") or "left", clicks=int(a.get("clicks") or 1))
    return f"Clicked at ({a['x']}, {a['y']})."


def _type_text(a: dict, ctx: ToolContext):
    vault = ctx.services.vault
    missing = []

    def rep(m):
        v = vault.resolve(m.group(1))
        if v is None:
            missing.append(m.group(1))
        return v or ""
    text = re.sub(r"\{\{secret:([A-Za-z_][A-Za-z0-9_]*)\}\}", rep, a["text"])
    if missing:
        return ToolResult(f"Unknown secret(s): {', '.join(missing)}", True)
    _gui().write(text, interval=0.01)
    return f"Typed {len(text)} characters."


def _press_keys(a: dict, ctx: ToolContext):
    keys = [k.strip() for k in a["keys"].split("+") if k.strip()]
    g = _gui()
    g.hotkey(*keys) if len(keys) > 1 else g.press(keys[0])
    return f"Pressed {a['keys']}."


TOOLS = [
    Tool("system_info", "Describe this computer: OS, user, folders, shell, time.", obj({}), _system_info, Risk.SAFE),
    Tool("open_target", "Open a URL in the browser, or a file/folder/app with its default program.",
         obj({"target": S}, ["target"]), _open_target, Risk.CAUTION),
]

DESKTOP_TOOLS = [
    # what is on screen may be a web page, so a screenshot counts as reading untrusted content
    Tool("screenshot", "Take a screenshot of the screen so you can see it.", obj({}), _screenshot, Risk.CAUTION, taints=True),
    Tool("mouse_click", "Click the mouse at a position given in pixels of the latest screenshot.",
         obj({"x": I, "y": I, "button": S, "clicks": I}, ["x", "y"]), _mouse_click, Risk.CAUTION),
    Tool("type_text", "Type text into the active window (plain keyboard characters). Use {{secret:NAME}} for stored secrets.",
         obj({"text": S}, ["text"]), _type_text, Risk.CAUTION),
    Tool("press_keys", "Press a key or a combination such as 'ctrl+s' or 'enter'.",
         obj({"keys": S}, ["keys"]), _press_keys, Risk.CAUTION),
]


def desktop_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("pyautogui") is not None
    except Exception:
        return False

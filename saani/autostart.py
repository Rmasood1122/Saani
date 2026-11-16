"""Start Saani automatically when you log in (per-user, no administrator rights).

* Windows : HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
* macOS   : ~/Library/LaunchAgents/com.saani.daemon.plist
* Linux   : ~/.config/autostart/saani.desktop  (XDG autostart)
"""
from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
from pathlib import Path, PureWindowsPath

LABEL = "com.saani.daemon"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE = "Saani"


def launch_args(home: Path, platform: str | None = None, executable: str | None = None) -> list[str]:
    platform = platform or sys.platform
    exe = executable or sys.executable
    if platform.startswith("win"):
        w = PureWindowsPath(exe)
        if w.name.lower() == "python.exe":      # pythonw.exe runs without a console window
            cand = str(w.with_name("pythonw.exe"))
            if executable is not None or os.path.exists(cand):
                exe = cand
    return [exe, "-m", "saani", "--home", str(home), "daemon"]


def _win_cmdline(args: list[str]) -> str:
    return subprocess.list2cmdline(args)


def linux_desktop_file(args: list[str]) -> str:
    return ("[Desktop Entry]\nType=Application\nName=Saani\nComment=Personal multi-agent AI assistant\n"
            f"Exec={' '.join(shlex.quote(a) for a in args)}\nTerminal=false\n"
            "X-GNOME-Autostart-enabled=true\nX-GNOME-Autostart-Delay=5\n")


def mac_plist(args: list[str], logs: Path) -> bytes:
    return plistlib.dumps({
        "Label": LABEL, "ProgramArguments": args, "RunAtLoad": True, "KeepAlive": False,
        "ProcessType": "Interactive",
        "StandardOutPath": str(logs / "daemon.out.log"), "StandardErrorPath": str(logs / "daemon.err.log"),
    })


def _linux_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "autostart" / "saani.desktop"


def _mac_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def install(config, platform: str | None = None) -> str:
    platform = platform or sys.platform
    args = launch_args(config.home, platform)
    if platform.startswith("win"):
        import winreg  # type: ignore
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, VALUE, 0, winreg.REG_SZ, _win_cmdline(args))
        return f"Registered in HKCU\\{RUN_KEY} as '{VALUE}'. Saani will start at your next login."
    if platform == "darwin":
        p = _mac_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(mac_plist(args, config.logs))
        uid = os.getuid() if hasattr(os, "getuid") else 0
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(p)], capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(p)], capture_output=True)
        if r.returncode != 0:
            subprocess.run(["launchctl", "load", "-w", str(p)], capture_output=True)
        return f"Installed LaunchAgent {p}. Saani starts at login (and has been started now)."
    p = _linux_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(linux_desktop_file(args), encoding="utf-8")
    return f"Installed {p}. Saani will start at your next desktop login."


def remove(config, platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform.startswith("win"):
        import winreg  # type: ignore
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, VALUE)
        except FileNotFoundError:
            return "Autostart was not installed."
        return "Autostart removed."
    p = _mac_path() if platform == "darwin" else _linux_path()
    if not p.exists():
        return "Autostart was not installed."
    if platform == "darwin":
        uid = os.getuid() if hasattr(os, "getuid") else 0
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(p)], capture_output=True)
    p.unlink()
    return f"Autostart removed ({p})."


def status(config, platform: str | None = None) -> tuple[bool, str]:
    platform = platform or sys.platform
    if platform.startswith("win"):
        import winreg  # type: ignore
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                val, _ = winreg.QueryValueEx(k, VALUE)
            return True, f"enabled: {val}"
        except FileNotFoundError:
            return False, "not installed"
    p = _mac_path() if platform == "darwin" else _linux_path()
    return (True, f"enabled: {p}") if p.exists() else (False, "not installed")

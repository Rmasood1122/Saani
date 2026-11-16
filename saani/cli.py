"""Command line interface: `saani <command>`."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__
from .config import Config
from .interaction import TerminalInteraction
from .llm import LLMError
from .textutil import clip


# ------------------------------------------------------------------ console
class Console:
    """Prints agent activity; holds lines back while a permission prompt is open."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._backlog: list[str] = []

    def line(self, text: str) -> None:
        lock = TerminalInteraction._lock
        if lock.acquire(blocking=False):
            try:
                for old in self._backlog:
                    print(old, flush=True)
                self._backlog.clear()
                print(text, flush=True)
            finally:
                lock.release()
        else:
            self._backlog.append(text)

    def on_event(self, ev: dict) -> None:
        k = ev["kind"]
        if k == "plan":
            how = {"skill": " (straight from a saved skill)", "fallback": " (fallback plan)"}.get(ev.get("source"), "")
            reuse = f"  [adapting skill '{ev['skill']}']" if ev.get("skill") and ev.get("source") == "planner" else ""
            self.line(f"[plan] {len(ev['agents'])} agent(s): {ev['shape']}{how}{reuse}")
            for a in ev["agents"]:
                self.line(f"         - {a['name']}: {a['role']}")
            for n in ev.get("notes") or []:
                self.line(f"         ! {n}")
        elif k == "agent_started" and self.verbose:
            self.line(f"[{ev['agent']}] started")
        elif k == "tool_call" and self.verbose:
            self.line(f"[{ev['agent']}] {ev['tool']}  {ev.get('summary', '')}" + ("" if ev["allowed"] else "   << not allowed"))
        elif k == "agent_finished":
            self.line(f"[{ev['agent']}] {ev['status']} after {ev['steps']} step(s)" + (f" - {ev['error']}" if ev.get("error") else ""))
        elif k == "agent_retry":
            self.line(f"[{ev['agent']}] retrying after: {ev['error']}")
        elif k == "verify_failed":
            self.line("[review] the verifier found problems; a fixer agent is working on them")
        elif k == "skill_saved":
            self.line(f"[learned] saved reusable skill '{ev['skill']}'")
        elif k == "notify":
            self.line(f"[note] {ev.get('text', '')}")


def _services(cfg: Config, *, client=None):
    from .app import build
    try:
        return build(cfg, TerminalInteraction(), client)
    except LLMError as e:
        raise SystemExit(f"{e}")


def _run_interruptible(orch, console: Console, text: str, **kw):
    """Run a task on a worker thread so Ctrl+C can cancel it cleanly."""
    box: dict = {}
    tid = orch.new_task_id()

    def work():
        box["res"] = orch.run_task(text, task_id=tid, **kw)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.2)
    except KeyboardInterrupt:
        print("\nStopping (waiting for the current step to finish)...", flush=True)
        orch.cancel(tid)
        t.join(30)
    console.line("")
    return box.get("res")


def _print_result(res) -> int:
    if res is None:
        print("The task was interrupted.")
        return 130
    s = res.stats
    print("=" * 72)
    print(res.result.strip() or "(no result)")
    print("=" * 72)
    extra = "".join([f" | reused skill: {s['skill_used']}" if s.get("skill_used") else "",
                     f" | learned skill: {s['skill_saved']}" if s.get("skill_saved") else "",
                     f" | review: {s['verified']}" if s.get("verified") else ""])
    print(f"status: {res.status} | task {res.task_id} | {s.get('agents', 0)} agent(s) | {s.get('llm_calls', 0)} model calls | "
          f"{s.get('seconds', 0)}s{extra}")
    print(f"folder: {res.workdir}")
    return 0 if res.status == "done" else 1


# ------------------------------------------------------------------ commands
def cmd_run(cfg, a):
    from .core.orchestrator import Orchestrator
    svc = _services(cfg)
    console = Console(not a.quiet)
    svc.events.subscribe(console.on_event)
    return _print_result(_run_interruptible(Orchestrator(svc), console, " ".join(a.task), workdir=a.dir, skill=a.skill))


def cmd_chat(cfg, a):
    from .core.orchestrator import Orchestrator
    svc = _services(cfg)
    console = Console(not a.quiet)
    svc.events.subscribe(console.on_event)
    orch = Orchestrator(svc)
    name = cfg.get("name", "Saani")
    print(f"{name} is ready. Tell me what to do (empty line or 'exit' to leave).")
    history: list[str] = []
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in ("", "exit", "quit", "bye"):
            return 0
        res = _run_interruptible(orch, console, text, workdir=a.dir, source="chat", history="\n\n".join(history[-3:]))
        if res is None:
            continue
        print(f"\n{name}> {res.result.strip()}\n   ({res.status}; task {res.task_id}; {res.stats.get('agents', 0)} agent(s))")
        history.append(f"Owner: {clip(text, 500)}\n{name}: {clip(res.result, 1200)}")


def cmd_demo(cfg, a):
    from .core.orchestrator import Orchestrator
    from .simllm import DEMO_TASK, demo_client
    print("DEMO MODE - this replays a scripted 'model' so you can watch the machinery work offline:\n"
          "planning, agents created on the fly, parallel work, a permission prompt, and a skill being saved.\n")
    svc = _services(cfg, client=demo_client())
    console = Console(True)
    svc.events.subscribe(console.on_event)
    print(f"task> {DEMO_TASK}\n")
    return _print_result(_run_interruptible(Orchestrator(svc), console, DEMO_TASK, source="demo"))


def cmd_init(cfg, a):
    from . import autostart
    from .persona import Persona
    from .security.secrets import Vault
    print(f"Setting up Saani in {cfg.home}\n")
    cfg.set("user_name", input(f"Your name [{cfg.get('user_name') or getpass.getuser()}]: ").strip()
            or cfg.get("user_name") or getpass.getuser())
    print("\nWhich model provider?\n  1) Anthropic (Claude)   2) OpenAI   3) Other OpenAI-compatible (Ollama, OpenRouter, Groq, ...)   4) Demo only")
    choice = input("Choose [1]: ").strip() or "1"
    vault = Vault(cfg.home)
    if choice == "1":
        cfg.set("llm.provider", "anthropic"); cfg.set("llm.base_url", ""); cfg.set("llm.api_key_env", "ANTHROPIC_API_KEY")
        cfg.set("llm.models", {"planner": "claude-opus-5-5", "worker": "claude-sonnet-5-5", "fast": "claude-haiku-5-5"})
    elif choice == "2":
        cfg.set("llm.provider", "openai"); cfg.set("llm.base_url", ""); cfg.set("llm.api_key_env", "OPENAI_API_KEY")
        m = input("Model name for all tiers (see your provider's model list): ").strip()
        cfg.set("llm.models", {"planner": m, "worker": m, "fast": m})
    elif choice == "3":
        cfg.set("llm.provider", "openai")
        cfg.set("llm.base_url", input("Base URL [http://localhost:11434/v1]: ").strip() or "http://localhost:11434/v1")
        cfg.set("llm.api_key_env", "SAANI_LLM_API_KEY")
        m = input("Model name (it must support tool calling): ").strip()
        cfg.set("llm.models", {"planner": m, "worker": m, "fast": m})
    else:
        cfg.set("llm.provider", "demo")
    env = cfg.get("llm.api_key_env")
    if choice in ("1", "2", "3") and not os.environ.get(env):
        key = getpass.getpass(f"API key (stored in your {vault.backend} vault, hidden while typing; Enter to skip): ").strip()
        if key:
            vault.set(env, key)
    cfg.save()
    persona = Persona(cfg.persona_path, cfg.get("user_name"))
    if (input("\nAnswer 7 short questions so agents think like you? [Y/n]: ").strip().lower() or "y") == "y":
        persona.interview(lambda q: input(f"\n{q}\n> "))
        print(f"Saved to {cfg.persona_path} (edit it any time).")
    else:
        persona.ensure()
    if (input("\nStart Saani automatically when you log in? [Y/n]: ").strip().lower() or "y") == "y":
        try:
            print(autostart.install(cfg))
        except Exception as e:
            print(f"Could not install autostart: {e}")
    print("\nDone. Try:  saani doctor   |   saani demo   |   saani run \"...\"   |   saani daemon")
    return 0


def cmd_daemon(cfg, a):
    from .daemon import Daemon, call_daemon, dashboard_url
    if call_daemon(cfg, "/api/health", timeout=2):
        print("Saani is already running.  Dashboard:", dashboard_url(cfg))
        return 0
    try:
        d = Daemon(cfg, enable_voice=not a.no_voice)
    except LLMError as e:
        raise SystemExit(str(e))
    print(f"{cfg.get('name', 'Saani')} is running.\nDashboard: {dashboard_url(cfg)}\n(Ctrl+C to stop)", flush=True)
    if cfg.get("daemon.open_dashboard_on_start"):
        threading.Timer(1.0, lambda: webbrowser.open(dashboard_url(cfg))).start()
    try:
        d.serve()
    except OSError as e:
        raise SystemExit(f"Could not start the service on port {cfg.get('daemon.port')}: {e}")
    return 0


def cmd_open(cfg, a):
    from .daemon import call_daemon, dashboard_url
    if not call_daemon(cfg, "/api/health", timeout=2):
        print("Saani is not running. Start it with:  saani daemon")
        return 1
    webbrowser.open(dashboard_url(cfg))
    print("Opened the dashboard in your browser.")
    return 0


def cmd_status(cfg, a):
    from .daemon import call_daemon
    s = call_daemon(cfg, "/api/state?since=999999999")
    if not s:
        print("not running")
        return 1
    print(f"running | up {s['uptime']}s | voice: {s['voice']} | {len(s['pending'])} waiting for you | "
          f"{sum(1 for t in s['tasks'] if t['status'] == 'running')} task(s) running")
    return 0


def cmd_stop(cfg, a):
    from .daemon import call_daemon
    print("stopped" if call_daemon(cfg, "/api/shutdown", {}) else "not running")
    return 0


def cmd_submit(cfg, a):
    from .daemon import call_daemon
    r = call_daemon(cfg, "/api/task", {"text": " ".join(a.task), "dir": a.dir or "", "source": "cli"})
    if not r:
        print("Saani is not running. Start it with `saani daemon`, or use `saani run`.")
        return 1
    print(f"queued as task {r['task_id']} - watch it with `saani open`")
    return 0


def cmd_listen(cfg, a):
    from . import voice as V
    missing = V.missing_voice_deps(cfg)
    if missing:
        raise SystemExit('Voice needs extra packages:  pip install "saani[voice]"   (missing: ' + ", ".join(missing) + ")")
    stt = V.make_stt(cfg)
    seg, stop = V.Segmenter(), threading.Event()
    print("Calibrating for a second, then speak...", flush=True)
    for frame in V.mic_frames(stop):
        pcm = seg.feed(frame)
        if pcm:
            stop.set()
            text = stt.transcribe(pcm)
            print(f"heard: {text!r}")
            heard, cmd = V.extract_command(text, cfg.get("voice.wake_words", []))
            print(f"wake word: {'yes' if heard else 'no'}   command: {cmd!r}")
            return 0
    return 1


def cmd_autostart(cfg, a):
    from . import autostart
    if a.action == "install":
        print(autostart.install(cfg))
    elif a.action == "remove":
        print(autostart.remove(cfg))
    else:
        print(autostart.status(cfg)[1])
    return 0


def cmd_persona(cfg, a):
    from .persona import Persona
    p = Persona(cfg.persona_path, cfg.get("user_name"))
    if a.action == "init":
        p.interview(lambda q: input(f"\n{q}\n> "))
        print(f"Saved to {cfg.persona_path}")
    elif a.action == "add":
        p.add_preference(" ".join(a.text))
        print("Added to learned preferences.")
    elif a.action == "path":
        print(cfg.persona_path)
    else:
        p.ensure()
        print(p.text(100000))
    return 0


def cmd_feedback(cfg, a):
    from .memory import Store
    from .persona import Persona
    text = " ".join(a.text)
    Persona(cfg.persona_path, cfg.get("user_name")).add_preference(text)
    Store(cfg.db_path).add_memory("preference", text, {"source": "feedback"})
    print("Noted. Agents will follow this from now on.")
    return 0


def cmd_skills(cfg, a):
    from .skills import SkillLibrary
    lib = SkillLibrary(cfg.skills_dir)
    if a.action == "show":
        s = lib.get(a.name or "")
        print(json.dumps(s.__dict__, indent=2, ensure_ascii=False) if s else "no such skill")
    elif a.action == "delete":
        print("deleted" if lib.delete(a.name or "") else "no such skill")
    else:
        skills = lib.all()
        print("\n".join(f"- {s.card()}" for s in skills) if skills else "No skills yet. Saani saves one after each repeatable task.")
    return 0


def cmd_agents(cfg, a):
    from .skills import AgentRegistry
    agents = AgentRegistry(cfg.agents_dir).all()
    print("\n".join(f"- {x['slug']}: {x['role'][:100]}  (tools: {', '.join(x['tools'])}; used {x['uses']}x)" for x in agents)
          if agents else "No agent templates yet. Every agent Saani creates is saved here for reuse.")
    return 0


def cmd_tools(cfg, a):
    from .simllm import SimLLM
    svc = _services(cfg, client=SimLLM())
    for n in svc.tools.names():
        t = svc.tools.get(n)
        print(f"- {n:<16} [{t.risk.name:<7}] {t.description.splitlines()[0][:110]}")
    return 0


def cmd_memory(cfg, a):
    from .memory import Store
    store = Store(cfg.db_path)
    if a.action == "add":
        store.add_memory("fact", " ".join(a.text))
        print("saved")
    else:
        rows = store.search(" ".join(a.text), limit=8)
        print("\n".join(f"- [{r['kind']}] {clip(r['text'], 300)}" for r in rows) if rows else "nothing found")
    return 0


def cmd_schedule(cfg, a):
    from .memory import Store
    from .scheduler import next_run
    store = Store(cfg.db_path)
    if a.action == "add":
        kind, spec = ("every", str(a.every)) if a.every else ("daily", a.daily) if a.daily else ("weekdays", a.weekdays)
        if not spec:
            raise SystemExit("choose one of --every MINUTES, --daily HH:MM, --weekdays HH:MM")
        try:
            nxt = next_run(kind, spec, time.time())
        except ValueError as e:
            raise SystemExit(str(e))
        sid = store.schedule_add(a.name, " ".join(a.task), kind, spec, nxt)
        print(f"schedule #{sid} added; first run {time.strftime('%Y-%m-%d %H:%M', time.localtime(nxt))}. "
              "It runs while the background service is running. Unattended runs can only do what does not need "
              "approval, unless you approve on the dashboard in time.")
    elif a.action == "remove":
        print("removed" if store.schedule_remove(int(a.name)) else "no such schedule")
    else:
        rows = store.schedule_list()
        print("\n".join(f"#{s['id']} {s['name']}: {s['kind']} {s['spec']} -> {s['task'][:80]}  "
                        f"(next {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['next_run']))})" for s in rows) or "No schedules.")
    return 0


def cmd_secret(cfg, a):
    from .security.secrets import Vault
    v = Vault(cfg.home)
    if a.action == "set":
        value = getpass.getpass(f"Value for {a.name} (hidden): ")
        try:
            v.set(a.name, value)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"stored in the {v.backend} vault")
    elif a.action == "delete":
        v.delete(a.name)
        print("deleted")
    else:
        print("\n".join(v.names()) or "(no secrets stored)")
        print(f"[vault backend: {v.backend}]")
    return 0


def cmd_audit(cfg, a):
    from .security.audit import AuditLog
    log = AuditLog(cfg.audit_path)
    if a.action == "verify":
        ok, msg = log.verify()
        print(("OK: " if ok else "FAILED: ") + msg)
        return 0 if ok else 1
    for e in log.entries(a.n):
        print(f"{time.strftime('%m-%d %H:%M:%S', time.localtime(e.get('ts', 0)))} {e.get('decision', '?'):<5} "
              f"{e.get('risk', ''):<8} {e.get('agent', '')}: {e.get('tool', '')} {(str(e.get('action', '')).splitlines() or [''])[0][:90]}"
              f"  (by {e.get('by', '')})")
    return 0


def cmd_trash(cfg, a):
    """Everything agents deleted or overwrote with their file tools is kept here."""
    import shutil
    index = cfg.trash / "index.jsonl"
    rows = []
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    if a.action == "restore":
        if not a.number or not 1 <= a.number <= len(rows):
            print("Give the number shown by `saani trash`.")
            return 1
        r = rows[a.number - 1]
        stored, original = Path(r["stored"]), Path(r["original"])
        if not stored.exists():
            print(f"The kept copy is gone: {stored}")
            return 1
        if original.exists() and not a.force:
            print(f"{original} exists now. Re-run with --force to replace it with the kept copy.")
            return 1
        original.parent.mkdir(parents=True, exist_ok=True)
        if stored.is_dir():
            shutil.copytree(stored, original, dirs_exist_ok=True)
        else:
            shutil.copy2(stored, original)
        print(f"Restored {original}")
        return 0
    if not rows:
        print(f"Nothing in the trash ({cfg.trash}).")
        return 0
    for i, r in enumerate(rows, 1):
        if i <= len(rows) - a.n:
            continue
        gone = "" if Path(r["stored"]).exists() else "  [copy missing]"
        print(f"{i:>4}  {time.strftime('%m-%d %H:%M', time.localtime(r.get('ts', 0)))}  {r.get('kind', '?'):<8} "
              f"{r.get('original', '')}  (task {r.get('task', '')}, by {r.get('agent', '')}){gone}")
    print(f"\nRestore with:  saani trash restore <number>      Kept copies live in {cfg.trash}")
    return 0


def cmd_trust(cfg, a):
    dirs = list(cfg.get("security.trusted_dirs", []) or [])
    if a.action in ("add", "remove"):
        if not a.folder:
            print("Give a folder.")
            return 1
        folder = str(Path(a.folder).expanduser().resolve())
        if a.action == "add":
            if not Path(folder).is_dir():
                print(f"No such folder: {folder}")
                return 1
            if Path(folder) in {Path.home().resolve(), Path(Path(folder).anchor)}:
                print("Trust project folders, not your whole home folder or a drive root.")
                return 1
            if folder not in dirs:
                dirs.append(folder)
        elif folder in dirs:
            dirs.remove(folder)
        else:
            print("That folder was not trusted.")
            return 1
        cfg.set("security.trusted_dirs", dirs)
        cfg.save()
        print("Saved. Restart the background service (`saani stop`, then `saani daemon`) for it to take effect there.")
    print("Agents may create and edit files without asking in:")
    print(f"- {cfg.workspace}  (Saani's own workspace)")
    for d in dirs:
        print(f"- {d}")
    print("Secrets, startup files and deletions outside these folders still ask every time.")
    return 0


def cmd_rules(cfg, a):
    rules = cfg.get("security.allow_rules", [])
    if a.action == "remove":
        if a.key in rules:
            rules.remove(a.key)
            cfg.set("security.allow_rules", rules)
            cfg.save()
            print("removed")
        else:
            print("no such rule")
    else:
        print(f"permission mode: {cfg.get('security.mode')}")
        print("\n".join(f"- {r}" for r in rules) or "(no 'always allow' rules saved)")
    return 0


def cmd_check(cfg, a):
    from .security.policy import Policy
    v = Policy(cfg.home, cfg.workspace, cfg.get("security.trusted_dirs", []), cfg.get("daemon.port", 8765)).classify_shell(" ".join(a.command))
    print(f"{v.risk.name}: {v.reason}" + (f"   (rule: {v.rule_key}{', can be always-allowed' if v.always_ok else ''})" if v.rule_key else ""))
    return 0


def cmd_history(cfg, a):
    from .memory import Store
    store = Store(cfg.db_path)
    if a.task_id:
        t = store.task_get(a.task_id)
        if not t:
            print("no such task")
            return 1
        print(f"task {t['id']} [{t['status']}] via {t['source']}\n{t['text']}\n\n{t['result']}\n\nfolder: {t['workdir']}\nstats: {t['stats']}")
        return 0
    for t in store.task_list(a.n):
        print(f"{t['id']}  {time.strftime('%m-%d %H:%M', time.localtime(t['created']))}  {t['status']:<11} {' '.join(t['text'].split())[:90]}")
    return 0


def cmd_config(cfg, a):
    if a.action == "set":
        try:
            value = json.loads(a.value)
        except ValueError:
            value = a.value
        if a.key == "security.mode" and value not in ("strict", "ask", "trusted"):
            raise SystemExit("security.mode must be strict, ask or trusted")
        cfg.set(a.key, value)
        cfg.save()
        print(f"{a.key} = {json.dumps(value)}")
    elif a.key:
        print(json.dumps(cfg.get(a.key), indent=2))
    else:
        print(json.dumps(cfg.data, indent=2))
    return 0


def cmd_doctor(cfg, a):
    from .doctor import run_checks
    marks = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
    bad = 0
    for status, name, detail in run_checks(cfg):
        print(f"{marks[status]} {name:<20} {detail}")
        bad += status == "fail"
    return 1 if bad else 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saani", description="Saani - your second self: a personal multi-agent AI system.")
    p.add_argument("--home", help="data folder (default ~/.saani or $SAANI_HOME)")
    p.add_argument("--version", action="version", version=f"saani {__version__}")
    sub = p.add_subparsers(dest="cmd")

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    add("init", cmd_init, "first-time setup: model, API key, persona, autostart")
    for name, fn, h in (("run", cmd_run, "run one task now, in this terminal"),):
        sp = add(name, fn, h)
        sp.add_argument("task", nargs="+")
        sp.add_argument("--dir", help="project folder to work in (writes there need no approval)")
        sp.add_argument("--skill", help="run with this saved skill, skipping planning")
        sp.add_argument("-q", "--quiet", action="store_true")
    sp = add("chat", cmd_chat, "interactive session")
    sp.add_argument("--dir"); sp.add_argument("-q", "--quiet", action="store_true")
    add("demo", cmd_demo, "offline demonstration (no API key needed)")
    sp = add("daemon", cmd_daemon, "run the background service (voice, dashboard, schedules)")
    sp.add_argument("--no-voice", action="store_true")
    add("open", cmd_open, "open the dashboard")
    add("status", cmd_status, "is the background service running?")
    add("stop", cmd_stop, "stop the background service")
    sp = add("submit", cmd_submit, "hand a task to the background service")
    sp.add_argument("task", nargs="+"); sp.add_argument("--dir")
    add("listen", cmd_listen, "test the microphone and speech recognition once")
    sp = add("autostart", cmd_autostart, "start Saani when you log in")
    sp.add_argument("action", choices=["install", "remove", "status"])
    sp = add("persona", cmd_persona, "how you think and work (what makes agents act like you)")
    sp.add_argument("action", nargs="?", default="show", choices=["show", "init", "add", "path"]); sp.add_argument("text", nargs="*")
    sp = add("feedback", cmd_feedback, "teach Saani a preference, e.g. saani feedback \"always write tests first\"")
    sp.add_argument("text", nargs="+")
    sp = add("skills", cmd_skills, "reusable skills learned from past tasks")
    sp.add_argument("action", nargs="?", default="list", choices=["list", "show", "delete"]); sp.add_argument("name", nargs="?")
    add("agents", cmd_agents, "agent templates created so far")
    add("tools", cmd_tools, "tools agents can use, with risk level")
    sp = add("memory", cmd_memory, "search or add long-term memory")
    sp.add_argument("action", choices=["search", "add"]); sp.add_argument("text", nargs="+")
    sp = add("schedule", cmd_schedule, "recurring tasks")
    sp.add_argument("action", choices=["add", "list", "remove"]); sp.add_argument("name", nargs="?"); sp.add_argument("task", nargs="*")
    sp.add_argument("--every", type=int, help="minutes"); sp.add_argument("--daily", help="HH:MM"); sp.add_argument("--weekdays", help="HH:MM")
    sp = add("secret", cmd_secret, "passwords and keys in the vault (agents only ever see names)")
    sp.add_argument("action", choices=["set", "list", "delete"]); sp.add_argument("name", nargs="?")
    sp = add("audit", cmd_audit, "the tamper-evident log of everything agents tried")
    sp.add_argument("action", choices=["verify", "tail"]); sp.add_argument("-n", type=int, default=30)
    sp = add("trash", cmd_trash, "files agents deleted or overwrote (kept so you can restore them)")
    sp.add_argument("action", nargs="?", default="list", choices=["list", "restore"]); sp.add_argument("number", nargs="?", type=int)
    sp.add_argument("-n", type=int, default=30); sp.add_argument("--force", action="store_true")
    sp = add("trust", cmd_trust, "project folders where agents may write without asking")
    sp.add_argument("action", nargs="?", default="list", choices=["list", "add", "remove"]); sp.add_argument("folder", nargs="?")
    sp = add("rules", cmd_rules, "your saved 'always allow' rules")
    sp.add_argument("action", nargs="?", default="list", choices=["list", "remove"]); sp.add_argument("key", nargs="?")
    sp = add("check", cmd_check, "see how the safety policy rates a shell command")
    sp.add_argument("command", nargs=argparse.REMAINDER)
    sp = add("history", cmd_history, "past tasks (give an id to see one)")
    sp.add_argument("task_id", nargs="?"); sp.add_argument("-n", type=int, default=20)
    sp = add("config", cmd_config, "show or change settings")
    sp.add_argument("action", nargs="?", default="get", choices=["get", "set"]); sp.add_argument("key", nargs="?"); sp.add_argument("value", nargs="?")
    add("doctor", cmd_doctor, "check the installation")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.home:
        os.environ["SAANI_HOME"] = str(Path(a.home).expanduser())
    if not getattr(a, "fn", None):
        parser.print_help()
        return 0
    cfg = Config.load()
    try:
        return int(a.fn(cfg, a) or 0)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

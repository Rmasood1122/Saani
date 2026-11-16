import array
import io
import json
import math
import os
import plistlib
import sys
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
import wave
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

from saani import autostart, voice
from saani.config import Config
from saani.daemon import Daemon
from saani.memory import Store
from saani.persona import Persona
from saani.scheduler import Scheduler, next_run
from saani.simllm import SimLLM, demo_client, DEMO_TASK
from saani.skills import AgentRegistry, Skill, SkillLibrary
from tests.helpers import SaaniCase, agent, free_port, plan


def tone(ms: int, amp: int = 8000) -> list[bytes]:
    n = voice.FRAME_BYTES // 2
    frame = array.array("h", [int(amp * math.sin(i / 3)) for i in range(n)]).tobytes()
    return [frame] * (ms // voice.FRAME_MS)


def quiet(ms: int) -> list[bytes]:
    return tone(ms, 20)


class VoiceTests(unittest.TestCase):
    def test_rms(self):
        self.assertEqual(voice.rms(b""), 0.0)
        self.assertLess(voice.rms(quiet(30)[0]), 30)
        self.assertGreater(voice.rms(tone(30)[0]), 3000)

    def test_segmenter_finds_utterances(self):
        seg = voice.Segmenter()
        out = [seg.feed(f) for f in quiet(1200) + tone(900) + quiet(1200) + tone(600) + quiet(1200)]
        utterances = [o for o in out if o]
        self.assertEqual(len(utterances), 2)
        self.assertGreater(len(utterances[0]), len(utterances[1]))
        self.assertGreater(seg.threshold, 300)

    def test_segmenter_ignores_clicks_and_caps_length(self):
        seg = voice.Segmenter()
        self.assertEqual([o for o in (seg.feed(f) for f in quiet(1200) + tone(60) + quiet(1500)) if o], [])
        seg = voice.Segmenter(max_s=1.0)
        got = [o for o in (seg.feed(f) for f in quiet(1200) + tone(5000)) if o]
        self.assertGreaterEqual(len(got), 3)                       # endless speech is cut into chunks, not buffered forever

    def test_wake_word(self):
        ww = ["saani", "sani", "sunny"]
        cases = {"Saani, fix the failing tests": "fix the failing tests", "hey sani open my project": "open my project",
                 "Okay, Sunny. What's new in AI?": "What's new in AI?", "SAANI": "", "saani!": ""}
        for said, cmd in cases.items():
            self.assertEqual(voice.extract_command(said, ww), (True, cmd), said)
        for said in ("please fix the tests", "I told saani yesterday", "sanitize the input", "", "sunnyvale weather"):
            self.assertEqual(voice.extract_command(said, ww), (False, ""), said)

    def test_wake_gate_two_step(self):
        g = voice.WakeGate(["saani"], arm_seconds=8)
        self.assertIsNone(g.hear("random chatter", now=0))
        self.assertIsNone(g.hear("Saani.", now=10))
        self.assertTrue(g.just_armed)
        self.assertEqual(g.hear("check my email", now=14), "check my email")
        self.assertIsNone(g.hear("and another thing", now=15))        # the gate closed again
        g.hear("saani", now=100)
        self.assertIsNone(g.hear("too late", now=120))

    def test_voice_loop_end_to_end_with_fakes(self):
        class FakeSTT(voice.STT):
            texts = ["the tv is talking", "Saani, summarise my inbox", "Saani", "book a meeting"]

            def transcribe(self, pcm):
                return self.texts.pop(0)

        class FakeTTS(voice.TTS):
            said = []

            def speak(self, text):
                self.said.append(text)
        commands = []
        frames = quiet(1200) + (tone(600) + quiet(1100)) * 4
        loop = voice.VoiceLoop(lambda stop: iter(frames), FakeSTT(), FakeTTS(), ["saani"], commands.append)
        loop.run()
        self.assertEqual(commands, ["summarise my inbox", "book a meeting"])
        self.assertEqual(FakeTTS.said, ["Yes?"])

    def test_voice_loop_survives_stt_failure(self):
        class Broken(voice.STT):
            def transcribe(self, pcm):
                raise RuntimeError("model missing")
        errors = []
        loop = voice.VoiceLoop(lambda stop: iter(quiet(1200) + tone(600) + quiet(1100)), Broken(), voice.NullTTS(), ["saani"],
                               lambda c: None, on_error=errors.append)
        loop.run()
        self.assertIn("speech recognition failed", errors[0])

    def test_wav_and_multipart(self):
        data = voice.pcm_to_wav(b"".join(tone(300)))
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))
        body, ctype = voice.multipart({"model": "whisper-1"}, {"file": ("a.wav", b"RIFF", "audio/wav")})
        boundary = ctype.split("boundary=")[1].encode()
        self.assertEqual(body.count(b"--" + boundary), 3)
        self.assertIn(b'name="file"; filename="a.wav"', body)

    def test_tts_factory_never_raises(self):
        self.assertFalse(voice.make_tts("off").available)
        voice.make_tts("auto").speak("")      # whatever this machine has, it must not blow up


class SchedulerTests(SaaniCase):
    def test_next_run(self):
        now = datetime(2026, 11, 16, 8, 30).timestamp()         # a Monday
        self.assertEqual(next_run("every", "45", now), now + 2700)
        self.assertEqual(datetime.fromtimestamp(next_run("daily", "09:00", now)), datetime(2026, 11, 16, 9, 0))
        self.assertEqual(datetime.fromtimestamp(next_run("daily", "08:00", now)), datetime(2026, 11, 17, 8, 0))
        fri = datetime(2026, 11, 20, 18, 0).timestamp()
        self.assertEqual(datetime.fromtimestamp(next_run("weekdays", "09:00", fri)), datetime(2026, 11, 23, 9, 0))
        for kind, spec in (("every", "0"), ("every", "x"), ("daily", "25:00"), ("daily", "9am"), ("hourly", "1")):
            with self.assertRaises(ValueError):
                next_run(kind, spec, now)

    def test_tick_fires_due_jobs_once(self):
        store = Store(self.tmp / "s.db")
        fired = []
        store.schedule_add("news", "research AI news", "every", "60", next_run=100)
        store.schedule_add("later", "x", "every", "60", next_run=10_000)
        s = Scheduler(store, lambda task, name: fired.append((name, task)))
        self.assertEqual(s.tick(now=500), ["news"])
        self.assertEqual(s.tick(now=501), [])                        # not again until the next slot
        self.assertEqual(fired, [("news", "research AI news")])
        self.assertEqual(store.schedule_list()[0]["next_run"], 500 + 3600)
        self.assertTrue(store.schedule_remove(1))
        self.assertFalse(store.schedule_remove(1))

    def test_failing_job_does_not_stop_the_scheduler(self):
        store = Store(self.tmp / "s.db")
        store.schedule_add("bad", "x", "every", "1", next_run=1)
        store.schedule_add("good", "y", "every", "1", next_run=1)
        fired = []

        def submit(task, name):
            if name == "bad":
                raise RuntimeError("boom")
            fired.append(name)
        self.assertEqual(Scheduler(store, submit).tick(now=100), ["good"])


class MemoryTests(SaaniCase):
    def test_search_ranks_relevant_memories(self):
        s = Store(self.tmp / "m.db")
        s.add_memory("fact", "The staging server for client Zeta runs on Render")
        s.add_memory("fact", "Grocery list: mangoes and rice")
        s.add_memory("preference", "Deploys happen on Fridays only after tests pass")
        self.assertIn("Render", s.search("where is the Zeta staging server")[0]["text"])
        self.assertEqual(len(s.search("deploying", kinds=["preference"])), 1)
        self.assertEqual(s.search(""), [])
        self.assertEqual(s.search("the and of"), [])                  # only stop-words
        self.assertEqual(s.search('weird "quotes" AND (syntax*'), []) # FTS operators in user text are harmless

    def test_fallback_search_without_fts(self):
        s = Store(self.tmp / "m.db")
        s.fts = False
        s.add_memory("fact", "Invoices are sent through Wave accounting")
        self.assertIn("Wave", s.search("how do I send an invoice")[0]["text"])

    def test_tasks_and_interruption(self):
        s = Store(self.tmp / "m.db")
        s.task_create("t1", "do x")
        s.task_update("t1", stats={"a": 1}, status="done")
        self.assertEqual(json.loads(s.task_get("t1")["stats"]), {"a": 1})
        s.task_create("t2", "do y")
        self.assertEqual(s.mark_interrupted(), 1)
        self.assertEqual(s.task_get("t2")["status"], "interrupted")
        self.assertEqual([t["id"] for t in s.task_list()], ["t2", "t1"])

    def test_skill_matching(self):
        lib = SkillLibrary(self.tmp / "skills")
        lib.save(Skill(name="Fix Failing Tests", description="Find and fix failing unit tests in a repository",
                       agents=[{"name": "Fixer"}], examples=["Fix the failing tests in my API project"]))
        lib.save(Skill(name="apply-to-jobs", description="Find matching job posts and draft applications",
                       agents=[{"name": "Scout"}], examples=["Find remote AI engineer jobs and draft applications"]))
        top = lib.match("fix failing tests in the billing project")
        self.assertEqual(top[0][0].name, "fix-failing-tests")
        self.assertGreater(top[0][1], 0.5)
        self.assertEqual(lib.match("find AI engineering jobs I can apply to")[0][0].name, "apply-to-jobs")
        self.assertEqual(lib.match("bake a chocolate cake"), [])
        lib.record_use("fix-failing-tests", True, "fix tests again", ["run the suite first"])
        s = lib.get("fix-failing-tests")
        self.assertEqual((s.uses, s.successes, s.lessons), (1, 1, ["run the suite first"]))
        self.assertTrue(lib.delete("fix-failing-tests"))
        self.assertIsNone(lib.get("fix-failing-tests"))

    def test_agent_registry(self):
        reg = AgentRegistry(self.tmp / "agents")
        reg.save("Bug Hunter", "Finds the root cause of bugs in Python services", "trace it", ["read_file", "run_shell"])
        reg.save("Bug Hunter", "Finds the root cause of bugs in Python services", "trace it", ["read_file"])
        self.assertEqual(reg.get("bug-hunter")["uses"], 2)
        self.assertEqual(reg.match("there is a bug in my python service")[0]["slug"], "bug-hunter")
        self.assertEqual(reg.match("write a poem"), [])

    def test_persona(self):
        p = Persona(self.tmp / "persona.md", "Ahmad")
        p.ensure()
        p.add_preference("Reply to clients within one business day")
        p.add_preference("Reply to clients within one business day")      # no duplicates
        self.assertEqual(p.text().count("one business day"), 1)
        answers = iter(["AI engineer", "ship products", "fast", "Python", "direct", "remote senior roles", "never spend money"])
        p.interview(lambda q: next(answers))
        text = p.text()
        self.assertIn("## How I decide\nfast", text)
        self.assertIn("one business day", text)                            # learned preferences survive a re-interview


class AutostartTests(SaaniCase):
    def cfg(self):
        return Config({}, self.home)

    def test_linux_install_status_remove(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.tmp / "xdg")}):
            self.assertEqual(autostart.status(self.cfg(), "linux")[0], False)
            autostart.install(self.cfg(), "linux")
            f = self.tmp / "xdg" / "autostart" / "saani.desktop"
            text = f.read_text()
            self.assertIn("-m saani --home", text)
            self.assertIn(str(self.home), text)
            self.assertIn(sys.executable, text)
            self.assertIn(" daemon", text.split("Exec=")[1].splitlines()[0])
            self.assertTrue(autostart.status(self.cfg(), "linux")[0])
            autostart.remove(self.cfg(), "linux")
            self.assertFalse(f.exists())
            self.assertEqual(autostart.remove(self.cfg(), "linux"), "Autostart was not installed.")

    def test_linux_exec_line_quotes_spaces(self):
        line = autostart.linux_desktop_file(["/usr/bin/python3", "-m", "saani", "--home", "/home/a b/.saani", "daemon"])
        self.assertIn("--home '/home/a b/.saani' daemon", line)

    def test_macos_plist(self):
        args = autostart.launch_args(self.home, "darwin", "/usr/bin/python3")
        d = plistlib.loads(autostart.mac_plist(args, self.home / "logs"))
        self.assertEqual(d["ProgramArguments"], ["/usr/bin/python3", "-m", "saani", "--home", str(self.home), "daemon"])
        self.assertTrue(d["RunAtLoad"])
        self.assertEqual(d["Label"], "com.saani.daemon")

    def test_windows_registry_run_key(self):
        store = {}
        fake = types.SimpleNamespace(
            HKEY_CURRENT_USER=1, KEY_SET_VALUE=2, REG_SZ=1,
            OpenKey=lambda *a, **k: mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False),
            SetValueEx=lambda k, name, _, typ, val: store.__setitem__(name, val),
            DeleteValue=lambda k, name: store.pop(name) if name in store else (_ for _ in ()).throw(FileNotFoundError()),
            QueryValueEx=lambda k, name: (store[name], 1) if name in store else (_ for _ in ()).throw(FileNotFoundError()))
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            self.assertEqual(autostart.status(self.cfg(), "win32"), (False, "not installed"))
            autostart.install(self.cfg(), "win32")
            self.assertIn("-m saani --home", store["Saani"])
            self.assertTrue(store["Saani"].endswith(" daemon"))
            self.assertTrue(autostart.status(self.cfg(), "win32")[0])
            autostart.remove(self.cfg(), "win32")
            self.assertEqual(store, {})
            self.assertEqual(autostart.remove(self.cfg(), "win32"), "Autostart was not installed.")

    def test_windows_uses_pythonw_and_quotes_paths(self):
        args = autostart.launch_args(Path(r"C:\Users\A B\.saani"), "win32", r"C:\Program Files\Python313\python.exe")
        self.assertEqual(args[0], r"C:\Program Files\Python313\pythonw.exe")
        self.assertTrue(autostart._win_cmdline(args).startswith('"C:\\Program Files\\Python313\\pythonw.exe" -m saani --home "'))


class DaemonTests(SaaniCase):
    def start(self, llm, **settings):
        cfg = Config({}, self.home)
        cfg.set("llm.provider", "demo")
        cfg.set("verify", False)
        cfg.set("daemon.port", free_port())
        for k, v in settings.items():
            cfg.set(k.replace("__", "."), v)
        cfg.save()
        self.d = Daemon(cfg, client=llm, enable_voice=False)
        self.d.serve(block=False)
        self.addCleanup(self.d.shutdown)
        self.base = f"http://127.0.0.1:{cfg.get('daemon.port')}"
        return self.d

    def http(self, path, body=None, token=True, headers=None, method=None):
        h = {"X-Requested-With": "saani", **(headers or {})}
        if token:
            h["X-Saani-Token"] = self.d.token if token is True else token
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=h, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=5) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, {}

    def wait(self, pred, timeout=8.0):
        end = time.time() + timeout
        while time.time() < end:
            v = pred()
            if v:
                return v
            time.sleep(0.05)
        self.fail("timed out waiting for the daemon")

    def test_auth_and_origin_checks(self):
        self.start(SimLLM())
        self.assertEqual(self.http("/api/health", token=False)[0], 200)
        self.assertEqual(self.http("/api/state", token=False)[0], 401)
        self.assertEqual(self.http("/api/state", token="wrong-token")[0], 401)
        self.assertEqual(self.http("/api/state")[0], 200)
        self.assertEqual(self.http("/api/state", headers={"Host": "evil.example.com"})[0], 403)      # DNS rebinding
        self.assertEqual(self.http("/api/task", {"text": "x"}, token=False)[0], 401)
        self.assertEqual(self.http("/api/task", {"text": "x"}, headers={"X-Requested-With": ""})[0], 403)   # cross-site post
        self.assertEqual(self.http("/api/task", {"text": "  "})[0], 400)
        self.assertEqual(self.http("/api/task", {"text": "x", "dir": "/no/such/dir"})[0], 400)
        self.assertEqual(self.http("/api/nope", {})[0], 404)
        self.assertEqual(self.http("/", token=False)[0], 401)
        if os.name != "nt":
            self.assertEqual(os.stat(self.home / "daemon.token").st_mode & 0o777, 0o600)

    def test_task_with_approval_through_the_api(self):
        self.start(demo_client())
        code, r = self.http("/api/task", {"text": DEMO_TASK})
        self.assertEqual(code, 200)
        pending = self.wait(lambda: self.http("/api/state")[1]["pending"])
        self.assertEqual((pending[0]["kind"], pending[0]["tool"], pending[0]["risk"]), ("approval", "run_python", "CAUTION"))
        self.assertEqual(self.http("/api/approval", {"id": "bogus", "allow": True})[0], 404)
        self.assertEqual(self.http("/api/approval", {"id": pending[0]["id"], "allow": True, "scope": "once"})[0], 200)
        task = self.wait(lambda: next((t for t in self.http("/api/state")[1]["tasks"] if t["status"] != "running"), None))
        self.assertEqual((task["id"], task["status"]), (r["task_id"], "done"))
        self.assertIn("3 lines, 14 words", task["result"])
        kinds = {e["kind"] for e in self.http("/api/state?since=0")[1]["events"]}
        self.assertTrue({"task_started", "plan", "attention", "tool_call", "task_finished"} <= kinds)
        self.assertTrue((self.home / "logs" / "events.log").exists())

    def test_denied_and_timed_out_approvals(self):
        script = {"Cleaner": [{"tool": "run_shell", "args": {"command": "pip install leftpad"}},
                              lambda m: "result: " + m[-1].content[:40]]}
        self.start(SimLLM(plan(agent("a", "Cleaner", ["run_shell"])), script), security__approval_timeout_s=0.4)
        self.http("/api/task", {"text": "install it"})
        task = self.wait(lambda: next((t for t in self.http("/api/state")[1]["tasks"] if t["status"] != "running"), None))
        self.assertIn("PERMISSION DENIED", task["result"])                       # nobody answered -> not executed
        self.assertEqual(self.d.svc.audit.entries()[-1]["by"], "timeout")

    def test_question_and_secret_via_dashboard(self):
        script = {"Asker": [{"tool": "ask_user", "args": {"question": "Which branch?"}},
                            {"tool": "request_secret", "args": {"name": "DEPLOY_KEY", "reason": "deploy"}},
                            lambda m: "got: " + m[2].content + " / " + m[-1].content]}
        self.start(SimLLM(plan(agent("a", "Asker", ["ask_user", "request_secret"])), script))
        self.http("/api/task", {"text": "ask me"})
        q = self.wait(lambda: self.http("/api/state")[1]["pending"])[0]
        self.assertEqual((q["kind"], q["question"]), ("question", "Which branch?"))
        self.http("/api/answer", {"id": q["id"], "text": "main"})
        s = self.wait(lambda: [p for p in self.http("/api/state")[1]["pending"] if p["kind"] == "secret"])[0]
        self.http("/api/answer", {"id": s["id"], "text": "super-secret-deploy-key"})
        task = self.wait(lambda: next((t for t in self.http("/api/state")[1]["tasks"] if t["status"] != "running"), None))
        self.assertIn("The user answered: main", task["result"])
        self.assertNotIn("super-secret-deploy-key", json.dumps(self.http("/api/state?since=0")[1]))
        self.assertEqual(self.d.svc.vault.get("DEPLOY_KEY"), "super-secret-deploy-key")

    def test_cancel_and_voice_routing(self):
        loop = {"Slow": [{"tool": "list_dir", "args": {"path": "."}}] * 200}
        d = self.start(SimLLM(plan(agent("a", "Slow", ["list_dir"])), loop, delay=0.05), limits__max_steps_per_agent=300)
        tid = self.http("/api/task", {"text": "loop"})[1]["task_id"]
        self.wait(lambda: tid in d.orch.running())
        d.on_voice_command("stop everything")
        task = self.wait(lambda: next((t for t in self.http("/api/state")[1]["tasks"] if t["status"] != "running"), None))
        self.assertEqual(task["status"], "cancelled")
        d.on_voice_command("what is on my calendar")                             # an ordinary request becomes a task
        self.wait(lambda: any(t["source"] == "voice" for t in self.http("/api/state")[1]["tasks"]))

    def test_voice_can_deny_but_not_approve_by_default(self):
        script = {"A": [{"tool": "run_shell", "args": {"command": "pip install x"}}, "end"]}
        d = self.start(SimLLM(plan(agent("a", "A", ["run_shell"])), script))
        self.http("/api/task", {"text": "go"})
        self.wait(lambda: d.inter.pending())
        d.on_voice_command("approve")
        self.assertEqual(len(d.inter.pending()), 1)                              # still waiting: voice approval is off
        d.on_voice_command("deny")
        self.wait(lambda: not d.inter.pending())
        self.assertEqual(d.svc.audit.entries()[-1]["by"], "voice")
        self.assertEqual(d.svc.audit.entries()[-1]["decision"], "deny")

    def test_refuses_non_loopback_bind(self):
        cfg = Config({}, self.home)
        cfg.set("llm.provider", "demo")
        cfg.set("daemon.host", "0.0.0.0")
        with self.assertRaises(SystemExit):
            Daemon(cfg, client=SimLLM(), enable_voice=False).serve(block=False)


class CliTests(SaaniCase):
    def cli(self, *argv, stdin=""):
        from saani.cli import main
        out = io.StringIO()
        with redirect_stdout(out), mock.patch("builtins.input", side_effect=list(stdin.splitlines()) + [EOFError()]):
            try:
                code = main(["--home", str(self.home), *argv])
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def test_check_config_rules(self):
        self.assertIn("BLOCKED", self.cli("check", "rm", "-rf", "/")[1])
        self.assertIn("SAFE", self.cli("check", "git", "status")[1])
        self.assertEqual(self.cli("config", "set", "security.mode", "yolo")[0], "security.mode must be strict, ask or trusted")
        self.cli("config", "set", "security.mode", "strict")
        self.assertIn("strict", self.cli("config", "get", "security.mode")[1])
        self.assertIn("permission mode: strict", self.cli("rules")[1])

    def test_demo_run_then_history_skills_audit(self):
        code, out = self.cli("demo", stdin="y\n")
        self.assertEqual(code, 0, out)
        self.assertIn("[plan] 3 agent(s): Builder | Doc Writer -> Reviewer", out)
        self.assertIn("PERMISSION NEEDED [CAUTION]", out)
        self.assertIn("3 lines, 14 words", out)
        self.assertIn("build-small-python-cli", self.cli("skills")[1])
        self.assertIn("reviewer", self.cli("agents")[1])
        self.assertIn("done", self.cli("history")[1])
        self.assertEqual(self.cli("audit", "verify")[0], 0)
        self.assertIn("run_python", self.cli("audit", "tail")[1])

    def test_demo_denied_reports_unverified(self):
        code, out = self.cli("demo", stdin="n\n")
        self.assertIn("could NOT run it because permission was denied", out)

    def test_feedback_schedule_memory(self):
        self.cli("feedback", "Always", "write", "tests", "first")
        self.assertIn("Always write tests first", self.cli("persona")[1])
        self.assertIn("schedule #1 added", self.cli("schedule", "add", "news", "Research", "AI", "news", "--daily", "09:00")[1])
        self.assertIn("news: daily 09:00", self.cli("schedule", "list")[1])
        self.assertEqual(self.cli("schedule", "add", "bad", "x", "--daily", "99:99")[0], "time must look like 09:00 (24-hour clock)")
        self.cli("memory", "add", "Client", "Zeta", "prefers", "weekly", "updates")
        self.assertIn("Zeta", self.cli("memory", "search", "zeta", "updates")[1])

    def test_run_without_key_explains_itself(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            code, _ = self.cli("run", "hello")
        self.assertIn("No API key", str(code))

    def test_doctor_and_tools(self):
        self.cli("config", "set", "llm.provider", "demo")
        code, out = self.cli("doctor")
        self.assertIn("[ ok ] Audit log", out)
        self.assertIn("run_shell", self.cli("tools")[1])


if __name__ == "__main__":
    unittest.main()

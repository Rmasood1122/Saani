"""A deterministic stand-in for a language model.

Used by the test-suite, the eval harness and ``saani demo`` so that the whole
machine (planning, dynamic agents, permissions, memory, skills) can be exercised
offline, without an API key and with repeatable results. It does not think: it
replays a script. Real intelligence comes from the provider you configure.
"""
from __future__ import annotations

import re
import threading
import time

from .llm import LLMClient, LLMResponse, Msg, ToolCall


class SimLLM(LLMClient):
    name = "sim"

    def __init__(self, plan=None, scripts: dict | None = None, synth: str | None = None,
                 skill: dict | None = None, delay: float = 0.0):
        self.plan = plan
        self.scripts = scripts or {}
        self.synth = synth
        self.skill = skill
        self.delay = delay
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def count(self, kind: str | None = None) -> int:
        with self._lock:
            return sum(1 for c in self.calls if kind is None or c["kind"] == kind)

    def seen_text(self) -> str:
        """Everything that was ever sent to the 'model' - for leak checks."""
        with self._lock:
            return "\n".join(c["system"] + "\n" + "\n".join(m.content for m in c["messages"]) for c in self.calls)

    def complete(self, *, system, messages, tools=None, model="", max_tokens=4096, force_tool=None):
        kind = ("plan" if force_tool == "submit_plan" else "skill" if force_tool == "submit_skill"
                else "synth" if "reporting back to" in system else "agent")
        with self._lock:
            attempt = sum(1 for c in self.calls if c["kind"] == kind)
            self.calls.append({"kind": kind, "system": system, "messages": list(messages), "model": model,
                               "tools": [t["name"] for t in tools or []], "t": time.time()})
        if self.delay and kind == "agent":
            time.sleep(self.delay)
        first = messages[0].content if messages else ""
        if kind == "plan":
            plan = self.plan(first, attempt) if callable(self.plan) else self.plan
            if isinstance(plan, str):
                return LLMResponse(text=plan)
            return LLMResponse(tool_calls=[ToolCall("plan", "submit_plan", plan or {})])
        if kind == "skill":
            return LLMResponse(tool_calls=[ToolCall("skill", "submit_skill", self.skill or {"worth_saving": False})])
        if kind == "synth":
            if callable(self.synth):
                return LLMResponse(text=self.synth(messages))
            return LLMResponse(text=self.synth or "Summary of the team's work:\n" + messages[-1].content[-600:])
        m = re.match(r"You are (.+?), a specialist agent", system)
        name = m.group(1) if m else "?"
        script = self.scripts.get(name)
        if script is None:
            script = self.scripts.get("*", ["Done."])
        idx = sum(1 for x in messages if x.role == "assistant")
        step = script[idx] if idx < len(script) else "Done."
        if callable(step):
            step = step(messages)
        if isinstance(step, str):
            return LLMResponse(text=step)
        steps = step if isinstance(step, list) else [step]
        return LLMResponse(tool_calls=[ToolCall(f"{name}-{idx}-{j}".replace(" ", "_"), s["tool"], dict(s.get("args", {})))
                                       for j, s in enumerate(steps)])


def last_tool_output(messages: list[Msg]) -> str:
    for m in reversed(messages):
        if m.role == "tool":
            return m.content
    return ""


# ------------------------------------------------------------------- demo
_WC = '''"""wc.py - count lines, words and characters in a text file."""
import sys


def count(text: str) -> tuple[int, int, int]:
    return len(text.splitlines()), len(text.split()), len(text)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.txt"
    with open(path, encoding="utf-8") as f:
        lines, words, chars = count(f.read())
    print(f"{lines} lines, {words} words, {chars} characters in {path}")
'''
_SAMPLE = "Saani means the second one.\nIt works the way you would,\nonly in parallel.\n"
_README = "# wc\n\nA tiny word counter.\n\n    python wc.py sample.txt\n"
_CHECK = ("import wc\ntext = open('sample.txt', encoding='utf-8').read()\n"
          "print('%d lines, %d words, %d characters' % wc.count(text))\n")

DEMO_TASK = "Build a tiny word-count command line tool in Python, document it, then test and review it."


def _demo_plan(first: str, attempt: int):
    if "word-count" not in first and "word count" not in first:
        return {"understanding": "Demo mode has no real model.", "complexity": "trivial", "agents": [
            {"id": "a1", "name": "Demo Agent", "role": "Explains that no model is connected", "instructions": "", "tools": []}]}
    return {
        "understanding": "A small, working, documented and reviewed word-count CLI.",
        "complexity": "simple",
        "agents": [
            {"id": "build", "name": "Builder", "role": "Python developer who writes small, clean command line tools",
             "instructions": "Write wc.py and a sample input file in the task folder.", "tools": ["write_file", "read_file"]},
            {"id": "docs", "name": "Doc Writer", "role": "Technical writer who documents tools briefly",
             "instructions": "Write README.md explaining how to run the tool.", "tools": ["write_file"]},
            {"id": "review", "name": "Reviewer", "role": "Tester who runs the tool and reviews the code",
             "instructions": "Read wc.py, run it on the sample, and report whether it works.",
             "tools": ["read_file", "run_python", "list_dir"], "depends_on": ["build", "docs"]},
        ],
        "synthesis": "Report what was built, where it is, and the test result.",
        "success_criteria": ["wc.py runs and prints counts for sample.txt"],
    }


def demo_client() -> SimLLM:
    scripts = {
        "Builder": [
            [{"tool": "write_file", "args": {"path": "wc.py", "content": _WC}},
             {"tool": "write_file", "args": {"path": "sample.txt", "content": _SAMPLE}}],
            "Wrote wc.py (count lines, words, characters) and sample.txt in the task folder.",
        ],
        "Doc Writer": [
            {"tool": "write_file", "args": {"path": "README.md", "content": _README}},
            "Wrote README.md with a one-line usage example.",
        ],
        "Reviewer": [
            {"tool": "read_file", "args": {"path": "wc.py"}},
            {"tool": "run_python", "args": {"code": _CHECK}},
            lambda msgs: ("Reviewed wc.py and ran it on sample.txt. Output: " + " ".join(last_tool_output(msgs).split())[:160]
                          if "PERMISSION DENIED" not in last_tool_output(msgs)
                          else "Reviewed wc.py by reading it. I could NOT run it because permission was denied, so the "
                               "test result is unverified."),
        ],
        "Demo Agent": ["This is demo mode: no language model is connected, so I can only run the built-in demo task. "
                       "Run `saani init` to connect a real model."],
    }
    return SimLLM(plan=_demo_plan, scripts=scripts, delay=0.4,
                  synth=lambda msgs: "Done. Here is what the team did:\n\n" + msgs[-1].content.split("# Agent reports", 1)[-1].strip(),
                  skill={"worth_saving": True, "name": "build-small-python-cli", "title": "Build a small Python CLI",
                         "description": "Build, document, test and review a small Python command line tool.",
                         "when_to_use": "When asked for a small script or command line utility with a quick test.",
                         "lessons": ["Write a sample input file so the reviewer can run the tool immediately.",
                                     "Have the reviewer execute the tool, not just read it."]})

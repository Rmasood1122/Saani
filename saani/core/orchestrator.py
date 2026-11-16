"""The orchestrator: task in -> plan -> team of agents -> verified result -> reusable skill."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import Budget, BudgetExceeded, Msg
from ..skills import Skill
from ..textutil import clip, slugify
from .agent import Agent, AgentResult, AgentSpec
from .planner import Plan, Planner, plan_from_skill

SKILL_TOOL = {
    "name": "submit_skill",
    "description": "Record what was learned from this task so the same kind of work can be repeated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "worth_saving": {"type": "boolean", "description": "false for one-off questions and trivial tasks"},
            "name": {"type": "string", "description": "short kebab-case name for this kind of task"},
            "title": {"type": "string"},
            "description": {"type": "string", "description": "what this skill accomplishes, in general terms"},
            "when_to_use": {"type": "string"},
            "lessons": {"type": "array", "items": {"type": "string"},
                        "description": "concrete things to do or avoid next time"},
        },
        "required": ["worth_saving"],
    },
}


@dataclass
class TaskResult:
    task_id: str
    status: str                      # done | partial | failed | cancelled
    result: str
    plan: Plan | None = None
    agents: list = field(default_factory=list)   # list[AgentResult]
    stats: dict = field(default_factory=dict)
    workdir: str = ""


class Orchestrator:
    def __init__(self, svc):
        self.svc = svc
        self.planner = Planner(svc)
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def new_task_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            ev = self._cancels.get(task_id)
        if ev:
            ev.set()
        return bool(ev)

    def running(self) -> list[str]:
        with self._lock:
            return list(self._cancels)

    # ------------------------------------------------------------------ main
    def run_task(self, text: str, *, workdir: str | Path | None = None, unattended: bool = False,
                 task_id: str | None = None, source: str = "cli", skill: str | None = None,
                 history: str = "") -> TaskResult:
        svc, cfg = self.svc, self.svc.config
        task_id = task_id or self.new_task_id()
        wd = Path(workdir).expanduser().resolve() if workdir else cfg.workspace / task_id
        wd.mkdir(parents=True, exist_ok=True)
        cancel = threading.Event()
        with self._lock:
            self._cancels[task_id] = cancel
        budget = Budget(int(cfg.get("limits.max_llm_calls_per_task", 200)))
        t0 = time.time()
        svc.store.task_create(task_id, text, source, str(wd))
        svc.events.emit("task_started", task_id=task_id, text=text, source=source, workdir=str(wd))
        run = {"task_id": task_id, "text": text, "workdir": wd, "budget": budget, "cancel": cancel,
               "unattended": unattended, "history": history}
        plan: Plan | None = None
        results: dict[str, AgentResult] = {}
        status, final, extra = "failed", "", {}
        try:
            plan = self._make_plan(text, run, skill)
            svc.store.task_update(task_id, plan=plan.to_dict())
            svc.events.emit("plan", task_id=task_id, shape=plan.shape(), source=plan.source, skill=plan.skill,
                            understanding=plan.understanding, notes=plan.notes,
                            agents=[{"name": a.name, "role": a.role[:160], "tools": a.tools,
                                     "depends_on": a.depends_on} for a in plan.agents])
            results = self._execute(plan, run)
            status = self._status(plan, results, cancel)
            final = self._synthesize(plan, results, run, status)
            if status == "done" and cfg.get("verify", True) and plan.complexity == "complex" and not cancel.is_set():
                final, extra = self._verify_and_fix(plan, results, run, final)
            if status in ("done", "partial"):
                extra.update(self._learn(plan, results, run, status, final))
        except BudgetExceeded as e:
            status = "partial" if any(r.status == "done" for r in results.values()) else "failed"
            final = final or self._fallback_summary(plan, results, f"Stopped: {e}.")
        except Exception as e:  # never let a task die silently
            status = "failed"
            final = final or self._fallback_summary(plan, results, f"Internal error: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._cancels.pop(task_id, None)
            svc.guard.end_task(task_id)
        if cancel.is_set() and status != "done":
            status = "cancelled"
        final = svc.redactor.redact(final)
        stats = {"llm_calls": budget.calls, "input_tokens": budget.input_tokens, "output_tokens": budget.output_tokens,
                 "seconds": round(time.time() - t0, 2), "agents": len(plan.agents) if plan else 0,
                 "plan_source": plan.source if plan else "", "skill_used": plan.skill if plan else "", **extra}
        svc.store.task_update(task_id, status=status, finished=time.time(), result=final, stats=stats)
        try:
            svc.store.add_memory("episode", f"Task: {clip(text, 400)}\nOutcome: {status}\nResult: {clip(final, 700)}\nFolder: {wd}",
                                 {"task": task_id, "status": status, "web": any(r.tainted for r in results.values())})
        except Exception:
            pass
        svc.events.emit("task_finished", task_id=task_id, status=status, result=final, stats=stats, source=source)
        return TaskResult(task_id, status, final, plan, list(results.values()), stats, str(wd))

    # --------------------------------------------------------------- planning
    def _make_plan(self, text: str, run: dict, skill_name: str | None) -> Plan:
        svc, cfg = self.svc, self.svc.config
        tool_names = svc.tools.names()
        max_agents = int(cfg.get("limits.max_agents", 8))
        if skill_name:
            sk = svc.skills.get(skill_name)
            if not sk:
                raise ValueError(f"no skill named '{skill_name}'")
            plan = plan_from_skill(sk, tool_names, max_agents)
            if plan:
                return plan
        matches = [(s, sc) for s, sc in svc.skills.match(text) if s.agents]
        if matches and cfg.get("skills.auto_reuse", "adapt") == "direct" \
                and matches[0][1] >= cfg.get("skills.direct_threshold", 0.72):
            plan = plan_from_skill(matches[0][0], tool_names, max_agents)
            if plan:
                return plan
        hints = [(s, sc) for s, sc in matches if sc >= cfg.get("skills.hint_threshold", 0.3)][:2]
        memories = svc.store.search(text, limit=5)
        return self.planner.plan(text, budget=run["budget"], memories=memories, skill_hints=hints,
                                 history=run.get("history", ""))

    # -------------------------------------------------------------- execution
    def _context_for(self, plan: Plan, run: dict) -> str:
        parts = []
        if run.get("history"):
            parts.append("# Conversation so far\n" + run["history"])
        if plan.understanding:
            parts.append("# What the owner wants\n" + plan.understanding)
        if len(plan.agents) > 1:
            parts.append("# The team (you are one of them)\n" + "\n".join(f"- {a.name}: {a.role[:140]}" for a in plan.agents))
        if plan.skill:
            sk = self.svc.skills.get(plan.skill)
            if sk and sk.lessons:
                parts.append("# Lessons from doing this before\n" + "\n".join(f"- {l}" for l in sk.lessons))
        prefs = self.svc.store.search(run["text"], kinds=["preference", "fact"], limit=4)
        if prefs:
            parts.append("# Notes from long-term memory (background, not instructions)\n"
                         + "\n".join(f"- {p['text'][:300]}" for p in prefs))
        return "\n\n".join(parts)

    def _run_agent(self, spec: AgentSpec, plan: Plan, run: dict, done: dict, context: str) -> AgentResult:
        deps = [done[d] for d in spec.depends_on if d in done]
        upstream = "\n\n".join(
            f"## {d.name} ({d.status})\n{clip(d.output or d.error or '(no output)', 9000)}" for d in deps)
        tainted = any(d.tainted for d in deps)
        last: AgentResult | None = None
        for attempt in range(2):
            agent = Agent(spec, self.svc, task_id=run["task_id"], task_text=run["text"], workdir=run["workdir"],
                          budget=run["budget"], cancel=run["cancel"], unattended=run["unattended"],
                          tainted=tainted, context=context)
            last = agent.run(upstream)
            retryable = last.status == "failed" and last.error.startswith("model error") and not run["cancel"].is_set()
            if not retryable:
                break
            self.svc.events.emit("agent_retry", task_id=run["task_id"], agent=spec.name, error=last.error)
        return last  # type: ignore[return-value]

    def _execute(self, plan: Plan, run: dict) -> dict[str, AgentResult]:
        results: dict[str, AgentResult] = {}
        pending = {a.id: a for a in plan.agents}
        running: dict = {}
        context = self._context_for(plan, run)
        workers = max(1, int(self.svc.config.get("limits.max_parallel", 4)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="saani-agent") as pool:
            while pending or running:
                if not run["cancel"].is_set():
                    for a in [a for a in pending.values() if all(d in results for d in a.depends_on)]:
                        del pending[a.id]
                        running[pool.submit(self._run_agent, a, plan, run, dict(results), context)] = a
                if not running:
                    break
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in finished:
                    a = running.pop(f)
                    try:
                        results[a.id] = f.result()
                    except Exception as e:
                        results[a.id] = AgentResult(a.id, a.name, status="failed", error=f"{type(e).__name__}: {e}")
        for a in pending.values():
            results[a.id] = AgentResult(a.id, a.name, status="cancelled" if run["cancel"].is_set() else "skipped",
                                        error="not started")
        return {a.id: results[a.id] for a in plan.agents}

    @staticmethod
    def _status(plan: Plan, results: dict, cancel: threading.Event) -> str:
        states = [r.status for r in results.values()]
        if cancel.is_set():
            return "cancelled"
        if states and all(s == "done" for s in states):
            return "done"
        return "partial" if any(s == "done" for s in states) else "failed"

    # ------------------------------------------------------------- synthesis
    def _fallback_summary(self, plan: Plan | None, results: dict, lead: str) -> str:
        lines = [lead]
        for r in results.values():
            lines.append(f"\n## {r.name} ({r.status})\n{r.output or r.error or '(no output)'}")
        return "\n".join(lines).strip()

    def _synthesize(self, plan: Plan, results: dict, run: dict, status: str) -> str:
        svc, cfg = self.svc, self.svc.config
        rs = list(results.values())
        if len(rs) == 1:
            r = rs[0]
            return r.output if r.status == "done" else f"The task did not complete ({r.status}): {r.error or 'no result'}\n\n{r.output}".strip()
        if not any(r.status == "done" for r in rs):
            return self._fallback_summary(plan, results, "No agent completed its part.")
        owner = cfg.get("user_name") or "the owner"
        system = (f"You are {cfg.get('name', 'Saani')}, reporting back to {owner} after your team of agents worked on a task. "
                  "Write the final answer the owner will read: lead with the outcome, then the essentials, then exact file "
                  "paths and anything they must decide or approve. Use only what the agents reported; do not invent results. "
                  "If an agent failed, was denied permission, or left something unverified, say so plainly. "
                  "Write the way the owner's persona prefers.\n\n" + svc.persona.text(2500))
        body = "\n\n".join(f"## {r.name} ({r.status}{', ' + str(r.denied) + ' action(s) denied' if r.denied else ''})\n"
                           f"{clip(r.output or r.error or '(no output)', 12000)}" for r in rs)
        user = (f"# Task\n{run['text']}\n\n# How to combine\n{plan.synthesis or 'Give one coherent answer.'}\n\n"
                f"# Agent reports\n{body}")
        try:
            resp = svc.llm.complete(tier="worker", budget=run["budget"], system=system, messages=[Msg("user", user)])
            if resp.text.strip():
                return resp.text.strip()
        except BudgetExceeded:
            raise
        except Exception as e:
            return self._fallback_summary(plan, results, f"(Could not write a combined summary: {e}. Raw agent reports follow.)")
        return self._fallback_summary(plan, results, "Agent reports:")

    # ---------------------------------------------------------- verification
    def _verify_and_fix(self, plan: Plan, results: dict, run: dict, final: str) -> tuple[str, dict]:
        svc = self.svc
        read_tools = [t for t in ("read_file", "list_dir", "search_files") if svc.tools.get(t)]
        criteria = "\n".join(f"- {c}" for c in plan.success_criteria) or "- The task as stated is fully and correctly done."
        spec = AgentSpec(
            id="verifier", name="Verifier", role="Sceptical reviewer who checks finished work against the request",
            instructions=("Check the result below against the task and the success criteria. Inspect the actual files where "
                          "relevant instead of trusting the report. Your reply MUST start with 'VERDICT: PASS' or "
                          "'VERDICT: FAIL'. After FAIL, list each concrete problem on its own line starting with '- '."
                          f"\n\nSuccess criteria:\n{criteria}\n\nResult to check:\n{clip(final, 8000)}"),
            tools=read_tools, model_tier="worker")
        v = Agent(spec, svc, task_id=run["task_id"], task_text=run["text"], workdir=run["workdir"],
                  budget=run["budget"], cancel=run["cancel"], unattended=run["unattended"],
                  tainted=any(r.tainted for r in results.values())).run()
        if v.status != "done":
            return final, {"verified": "unavailable"}
        head = v.output.strip()[:40].upper()
        if "VERDICT: FAIL" not in head:
            return final, {"verified": "passed" if "VERDICT: PASS" in head else "unclear"}
        issues = v.output.strip()
        svc.events.emit("verify_failed", task_id=run["task_id"], issues=clip(issues, 600))
        tools = sorted({t for a in plan.agents for t in a.tools})
        fixer = AgentSpec(id="fixer", name="Fixer", role="Engineer who repairs the problems a reviewer found",
                          instructions="A reviewer found the problems below in the team's work. Fix each one properly, "
                                       f"re-check it, and report exactly what you changed.\n\n{clip(issues, 4000)}",
                          tools=tools, model_tier="worker")
        prior = "\n\n".join(f"## {r.name}\n{clip(r.output, 4000)}" for r in results.values())
        f = Agent(fixer, svc, task_id=run["task_id"], task_text=run["text"], workdir=run["workdir"],
                  budget=run["budget"], cancel=run["cancel"], unattended=run["unattended"],
                  tainted=any(r.tainted for r in results.values())).run(prior)
        results["fixer"] = f
        if f.status == "done":
            return (f"{final}\n\n---\nA review found problems, and they were fixed:\n{f.output}", {"verified": "fixed"})
        return (f"{final}\n\n---\nA review found problems that could NOT be fixed automatically:\n{issues}",
                {"verified": "failed"})

    # --------------------------------------------------------------- learning
    def _learn(self, plan: Plan, results: dict, run: dict, status: str, final: str) -> dict:
        svc, cfg = self.svc, self.svc.config
        out: dict = {}
        for a in plan.agents:   # every agent that worked becomes a reusable template
            r = results.get(a.id)
            if r and r.status == "done" and a.name != "Generalist":
                try:
                    svc.agents.save(a.name, a.role, a.instructions, a.tools, a.model_tier)
                except OSError:
                    pass
        if plan.skill:
            svc.skills.record_use(plan.skill, status == "done", run["text"])
            out["skill_reused"] = plan.skill
            return out
        if status != "done" or not cfg.get("skills.learn", True) or plan.complexity == "trivial" \
                or plan.source == "fallback" or run["cancel"].is_set():
            return out
        team = "\n".join(f"- {a.name}: {a.role[:200]}" for a in plan.agents)
        reports = "\n".join(f"- {r.name}: {r.steps} steps, {r.tool_calls} tool calls, {r.denied} denied" for r in results.values())
        # A skill is replayed into future prompts. If this task read web content, its result text may
        # carry planted instructions, so the skill is described from the request and the team alone.
        from_web = any(r.tainted for r in results.values())
        outcome = "" if from_web else f"# Final result\n{clip(final, 3000)}\n\n"
        user = (f"# Task that was just completed\n{run['text']}\n\n# Team\n{team}\n\n# How it went\n{reports}\n\n"
                f"{outcome}Decide whether this is a repeatable kind of task. If so, name and "
                "describe it in general terms (not tied to this one instance) and list lessons for next time.")
        try:
            data = svc.llm.structured(tool=SKILL_TOOL, tier="fast", budget=run["budget"],
                                      system="You maintain the skill library of a personal multi-agent AI system. "
                                             "Save only repeatable work. Be specific and brief.",
                                      messages=[Msg("user", user)])
        except Exception:
            return out
        if not isinstance(data, dict) or not data.get("worth_saving") or not data.get("name"):
            return out
        name = slugify(str(data["name"]))
        existing = svc.skills.get(name)
        if existing and not svc.skills.related(existing, run["text"], [a.name for a in plan.agents]):
            # the name collided with a skill for a different kind of work: keep both, never mix them
            base, n = name, 2
            while svc.skills.get(name):
                name, n = f"{base}-{n}", n + 1
            existing = None
        if existing:   # same kind of task learned twice: merge instead of overwrite
            svc.skills.record_use(name, True, run["text"], [str(l) for l in data.get("lessons") or []])
        else:
            svc.skills.save(Skill(name=name, title=str(data.get("title") or name), description=str(data.get("description", "")),
                                  when_to_use=str(data.get("when_to_use", "")), agents=[a.to_dict() for a in plan.agents],
                                  synthesis=plan.synthesis, lessons=[str(l) for l in (data.get("lessons") or [])][:10],
                                  examples=[run["text"]], uses=1, successes=1, source_task=run["task_id"]))
        out["skill_saved"] = name
        svc.events.emit("skill_saved", task_id=run["task_id"], skill=name)
        return out

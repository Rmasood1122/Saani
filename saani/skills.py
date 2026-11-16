"""Reusable work: the skill library and the agent-template registry.

A *skill* is what Saani keeps from a finished task so the same kind of work is
never planned from scratch again: the team of agents that worked, their
instructions, and the lessons learned. Skills are plain JSON files you can read,
edit, share or delete.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .textutil import cosine, slugify


@dataclass
class Skill:
    name: str
    title: str = ""
    description: str = ""
    when_to_use: str = ""
    agents: list = field(default_factory=list)       # plan template (list of agent dicts)
    synthesis: str = ""
    lessons: list = field(default_factory=list)
    examples: list = field(default_factory=list)     # task texts this skill solved
    created: float = field(default_factory=time.time)
    uses: int = 0
    successes: int = 0
    source_task: str = ""

    def card(self) -> str:
        team = ", ".join(a.get("name", "?") for a in self.agents)
        return f"{self.name}: {self.description} (team: {team}; used {self.uses}x)"


class SkillLibrary:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    def _file(self, name: str) -> Path:
        return self.root / slugify(name) / "skill.json"

    def all(self) -> list[Skill]:
        out = []
        if not self.root.exists():
            return out
        for f in sorted(self.root.glob("*/skill.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                out.append(Skill(**{k: v for k, v in data.items() if k in Skill.__dataclass_fields__}))
            except (OSError, ValueError, TypeError):
                continue
        return out

    def get(self, name: str) -> Skill | None:
        f = self._file(name)
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            return Skill(**{k: v for k, v in data.items() if k in Skill.__dataclass_fields__})
        except (OSError, ValueError, TypeError):
            return None

    def save(self, skill: Skill) -> Skill:
        with self._lock:
            skill.name = slugify(skill.name)
            f = self._file(skill.name)
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(skill), indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(f)
            return skill

    def delete(self, name: str) -> bool:
        d = self._file(name).parent
        if d.exists() and d.parent == self.root:
            shutil.rmtree(d)
            return True
        return False

    def match(self, task: str, limit: int = 3) -> list[tuple[Skill, float]]:
        """Rank skills by similarity to the task: best of (example similarity, description similarity)."""
        scored = []
        for s in self.all():
            ex = max((cosine(task, e) for e in s.examples), default=0.0)
            desc = cosine(task, f"{s.title} {s.description} {s.when_to_use}")
            score = max(ex, 0.85 * desc)
            if score > 0:
                scored.append((s, round(score, 3)))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    @staticmethod
    def related(skill: Skill, task: str, team: list | None = None) -> bool:
        """Is this task the same kind of work as the skill? Same team, or clearly similar wording."""
        if team is not None and sorted(team) == sorted(a.get("name", "") for a in skill.agents):
            sim = max((cosine(task, e) for e in skill.examples), default=0.0)
            return sim >= 0.15 or cosine(task, f"{skill.title} {skill.description} {skill.when_to_use}") >= 0.15
        return max((cosine(task, e) for e in skill.examples), default=0.0) >= 0.45

    def record_use(self, name: str, success: bool, task: str = "", lessons: list | None = None) -> None:
        with self._lock:
            s = self.get(name)
            if not s:
                return
            s.uses += 1
            s.successes += int(success)
            if task and task not in s.examples:
                s.examples = (s.examples + [task])[-8:]
            for lesson in lessons or []:
                if lesson and lesson not in s.lessons:
                    s.lessons = (s.lessons + [lesson])[-12:]
            self.save(s)


class AgentRegistry:
    """Every agent Saani invents is kept as a reusable template."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    def all(self) -> list[dict]:
        out = []
        if not self.root.exists():
            return out
        for f in sorted(self.root.glob("*.json")):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def get(self, slug: str) -> dict | None:
        f = self.root / f"{slugify(slug)}.json"
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, name: str, role: str, instructions: str, tools: list, model_tier: str = "worker") -> str:
        slug = slugify(name)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            prev = self.get(slug) or {}
            data = {"slug": slug, "name": name, "role": role, "instructions": instructions,
                    "tools": sorted(set(tools)), "model_tier": model_tier,
                    "uses": int(prev.get("uses", 0)) + 1,
                    "created": prev.get("created", time.time()), "updated": time.time()}
            (self.root / f"{slug}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return slug

    def match(self, task: str, limit: int = 5) -> list[dict]:
        scored = [(cosine(task, f"{a.get('name', '')} {a.get('role', '')}"), a) for a in self.all()]
        scored.sort(key=lambda x: (-x[0], -x[1].get("uses", 0)))
        return [a for s, a in scored[:limit] if s > 0.05]

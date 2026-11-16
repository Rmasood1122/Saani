"""Persona: the written-down version of how the owner thinks, decides and works.

This file is what makes agents act "like you". It is plain Markdown that you can
edit by hand; Saani also appends preferences it learns from your feedback.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

LEARNED_HEADER = "## Learned preferences"

QUESTIONS: list[tuple[str, str]] = [
    ("Identity", "Who are you professionally? (role, what you build, who your clients are)"),
    ("Goals and priorities", "What are your top goals right now, in priority order?"),
    ("How I decide", "How do you make decisions? (speed vs. perfection, risk appetite, what you check first, what makes you say no)"),
    ("How I work", "Your tech stack, tools and quality standards? (languages, frameworks, testing habits, naming, commit style)"),
    ("How I communicate", "How do you write to clients and employers? (tone, length, language, sign-off)"),
    ("Job search", "What roles are you looking for, and what are your deal-breakers? (title, location/remote, salary floor)"),
    ("Never do", "What should your assistant never do without asking you first?"),
]

TEMPLATE = """# Persona

This describes how {owner} thinks and works. Every agent reads it before acting.
Edit it freely - the more specific it is, the more the system behaves like you.

## Identity
(not filled in yet - run `saani persona init`)

## Goals and priorities

## How I decide

## How I work

## How I communicate

## Never do
- Never spend money, send messages as me, or publish anything without my approval.

{learned}
"""


class Persona:
    def __init__(self, path: Path, owner: str = ""):
        self.path = Path(path)
        self.owner = owner or "the owner"
        self._lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.exists()

    def ensure(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(TEMPLATE.format(owner=self.owner, learned=LEARNED_HEADER + "\n"), encoding="utf-8")

    def text(self, max_chars: int = 7000) -> str:
        try:
            t = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if len(t) <= max_chars:
            return t
        # keep the start (identity, rules) and the end (most recent learned preferences)
        return t[: max_chars * 2 // 3] + "\n...\n" + t[-max_chars // 3:]

    def add_preference(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        with self._lock:
            self.ensure()
            body = self.path.read_text(encoding="utf-8")
            if LEARNED_HEADER not in body:
                body = body.rstrip() + f"\n\n{LEARNED_HEADER}\n"
            if text.lower() in body.lower():
                return
            body = body.rstrip() + f"\n- {text} ({time.strftime('%Y-%m-%d')})\n"
            self.path.write_text(body, encoding="utf-8")

    def interview(self, ask: Callable[[str], str | None]) -> None:
        """Build the persona from a short interview. ``ask`` returns the answer text."""
        sections = []
        for title, q in QUESTIONS:
            ans = (ask(q) or "").strip()
            sections.append(f"## {title}\n{ans or '(not answered)'}\n")
        old_learned = ""
        if self.path.exists():
            body = self.path.read_text(encoding="utf-8")
            if LEARNED_HEADER in body:
                old_learned = body.split(LEARNED_HEADER, 1)[1].strip()
        out = (f"# Persona\n\nThis describes how {self.owner} thinks and works. "
               "Every agent reads it before acting.\n\n" + "\n".join(sections)
               + f"\n{LEARNED_HEADER}\n{old_learned}\n")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(out, encoding="utf-8")

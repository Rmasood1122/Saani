"""Training data: conversations rendered into token sequences.

A conversation is a list of turns:
    {"role": "system"|"user"|"assistant"|"tool", "content": str,
     "tool_calls": [{"name": str, "args": dict}]}      # assistant only

Rendered with the tokenizer's special tokens:
    <|sys|>system text<|usr|>user text<|ast|>answer<|call|>{"name":..,"args":{..}}<|end|>
    <|tool|>tool result text<|ast|>next answer<|end|> ... <|eos|>

The loss is computed only on assistant tokens (what the model should say), never on
the prompt, so the model learns to answer rather than to repeat the question.

Sources:
  * Saani traces  (~/.saani/traces/*.jsonl) - real exchanges Saani recorded with
    its main model; the most valuable data because they are your tasks.
  * plain text    (.txt / .md files) - documents, trained as plain language modelling.
  * demo          - a synthetic set of small Saani-style tasks so a nano model can be
    trained end to end in minutes on a CPU.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .tokenizer import Tokenizer

Turn = dict
MAX_TOOL_RESULT_CHARS = 1500          # long tool outputs are cut; the model learns the shape, not the dump
MAX_SYSTEM_CHARS = 4000


# --------------------------------------------------------------------- render
def render(tok: Tokenizer, turns: list[Turn], max_len: int) -> tuple[list[int], list[int]]:
    """Return (ids, mask); mask[i]=1 where ids[i] is an assistant token to learn."""
    ids: list[int] = []
    mask: list[int] = []

    def put(text_ids: list[int], learn: bool):
        ids.extend(text_ids); mask.extend([1 if learn else 0] * len(text_ids))

    for t in turns:
        role, content = t.get("role"), (t.get("content") or "")
        if role == "system":
            put([tok.token("<|sys|>")] + tok.encode(content[:MAX_SYSTEM_CHARS]), False)
        elif role == "user":
            put([tok.token("<|usr|>")] + tok.encode(content), False)
        elif role == "tool":
            put([tok.token("<|tool|>")] + tok.encode(content[:MAX_TOOL_RESULT_CHARS]), False)
        elif role == "assistant":
            put([tok.token("<|ast|>")], False)
            if content:
                put(tok.encode(content), True)
            for c in t.get("tool_calls") or []:
                body = json.dumps({"name": c["name"], "args": c.get("args", {})}, ensure_ascii=False)
                put([tok.token("<|call|>")] + tok.encode(body), True)
            put([tok.token("<|end|>")], True)
    put([tok.token("<|eos|>")], True)
    if len(ids) > max_len:                      # keep the end: that is where the answer is
        ids, mask = ids[-max_len:], mask[-max_len:]
    return ids, mask


def render_prompt(tok: Tokenizer, turns: list[Turn]) -> list[int]:
    """Prompt for generation: everything up to and including the opening <|ast|>."""
    ids, _ = render(tok, turns, 1 << 30)
    # drop the trailing <|eos|> that render() adds, then open an assistant turn
    ids = ids[:-1]
    return ids + [tok.token("<|ast|>")]


# -------------------------------------------------------------------- sources
def from_saani_traces(folder: str | Path) -> Iterator[list[Turn]]:
    """Each Saani trace line becomes one conversation ending in the recorded answer."""
    for path in sorted(Path(folder).glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                turns: list[Turn] = []
                system = rec.get("system") or ""
                tools = rec.get("tools") or []
                if tools:
                    system = system.rstrip() + "\n\n" + tools_block(tools)
                if system:
                    turns.append({"role": "system", "content": system})
                for m in rec.get("messages") or []:
                    role = m.get("role")
                    if role == "assistant":
                        turns.append({"role": "assistant", "content": m.get("content") or "",
                                      "tool_calls": [{"name": c["name"], "args": c.get("args", {})}
                                                     for c in m.get("tool_calls") or []]})
                    elif role == "tool":
                        pre = "ERROR: " if m.get("is_error") else ""
                        turns.append({"role": "tool", "content": pre + (m.get("content") or "")})
                    else:
                        turns.append({"role": "user", "content": m.get("content") or ""})
                r = rec.get("response") or {}
                turns.append({"role": "assistant", "content": r.get("text") or "",
                              "tool_calls": [{"name": c["name"], "args": c.get("args", {})}
                                             for c in r.get("tool_calls") or []]})
                yield turns


def tools_block(tools: list[dict]) -> str:
    """Compact description of the tools on offer; the server writes the same block."""
    lines = ["Tools you can call:"]
    for t in tools:
        props = ((t.get("input_schema") or {}).get("properties") or {})
        args = ", ".join(props.keys())
        desc = (t.get("description") or "").strip().split("\n")[0][:120]
        lines.append(f"- {t['name']}({args}): {desc}")
    return "\n".join(lines)


def from_text_files(folder: str | Path, chunk_chars: int = 3000) -> Iterator[list[Turn]]:
    """Plain documents become 'assistant' text with no prompt: ordinary language modelling."""
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in (".txt", ".md") or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i in range(0, len(text), chunk_chars):
            piece = text[i:i + chunk_chars].strip()
            if len(piece) > 50:
                yield [{"role": "assistant", "content": piece}]


# ------------------------------------------------------------------- demo set
_DEMO_TOOLS = [
    {"name": "read_file", "description": "Read a text file", "input_schema": {"properties": {"path": {}}}},
    {"name": "write_file", "description": "Write a text file", "input_schema": {"properties": {"path": {}, "content": {}}}},
    {"name": "list_dir", "description": "List a folder", "input_schema": {"properties": {"path": {}}}},
    {"name": "shell", "description": "Run a shell command", "input_schema": {"properties": {"command": {}}}},
    {"name": "finish", "description": "Report the result", "input_schema": {"properties": {"result": {}}}},
]
_FILES = ["notes.md", "todo.txt", "report.md", "README.md", "config.json", "main.py", "app.py", "tests/test_app.py",
          "data/sales.csv", "docs/plan.md", "src/server.py", "ideas.txt"]
_FOLDERS = [".", "src", "docs", "data", "tests", "project"]
_FACTS = {"What is 2+2?": "4", "What is 10 times 3?": "30", "What is 7 plus 5?": "12", "What is 9 minus 4?": "5",
          "What is the capital of France?": "Paris", "What is the capital of Japan?": "Tokyo",
          "What colour is the sky on a clear day?": "Blue", "How many days are in a week?": "7",
          "What is the opposite of hot?": "Cold", "What language is Saani written in?": "Python",
          "Which planet do we live on?": "Earth", "How many hours are in a day?": "24"}
_GREETS = [("Hello", "Hello. What should I do?"), ("Hi Saani", "Hi. Give me a task and I will get started."),
           ("Thanks", "You're welcome."), ("Good morning", "Good morning. What is first today?"),
           ("Are you there?", "Yes. What do you need?"), ("Who are you?", "I am Saani, your assistant.")]


def demo_conversations(n: int = 4000, seed: int = 0) -> list[list[Turn]]:
    """Synthetic Saani-style tasks: small enough to learn in minutes, varied enough to test tool calls."""
    rng = random.Random(seed)
    sys_prompt = "You are Saani, a careful assistant. Use tools when a task needs them, then finish.\n\n" + tools_block(_DEMO_TOOLS)
    out: list[list[Turn]] = []
    for _ in range(n):
        kind = rng.choice(["read", "read", "write", "list", "shell", "fact", "fact", "greet", "finish_after_read"])
        f, d = rng.choice(_FILES), rng.choice(_FOLDERS)
        sysm = {"role": "system", "content": sys_prompt}
        if kind == "read":
            u = rng.choice([f"Read {f}", f"Open {f} and tell me what is in it", f"Show me the contents of {f}",
                            f"What does {f} say?"])
            out.append([sysm, {"role": "user", "content": u},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": "read_file", "args": {"path": f}}]}])
        elif kind == "finish_after_read":
            body = rng.choice(["buy milk\ncall Sam", "Q3 revenue: 4875\nQ2 revenue: 3960", "- fix login bug\n- ship v2",
                               "print('hello')", "name: demo\nversion: 1"])
            out.append([sysm, {"role": "user", "content": f"Read {f} and summarise it"},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": "read_file", "args": {"path": f}}]},
                        {"role": "tool", "content": body},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": "finish", "args": {"result": f"{f} contains: " + body.replace("\n", "; ")}}]}])
        elif kind == "write":
            text = rng.choice(["hello world", "TODO: review the plan", "meeting at 10", "done", "draft v1"])
            u = rng.choice([f"Write '{text}' to {f}", f"Create {f} containing {text}", f"Save the text {text} into {f}"])
            out.append([sysm, {"role": "user", "content": u},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": "write_file", "args": {"path": f, "content": text}}]}])
        elif kind == "list":
            u = rng.choice([f"List the files in {d}", f"What is inside {d}?", f"Show the folder {d}"])
            out.append([sysm, {"role": "user", "content": u},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": "list_dir", "args": {"path": d}}]}])
        elif kind == "shell":
            cmd = rng.choice(["pytest -q", "git status", "python main.py", "ls -la", "npm test", "pip list"])
            u = rng.choice([f"Run {cmd}", f"Execute the command {cmd}", f"Please run `{cmd}`"])
            out.append([sysm, {"role": "user", "content": u},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": "shell", "args": {"command": cmd}}]}])
        elif kind == "fact":
            q, a = rng.choice(list(_FACTS.items()))
            out.append([sysm, {"role": "user", "content": q}, {"role": "assistant", "content": a}])
        else:
            q, a = rng.choice(_GREETS)
            out.append([sysm, {"role": "user", "content": q}, {"role": "assistant", "content": a}])
    return out


# --------------------------------------------------------------------- packing
def pack(tok: Tokenizer, convs: Iterable[list[Turn]], block: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Concatenate rendered conversations into one token stream plus a loss mask.

    Returns (ids, mask, n_conversations). The trainer cuts random windows of `block`
    tokens from the stream; <|eos|> separates conversations.
    """
    ids_all: list[int] = []
    mask_all: list[int] = []
    n = 0
    for turns in convs:
        ids, mask = render(tok, turns, block)
        ids_all.extend(ids); mask_all.extend(mask); n += 1
    return np.array(ids_all, dtype=np.uint16 if tok.vocab_size < 65535 else np.int32), np.array(mask_all, dtype=np.uint8), n


def conversation_texts(convs: Iterable[list[Turn]]) -> Iterator[str]:
    """Raw text of conversations, for tokenizer training."""
    for turns in convs:
        for t in turns:
            if t.get("content"):
                yield t["content"]
            for c in t.get("tool_calls") or []:
                yield json.dumps({"name": c["name"], "args": c.get("args", {})}, ensure_ascii=False)

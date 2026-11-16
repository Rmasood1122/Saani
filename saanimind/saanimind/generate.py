"""Turn a trained model into something you can talk to: load, prompt, sample, parse tool calls."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .data import render_prompt, tools_block
from .model import GPT
from .tokenizer import Tokenizer
from .train import load_model, pick_device

_CALL_RE = re.compile(r"<\|call\|>(.*?)(?=<\|call\|>|<\|end\|>|<\|eos\|>|$)", re.S)


class Mind:
    """A loaded model + tokenizer."""

    def __init__(self, run_dir: str | Path, device: str = "auto"):
        self.run_dir = Path(run_dir)
        self.device = pick_device(device)
        self.tok = Tokenizer.load(self.run_dir / "tokenizer.json")
        self.model: GPT = load_model(self.run_dir, self.device)
        self.stop = {self.tok.token("<|end|>"), self.tok.token("<|eos|>")}
        self.name = (self.run_dir.name or "saanimind")

    def info(self) -> dict:
        c = self.model.cfg
        return {"name": self.name, "params": self.model.n_params(), "layers": c.n_layer, "heads": c.n_head,
                "width": c.n_embd, "context": c.block_size, "vocab": c.vocab_size, "device": str(self.device)}

    def reply(self, turns: list[dict], tools: list[dict] | None = None, *, max_new_tokens: int = 256,
              temperature: float = 0.7, top_p: float = 0.9) -> dict:
        """Returns {"text": str, "tool_calls": [{"name", "args"}], "raw": str, "tokens": int}."""
        turns = list(turns)
        if tools:
            block = tools_block(tools)
            if turns and turns[0].get("role") == "system":
                turns[0] = {"role": "system", "content": (turns[0].get("content") or "").rstrip() + "\n\n" + block}
            else:
                turns.insert(0, {"role": "system", "content": block})
        ids = render_prompt(self.tok, turns)
        ids = ids[-(self.model.cfg.block_size - 1):]         # leave room to generate
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        budget = min(max_new_tokens, self.model.cfg.block_size - len(ids))
        out: list[int] = []
        for t in self.model.generate(x, budget, temperature=temperature, top_p=top_p, stop_ids=self.stop):
            out.append(t)
        raw = self.tok.decode(out)
        return {**parse_output(raw), "raw": raw, "tokens": len(out)}


def parse_output(raw: str) -> dict:
    """Split generated text into plain text and tool calls; broken JSON becomes text."""
    text = raw
    calls: list[dict] = []
    first = raw.find("<|call|>")
    if first >= 0:
        text = raw[:first]
        for m in _CALL_RE.finditer(raw):
            body = m.group(1).strip()
            try:
                d = json.loads(body)
                if isinstance(d, dict) and isinstance(d.get("name"), str):
                    calls.append({"name": d["name"], "args": d.get("args") if isinstance(d.get("args"), dict) else {}})
                    continue
            except ValueError:
                pass
            text += " " + body
    for s in ("<|end|>", "<|eos|>", "<|ast|>", "<|usr|>", "<|sys|>", "<|tool|>"):
        text = text.replace(s, "")
    return {"text": text.strip(), "tool_calls": calls}

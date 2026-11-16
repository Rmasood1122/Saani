"""saanimind command line: prepare -> train -> chat / serve."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .data import (conversation_texts, demo_conversations, from_saani_traces, from_text_files, pack)
from .model import PRESETS, ModelConfig
from .tokenizer import Tokenizer


def default_traces() -> Path:
    return Path.home() / ".saani" / "traces"


def cmd_prepare(a) -> int:
    convs: list[list[dict]] = []
    sources: list[str] = []
    traces = Path(a.traces) if a.traces else default_traces()
    if traces.exists() and any(traces.glob("*.jsonl")):
        n0 = len(convs); convs.extend(from_saani_traces(traces))
        sources.append(f"{len(convs) - n0} conversations from Saani traces at {traces}")
    elif a.traces:
        print(f"no trace files in {traces}"); return 1
    for folder in a.text or []:
        n0 = len(convs); convs.extend(from_text_files(folder))
        sources.append(f"{len(convs) - n0} text chunks from {folder}")
    if a.demo or not convs:
        if not convs and not a.demo:
            print("No traces or text found; using the built-in demo set. Turn on `llm.record_traces` in Saani to collect real data.")
        n0 = len(convs); convs.extend(demo_conversations(a.demo_size, a.seed))
        sources.append(f"{len(convs) - n0} synthetic demo conversations")
    rng = random.Random(a.seed); rng.shuffle(convs)
    n_val = max(1, int(len(convs) * a.val_fraction))
    val, train = convs[:n_val], convs[n_val:]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("training tokenizer ...")
    tok = Tokenizer()
    tok.train(list(conversation_texts(train)), a.vocab_size, verbose=True)
    tok.save(out / "tokenizer.json")
    print(f"vocabulary: {tok.vocab_size} tokens")
    for name, part in (("train", train), ("val", val)):
        ids, mask, n = pack(tok, part, a.block_size)
        np.save(out / f"{name}_ids.npy", ids); np.save(out / f"{name}_mask.npy", mask)
        print(f"{name}: {n} conversations, {len(ids):,} tokens ({int(mask.sum()):,} to learn)")
    (out / "meta.json").write_text(json.dumps({"vocab_size": tok.vocab_size, "block_size": a.block_size,
                                               "n_conversations": len(train), "sources": sources}, indent=2))
    print("sources:\n  " + "\n  ".join(sources))
    print(f"wrote {out}/   next:  saanimind train --data {out} --out runs/{a.size_hint}")
    return 0


def cmd_train(a) -> int:
    from .train import train
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    preset = dict(PRESETS[a.size])
    preset["block_size"] = min(preset["block_size"], meta["block_size"]) if not a.block_size else a.block_size
    cfg = ModelConfig(vocab_size=meta["vocab_size"], dropout=a.dropout, **preset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(Path(a.data) / "tokenizer.json", out / "tokenizer.json")
    r = train(a.data, out, cfg, steps=a.steps, batch=a.batch, lr=a.lr, warmup=a.warmup, minutes=a.minutes,
              eval_every=a.eval_every, device=a.device, resume=a.resume)
    print(f"done: {r['steps']} steps, best val loss {r['best_val_loss']:.3f}, {r['params']/1e6:.2f}M params, "
          f"{r['minutes']:.1f} min\nnext:  saanimind chat {out}   or   saanimind serve {out}")
    return 0


def cmd_info(a) -> int:
    from .generate import Mind
    m = Mind(a.run, a.device)
    for k, v in m.info().items():
        print(f"{k:>8}: {v:,}" if isinstance(v, int) else f"{k:>8}: {v}")
    h = Path(a.run) / "history.json"
    if h.exists():
        hist = json.loads(h.read_text())
        if hist:
            print(f"{'val':>8}: {hist[-1].get('val_loss', float('nan')):.3f} at step {hist[-1]['step']}")
    return 0


def cmd_chat(a) -> int:
    from .generate import Mind
    m = Mind(a.run, a.device)
    print(f"{m.name}: {m.info()['params']/1e6:.2f}M params. Type a message; empty line to quit.")
    turns: list[dict] = [{"role": "system", "content": a.system}] if a.system else []
    while True:
        try:
            u = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            break
        turns.append({"role": "user", "content": u})
        r = m.reply(turns, max_new_tokens=a.max_tokens, temperature=a.temperature)
        if r["tool_calls"]:
            for c in r["tool_calls"]:
                print(f"  [tool call] {c['name']} {json.dumps(c['args'])}")
        if r["text"]:
            print(r["text"])
        turns.append({"role": "assistant", "content": r["text"], "tool_calls": r["tool_calls"]})
        if not a.keep_history:
            turns = turns[:1] if a.system else []
    return 0


def cmd_serve(a) -> int:
    from .generate import Mind
    from .server import Server
    m = Mind(a.run, a.device)
    Server(m, a.host, a.port, temperature=a.temperature, max_new_tokens=a.max_tokens).serve()
    return 0


def cmd_eval(a) -> int:
    """How often does the model pick the right tool on held-out demo tasks? A quick sanity score."""
    from .generate import Mind
    m = Mind(a.run, a.device)
    convs = demo_conversations(a.n, seed=12345)
    right = total = 0
    for turns in convs:
        prompt, answer = turns[:-1], turns[-1]
        want = [c["name"] for c in answer.get("tool_calls") or []]
        if not want:
            continue
        r = m.reply(prompt, max_new_tokens=96, temperature=0.0)
        got = [c["name"] for c in r["tool_calls"]]
        total += 1
        right += int(got[:1] == want[:1])
    print(f"first tool call correct on {right}/{total} held-out demo tasks ({100*right/max(1,total):.0f}%)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="saanimind", description="Train and serve your own small language model.")
    p.add_argument("--version", action="version", version=f"saanimind {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="build a dataset from Saani traces, text files and/or the demo set")
    s.add_argument("--traces", help="folder of Saani trace files (default ~/.saani/traces)")
    s.add_argument("--text", action="append", help="folder of .txt/.md files (repeatable)")
    s.add_argument("--demo", action="store_true", help="add the synthetic demo set")
    s.add_argument("--demo-size", type=int, default=4000)
    s.add_argument("--out", default="data")
    s.add_argument("--vocab-size", type=int, default=4096)
    s.add_argument("--block-size", type=int, default=512)
    s.add_argument("--val-fraction", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--size-hint", default="tiny", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train", help="train a model on a prepared dataset")
    s.add_argument("--data", default="data")
    s.add_argument("--out", default="runs/tiny")
    s.add_argument("--size", choices=list(PRESETS), default="tiny")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--minutes", type=float, help="stop after this many minutes even if steps remain")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--lr", type=float, default=6e-4)
    s.add_argument("--warmup", type=int, default=100)
    s.add_argument("--eval-every", type=int, default=200)
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--block-size", type=int, help="override the preset context length")
    s.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_train)

    for name, fn, help_ in (("info", cmd_info, "describe a trained model"), ("chat", cmd_chat, "talk to a trained model"),
                            ("serve", cmd_serve, "serve a model to Saani over the OpenAI chat API"),
                            ("eval", cmd_eval, "score tool choice on held-out demo tasks")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("run", help="run folder, e.g. runs/tiny")
        s.add_argument("--device", default="auto")
        if name == "chat":
            s.add_argument("--system", default="You are Saani, a careful assistant.")
            s.add_argument("--max-tokens", type=int, default=200)
            s.add_argument("--temperature", type=float, default=0.7)
            s.add_argument("--keep-history", action="store_true")
        if name == "serve":
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=8900)
            s.add_argument("--max-tokens", type=int, default=400)
            s.add_argument("--temperature", type=float, default=0.5)
        if name == "eval":
            s.add_argument("--n", type=int, default=300)
        s.set_defaults(fn=fn)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

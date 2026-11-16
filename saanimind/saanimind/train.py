"""Training loop: AdamW, warm-up + cosine decay, gradient clipping, checkpoints, resume."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import GPT, ModelConfig


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Batches:
    """Random windows from the packed token stream."""

    def __init__(self, ids: np.ndarray, mask: np.ndarray, block: int, batch: int, device, seed: int = 0):
        self.ids, self.mask, self.block, self.batch, self.device = ids, mask, block, batch, device
        self.rng = np.random.default_rng(seed)
        if len(ids) < block + 2:
            raise ValueError(f"only {len(ids)} tokens of data; need more than block_size {block}")

    def next(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.ids) - self.block - 1, size=self.batch)
        x = np.stack([self.ids[s:s + self.block] for s in starts]).astype(np.int64)
        y = np.stack([self.ids[s + 1:s + 1 + self.block] for s in starts]).astype(np.int64)
        m = np.stack([self.mask[s + 1:s + 1 + self.block] for s in starts]).astype(np.int64)
        t = lambda a: torch.from_numpy(a).to(self.device)
        return t(x), t(y), t(m)


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return max_lr * min_ratio
    p = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def save_checkpoint(path: Path, model: GPT, opt, step: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "step": step,
                "optimizer": opt.state_dict() if opt is not None else None, **(extra or {})}, path)


def load_model(run_dir: str | Path, device=None) -> GPT:
    ck = torch.load(Path(run_dir) / "model.pt", map_location=device or "cpu", weights_only=False)
    model = GPT(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(device or "cpu")


def train(data_dir: str | Path, run_dir: str | Path, cfg: ModelConfig, *, steps: int = 2000, batch: int = 32,
          lr: float = 6e-4, warmup: int = 100, minutes: float | None = None, eval_every: int = 200,
          device: str = "auto", resume: bool = False, log=print, seed: int = 0) -> dict:
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    dev = pick_device(device)
    torch.manual_seed(seed)
    ids = np.load(data_dir / "train_ids.npy"); mask = np.load(data_dir / "train_mask.npy")
    vids = np.load(data_dir / "val_ids.npy"); vmask = np.load(data_dir / "val_mask.npy")
    meta = json.loads((data_dir / "meta.json").read_text())
    cfg.vocab_size = meta["vocab_size"]
    model = GPT(cfg).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    step0 = 0
    if resume and (run_dir / "model.pt").exists():
        ck = torch.load(run_dir / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); step0 = ck.get("step", 0)
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        log(f"resumed from step {step0}")
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_b = Batches(ids, mask, cfg.block_size, batch, dev, seed)
    val_b = Batches(vids, vmask, cfg.block_size, min(batch, 16), dev, seed + 1)
    log(f"model: {model.n_params()/1e6:.2f}M params, device {dev}, data {len(ids):,} tokens "
        f"({meta.get('n_conversations', '?')} conversations), vocab {cfg.vocab_size}")
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    t0 = time.time()
    best = float("inf")
    history: list[dict] = []
    step = step0
    try:
        while step < steps:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr, warmup, steps)
            x, y, m = train_b.next()
            model.train()
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            elapsed = (time.time() - t0) / 60
            if step % 20 == 0 or step == 1:
                log(f"step {step}/{steps}  loss {loss.item():.3f}  lr {opt.param_groups[0]['lr']:.2e}  {elapsed:.1f} min")
            if step % eval_every == 0 or step == steps:
                vl = evaluate(model, val_b, 10)
                history.append({"step": step, "train_loss": loss.item(), "val_loss": vl, "minutes": elapsed})
                log(f"  val loss {vl:.3f}")
                if vl < best:
                    best = vl
                    save_checkpoint(run_dir / "model.pt", model, opt, step, {"val_loss": vl})
                    log("  saved (best so far)")
            if minutes is not None and elapsed >= minutes:
                log(f"time budget of {minutes} min reached at step {step}")
                break
    except KeyboardInterrupt:
        log("interrupted; saving where we are")
    if not (run_dir / "model.pt").exists() or history and history[-1]["step"] != step:
        vl = evaluate(model, val_b, 10)
        if vl <= best or not (run_dir / "model.pt").exists():
            save_checkpoint(run_dir / "model.pt", model, opt, step, {"val_loss": vl})
        history.append({"step": step, "val_loss": vl, "minutes": (time.time() - t0) / 60})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"steps": step, "best_val_loss": best if best < float("inf") else history[-1]["val_loss"],
            "params": model.n_params(), "minutes": (time.time() - t0) / 60}


@torch.no_grad()
def evaluate(model: GPT, batches: Batches, n: int) -> float:
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y, m = batches.next()
        _, loss = model(x, y, m)
        tot += loss.item()
    model.train()
    return tot / n

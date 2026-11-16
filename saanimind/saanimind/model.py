"""A decoder-only transformer (the GPT family), written from scratch in PyTorch.

Pieces: token embedding -> N blocks of [RMSNorm, causal self-attention with rotary
positions, RMSNorm, SwiGLU feed-forward] -> RMSNorm -> tied output projection.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256       # longest sequence the model has seen in training
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Rough parameter counts (with a 4k vocabulary): nano 0.6M, tiny 6M, small 30M, base 110M.
PRESETS: dict[str, dict] = {
    "nano":  dict(n_layer=3, n_head=4, n_embd=96,  block_size=256),
    "tiny":  dict(n_layer=6, n_head=6, n_embd=256, block_size=512),
    "small": dict(n_layer=10, n_head=8, n_embd=512, block_size=1024),
    "base":  dict(n_layer=16, n_head=12, n_embd=768, block_size=2048),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); rotate pairs (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :, :].to(x.dtype), sin[None, None, :, :].to(x.dtype)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide by n_head"
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "head size must be even for rotary positions"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class FeedForward(nn.Module):
    """SwiGLU: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.n_embd), Attention(cfg)
        self.n2, self.ff = RMSNorm(cfg.n_embd), FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ff(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight             # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():        # scale residual projections by depth
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._rope: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_for(self, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (T, str(device))
        if key not in self._rope:
            cos, sin = rope_cache(T, self.cfg.n_embd // self.cfg.n_head, device)
            self._rope = {key: (cos, sin)}
        return self._rope[key]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence of {T} tokens is longer than block_size {self.cfg.block_size}")
        cos, sin = self._rope_for(T, idx.device)
        x = self.drop(self.tok(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                      ignore_index=-1, reduction="none").view(B, T)
            if loss_mask is not None:
                m = loss_mask.float()
                loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
            else:
                valid = (targets != -1).float()
                loss = (per_tok * valid).sum() / valid.sum().clamp(min=1.0)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_p: float = 0.95, repetition_penalty: float = 1.1, stop_ids: set[int] | None = None):
        """Sample tokens one at a time. Yields each new token id."""
        stop_ids = stop_ids or set()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                seen = torch.unique(ctx[0])
                pen = logits[0, seen]
                logits[0, seen] = torch.where(pen > 0, pen / repetition_penalty, pen * repetition_penalty)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if 0 < top_p < 1:
                    sp, si = probs.sort(descending=True)
                    keep = (sp.cumsum(-1) - sp) < top_p
                    sp = sp * keep
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
            tok = int(nxt[0, 0])
            yield tok
            if tok in stop_ids:
                break

"""Byte-level BPE tokenizer, written from scratch (no external tokenizer library).

Every byte is a base token (256 of them), so any text can be encoded and decoded
exactly. Training learns merges of frequent adjacent pairs, the same idea as the
GPT-2 tokenizer. A handful of special tokens mark the parts of a conversation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Conversation markers. They are never produced by byte merging, so text cannot forge them.
SPECIAL = ["<|pad|>", "<|sys|>", "<|usr|>", "<|ast|>", "<|tool|>", "<|call|>", "<|end|>", "<|eos|>"]

# Pre-split so merges never cross a word/space boundary in odd ways (simplified GPT-2 pattern).
_PRE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+""")


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.special = {tok: i for i, tok in enumerate(SPECIAL)}
        self.n_special = len(SPECIAL)
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self._cache: dict[str, list[int]] = {}
        self._rebuild()

    # ----------------------------------------------------------------- basics
    @property
    def vocab_size(self) -> int:
        return self.n_special + 256 + len(self.merges)

    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {self.n_special + b: bytes([b]) for b in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[self.n_special + 256 + i] = self.vocab[a] + self.vocab[b]
        self._cache.clear()

    def token(self, name: str) -> int:
        return self.special[name]

    # --------------------------------------------------------------- training
    def train(self, texts: list[str], vocab_size: int, verbose: bool = False) -> None:
        """Learn merges until the vocabulary reaches vocab_size."""
        target_merges = vocab_size - self.n_special - 256
        if target_merges <= 0:
            raise ValueError("vocab_size must be larger than %d" % (self.n_special + 256))
        words: Counter[tuple[int, ...]] = Counter()
        for t in texts:
            for w in _PRE.findall(t):
                words[tuple(self.n_special + b for b in w.encode("utf-8"))] += 1
        self.merges = []
        seq: dict[tuple[int, ...], list[int]] = {w: list(w) for w in words}
        while len(self.merges) < target_merges:
            pairs: Counter[tuple[int, int]] = Counter()
            for w, ids in seq.items():
                f = words[w]
                for a, b in zip(ids, ids[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:
                break
            new_id = self.n_special + 256 + len(self.merges)
            self.merges.append(best)
            for w, ids in seq.items():
                if len(ids) < 2:
                    continue
                seq[w] = _merge(ids, best, new_id)
            if verbose and len(self.merges) % 200 == 0:
                print(f"  merges: {len(self.merges)}/{target_merges}")
        self._rebuild()

    # --------------------------------------------------------------- encoding
    def _encode_word(self, w: str) -> list[int]:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        ids = [self.n_special + b for b in w.encode("utf-8")]
        while len(ids) > 1:
            best = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            ids = _merge(ids, best, self.n_special + 256 + self.ranks[best])
        self._cache[w] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for w in _PRE.findall(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        parts: list[str] = []
        for i in ids:
            if i < self.n_special:
                if buf:
                    parts.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
                parts.append(SPECIAL[i])
            else:
                buf += self.vocab.get(i, b"")
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    # ------------------------------------------------------------------ files
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"special": SPECIAL, "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("special") != SPECIAL:
            raise ValueError("tokenizer file was made with a different special-token set")
        return cls([tuple(m) for m in d["merges"]])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

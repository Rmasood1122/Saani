"""Small text helpers shared across the system (no dependencies)."""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set(
    "a an the and or but if then else of to in on at by for with from into over under is are was "
    "were be been being it its this that these those i me my we our you your he she they them "
    "their do does did done can could should would will shall may might must not no yes as so "
    "than too very just also about up down out off again please need want some any all".split()
)


def stem(w: str) -> str:
    """Very light stemming so 'invoices'/'invoice' and 'deploying'/'deploy' meet."""
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if w in STOP or len(w) < 2:
            continue
        out.append(stem(w))
    return out


def cosine(a: str, b: str) -> float:
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb)


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:max_len].strip("-")) or "item"


def clip(text: str, limit: int) -> str:
    """Keep head and tail of long text so both the start and the end survive."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} characters omitted]...\n{text[-tail:]}"


def first_sentences(text: str, n: int = 2, max_chars: int = 320) -> str:
    flat = re.sub(r"\s+", " ", re.sub(r"[#*`_>|-]{1,}", " ", text or "")).strip()
    parts = re.split(r"(?<=[.!?])\s+", flat)
    return " ".join(parts[:n])[:max_chars]

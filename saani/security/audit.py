"""Append-only, hash-chained audit log of every action an agent attempted."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


def _digest(prev: str, body: dict) -> str:
    blob = prev + json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._prev = "genesis"
        if self.path.exists():
            last = None
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = line
            if last:
                try:
                    self._prev = json.loads(last)["hash"]
                except (ValueError, KeyError):
                    self._prev = "corrupt"

    def record(self, **fields) -> dict:
        with self._lock:
            body = {"ts": round(time.time(), 3), **fields}
            entry = {**body, "prev": self._prev, "hash": _digest(self._prev, body)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._prev = entry["hash"]
            return entry

    def entries(self, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        out.append({"corrupt": line[:200]})
        return out[-limit:] if limit else out

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain. Returns (ok, message)."""
        prev = "genesis"
        n = 0
        for n, e in enumerate(self.entries(), 1):
            if "corrupt" in e:
                return False, f"entry {n} is not valid JSON"
            body = {k: v for k, v in e.items() if k not in ("prev", "hash")}
            if e.get("prev") != prev or e.get("hash") != _digest(prev, body):
                return False, f"chain broken at entry {n}: the log was edited or truncated"
            prev = e["hash"]
        return True, f"{n} entries, chain intact"

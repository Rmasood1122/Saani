"""A tiny in-process event bus so the CLI, dashboard and voice can follow what agents do."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class EventBus:
    def __init__(self, keep: int = 3000):
        self._buf: deque = deque(maxlen=keep)
        self._subs: list[Callable[[dict], None]] = []
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, kind: str, **data) -> dict:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "ts": round(time.time(), 2), "kind": kind, **data}
            self._buf.append(ev)
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(ev)
            except Exception:
                pass  # a broken listener must never break an agent
        return ev

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        with self._lock:
            self._subs.append(fn)

    def since(self, seq: int = 0, limit: int = 300) -> list[dict]:
        with self._lock:
            return [e for e in self._buf if e["seq"] > seq][-limit:]

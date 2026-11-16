"""Recurring tasks: 'every N minutes', 'daily at HH:MM', 'weekdays at HH:MM'."""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta
from typing import Callable


def parse_spec(kind: str, spec: str) -> None:
    if kind == "every":
        if not str(spec).isdigit() or int(spec) < 1:
            raise ValueError("'every' needs a whole number of minutes, e.g. --every 60")
    elif kind in ("daily", "weekdays"):
        m = re.match(r"^(\d{1,2}):(\d{2})$", str(spec))
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            raise ValueError("time must look like 09:00 (24-hour clock)")
    else:
        raise ValueError("kind must be every, daily or weekdays")


def next_run(kind: str, spec: str, now: float) -> float:
    parse_spec(kind, spec)
    if kind == "every":
        return now + int(spec) * 60
    h, m = map(int, spec.split(":"))
    base = datetime.fromtimestamp(now)
    cand = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if cand <= base:
        cand += timedelta(days=1)
    while kind == "weekdays" and cand.weekday() >= 5:
        cand += timedelta(days=1)
    return cand.timestamp()


class Scheduler(threading.Thread):
    def __init__(self, store, submit: Callable[[str, str], object], poll_s: float = 20.0):
        super().__init__(daemon=True, name="saani-scheduler")
        self.store, self.submit, self.poll_s = store, submit, poll_s
        self.stop_event = threading.Event()

    def tick(self, now: float | None = None) -> list[str]:
        """Run everything that is due. Returns the names that fired."""
        now = time.time() if now is None else now
        fired = []
        for s in self.store.schedule_list():
            if not s["enabled"] or s["next_run"] > now:
                continue
            # schedule the next run first, so a crash can never cause a tight re-run loop;
            # runs missed while the computer was off are skipped, not replayed in a burst
            self.store.schedule_mark(s["id"], now, next_run(s["kind"], s["spec"], now))
            try:
                self.submit(s["task"], s["name"])
                fired.append(s["name"])
            except Exception:
                pass
        return fired

    def run(self) -> None:
        while not self.stop_event.wait(self.poll_s):
            try:
                self.tick()
            except Exception:
                pass

    def stop(self) -> None:
        self.stop_event.set()

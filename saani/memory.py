"""Long-term memory: tasks, searchable memories, schedules (SQLite, stdlib only)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .textutil import cosine, tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, text TEXT NOT NULL,
  meta TEXT DEFAULT '{}', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT DEFAULT 'cli', status TEXT DEFAULT 'running',
  created REAL NOT NULL, finished REAL, result TEXT DEFAULT '', plan TEXT DEFAULT '', stats TEXT DEFAULT '{}',
  workdir TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS schedules(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, task TEXT NOT NULL, kind TEXT NOT NULL,
  spec TEXT NOT NULL, enabled INTEGER DEFAULT 1, last_run REAL DEFAULT 0, next_run REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            try:
                self._db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(text)")
                self.fts = True
            except sqlite3.OperationalError:
                self.fts = False

    def mark_interrupted(self) -> int:
        """Called when the daemon starts: tasks left 'running' by a crash or power-off are not running."""
        with self._lock:
            return self._db.execute("UPDATE tasks SET status='interrupted', finished=? WHERE status='running'",
                                    (time.time(),)).rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- memories --------------------------------------------------------
    def add_memory(self, kind: str, text: str, meta: dict | None = None) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO memories(kind,text,meta,created) VALUES(?,?,?,?)",
                                   (kind, text, json.dumps(meta or {}), time.time()))
            rid = int(cur.lastrowid)
            if self.fts:
                self._db.execute("INSERT INTO mem_fts(rowid,text) VALUES(?,?)", (rid, text))
            return rid

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        try:
            d["meta"] = json.loads(d.get("meta") or "{}")
        except ValueError:
            d["meta"] = {}
        return d

    def search(self, query: str, kinds: list[str] | None = None, limit: int = 5) -> list[dict]:
        toks = list(dict.fromkeys(tokens(query)))[:16]
        if not toks:
            return []
        kind_sql = f" AND m.kind IN ({','.join('?' * len(kinds))})" if kinds else ""
        with self._lock:
            if self.fts:
                # prefix match so the light stemming in tokens() still finds full words
                match = " OR ".join(f'"{t}"*' for t in toks)
                try:
                    rows = self._db.execute(
                        "SELECT m.* FROM mem_fts f JOIN memories m ON m.id=f.rowid "
                        f"WHERE mem_fts MATCH ?{kind_sql} ORDER BY bm25(mem_fts) LIMIT ?",
                        (match, *(kinds or []), limit)).fetchall()
                    return [self._row(r) for r in rows]
                except sqlite3.OperationalError:
                    pass
            rows = self._db.execute(
                f"SELECT m.* FROM memories m WHERE 1=1{kind_sql} ORDER BY id DESC LIMIT 800",
                tuple(kinds or [])).fetchall()
        scored = sorted(((cosine(query, r["text"]), r) for r in rows), key=lambda x: -x[0])
        return [self._row(r) for s, r in scored[:limit] if s > 0]

    def recent(self, kind: str, limit: int = 10) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM memories WHERE kind=? ORDER BY id DESC LIMIT ?",
                                    (kind, limit)).fetchall()
        return [self._row(r) for r in rows]

    def delete_memory(self, mem_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            if self.fts:
                self._db.execute("DELETE FROM mem_fts WHERE rowid=?", (mem_id,))

    # ---- tasks -----------------------------------------------------------
    def task_create(self, task_id: str, text: str, source: str = "cli", workdir: str = "") -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO tasks(id,text,source,status,created,workdir) VALUES(?,?,?,?,?,?)",
                             (task_id, text, source, "running", time.time(), workdir))

    def task_update(self, task_id: str, **fields) -> None:
        if not fields:
            return
        for k in ("plan", "stats"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k], default=str)
        cols = ",".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), task_id))

    def task_get(self, task_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(r) if r else None

    def task_list(self, limit: int = 30) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM tasks ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- schedules -------------------------------------------------------
    def schedule_add(self, name: str, task: str, kind: str, spec: str, next_run: float) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO schedules(name,task,kind,spec,next_run) VALUES(?,?,?,?,?)",
                                   (name, task, kind, spec, next_run))
            return int(cur.lastrowid)

    def schedule_list(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM schedules ORDER BY id").fetchall()]

    def schedule_remove(self, sid: int) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM schedules WHERE id=?", (sid,)).rowcount > 0

    def schedule_mark(self, sid: int, last_run: float, next_run: float) -> None:
        with self._lock:
            self._db.execute("UPDATE schedules SET last_run=?, next_run=? WHERE id=?", (last_run, next_run, sid))

    # ---- kv ----------------------------------------------------------------
    def kv_get(self, k: str, default: str | None = None) -> str | None:
        with self._lock:
            r = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r["v"] if r else default

    def kv_set(self, k: str, v: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (k, v))

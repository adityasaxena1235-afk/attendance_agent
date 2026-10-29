"""Local SQLite storage. Only derived metrics are stored: no frames, no audio, no faces."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    started_at TEXT,
    duration_s REAL,
    source TEXT,
    summary_json TEXT
);
"""


class Storage:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_session(self, session_id: str, name: str, started_at: datetime,
                     duration_s: float, source: str, report: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?)",
                      (session_id, name, started_at.isoformat(timespec="seconds"),
                       duration_s, source, json.dumps(report)))

    def list_sessions(self) -> List[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id,name,started_at,duration_s,source,summary_json "
                             "FROM sessions ORDER BY started_at DESC").fetchall()
        out = []
        for r in rows:
            rep = json.loads(r["summary_json"])
            out.append({
                "id": r["id"], "name": r["name"], "started_at": r["started_at"],
                "duration_s": r["duration_s"], "source": r["source"],
                "attendees": len(rep.get("attendance", [])),
                "mean_engagement": rep.get("trends", {}).get("mean"),
            })
        return out

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if r is None:
            return None
        rep = json.loads(r["summary_json"])
        rep.update({"id": r["id"], "name": r["name"], "started_at": r["started_at"],
                    "duration_s": r["duration_s"], "source": r["source"]})
        return rep

    def delete_session(self, session_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount > 0

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE started_at < ?", (cutoff,)).rowcount

"""Access log: records every HTTP request, WebSocket connection, and SocketIO
action, tagged by whether it came from the web interface or the API.

Extracted from speech_to_text.py so it can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only; the SQLite path
and limits are passed in explicitly, and a thin wrapper in speech_to_text.py
supplies the live values.

The store is a single SQLite table capped to ``max_rows`` — inserts prune the
oldest rows once the table grows past the cap, so the log never grows
unbounded. All writes go through one connection guarded by a lock, which is
fine for the request volume of a single-server event operator tool.
"""

import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# Traffic origin, derived from the request path / transport.
SOURCE_WEB = "web"        # a page route or its assets (served to a browser UI)
SOURCE_API = "api"        # a programmatic /api/... endpoint
SOURCE_SOCKET = "socket"  # a WebSocket connection or SocketIO event

# What kind of event a row records.
KIND_HTTP = "http"              # an HTTP request/response
KIND_CONNECTION = "connection"  # a WebSocket connect/disconnect
KIND_ACTION = "action"          # a named SocketIO event ("action")

# Columns returned by query(), in insertion order — kept as a constant so the
# route and template agree on the shape without duplicating the list.
FIELDS = (
    "id", "ts", "source", "kind", "method", "path",
    "status", "ip", "user_agent", "duration_ms", "detail",
)


def classify_source(path: str, *, api_prefix: str = "/api/") -> str:
    """Tag an HTTP request path as web-interface or API traffic.

    Anything under ``api_prefix`` is API; everything else that reaches a page
    route (or its assets) is treated as web-interface traffic. SocketIO
    transport is tagged separately by the caller via :data:`SOURCE_SOCKET`.
    """
    if path.startswith(api_prefix):
        return SOURCE_API
    return SOURCE_WEB


class RequestLog:
    """SQLite-backed, size-capped access log."""

    def __init__(self, db_path: str, *, max_rows: int = 50_000, prune_every: int = 200) -> None:
        """Open (creating if needed) the log database at ``db_path``.

        ``max_rows`` is the hard cap on retained rows; ``prune_every`` is how
        many inserts to allow between prune sweeps (pruning on every insert
        would double the write cost for no benefit).
        """
        self.db_path = db_path
        self.max_rows = max(1, int(max_rows))
        self.prune_every = max(1, int(prune_every))
        self._lock = threading.Lock()
        self._since_prune = 0
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                method TEXT,
                path TEXT,
                status INTEGER,
                ip TEXT,
                user_agent TEXT,
                duration_ms REAL,
                detail TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_id ON access_log(id)")
        self._conn.commit()

    def log(
        self,
        *,
        source: str,
        kind: str,
        method: Optional[str] = None,
        path: Optional[str] = None,
        status: Optional[int] = None,
        ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        duration_ms: Optional[float] = None,
        detail: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Record one event. Never raises — logging must not break a response."""
        try:
            row_ts = time.time() if ts is None else ts
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO access_log
                        (ts, source, kind, method, path, status, ip, user_agent, duration_ms, detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row_ts, source, kind, method, path, status, ip, user_agent, duration_ms, detail),
                )
                self._since_prune += 1
                if self._since_prune >= self.prune_every:
                    self._prune_locked()
                    self._since_prune = 0
                self._conn.commit()
        except Exception:
            # Best-effort: a logging failure must never propagate into the
            # request/socket handler it is instrumenting.
            pass

    def _prune_locked(self) -> None:
        """Delete rows beyond ``max_rows``, keeping the newest. Caller holds the lock."""
        self._conn.execute(
            """
            DELETE FROM access_log
            WHERE id <= (
                SELECT id FROM access_log ORDER BY id DESC LIMIT 1 OFFSET ?
            )
            """,
            (self.max_rows,),
        )

    def query(
        self,
        *,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return the most recent matching rows, newest first, as dicts.

        ``source``/``kind`` are exact-match filters; ``search`` is a
        case-insensitive substring match against path, IP, and detail.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if search:
            like = f"%{search}%"
            clauses.append("(path LIKE ? OR ip LIKE ? OR detail LIKE ?)")
            params.extend([like, like, like])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT {', '.join(FIELDS)} FROM access_log{where} ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        return [dict(zip(FIELDS, row)) for row in rows]

    def count(self) -> int:
        """Total number of rows currently retained."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM access_log")
            return int(cursor.fetchone()[0])

    def clear(self) -> None:
        """Delete every row."""
        with self._lock:
            self._conn.execute("DELETE FROM access_log")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection (mainly for tests)."""
        with self._lock:
            self._conn.close()

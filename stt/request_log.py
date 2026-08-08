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
    "status", "ip", "user_agent", "duration_ms", "detail", "origin",
)

# How the request reached us, as classified by stt.request_origin. Distinct from
# `source` (which describes the path/transport): a row can be api+tunnel or
# web+lan. Kept as a plain column so the logs page can filter on it.
ORIGIN_LOCAL = "local"
ORIGIN_LAN = "lan"
ORIGIN_TUNNEL = "tunnel"


def classify_source(path: str, *, api_prefix: str = "/api/") -> str:
    """Tag an HTTP request path as web-interface or API traffic.

    Anything under ``api_prefix`` is API; everything else that reaches a page
    route (or its assets) is treated as web-interface traffic. SocketIO
    transport is tagged separately by the caller via :data:`SOURCE_SOCKET`.
    """
    if path.startswith(api_prefix):
        return SOURCE_API
    return SOURCE_WEB


def _percentile(sorted_values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile of an already-ascending list, or ``None``.

    ``q`` is a fraction in [0, 1]. Small samples are fine — the nearest-rank
    method avoids interpolation surprises for the handful of requests a single
    operator box sees in a short window.
    """
    if not sorted_values:
        return None
    rank = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return round(float(sorted_values[rank]), 2)


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
        # An access log is diagnostics, not the recording. WAL alone still leaves
        # synchronous at FULL, which fsyncs on every request — and with a display client
        # and the health page polling, a service is thousands of fsyncs on the same disk
        # the session audio is being written to. NORMAL matches what the session database
        # itself uses; the worst case it admits is losing the last few log rows on a power
        # cut, which is not something anyone reconstructs a service from.
        self._conn.execute("PRAGMA synchronous=NORMAL")
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
                detail TEXT,
                origin TEXT
            )
            """
        )
        self._migrate_locked()
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_id ON access_log(id)")
        self._conn.commit()

    def _migrate_locked(self) -> None:
        """Add columns missing from a database created by an older version.

        CREATE TABLE IF NOT EXISTS does nothing to an existing table, so a box
        that has been logging since before a column existed would fail every
        insert. Deployed boxes upgrade themselves in place, so this has to be
        handled here rather than by recreating the database (which would throw
        away the operator's history).
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(access_log)")}
        for column, decl in (("origin", "TEXT"),):
            if column not in existing:
                self._conn.execute(f"ALTER TABLE access_log ADD COLUMN {column} {decl}")

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
        origin: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Record one event. Never raises — logging must not break a response."""
        try:
            row_ts = time.time() if ts is None else ts
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO access_log
                        (ts, source, kind, method, path, status, ip, user_agent, duration_ms, detail, origin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row_ts, source, kind, method, path, status, ip, user_agent, duration_ms, detail, origin),
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
        ip: Optional[str] = None,
        origin: Optional[str] = None,
        search: Optional[str] = None,
        exclude_paths: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return the most recent matching rows, newest first, as dicts.

        ``source``/``kind``/``ip``/``origin`` are exact-match filters; ``search`` is a
        case-insensitive substring match against path, IP, and detail;
        ``exclude_paths`` drops rows whose path exactly equals any of the given
        paths (used to hide high-frequency dashboard polling from the view).
        """
        clauses: List[str] = []
        params: List[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if ip:
            clauses.append("ip = ?")
            params.append(ip)
        if origin:
            clauses.append("origin = ?")
            params.append(origin)
        if search:
            like = f"%{search}%"
            clauses.append("(path LIKE ? OR ip LIKE ? OR detail LIKE ?)")
            params.extend([like, like, like])
        if exclude_paths:
            placeholders = ", ".join("?" for _ in exclude_paths)
            clauses.append(f"(path IS NULL OR path NOT IN ({placeholders}))")
            params.extend(exclude_paths)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT {', '.join(FIELDS)} FROM access_log{where} ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        return [dict(zip(FIELDS, row)) for row in rows]

    def distinct_ips(
        self,
        *,
        exclude_paths: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Every distinct client IP in the log, busiest first.

        Each entry is ``{"ip", "count", "last_ts"}``. ``exclude_paths`` drops
        the same polling noise :meth:`query` hides, so an address that only ever
        polled the dashboard doesn't show up as a client when the viewer is
        hiding polling. Rows with no IP (some socket events) are skipped.
        """
        clauses = ["ip IS NOT NULL", "ip != ''"]
        params: List[Any] = []
        if exclude_paths:
            placeholders = ", ".join("?" for _ in exclude_paths)
            clauses.append(f"(path IS NULL OR path NOT IN ({placeholders}))")
            params.extend(exclude_paths)
        sql = (
            "SELECT ip, COUNT(*) AS n, MAX(ts) AS last_ts FROM access_log "
            f"WHERE {' AND '.join(clauses)} "
            "GROUP BY ip ORDER BY n DESC, last_ts DESC LIMIT ?"
        )
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [{"ip": row[0], "count": int(row[1]), "last_ts": float(row[2])} for row in rows]

    def count(self) -> int:
        """Total number of rows currently retained."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM access_log")
            return int(cursor.fetchone()[0])

    def stats(self, window_seconds: float, *, now: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate request health over the last ``window_seconds``.

        Returns request count, per-minute rate, error counts/rates split into
        server (5xx) and client (4xx) — plus a combined total (>= 400) — and
        p50/p95 response-latency in milliseconds: everything the health
        dashboard needs from the access log in one query. ``now`` is
        injectable for deterministic tests. Rows with a NULL ``duration_ms``
        (e.g. socket connections) are excluded from the latency percentiles but
        still counted as requests.

        An empty window yields zeros / ``None`` percentiles, never an error.
        """
        now_ts = time.time() if now is None else now
        cutoff = now_ts - max(0.0, float(window_seconds))
        with self._lock:
            total = int(self._conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE ts >= ?", (cutoff,),
            ).fetchone()[0])
            errors = int(self._conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE ts >= ? AND status >= 400",
                (cutoff,),
            ).fetchone()[0])
            server_errors = int(self._conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE ts >= ? AND status >= 500",
                (cutoff,),
            ).fetchone()[0])
            durations = [
                row[0] for row in self._conn.execute(
                    "SELECT duration_ms FROM access_log "
                    "WHERE ts >= ? AND duration_ms IS NOT NULL "
                    "ORDER BY duration_ms ASC",
                    (cutoff,),
                ).fetchall()
            ]

        minutes = window_seconds / 60.0 if window_seconds > 0 else 0.0
        client_errors = errors - server_errors
        return {
            "window_seconds": int(window_seconds),
            "requests": total,
            "req_per_min": round(total / minutes, 1) if minutes > 0 else 0.0,
            # error_count / error_rate = all >= 400 (kept for callers that want
            # the total). Health status keys off server_* (5xx) only: a 4xx is a
            # client/auth problem (e.g. a non-whitelisted viewer polling), not a
            # server fault, and must not make the server look unhealthy.
            "error_count": errors,
            "error_rate": round(errors / total, 3) if total else 0.0,
            "server_error_count": server_errors,
            "server_error_rate": round(server_errors / total, 3) if total else 0.0,
            "client_error_count": client_errors,
            "client_error_rate": round(client_errors / total, 3) if total else 0.0,
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
        }

    def clear(self) -> None:
        """Delete every row."""
        with self._lock:
            self._conn.execute("DELETE FROM access_log")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection (mainly for tests)."""
        with self._lock:
            self._conn.close()

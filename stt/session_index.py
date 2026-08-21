"""Which recorded services exist, and which one did the caller mean.

Extracted so the rule that decides whether a request may open a database is a testable
function rather than a few lines inside a route. Stdlib-only; the caller supplies the
enumeration and the live path, so nothing here knows about the config or the filesystem
layout.

The security property lives in :func:`resolve_session`: a caller names a session by
basename, and that name is only ever *compared* against an enumeration the server built
itself. Nothing derived from the request is ever joined onto a directory, so there is no
path to confine and no traversal to sanitise — the class of bug is absent rather than
defended against. Deliberately different from stt/paths.py, which confines a caller-supplied
path because the file manager genuinely needs to accept one.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

# Session filenames come from database.filename_format, "%Y-%m-%d_%H%M%S" by default.
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def session_date(basename: str) -> str:
    """The service's date from its filename, or "" if the name is not dated.

    The same read _service_phase_first_sunday makes: the date is in the name, so a listing
    does not have to open every database to sort or label it.
    """
    m = _DATE_RE.match(os.path.basename(basename or ""))
    return m.group(1) if m else ""


def _same_file(a: str, b: str) -> bool:
    """Whether two paths name the same database, tolerating relative and symlinked forms."""
    if not a or not b:
        return False
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.abspath(a) == os.path.abspath(b)


def resolve_session(paths: Sequence[str], name: Optional[str],
                    live_path: Optional[str]) -> Optional[Tuple[str, bool]]:
    """``(path, is_live)`` for the session ``name``, or None if it is not one of ``paths``.

    An empty ``name`` means the running session, which is the default every page wants; it
    resolves to None when nothing is running, and the caller turns that into its own 404.

    ``is_live`` is what decides whether a write may checkpoint the database afterwards, so
    it is compared by inode where possible: the live path and the swept path can be the same
    file reached two ways, and treating the running session as an archive would mean
    retiring a WAL another process is still writing to.
    """
    wanted = os.path.basename((name or "").strip())
    if not wanted:
        if live_path and os.path.exists(live_path):
            return live_path, True
        return None
    for path in paths:
        if os.path.basename(path) == wanted:
            return path, _same_file(path, live_path or "")
    return None


def _table_exists(conn: "sqlite3.Connection", table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def describe(conn: "sqlite3.Connection") -> Dict[str, object]:
    """Row count, span and feature flags for one session, for the picker.

    Everything is best-effort: the archive holds databases written by older builds, and one
    that predates a table must show up in the list as a service you can still open and
    re-run — not vanish because a probe raised.
    """
    out: Dict[str, object] = {
        "rows": 0, "start_ms": 0, "end_ms": 0, "minutes": 0,
        "has_phase": False, "has_summaries": False,
    }
    try:
        row = conn.execute(
            "SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM transcriptions "
            "WHERE is_final = 1 AND ts_ms IS NOT NULL").fetchone()
    except sqlite3.Error:
        return out
    if not row or not row[0]:
        return out
    start, end = int(row[1] or 0), int(row[2] or 0)
    out["rows"] = int(row[0])
    out["start_ms"], out["end_ms"] = start, end
    out["minutes"] = max(0, round((end - start) / 60000.0))
    out["has_phase"] = _table_exists(conn, "service_phase_blocks")
    out["has_summaries"] = _table_exists(conn, "sermon_summaries")
    return out


def index(entries: Sequence[Tuple[str, Dict[str, object]]], live_path: Optional[str] = None
          ) -> List[Dict[str, object]]:
    """Listing rows from ``(path, describe(conn))`` pairs, dropping empty sessions.

    A session with no finalized rows is a recording that never happened — a start/stop, or a
    process that died before anyone spoke. Listing them buries the real services.
    """
    out: List[Dict[str, object]] = []
    for path, described in entries:
        if not described.get("rows"):
            continue
        out.append({
            "session_id": os.path.basename(path),
            "date": session_date(path),
            "live": _same_file(path, live_path or ""),
            **described,
        })
    return out

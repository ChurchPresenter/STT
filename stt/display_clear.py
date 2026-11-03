"""Clearing every caption screen without touching the recording.

An operator sometimes needs every display — the web viewers, the translation
view, ChurchPresenter — to start fresh in the middle of a service. Rolling to a
new session database (``/api/transcription/reset-session``) does that, but splits
the service into two files. This module is the other way: the session database
gets a **marker row** in its own ``display_clears`` table, and the live screens
show only captions that came after it. Nothing is deleted, edited or denied, so
exports, the File Manager and every other reader of the database still see the
whole service.

"After the clear" is two conditions, not one:

* ``id > after_row_id`` — everything already saved when Clear was pressed.
* ``start_time >= audio_cutoff`` — anything that was *in progress* then. The
  worker holds speech it has heard but not saved in its rolling audio buffer, its
  live hypothesis, and the held-back sentence fragment it prepends to the next
  row (carrying that fragment's original start time). All of it finalises later,
  with a higher id, but with a ``start_time`` before the moment of the clear —
  which is what this condition catches. A sentence straddling the clear is hidden
  whole; that is the intended reading of "nothing from before shows up".

The audio condition only applies to rows finalised within ``CUTOFF_HORIZON_MS``
of the marker. In-progress material always lands well inside that (the rolling
buffer holds at most 45 s, a held fragment at most its ``max_age_seconds``), and
the bound means a database reopened by a fresh transcriber, whose audio clock
restarts at zero, cannot have every later caption hidden for ever. A row with no
``ts_ms`` falls under the cutoff; one with no ``start_time`` never does, so an
untimed caption is never lost.

stdlib only, connections passed in: the monolith owns the paths and locks.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple

TABLE = "display_clears"

# How long after a clear the audio-time cutoff keeps hiding rows. See the module
# docstring: it must comfortably exceed the 45 s rolling buffer plus a held
# fragment's maximum age, and be short enough that a restarted clock recovers.
CUTOFF_HORIZON_MS = 120_000


@dataclass(frozen=True)
class ClearMarker:
    """One press of Clear, as stored in ``display_clears``."""

    id: int
    timestamp: str
    ts_ms: int
    after_row_id: int
    audio_cutoff: float


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create ``display_clears`` if this database has none yet."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT NOT NULL, "
        "ts_ms INTEGER NOT NULL, "
        "after_row_id INTEGER NOT NULL, "
        "audio_cutoff REAL NOT NULL)"
    )


def record_clear(conn: sqlite3.Connection, timestamp: str, ts_ms: int, audio_cutoff: float) -> ClearMarker:
    """Write a marker covering every caption saved so far, and commit it.

    The caller must be the connection that writes captions (the worker's), under
    its write lock, so no caption can land between reading ``MAX(id)`` and the
    insert.
    """
    ensure_table(conn)
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM transcriptions").fetchone()
    after_row_id = int(row[0]) if row else 0
    cutoff = max(0.0, float(audio_cutoff))
    cur = conn.execute(
        f"INSERT INTO {TABLE} (timestamp, ts_ms, after_row_id, audio_cutoff) VALUES (?, ?, ?, ?)",
        (timestamp, int(ts_ms), after_row_id, cutoff),
    )
    conn.commit()
    marker_id = cur.lastrowid
    if marker_id is None:  # pragma: no cover - sqlite always sets it after an INSERT
        raise sqlite3.DatabaseError("display_clears insert returned no row id")
    return ClearMarker(int(marker_id), timestamp, int(ts_ms), after_row_id, cutoff)


def latest_clear(conn: sqlite3.Connection) -> Optional[ClearMarker]:
    """The most recent marker, or None when the screens were never cleared.

    A database without the table (every session before this feature, the demo's
    recording) simply has no marker.
    """
    try:
        row = conn.execute(
            f"SELECT id, timestamp, ts_ms, after_row_id, audio_cutoff FROM {TABLE} ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return ClearMarker(int(row[0]), str(row[1]), int(row[2]), int(row[3]), float(row[4]))


def sql_filter(marker: Optional[ClearMarker]) -> Tuple[str, Tuple[float, ...]]:
    """An ``AND ...`` clause (and its parameters) keeping only post-clear rows.

    Written against the raw ``transcriptions`` columns ``id``, ``start_time`` and
    ``ts_ms``. Empty when there is no marker.
    """
    if marker is None:
        return "", ()
    clause = (
        " AND id > ?"
        " AND NOT (start_time IS NOT NULL AND start_time < ?"
        " AND (ts_ms IS NULL OR ts_ms <= ?))"
    )
    return clause, (marker.after_row_id, marker.audio_cutoff, marker.ts_ms + CUTOFF_HORIZON_MS)


def is_visible(row_id: int, start_time: Optional[float], ts_ms: Optional[int], marker: Optional[ClearMarker]) -> bool:
    """Whether a caption row belongs on screen after ``marker``; mirrors :func:`sql_filter`."""
    if marker is None:
        return True
    if row_id <= marker.after_row_id:
        return False
    if start_time is None or start_time >= marker.audio_cutoff:
        return True
    return ts_ms is not None and ts_ms > marker.ts_ms + CUTOFF_HORIZON_MS


def display_epoch(marker: Optional[ClearMarker]) -> int:
    """What clients compare to notice a clear: the marker's id, 0 when none."""
    return marker.id if marker is not None else 0

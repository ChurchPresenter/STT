"""Editing the rows of a session database: what may change, and what it costs.

A session database is the record of a service — captions, their translations and
their timings. Until now nothing could edit one except the live correction path,
which only ever touches the session being recorded right now
(``transcription_state["db_name"]``). Correcting last week's service, or writing
the script the demo replays, meant opening SQLite by hand.

This module is the decision layer for doing it from the File Manager. It is
stdlib-only, takes the database as a connection, and reads no globals — the
monolith supplies the path confinement (``safe_managed_path``) and the answer to
"is this the database being written right now".

Two rules it exists to enforce:

* **Only the columns a person can meaningfully correct.** A caption's text, its
  translation and its timings. Not ``id``, not ``segment_id``, not ``words_json``
  — a hand-edited word-timing blob desynchronises the live preview from the text
  it is highlighting, and nothing in the UI could show you that you had done it.
* **An edit that is not reversible is a decision, not a default.** The caller says
  whether to take a backup first; this module names it and makes it, and refuses
  outright to touch the database the recorder currently holds.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The table every session database keeps its captions in.
TABLE = "transcriptions"

# Column -> how to read a submitted value. Everything absent from this map is
# rejected: see the module docstring for why the list is this short.
EDITABLE: Dict[str, str] = {
    "text": "text",
    "translated_text": "text",
    "start_time": "number",
    "end_time": "number",
    "ts_ms": "integer",
}

# Columns a newly inserted row needs beyond the edited ones, so it looks like a
# row the recorder wrote rather than a half-built one the readers then skip.
# load_schedule and the export path both filter on is_final/denied.
INSERT_DEFAULTS: Dict[str, Any] = {
    "is_final": 1,
    "denied": 0,
    "needs_review": 0,
    "confidence": 1.0,
}

VALID_OPS = ("update", "insert", "delete")


class EditError(ValueError):
    """A submitted edit that must be refused, with a message meant for the operator."""


# --- reading what was submitted --------------------------------------------


def coerce(column: str, value: Any) -> Any:
    """A submitted cell as the column's type, or raise :class:`EditError`.

    Empty means NULL for a translation (a caption with no translation yet is a real
    state) and for timings, but never for ``text``: a row whose text is blank is
    invisible to every reader, so blanking it is a delete pretending to be an edit.
    """
    kind = EDITABLE.get(column)
    if kind is None:
        raise EditError(f"{column!r} is not an editable column.")
    if kind == "text":
        text = "" if value is None else str(value)
        if column == "text" and not text.strip():
            raise EditError("A caption cannot be empty — delete the row instead.")
        return text if text != "" else None
    raw = "" if value is None else str(value).strip()
    if raw == "":
        return None
    try:
        number = float(raw)
    except ValueError:
        raise EditError(f"{column} must be a number, not {raw!r}.") from None
    if kind == "integer":
        return int(number)
    return number


def parse_edit(raw: Mapping[str, Any]) -> Tuple[str, Optional[int], Dict[str, Any], Optional[int]]:
    """One submitted edit as ``(op, row_id, values, after_id)``."""
    op = str(raw.get("op", "")).strip().lower()
    if op not in VALID_OPS:
        raise EditError(f"Unknown edit {op!r}.")

    row_id: Optional[int] = None
    if op in ("update", "delete"):
        try:
            row_id = int(raw.get("id"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise EditError(f"An {op} needs the row's id.") from None

    after_id: Optional[int] = None
    if op == "insert" and raw.get("after_id") not in (None, ""):
        try:
            after_id = int(raw["after_id"])
        except (TypeError, ValueError):
            raise EditError("after_id must be a row id.") from None

    values: Dict[str, Any] = {}
    if op != "delete":
        submitted = raw.get("values") or {}
        if not isinstance(submitted, Mapping):
            raise EditError("values must be an object of column -> value.")
        for column, value in submitted.items():
            values[str(column)] = coerce(str(column), value)
        if not values:
            raise EditError("Nothing to change.")
    _check_timings(values)
    return op, row_id, values, after_id


def _check_timings(values: Mapping[str, Any]) -> None:
    """Refuse a segment that ends before it starts.

    Both have to be present in the same edit to compare them; a lone end_time moved
    behind a start_time already in the row is caught on the way in by the UI showing
    the row's other value, and is not worth a read here.
    """
    start, end = values.get("start_time"), values.get("end_time")
    if start is not None and end is not None and end < start:
        raise EditError("A segment cannot end before it starts.")


def parse_edits(raw: Any) -> List[Tuple[str, Optional[int], Dict[str, Any], Optional[int]]]:
    """Every submitted edit, in order, refusing the batch if any one of them is bad."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise EditError("edits must be a list.")
    if not raw:
        raise EditError("No edits were submitted.")
    parsed = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise EditError("Each edit must be an object.")
        parsed.append(parse_edit(item))
    return parsed


# --- what an edit must not touch -------------------------------------------


def is_live_database(path: str, live_db_name: Optional[str]) -> bool:
    """Whether ``path`` is the database the recorder is writing right now.

    Compared by realpath: the file manager hands back a resolved path and the
    recorder's name may be relative or reached through a symlink, and a mismatch
    here would let an edit land in the middle of a service.
    """
    if not live_db_name:
        return False
    try:
        return os.path.realpath(path) == os.path.realpath(live_db_name)
    except OSError:  # pragma: no cover - realpath on a vanished path
        return False


# --- keeping the original ---------------------------------------------------


def backup_path(db_path: str, now: Optional[float] = None) -> str:
    """Where a pre-edit copy of ``db_path`` goes.

    Keeps the ``.db`` extension so the copy is previewable in the same file
    manager — a backup you cannot open to check is not much of a backup — and
    stamps it so a second edit does not overwrite the first one's original.
    """
    stem, ext = os.path.splitext(db_path)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    return f"{stem}.before-{stamp}{ext or '.db'}"


def make_backup(db_path: str, now: Optional[float] = None) -> str:
    """Copy ``db_path`` aside and return the copy's path."""
    destination = backup_path(db_path, now)
    shutil.copyfile(db_path, destination)
    return destination


# --- applying ---------------------------------------------------------------


def _columns(conn: sqlite3.Connection) -> List[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{TABLE}")')]


def next_ts_ms(conn: sqlite3.Connection, after_id: Optional[int]) -> int:
    """When an inserted row should appear.

    Midway between the row it follows and the next one, so a line added in the
    middle of a service replays in the middle of it. Appended to the end when there
    is no next row, and at the very start when it follows nothing.
    """
    if after_id is None:
        first = conn.execute(
            f'SELECT MIN(ts_ms) FROM "{TABLE}" WHERE ts_ms IS NOT NULL').fetchone()[0]
        return int(first) - 1000 if first is not None else int(time.time() * 1000)
    row = conn.execute(
        f'SELECT ts_ms FROM "{TABLE}" WHERE id = ?', (after_id,)).fetchone()
    if row is None or row[0] is None:
        raise EditError(f"Row {after_id} has no position to insert after.")
    anchor = int(row[0])
    following = conn.execute(
        f'SELECT MIN(ts_ms) FROM "{TABLE}" WHERE ts_ms > ?', (anchor,)).fetchone()[0]
    return anchor + 1000 if following is None else anchor + (int(following) - anchor) // 2


def apply_edits(conn: sqlite3.Connection,
                edits: Iterable[Tuple[str, Optional[int], Dict[str, Any], Optional[int]]],
                ) -> Dict[str, int]:
    """Apply parsed edits inside one transaction and report what changed.

    All or nothing: a batch that half-applies leaves a transcript nobody can reason
    about, and the operator's own screen is the only record of what they meant.
    """
    available = set(_columns(conn))
    counts = {"updated": 0, "inserted": 0, "deleted": 0}
    for op, row_id, values, after_id in edits:
        unknown = sorted(set(values) - available)
        if unknown:
            raise EditError(f"This database has no column {unknown[0]!r}.")
        if op == "delete":
            cursor = conn.execute(f'DELETE FROM "{TABLE}" WHERE id = ?', (row_id,))
            if not cursor.rowcount:
                raise EditError(f"Row {row_id} is not there any more.")
            counts["deleted"] += cursor.rowcount
        elif op == "update":
            assignments = ", ".join(f'"{c}" = ?' for c in values)
            cursor = conn.execute(
                f'UPDATE "{TABLE}" SET {assignments} WHERE id = ?',
                (*values.values(), row_id))
            if not cursor.rowcount:
                raise EditError(f"Row {row_id} is not there any more.")
            counts["updated"] += cursor.rowcount
        else:
            row = dict(values)
            if row.get("ts_ms") is None:
                row["ts_ms"] = next_ts_ms(conn, after_id)
            for column, default in INSERT_DEFAULTS.items():
                if column in available:
                    row.setdefault(column, default)
            if "timestamp" in available:
                row.setdefault("timestamp", time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(int(row["ts_ms"]) / 1000.0)))
            names = ", ".join(f'"{c}"' for c in row)
            marks = ", ".join("?" for _ in row)
            cursor = conn.execute(
                f'INSERT INTO "{TABLE}" ({names}) VALUES ({marks})', tuple(row.values()))
            # segment_id is the id as a string everywhere the recorder writes it.
            if "segment_id" in available and cursor.lastrowid is not None:
                conn.execute(f'UPDATE "{TABLE}" SET segment_id = ? WHERE id = ?',
                             (str(cursor.lastrowid), cursor.lastrowid))
            counts["inserted"] += 1
    return counts

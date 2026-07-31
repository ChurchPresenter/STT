"""Retiring a session database's WAL sidecars safely.

A SQLite database in WAL mode is accompanied by ``-wal`` and ``-shm`` files.
They are removed when the last connection closes cleanly, and the server does
exactly that at the end of a transcription session. It does not get the chance
when the *process* is stopped mid-session — a service restart terminates the
worker before it can checkpoint — so sidecars survive next to a finished
session and, with nothing sweeping afterwards, stay there indefinitely.

**The rule this module exists to enforce: never delete a ``-wal`` file.**

A WAL can hold committed transactions the main database file does not have yet.
Removing it discards them silently, and the loss is invisible until someone
reads the transcript and finds the last minutes of a service missing. The safe
retirement is the opposite of a delete: *open* the database, which recovers the
WAL, fold it into the main file, and let SQLite remove the file itself — see
checkpoint_and_release for the exact sequence, which is not the obvious one.

Stdlib-only so it can be unit-tested without the monolith's import-time side
effects; paths are passed in explicitly.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Suffixes SQLite keeps beside a database. "-journal" belongs to rollback mode
# rather than WAL, and is included so a database that was switched between modes
# does not leave one behind unnoticed.
SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def resolve_sidecars(db_path: str) -> List[str]:
    """Existing sidecar paths for ``db_path``, in a stable order."""
    if not db_path:
        return []
    return [db_path + suffix for suffix in SIDECAR_SUFFIXES
            if os.path.exists(db_path + suffix)]


def checkpoint_and_release(db_path: str, timeout: float = 5.0) -> bool:
    """Fold the WAL into ``db_path`` and retire the sidecars.

    Returns True when no sidecars remain afterwards. False means the database
    is in use, unreadable, or not ours — in every one of those cases the right
    answer is to leave the files alone rather than force the issue.

    The sequence is deliberate, and was arrived at by measurement rather than
    assumption: a TRUNCATE checkpoint alone empties the WAL but leaves both
    files on disk, and so does simply closing the last connection. Switching the
    journal mode to DELETE is what makes SQLite remove the ``-wal`` itself — the
    important part, because it means this module never unlinks a file that could
    still hold data.

    Only the ``-shm`` is unlinked directly, and only once the WAL is gone: it is
    a shared-memory index *for* the WAL, so with no WAL and no open connection
    it holds nothing to lose. Changing the journal mode is persistent, which is
    correct for a finished session — a delivered database that never opens in
    WAL mode never grows sidecars at its destination either. Live sessions are
    excluded by the caller, and the server sets WAL explicitly when it creates a
    session, so new recordings are unaffected.

    ``timeout`` is short because this is background housekeeping: it must yield
    to a live writer rather than contend with one.
    """
    if not db_path or not os.path.isfile(db_path):
        return False
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout, isolation_level=None)
        # Opening already recovered the WAL; this writes it into the main file.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # And this removes it, having nothing left to preserve.
        conn.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.Error:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    for path in resolve_sidecars(db_path):
        if path.endswith("-wal"):
            continue  # never unlinked; if it is still here, say so by returning False
        try:
            os.remove(path)
        except OSError:
            pass
    return not resolve_sidecars(db_path)


def _iter_databases(base_dirs: Iterable[str]) -> Iterable[str]:
    """Every *.db under the given directories, deduplicated."""
    seen = set()
    for base in base_dirs or ():
        if not base or not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for name in files:
                if not name.endswith(".db"):
                    continue
                path = os.path.realpath(os.path.join(root, name))
                if path not in seen:
                    seen.add(path)
                    yield path


def sweep_orphaned_sidecars(base_dirs: Sequence[str],
                            skip_paths: Iterable[str] = (),
                            min_age_s: Optional[float] = None,
                            now: Optional[float] = None) -> Dict[str, Any]:
    """Retire sidecars under ``base_dirs``. Returns a summary, never raises.

    ``skip_paths`` names databases to leave alone — the live session above all,
    which is being written to and whose sidecars are load-bearing.

    ``min_age_s`` skips databases modified more recently than that, so a session
    that has only just stopped (and may still be having its SRT written) is not
    disturbed. ``now`` is injectable so tests need no sleeping.
    """
    skip = {os.path.realpath(p) for p in skip_paths if p}
    scanned = cleaned = skipped_active = skipped_recent = failed = 0
    errors: List[str] = []
    reference = now if now is not None else _now()

    try:
        for db_path in _iter_databases(base_dirs):
            if not resolve_sidecars(db_path):
                continue  # already tidy; nothing to open
            scanned += 1

            if db_path in skip:
                skipped_active += 1
                continue
            if min_age_s is not None:
                try:
                    if reference - os.path.getmtime(db_path) < min_age_s:
                        skipped_recent += 1
                        continue
                except OSError:
                    pass  # unreadable mtime is not a reason to skip the work

            if checkpoint_and_release(db_path):
                cleaned += 1
            else:
                failed += 1
                if len(errors) < 10:  # a summary, not a catalogue
                    errors.append(os.path.basename(db_path))
    except OSError as e:
        errors.append(f"{type(e).__name__}: {e}")

    return {
        "scanned": scanned,
        "cleaned": cleaned,
        "skipped_active": skipped_active,
        "skipped_recent": skipped_recent,
        "failed": failed,
        "errors": errors,
    }


def _now() -> float:
    import time
    return time.time()

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


# Every SQLite database begins with this, including one whose pages are later
# corrupt. Checking it costs one read and keeps us from *opening* a file that is
# not ours — which matters more than it sounds: on sqlite 3.50, merely opening
# and closing a file that has a stray "-wal" beside it deletes that sidecar,
# whatever the caller intended. Refusing to open is the only way to leave a
# foreign file untouched.
_SQLITE_MAGIC = b"SQLite format 3\x00"


def looks_like_sqlite(path: str) -> bool:
    """Whether ``path`` starts with the SQLite file header."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
    except OSError:
        return False


def header_says_wal(path: str) -> bool:
    """Whether the file header records WAL mode, without opening the database.

    ``PRAGMA journal_mode`` would open it read-write and create the very ``-shm``
    whose absence is the condition being tested — the probe would change what it
    is probing. Byte 18 of the header is the write format version; 2 means WAL.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    return len(header) >= 20 and header[18] == 2


def open_readonly(db_path: str, recover: bool = True,
                  timeout: float = 5.0) -> "sqlite3.Connection":
    """A read-only connection, retiring an un-checkpointed WAL first if that is what blocks.

    A database still in WAL mode whose ``-shm`` is absent is the shape a session leaves
    behind when the process died mid-recording — the case this module exists for. What
    ``mode=ro`` does with it depends on the SQLite build: some refuse it (SQLite must
    create that shared-memory index to read the WAL, and a read-only connection may not),
    and others open it and create ``-wal`` and ``-shm`` next to the file. Neither outcome
    is acceptable — the first hides exactly the services most worth reading, the second
    scatters sidecars beside a delivered database — so the WAL is retired up front and
    the failure is still handled below.

    With ``recover`` the failure is instead treated as "this session was never retired":
    :func:`checkpoint_and_release` folds its WAL in, which is the same thing the end of a
    normal session does, and the open is retried. The database is left tidier than it was
    found and nothing is discarded — the WAL is recovered, never deleted.

    ``recover=False`` is for a database another process may be writing: its WAL is not ours
    to retire, and the caller wants the error.

    The statement after connect is not decoration. ``sqlite3.connect`` is lazy, so a
    database that cannot actually be read hands back a healthy-looking connection and fails
    later, somewhere further from the cause.
    """
    def _open() -> "sqlite3.Connection":
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except Exception:
            conn.close()
            raise
        return conn

    # Retire a delivered session's WAL from the header, rather than waiting for
    # the open to fail. Whether mode=ro *can* open a WAL database whose -shm is
    # absent turns out to be a property of the SQLite build, not a rule: 3.51
    # refuses, while 3.50 — what the provisioned runtime ships, and what CI runs —
    # opens it happily and creates -wal and -shm beside the file, leaving behind
    # exactly the sidecars this module exists to retire. Only a database with no
    # sidecars at all is touched, so a live writer's WAL is still none of our
    # business; that case falls through to the open below unchanged.
    if (recover and looks_like_sqlite(db_path) and header_says_wal(db_path)
            and not resolve_sidecars(db_path)):
        checkpoint_and_release(db_path, timeout=timeout)

    try:
        return _open()
    except sqlite3.Error:
        if not recover:
            raise
        checkpoint_and_release(db_path, timeout=timeout)
        return _open()


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
    if not looks_like_sqlite(db_path):
        return False  # not ours; opening it would itself remove its sidecar
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout, isolation_level=None)
        # Prove it is readable before changing anything about it: connect is
        # lazy and PRAGMAs do not always touch the file, so a database with a
        # valid header but corrupt pages would otherwise reach the journal-mode
        # switch. Reading the schema is the cheapest statement that must hit it.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
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

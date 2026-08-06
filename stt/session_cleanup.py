"""Tell finished service recordings apart from test runs, for archive cleanup.

A backup directory accumulates both: real services, and the runs made while
setting up, testing a microphone, or trying a model. They look identical on
disk, and the recordings dominate the space (an hour of .ts/.wav dwarfs its
database), so the question "which of these can go" is worth answering
mechanically rather than by eye over hundreds of files.

The signal that separates them is the schedule. Services happen at known times
— Sunday 10:00 and 18:00, Wednesday 19:00 — so a session is part of a service
when it *starts inside that service's window*, and a test when it starts
outside every one of them. Duration alone is not enough: a service whose
operator restarted halfway leaves two short files, and deleting them by length
would throw away half a service.

Three verdicts, and only one of them is deletable:

* ``service``  — inside a window and long enough to be the service itself.
* ``fragment`` — inside a window but short: a restart, a false start, or the
  operator testing minutes before the service. Never deleted by this module;
  it is the case a human should look at.
* ``unusual``  — outside every window, but hours long and full of speech. A
  wedding, a conference, a service moved for a holiday. Also never deleted:
  the schedule describes the ordinary week, not every week.
* ``test``     — outside every window and not substantial. Safe to remove.

A session is a *set* of files, not one file: the database, its WAL/shared-memory
sidecars, the SRT/HTML exports, and the .ts/.wav recordings. The recording is
named for when capture began and the database for when the session did, so the
two stems differ by seconds — companions are therefore matched to the *nearest*
database in time rather than by an equal stem, which also keeps two sessions
started twenty seconds apart from swallowing each other's files.

Stdlib only, and read-only: nothing here deletes. It reports; the caller
decides.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# Both filename conventions seen in the wild: the current
# "%Y-%m-%d_%H%M%S" and the older "%Y-%m-%d__%H-%M-%S" with a
# _Recording/_Transcriptions suffix.
_NAME_PATTERNS = (
    re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"),
    re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2})__(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})"),
)

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

# Row timestamps are "%Y-%m-%d %H:%M:%S", but not all of them: a real archive
# holds rows whose timestamp is a single space. Anything unparseable is ignored
# rather than trusted, which is why the span is measured against the filename.
_ROW_TIME = "%Y-%m-%d %H:%M:%S"


class Slot(NamedTuple):
    """A recurring service: weekday (Mon=0) and local start time."""

    weekday: int
    hour: int
    minute: int

    def label(self) -> str:
        name = next(k for k, v in _WEEKDAYS.items() if v == self.weekday)
        return f"{name}@{self.hour:02d}:{self.minute:02d}"


DEFAULT_SLOTS: Tuple[Slot, ...] = (
    Slot(6, 10, 0),   # Sunday morning
    Slot(6, 18, 0),   # Sunday evening
    Slot(2, 19, 0),   # Wednesday evening
)


class Session(NamedTuple):
    db_path: str
    started_at: datetime
    minutes: float          # wall clock, filename start -> last usable row
    rows: int
    files: Tuple[str, ...]  # the database and everything that belongs with it
    total_bytes: int

    @property
    def name(self) -> str:
        return os.path.basename(self.db_path)


def parse_started_at(filename: str) -> Optional[datetime]:
    """Session start from a backup filename, or None if it is not one."""
    base = os.path.basename(filename)
    for pattern in _NAME_PATTERNS:
        m = pattern.match(base)
        if m:
            year, month, day = (int(part) for part in m.group("d").split("-"))
            try:
                return datetime(year, month, day,
                                int(m.group("h")), int(m.group("m")), int(m.group("s")))
            except ValueError:
                return None
    return None


def parse_slots(spec: str) -> Tuple[Slot, ...]:
    """"sun@10:00,wed@19:00" -> slots. Raises ValueError on anything else."""
    slots: List[Slot] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        day, _, clock = part.partition("@")
        if day not in _WEEKDAYS or ":" not in clock:
            raise ValueError(f"expected day@HH:MM, got {part!r}")
        hh, mm = clock.split(":", 1)
        slots.append(Slot(_WEEKDAYS[day], int(hh), int(mm)))
    if not slots:
        raise ValueError("no slots given")
    return tuple(slots)


def read_session_span(db_path: str, started_at: datetime) -> Tuple[float, int]:
    """(minutes, rows) for a session database. (0.0, 0) when it cannot be read.

    Measured from the filename's start to the last row whose timestamp parses,
    not from the row timestamps alone: archives contain rows with a blank
    timestamp, which would make the span look like decades. end_time is not used
    either — it is an audio-stream offset that has been seen to disagree with the
    wall clock by 30 minutes on a restarted capture.

    Opened immutable, not merely read-only. A plain read-only open still takes
    locks, and on a WAL database that means touching the -shm sidecar: scanning
    an archive rewrote the modification time of all 86 sessions in it, which is
    both a lie about the data and the very signal a caller might use to decide
    what is safe to delete. Immutable also happens to be the only way SQLite can
    open a file on an SMB mount, where those locks are unavailable.
    """
    for uri in (f"file:{db_path}?immutable=1", f"file:{db_path}?mode=ro"):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5)
        except sqlite3.Error:
            continue
        try:
            with conn:
                rows = conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
                last: Optional[datetime] = None
                for (value,) in conn.execute(
                        "SELECT timestamp FROM transcriptions ORDER BY id DESC LIMIT 200"):
                    try:
                        last = datetime.strptime(str(value), _ROW_TIME)
                        break
                    except (TypeError, ValueError):
                        continue
            if last is None:
                return 0.0, int(rows)
            return max(0.0, (last - started_at).total_seconds() / 60.0), int(rows)
        except sqlite3.Error:
            continue
        finally:
            conn.close()
    return 0.0, 0


def nearest_db(path_time: datetime, db_times: Sequence[Tuple[str, datetime]],
               limit_minutes: float = 30.0) -> Optional[str]:
    """The database a companion file belongs to, or None if nothing is near.

    Nearest in time rather than same-stem: a recording is named for when capture
    started, its database for when the session did, and in practice they differ
    by half a minute. Two sessions twenty seconds apart still resolve correctly,
    which a fixed window would not.
    """
    best: Optional[str] = None
    best_delta = timedelta(minutes=limit_minutes)
    for db_path, db_time in db_times:
        delta = abs(path_time - db_time)
        if delta <= best_delta:
            best, best_delta = db_path, delta
    return best


def scan(root: str) -> Tuple[List[Session], List[str]]:
    """Every session under ``root``, plus files that belong to no session."""
    dated: List[Tuple[str, datetime]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            started = parse_started_at(name)
            if started is not None:
                dated.append((os.path.join(dirpath, name), started))

    db_times = [(p, t) for p, t in dated if p.endswith(".db")]
    grouped: Dict[str, List[str]] = {p: [p] for p, _ in db_times}
    orphans: List[str] = []
    for path, when in dated:
        if path.endswith(".db"):
            continue
        owner = nearest_db(when, db_times)
        if owner is None:
            orphans.append(path)
        else:
            grouped[owner].append(path)

    sessions: List[Session] = []
    for db_path, started in sorted(db_times, key=lambda pair: pair[1]):
        minutes, rows = read_session_span(db_path, started)
        files = tuple(sorted(grouped[db_path]))
        sessions.append(Session(db_path, started, minutes, rows, files,
                                sum(_size(f) for f in files)))
    return sessions, sorted(orphans)


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def in_window(started_at: datetime, slots: Iterable[Slot],
              lead_minutes: float, span_minutes: float) -> Optional[Slot]:
    """The service this session starts inside, if any.

    The window opens before the slot (setting up, and a service that starts
    early) and stays open long enough to cover a restart partway through — but
    no longer. At three hours a 10:00 service still claimed a session started at
    13:00, which is an afternoon of testing, not a restart.
    """
    for slot in slots:
        if started_at.weekday() != slot.weekday:
            continue
        slot_at = started_at.replace(hour=slot.hour, minute=slot.minute,
                                     second=0, microsecond=0)
        if -lead_minutes <= (started_at - slot_at).total_seconds() / 60.0 <= span_minutes:
            return slot
    return None


def classify(session: Session, slots: Iterable[Slot] = DEFAULT_SLOTS,
             *, service_minutes: float = 90.0, lead_minutes: float = 120.0,
             span_minutes: float = 150.0, substantial_rows: int = 300) -> Tuple[str, str]:
    """(verdict, why) for one session. Only "test" is safe to delete.

    A session inside a service window is never called a test however short it
    is: that is where restarts and false starts land, and they are the operator's
    to judge, not this module's.

    Nor is a session holding a lot of transcribed speech, whatever day it falls
    on or how long it ran. Measured against a real archive, the schedule alone
    would have deleted a 154-minute Saturday recording of 2650 lines, and adding
    a length rule still deleted a 43-minute Friday one of 460. Line count is what
    separates them: a test leaves a long file with a handful of lines in it (83
    minutes, 99 lines), a gathering leaves hundreds however long it ran.
    """
    slot = in_window(session.started_at, slots, lead_minutes, span_minutes)
    if slot is None:
        if session.rows >= substantial_rows:
            return "unusual", (f"no service scheduled, but {session.rows} lines "
                               f"over {session.minutes:.0f} min")
        return "test", "outside every service window"
    if session.minutes >= service_minutes:
        return "service", f"{slot.label()} service, {session.minutes:.0f} min"
    return "fragment", f"{slot.label()} window but only {session.minutes:.0f} min"


# --- Still being written -----------------------------------------------------
#
# The one thing age has to protect. A session whose files are open right now
# must never be swept, and this lived in the CLI wrapper rather than here, so
# anything importing this module got the verdicts without the guard.

_SIDECARS = ("-shm", "-wal", "-journal")

#: A session written this recently is in progress and is never touched.
LIVE_MINUTES = 10.0


def last_written(paths: Iterable[str]) -> float:
    """Newest mtime among ``paths``, ignoring the SQLite sidecars.

    The ``-wal``/``-shm`` files are excluded deliberately: any reader disturbs
    them, so their mtime says when something looked at the session, not when the
    session was written. Returns 0.0 when nothing readable is left.
    """
    times = [os.path.getmtime(p) for p in paths
             if not p.endswith(_SIDECARS) and os.path.exists(p)]
    return max(times) if times else 0.0


def idle_minutes(session: Session, now: Optional[float] = None) -> float:
    """Minutes since any real file of ``session`` was last written.

    ``now`` is a POSIX timestamp, passed in so this is testable without waiting.
    """
    written = last_written(session.files)
    if written <= 0.0:
        return float("inf")     # nothing to date it by — not evidence of being live
    current = time.time() if now is None else now
    return (current - written) / 60.0


def classify_live(session: Session, slots: Iterable[Slot] = DEFAULT_SLOTS, *,
                  now: Optional[float] = None, live_minutes: float = LIVE_MINUTES,
                  service_minutes: float = 90.0, lead_minutes: float = 120.0,
                  span_minutes: float = 150.0,
                  substantial_rows: int = 300) -> Tuple[str, str]:
    """:func:`classify`, but a session still being written wins over every verdict.

    Callers that can delete should use this rather than :func:`classify`: the
    recording in progress is off-schedule and has few rows early on, which is
    exactly the shape of a ``test``.
    """
    idle = idle_minutes(session, now)
    if idle < live_minutes:
        return "in progress", f"written {idle:.0f} min ago"
    return classify(session, slots, service_minutes=service_minutes,
                    lead_minutes=lead_minutes, span_minutes=span_minutes,
                    substantial_rows=substantial_rows)


#: Verdicts a sweep must never remove, whatever the caller asks for.
#:
#: Only one. ``service`` is not here: the CLI this came from refuses to delete a
#: service because it runs unattended over a whole archive, but dev mode is an
#: operator naming one session in front of them, and an operator who has decided
#: a service recording can go is allowed to say so.
#:
#: ``in progress`` stays, and is not negotiable. That is not a judgement about
#: the recording's worth — the file is open and being appended to, so removing it
#: destroys a service *while it is being captured*, and no confirmation dialog
#: can make that a considered choice.
UNDELETABLE = frozenset({"in progress"})


def wav_seconds(path: str) -> Optional[float]:
    """Length of a RIFF/WAVE file, without decoding it.

    Stdlib and exact: the byte rate and the data chunk size are in the header,
    so this works on a share, on Windows, and with no ffmpeg installed.

    The header is not always the truth. These recordings are written as a stream
    and the size fields are never patched at the end, so a 476 MB file can
    declare 32000 bytes of audio — one second. ffprobe believes it, which is why
    such a file plays for a second and stops. When the declared size is far short
    of the file, the bytes on disk are what actually got recorded.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(44)
        if len(head) < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        byte_rate = int.from_bytes(head[28:32], "little")
        declared = int.from_bytes(head[40:44], "little")
        if not byte_rate:
            return None
        on_disk = max(os.path.getsize(path) - 44, 0)
        # Trailing chunks after the audio are normal, so only treat the header as
        # stale when it accounts for less than most of the file.
        data_size = declared if declared >= 0.9 * on_disk else on_disk
        return data_size / byte_rate
    except OSError:
        return None


class Refusal(NamedTuple):
    """A path the sweep declined, and the reason to show the operator."""

    name: str
    reason: str


def plan_deletion(sessions: Sequence[Dict[str, Any]], wanted: Iterable[str],
                  live_db: str = "") -> Tuple[List[str], List[Refusal]]:
    """Split requested sessions into files to remove and refusals.

    Every safety decision a sweep makes lives here rather than in the request
    handler, so each one is testable without a server:

    * a path that is not a session under the scanned directory is refused —
      the caller cannot name an arbitrary file;
    * ``service`` and ``in progress`` are refused however they were asked for;
    * the session currently being recorded is refused even if the classifier
      has not noticed yet;
    * a ``-wal`` with bytes in it is refused, because it holds rows that have
      not been folded into the database and deleting it loses them.

    ``sessions`` are the dicts a scan produced (``db_path``, ``files``,
    ``verdict``, ``name``). ``live_db`` is the database being written right now,
    or "" when nothing is recording.
    """
    by_path = {s["db_path"]: s for s in sessions}
    live = os.path.realpath(live_db) if live_db else ""

    to_delete: List[str] = []
    refused: List[Refusal] = []
    for raw in wanted:
        session = by_path.get(raw)
        if session is None:
            refused.append(Refusal(os.path.basename(raw or "?"),
                                   "not a session under this directory"))
            continue
        if session["verdict"] in UNDELETABLE:
            refused.append(Refusal(session["name"], session["verdict"]))
            continue
        if live and os.path.realpath(session["db_path"]) == live:
            refused.append(Refusal(session["name"], "this session is recording"))
            continue
        for path in session["files"]:
            if path.endswith("-wal") and _size(path) > 0:
                refused.append(Refusal(os.path.basename(path), "unmerged -wal"))
                continue
            to_delete.append(path)
    return to_delete, refused


def pick_recording(files: Iterable[str]) -> Optional[str]:
    """The audio file of a session, preferring the ``.wav``.

    Both are the same service: the ``.ts`` is the capture, the ``.wav`` is what
    was made from it to feed the recogniser. The ``.wav`` is the one to hand to
    another tool — it is plain PCM, and it is what a re-run transcribes.

    Choosing it has to be explicit rather than "the first audio file in the
    set". A session's files are sorted, and the capture is named for when
    recording began while everything else is named for when the session did —
    seconds earlier, but enough that ``2026-08-05_182841.ts`` sorts ahead of
    ``2026-08-05_182905.wav`` and a first-match picker silently returns the
    capture instead.
    """
    audio = [f for f in files if f.lower().endswith((".wav", ".ts"))]
    if not audio:
        return None
    wavs = [f for f in audio if f.lower().endswith(".wav")]
    return sorted(wavs)[0] if wavs else sorted(audio)[0]


def plan_orphan_deletion(orphans: Iterable[str], wanted: Iterable[str],
                         now: Optional[float] = None,
                         live_minutes: float = LIVE_MINUTES) -> Tuple[List[str], List[Refusal]]:
    """Split requested orphan files into ones to remove and refusals.

    An orphan is a recording or export with no database near it in time — the
    leftovers of a session whose record was already deleted. Nothing groups
    them, so a session sweep cannot reach them and they sit on disk unnoticed;
    they are usually the largest files in the folder.

    Two refusals, and the second is the one that matters: **a recording being
    made right now is an orphan.** Its database does not exist yet, so it looks
    exactly like an abandoned file, and the only thing separating them is that
    one of them is still being written.
    """
    known = set(orphans)
    current = time.time() if now is None else now

    to_delete: List[str] = []
    refused: List[Refusal] = []
    for path in wanted:
        if path not in known:
            refused.append(Refusal(os.path.basename(path or "?"), "not an orphan here"))
            continue
        written = last_written([path])
        if written > 0.0 and (current - written) / 60.0 < live_minutes:
            refused.append(Refusal(os.path.basename(path), "still being written"))
            continue
        to_delete.append(path)
    return to_delete, refused

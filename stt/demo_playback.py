"""Replay a recorded service into a live session, as if it were being spoken now.

The demo has no microphone and no model, but it must not look like a recording being
scrolled: the operator UI, the phase detector, the corrections path and the file
manager should all behave exactly as they do during a real service. So the player
works on the *writer* side. It creates a genuine session database and copies rows into
it at the pace they originally arrived, which means every reader downstream —
``get_new_entries``, the ``transcription_update`` emitter, ``_service_phase_tick``,
the session index — needs no knowledge that this is a demo at all.

The alternative (letting readers see the whole recording and filtering by time) was
rejected: it needs a hook in every reader, and each one would still hold the finished
service, so the phase page would show the ending before it happened.

Stdlib-only. The clock is injected, so the whole schedule can be driven a tick at a
time in tests without sleeping. Shared state is passed in; nothing here reads the
monolith's globals.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import threading
import time
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

# A caption always occupies some time on screen, even when the recording's own
# start/end stamps collapsed to nothing.
MIN_ROW_DURATION_S = 0.4

# Columns cleared when a recording is turned into an empty session to replay into.
_RESET_TABLES = (
    "transcriptions",
    "service_phase_bins",
    "service_phase_blocks",
    "service_phase_spans",
    "service_phase_corrections",
    "sermon_summaries",
)


class ScheduledRow:
    """One caption from the recording, with when it should reappear."""

    __slots__ = ("columns", "duration_s", "index", "offset_s")

    def __init__(self, index: int, offset_s: float, duration_s: float,
                 columns: Dict[str, Any]) -> None:
        self.index = index
        self.offset_s = offset_s
        self.duration_s = duration_s
        self.columns = columns

    @property
    def text(self) -> str:
        return str(self.columns.get("text") or "")

    @property
    def words_json(self) -> Optional[str]:
        value = self.columns.get("words_json")
        return str(value) if value else None

    @property
    def music_prob(self) -> float:
        try:
            return float(self.columns.get("music_prob") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScheduledRow(index={self.index}, offset_s={self.offset_s:.2f})"


class PlaybackConfig:
    """How a recording is replayed."""

    __slots__ = (
        "filename_format",
        "keep_loops",
        "loop",
        "loop_gap_s",
        "path_format",
        "session_dir",
        "source_db",
        "speed",
    )

    def __init__(self, source_db: str, session_dir: str, speed: float = 1.0,
                 loop: bool = True, loop_gap_s: float = 5.0, keep_loops: int = 3,
                 filename_format: str = "%Y-%m-%d_%H%M%S",
                 path_format: str = "%Y/%m") -> None:
        self.source_db = source_db
        self.session_dir = session_dir
        self.speed = speed
        self.loop = loop
        self.loop_gap_s = loop_gap_s
        self.keep_loops = keep_loops
        self.filename_format = filename_format
        self.path_format = path_format


class SystemClock:
    """Wall time. Tests substitute a clock they can advance by hand."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


# --- reading the recording -------------------------------------------------


def read_only_uri(path: str) -> str:
    """Open-for-replay URI. Same rule as stt/translation_replay: immutable, so the
    operator's file gains no WAL sidecar and is left byte-identical."""
    return "file:" + path.replace("?", "%3f").replace("#", "%23") + "?immutable=1"


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def load_schedule(source_db: str) -> List[ScheduledRow]:
    """Every caption worth replaying, in the order it was spoken.

    Only final, undenied rows with a timestamp: partials were superseded, denied rows
    were hidden from the display, and a row with no ``ts_ms`` (the blank marker row a
    session opens with) has no place on a timeline.
    """
    conn = sqlite3.connect(read_only_uri(source_db), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        columns = _table_columns(conn, "transcriptions")
        rows = conn.execute(
            "SELECT * FROM transcriptions "
            "WHERE is_final = 1 AND COALESCE(denied, 0) = 0 AND ts_ms IS NOT NULL "
            "AND TRIM(COALESCE(text, '')) != '' "
            "ORDER BY ts_ms ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    first_ms = int(rows[0]["ts_ms"])
    schedule: List[ScheduledRow] = []
    for index, row in enumerate(rows):
        values = {name: row[name] for name in columns}
        schedule.append(ScheduledRow(
            index=index,
            offset_s=(int(row["ts_ms"]) - first_ms) / 1000.0,
            duration_s=_row_duration(values),
            columns=values,
        ))
    return schedule


def _row_duration(columns: Dict[str, Any]) -> float:
    try:
        span = float(columns.get("end_time") or 0.0) - float(columns.get("start_time") or 0.0)
    except (TypeError, ValueError):
        span = 0.0
    return max(span, MIN_ROW_DURATION_S)


def schedule_length_s(schedule: Sequence[ScheduledRow]) -> float:
    """How long the replay runs for."""
    return schedule[-1].offset_s if schedule else 0.0


# --- building the session to replay into -----------------------------------


def session_path(session_dir: str, when: float, filename_format: str = "%Y-%m-%d_%H%M%S",
                 path_format: str = "%Y/%m") -> str:
    """Where a session starting at ``when`` would be written, following the real layout."""
    stamp = time.localtime(when)
    return os.path.join(session_dir, time.strftime(path_format, stamp),
                        time.strftime(filename_format, stamp) + ".db")


def create_session_db(source_db: str, dest: str, started_at: float) -> str:
    """Copy the recording's *schema* into a fresh, empty session database.

    Copying the file and emptying it means the schema is identical to a real session's
    by construction — including whichever ALTER TABLE migrations that recording has
    been through — so this can never drift from ``initialize_database``.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copyfile(source_db, dest)
    conn = sqlite3.connect(dest)
    try:
        existing = {row[0] for row in
                    conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in _RESET_TABLES:
            if table in existing:
                conn.execute(f"DELETE FROM {table}")
        if "sqlite_sequence" in existing:
            conn.execute("DELETE FROM sqlite_sequence")
        if "session_meta" in existing:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at))
            conn.execute("INSERT OR REPLACE INTO session_meta (key, value) VALUES (?, ?)",
                         ("session_started", stamp))
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    return dest


def rebase_row(row: ScheduledRow, session_start_ms: int, new_id: int) -> Dict[str, Any]:
    """The recording's row, restamped onto the session now running.

    Gaps between captions are preserved exactly — the replay reproduces the rhythm of
    the service, not just its contents. ``segment_id`` is set to ``str(new_id)`` because
    that is the identity the display and the corrections path agree on.
    """
    columns = dict(row.columns)
    ts_ms = session_start_ms + round(row.offset_s * 1000)
    columns["id"] = new_id
    columns["ts_ms"] = ts_ms
    columns["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000.0))
    columns["segment_id"] = str(new_id)
    columns["is_final"] = 1
    columns["denied"] = 0
    columns["corrected_by"] = None
    columns["needs_review"] = 0
    if columns.get("translated_text"):
        columns["translation_ts_ms"] = ts_ms
    return columns


def insert_rows(conn: sqlite3.Connection, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rebased rows into the live session."""
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    statement = (f"INSERT OR REPLACE INTO transcriptions ({', '.join(columns)}) "
                 f"VALUES ({placeholders})")
    conn.executemany(statement, [tuple(row[name] for name in columns) for row in rows])
    conn.commit()


# --- what is due, and what is being said right now -------------------------


def due_rows(schedule: Sequence[ScheduledRow], elapsed_s: float,
             cursor: int) -> Tuple[List[ScheduledRow], int]:
    """Rows whose moment has arrived, and the new cursor.

    Inclusive at the boundary: a row scheduled for exactly this instant belongs to it.
    """
    due: List[ScheduledRow] = []
    index = cursor
    while index < len(schedule) and schedule[index].offset_s <= elapsed_s:
        due.append(schedule[index])
        index += 1
    return due, index


def speaking_row(schedule: Sequence[ScheduledRow], elapsed_s: float,
                 cursor: int) -> Optional[ScheduledRow]:
    """The caption currently being spoken, if any.

    A row commits when it was saved, so the speech that produced it occupies the window
    ending at that moment — which is why the preview appears *before* the line lands.
    """
    if cursor >= len(schedule):
        return None
    row = schedule[cursor]
    start = row.offset_s - row.duration_s
    return row if start <= elapsed_s < row.offset_s else None


def _parse_words(words_json: Optional[str]) -> List[Dict[str, Any]]:
    if not words_json:
        return []
    try:
        parsed = json.loads(words_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [word for word in parsed if isinstance(word, dict) and "w" in word]


def partial_text(text: str, words_json: Optional[str], progress: float) -> str:
    """How much of a caption has been spoken at ``progress`` through its window.

    Word timings are used when the recording has them, so the preview reveals at the
    pace the words were actually said; otherwise it falls back to revealing whole
    words proportionally. Never returns a half-word — a flickering partial token reads
    as a rendering bug.
    """
    progress = min(max(progress, 0.0), 1.0)
    if progress >= 1.0:
        return text
    if progress <= 0.0:
        return ""

    words = _parse_words(words_json)
    if words:
        starts = [float(word.get("s_ms") or 0) for word in words]
        ends = [float(word.get("e_ms") or 0) for word in words]
        first, last = starts[0], max(ends or starts)
        if last > first:
            revealed_to = first + progress * (last - first)
            spoken = [word["w"] for word, start in zip(words, starts) if start <= revealed_to]
            if spoken:
                return "".join(str(word) for word in spoken).strip()
            return ""

    tokens = text.split()
    if not tokens:
        return ""
    take = int(len(tokens) * progress)
    return " ".join(tokens[:take])


class AudioLevel:
    """What the level meter should show this instant."""

    __slots__ = ("audio_type", "db", "energy", "level", "music_prob")

    def __init__(self, level: int, db: float, energy: float, audio_type: str,
                 music_prob: float) -> None:
        self.level = level
        self.db = db
        self.energy = energy
        self.audio_type = audio_type
        self.music_prob = music_prob

    def as_state(self) -> Dict[str, Any]:
        return {
            "audio_level": self.level,
            "audio_db": self.db,
            "audio_energy": self.energy,
            "audio_type": self.audio_type,
            "music_prob": self.music_prob,
        }


def audio_level_for(row: Optional[ScheduledRow], rng: random.Random) -> AudioLevel:
    """A plausible meter reading for whatever is happening now.

    Synthesised rather than recorded: the level was never stored, but a meter frozen
    at zero while captions appear is the one thing that would give the replay away.
    Driven by a seeded generator so a test can assert on it.
    """
    if row is None:
        level = rng.randint(3, 9)
        return AudioLevel(level=level, db=-52.0 + level * 0.2, energy=level * 4.0,
                          audio_type="Quiet", music_prob=0.0)
    music = row.music_prob
    level = rng.randint(45, 85)
    return AudioLevel(
        level=level,
        db=-30.0 + (level - 45) * 0.45,
        energy=level * 180.0,
        audio_type="Music" if music > 0.5 else "Speaking",
        music_prob=music,
    )


# --- the player ------------------------------------------------------------


class Player:
    """Replays a recording into successive live sessions.

    Quacks like the worker process the monolith otherwise manages — ``is_alive`` and
    ``pid`` — so ``start_transcription`` reuses it instead of spawning a real worker,
    and Start/Stop in the UI drive playback with no change to either route.
    """

    def __init__(self, config: PlaybackConfig, state: MutableMapping[str, Any],
                 control_queue: Any = None, clock: Optional[SystemClock] = None,
                 seed: int = 20260823) -> None:
        self._config = config
        self._state = state
        self._control_queue = control_queue
        self._clock = clock or SystemClock()
        self._rng = random.Random(seed)
        self._schedule: List[ScheduledRow] = []
        self._cursor = 0
        self._started_at = 0.0
        self._virtual_at_start = 0.0
        self._running = False
        self._db_path: Optional[str] = None
        self._sessions: List[str] = []
        self._next_id = 1
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    # -- worker-process shim --

    def is_alive(self) -> bool:
        return True

    @property
    def pid(self) -> int:
        return os.getpid()

    # -- lifecycle --

    def start(self, tick_s: float = 0.25) -> None:
        """Begin consuming the control queue in the background."""
        if self._thread is not None:
            return
        self._state["status"] = "stopped"
        self._state["message"] = "Ready"

        def _loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.tick()
                except Exception as exc:  # pragma: no cover - a demo must not die
                    print(f"[DEMO] playback error: {exc}", flush=True)
                self._clock.sleep(tick_s)

        self._thread = threading.Thread(target=_loop, daemon=True, name="demo-playback")
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()

    def begin_session(self) -> str:
        """Open a new live session and start the clock at the top of the recording."""
        with self._lock:
            if not self._schedule:
                self._schedule = load_schedule(self._config.source_db)
            now = time.time()
            path = session_path(self._config.session_dir, now,
                                self._config.filename_format, self._config.path_format)
            # A second session inside the same second would collide with the first.
            suffix = 1
            while os.path.exists(path):
                base, ext = os.path.splitext(path)
                path = f"{base}_{suffix}{ext}"
                suffix += 1
            create_session_db(self._config.source_db, path, now)

            self._db_path = path
            self._sessions.append(path)
            self._prune_sessions()
            self._cursor = 0
            self._next_id = 1
            self._started_at = self._clock.now()
            self._virtual_at_start = 0.0
            self._running = True

            self._state.update({
                "running": True,
                "status": "running",
                "message": "Transcribing",
                "error": None,
                "db_name": path,
                "session_id": os.path.splitext(os.path.basename(path))[0],
                "start_time": now,
                "loaded_model": self._demo_model_label(),
                "detection_mode": "panns",
                "live_text": "",
                "live_start": 0,
                "live_end": 0,
                "rows_saved": 0,
                "segments_total": 0,
            })
            return path

    def end_session(self) -> None:
        with self._lock:
            self._running = False
            self._state.update({
                "running": False,
                "status": "stopped",
                "message": "Transcription stopped",
                "live_text": "",
                "live_start": 0,
                "live_end": 0,
                "loaded_model": "",
                "audio_level": 0,
                "audio_db": -60,
                "audio_type": None,
            })

    def restart(self) -> None:
        self.end_session()
        self.begin_session()

    def set_speed(self, speed: float) -> None:
        """Change pace without losing position: the virtual clock is rebased, not reset."""
        with self._lock:
            if self._running:
                self._virtual_at_start = self.elapsed_s()
                self._started_at = self._clock.now()
            self._config.speed = max(0.1, min(speed, 20.0))

    def elapsed_s(self) -> float:
        """Virtual seconds into the recording.

        Measured from the session start rather than accumulated per tick, so a demo
        left running for hours does not drift.
        """
        if not self._running:
            return self._virtual_at_start
        return self._virtual_at_start + (self._clock.now() - self._started_at) * self._config.speed

    @property
    def running(self) -> bool:
        return self._running

    @property
    def db_path(self) -> Optional[str]:
        return self._db_path

    # -- one step --

    def tick(self) -> None:
        """Advance playback by however much time has passed. Tests call this directly."""
        self._drain_control_queue()
        if not self._running or not self._db_path:
            return

        elapsed = self.elapsed_s()
        with self._lock:
            due, cursor = due_rows(self._schedule, elapsed, self._cursor)
            if due:
                session_start_ms = int(self._state.get("start_time", time.time()) * 1000)
                rebased = []
                for row in due:
                    rebased.append(rebase_row(row, session_start_ms, self._next_id))
                    self._next_id += 1
                conn = sqlite3.connect(self._db_path)
                try:
                    insert_rows(conn, rebased)
                finally:
                    conn.close()
                self._cursor = cursor
                self._state["rows_saved"] = self._next_id - 1
                self._state["segments_total"] = self._next_id - 1

            current = speaking_row(self._schedule, elapsed, self._cursor)
            self._publish_preview(current, elapsed)
            self._state.update(audio_level_for(current, self._rng).as_state())

            if self._cursor >= len(self._schedule):
                self._handle_end(elapsed)

    def _publish_preview(self, row: Optional[ScheduledRow], elapsed_s: float) -> None:
        if row is None:
            self._state["live_text"] = ""
            self._state["live_start"] = 0
            self._state["live_end"] = 0
            return
        start = row.offset_s - row.duration_s
        progress = (elapsed_s - start) / row.duration_s if row.duration_s else 1.0
        self._state["live_text"] = partial_text(row.text, row.words_json, progress)
        self._state["live_start"] = float(row.columns.get("start_time") or 0.0)
        self._state["live_end"] = float(row.columns.get("end_time") or 0.0)

    def _handle_end(self, elapsed_s: float) -> None:
        finished_at = schedule_length_s(self._schedule)
        if elapsed_s < finished_at + self._config.loop_gap_s:
            return
        if self._config.loop:
            # A new session rather than a rewind: it exercises the real rollover, and
            # the file manager ends up with more than one service to look at.
            self.begin_session()
        else:
            self.end_session()

    def _prune_sessions(self) -> None:
        while len(self._sessions) > max(1, self._config.keep_loops):
            stale = self._sessions.pop(0)
            try:
                os.remove(stale)
            except OSError:
                pass

    def _demo_model_label(self) -> str:
        """What the recording says produced it, so the UI names a real model."""
        try:
            conn = sqlite3.connect(read_only_uri(self._config.source_db), uri=True)
            try:
                row = conn.execute(
                    "SELECT asr_model FROM transcriptions "
                    "WHERE asr_model IS NOT NULL AND asr_model != '' LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if row and row[0]:
                return str(row[0])
        except sqlite3.Error:
            pass
        return "large-v3"

    # -- control --

    def _drain_control_queue(self) -> None:
        """Honour the same start/stop commands the real worker consumes."""
        if self._control_queue is None:
            return
        while True:
            try:
                command = self._control_queue.get_nowait()
            except Exception:
                return
            if not isinstance(command, dict):
                continue
            action = command.get("command")
            if action == "start" and not self._running:
                self.begin_session()
            elif action == "stop" and self._running:
                self.end_session()


def translation_pairs(source_db: str) -> Dict[str, str]:
    """Every caption the recording holds alongside what it was translated to.

    The demo answers /api/translate out of this: the recording already contains real
    output from the real engine, so a visitor who pastes in a line from the transcript
    gets the genuine translation rather than an invented one.
    """
    pairs: Dict[str, str] = {}
    try:
        conn = sqlite3.connect(read_only_uri(source_db), uri=True)
    except sqlite3.Error:
        return pairs
    try:
        rows = conn.execute(
            "SELECT text, translated_text FROM transcriptions "
            "WHERE is_final = 1 AND COALESCE(denied, 0) = 0 "
            "AND TRIM(COALESCE(text, '')) != '' "
            "AND TRIM(COALESCE(translated_text, '')) != ''"
        ).fetchall()
    except sqlite3.Error:
        return pairs
    finally:
        conn.close()
    for text, translated in rows:
        source = str(text).strip()
        pairs.setdefault(source, str(translated).strip())
        pairs.setdefault(source.lower(), str(translated).strip())
    return pairs

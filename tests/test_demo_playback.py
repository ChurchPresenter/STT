"""Replaying a recorded service into a live session.

The clock is injected everywhere, so the whole schedule is driven a tick at a time
and nothing here sleeps.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sqlite3

import pytest

from stt import demo_playback


# The columns the player actually reads and rewrites. A real session has many more;
# create_session_db copies whatever the recording has, so the test only needs enough
# to exercise the scheduling and rebasing.
_DDL = """
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, text TEXT, start_time REAL, end_time REAL, confidence REAL,
    original_text TEXT, corrected_by TEXT, needs_review INTEGER,
    translated_text TEXT, translation_language TEXT, speech_type TEXT,
    audio_tag TEXT, music_prob REAL, denied INTEGER, ts_ms INTEGER,
    words_json TEXT, is_final INTEGER DEFAULT 1, partial_seq INTEGER,
    source_language TEXT, segment_id TEXT, words_source TEXT, session_id TEXT,
    denied_reason TEXT, marked INTEGER, translation_ts_ms INTEGER, asr_model TEXT
);
CREATE TABLE session_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE service_phase_blocks (id INTEGER PRIMARY KEY, phase TEXT);
"""


class FakeClock:
    """Monotonic time the test moves by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start
        self.slept = 0.0

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.slept += seconds

    def advance(self, seconds: float) -> None:
        self._now += seconds


def make_recording(path, rows, *, asr_model="large-v3"):
    """A recording with ``rows`` of (ts_ms, start_time, end_time, text, translation)."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_DDL)
    conn.execute("INSERT INTO session_meta (key, value) VALUES ('session_started', 'old')")
    conn.execute("INSERT INTO service_phase_blocks (id, phase) VALUES (1, 'sermon')")
    # The blank marker row a real session opens with: no ts_ms, no text.
    conn.execute("INSERT INTO transcriptions (id, text, is_final, denied) VALUES (1, ' ', 1, 0)")
    for index, (ts_ms, start, end, text, translation) in enumerate(rows, start=2):
        conn.execute(
            "INSERT INTO transcriptions (id, ts_ms, start_time, end_time, text, "
            "translated_text, is_final, denied, music_prob, asr_model, segment_id) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?)",
            (index, ts_ms, start, end, text, translation, 0.05, asr_model, str(index)))
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture()
def recording(tmp_path):
    return make_recording(tmp_path / "source.db", [
        (1_000_000, 0.0, 2.0, "Peace be with you", "Мир вам"),
        (1_004_000, 2.0, 5.0, "Let us pray together", "Помолимся вместе"),
        (1_010_000, 5.0, 8.0, "Please be seated", "Садитесь пожалуйста"),
    ])


def player_for(recording, tmp_path, clock, **kwargs):
    config = demo_playback.PlaybackConfig(
        source_db=recording, session_dir=str(tmp_path / "sessions"), **kwargs)
    state = {"start_time": 0}
    return demo_playback.Player(config, state, clock=clock), state


def session_rows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT id, ts_ms, text, translated_text, segment_id FROM transcriptions "
            "ORDER BY id").fetchall()
    finally:
        conn.close()


# --- reading the recording -------------------------------------------------


def test_load_schedule_skips_the_blank_marker_row(recording):
    schedule = demo_playback.load_schedule(recording)

    assert len(schedule) == 3
    assert [row.text for row in schedule] == [
        "Peace be with you", "Let us pray together", "Please be seated"]


def test_load_schedule_offsets_are_relative_to_the_first_caption(recording):
    schedule = demo_playback.load_schedule(recording)

    assert [row.offset_s for row in schedule] == [0.0, 4.0, 10.0]


def test_load_schedule_skips_partials_and_denied_rows(tmp_path):
    path = make_recording(tmp_path / "s.db", [(1000, 0.0, 1.0, "kept", None)])
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO transcriptions (id, ts_ms, text, is_final, denied) "
                 "VALUES (50, 2000, 'a partial', 0, 0)")
    conn.execute("INSERT INTO transcriptions (id, ts_ms, text, is_final, denied) "
                 "VALUES (51, 3000, 'hallucinated', 1, 1)")
    conn.commit()
    conn.close()

    assert [row.text for row in demo_playback.load_schedule(path)] == ["kept"]


def test_load_schedule_of_an_empty_recording_is_empty(tmp_path):
    path = make_recording(tmp_path / "empty.db", [])

    assert demo_playback.load_schedule(path) == []
    assert demo_playback.schedule_length_s([]) == 0.0


def test_row_duration_never_collapses_to_nothing(tmp_path):
    path = make_recording(tmp_path / "s.db", [(1000, 4.0, 4.0, "instant", None)])

    assert demo_playback.load_schedule(path)[0].duration_s == demo_playback.MIN_ROW_DURATION_S


def test_reading_leaves_the_recording_untouched(recording):
    before = os.stat(recording)
    with open(recording, "rb") as handle:
        contents = handle.read()

    demo_playback.load_schedule(recording)

    assert not os.path.exists(recording + "-wal")
    assert not os.path.exists(recording + "-shm")
    with open(recording, "rb") as handle:
        assert handle.read() == contents
    assert os.stat(recording).st_size == before.st_size


# --- building the session --------------------------------------------------


def test_create_session_db_keeps_the_schema_and_drops_the_content(recording, tmp_path):
    dest = str(tmp_path / "live.db")

    demo_playback.create_session_db(recording, dest, 1_700_000_000.0)

    conn = sqlite3.connect(dest)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(transcriptions)")}
        assert {"ts_ms", "words_json", "translated_text", "segment_id"} <= columns
        assert conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM service_phase_blocks").fetchone()[0] == 0
        started = conn.execute(
            "SELECT value FROM session_meta WHERE key='session_started'").fetchone()
        assert started[0] != "old"
    finally:
        conn.close()


def test_session_path_follows_the_real_layout(tmp_path):
    path = demo_playback.session_path(str(tmp_path), 1_700_000_000.0)

    assert path.endswith(".db")
    assert os.path.basename(os.path.dirname(path)).isdigit()      # %m
    assert len(os.path.basename(path)) == len("2026-08-02_090000.db")


def test_rebase_row_preserves_the_gaps_between_captions(recording):
    schedule = demo_playback.load_schedule(recording)
    start_ms = 5_000_000

    stamps = [demo_playback.rebase_row(row, start_ms, i + 1)["ts_ms"]
              for i, row in enumerate(schedule)]

    assert stamps == [5_000_000, 5_004_000, 5_010_000]
    original = [row.columns["ts_ms"] for row in schedule]
    assert [b - a for a, b in zip(stamps, stamps[1:])] == \
           [b - a for a, b in zip(original, original[1:])]


def test_rebase_row_gives_the_display_the_identity_it_expects(recording):
    row = demo_playback.load_schedule(recording)[0]

    rebased = demo_playback.rebase_row(row, 5_000_000, 7)

    assert rebased["id"] == 7
    assert rebased["segment_id"] == "7"
    assert rebased["is_final"] == 1 and rebased["denied"] == 0
    assert rebased["corrected_by"] is None


def test_rebase_row_keeps_the_timestamp_consistent_with_ts_ms(recording):
    import time

    row = demo_playback.load_schedule(recording)[0]

    rebased = demo_playback.rebase_row(row, 5_000_000, 1)

    expected = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rebased["ts_ms"] / 1000.0))
    assert rebased["timestamp"] == expected


def test_rebase_row_stamps_translations_as_arriving_with_the_caption(recording):
    row = demo_playback.load_schedule(recording)[0]

    rebased = demo_playback.rebase_row(row, 5_000_000, 1)

    assert rebased["translated_text"] == "Мир вам"
    assert rebased["translation_ts_ms"] == rebased["ts_ms"]


def test_rebase_row_does_not_mutate_the_schedule(recording):
    row = demo_playback.load_schedule(recording)[0]

    demo_playback.rebase_row(row, 5_000_000, 99)

    assert row.columns["id"] == 2
    assert row.columns["ts_ms"] == 1_000_000


# --- what is due -----------------------------------------------------------


def test_due_rows_includes_a_row_scheduled_for_exactly_now(recording):
    schedule = demo_playback.load_schedule(recording)

    due, cursor = demo_playback.due_rows(schedule, 4.0, 0)

    assert [row.text for row in due] == ["Peace be with you", "Let us pray together"]
    assert cursor == 2


def test_due_rows_resumes_from_the_cursor(recording):
    schedule = demo_playback.load_schedule(recording)

    due, cursor = demo_playback.due_rows(schedule, 10.0, 2)

    assert [row.text for row in due] == ["Please be seated"]
    assert cursor == 3


def test_due_rows_at_the_end_returns_nothing(recording):
    schedule = demo_playback.load_schedule(recording)

    assert demo_playback.due_rows(schedule, 999.0, 3) == ([], 3)


def test_speaking_row_is_the_caption_being_said_before_it_commits(recording):
    schedule = demo_playback.load_schedule(recording)

    # Second caption holds 3.0s of speech (start_time 2.0, end_time 5.0) and commits
    # at offset 4.0, so its window is [1.0, 4.0).
    assert demo_playback.speaking_row(schedule, 3.0, 1).text == "Let us pray together"
    assert demo_playback.speaking_row(schedule, 1.0, 1).text == "Let us pray together"
    # At its commit moment it is no longer "in progress".
    assert demo_playback.speaking_row(schedule, 4.0, 1) is None
    # Before the window opens, nothing is being said.
    assert demo_playback.speaking_row(schedule, 0.5, 1) is None


def test_speaking_row_past_the_end_is_nothing(recording):
    schedule = demo_playback.load_schedule(recording)

    assert demo_playback.speaking_row(schedule, 50.0, 3) is None


# --- the live preview ------------------------------------------------------


def test_partial_text_grows_monotonically_and_completes():
    text = "Peace be with you brothers and sisters"

    seen = [demo_playback.partial_text(text, None, p / 10.0) for p in range(11)]

    assert seen[0] == ""
    assert seen[-1] == text
    for earlier, later in zip(seen, seen[1:]):
        assert later.startswith(earlier)


def test_partial_text_never_reveals_half_a_word():
    text = "Peace be with you"

    for step in range(11):
        revealed = demo_playback.partial_text(text, None, step / 10.0)
        assert revealed == "" or text.startswith(revealed)
        assert not revealed or revealed.split()[-1] in text.split()


def test_partial_text_uses_word_timings_when_the_recording_has_them():
    words = json.dumps([
        {"w": " Peace", "s_ms": 1000, "e_ms": 1400},
        {"w": " be", "s_ms": 1500, "e_ms": 1700},
        {"w": " with", "s_ms": 1800, "e_ms": 2000},
        {"w": " you", "s_ms": 2800, "e_ms": 3000},
    ])

    # "you" starts late in the window, so a half-progress reveal must not include it.
    halfway = demo_playback.partial_text("Peace be with you", words, 0.5)

    assert halfway == "Peace be with"
    assert demo_playback.partial_text("Peace be with you", words, 1.0) == "Peace be with you"


def test_partial_text_falls_back_when_words_json_is_unusable():
    for broken in ("not json", "[]", '{"not": "a list"}', "[1, 2, 3]"):
        assert demo_playback.partial_text("one two three four", broken, 0.5) == "one two"


def test_partial_text_handles_empty_and_out_of_range_progress():
    assert demo_playback.partial_text("hello", None, -1.0) == ""
    assert demo_playback.partial_text("hello", None, 5.0) == "hello"
    assert demo_playback.partial_text("", None, 0.5) == ""


# --- the meter -------------------------------------------------------------


def test_audio_level_is_quiet_between_captions():
    level = demo_playback.audio_level_for(None, random.Random(1))

    assert level.audio_type == "Quiet"
    assert 0 <= level.level < 20
    assert level.db < -45


def test_audio_level_is_speaking_during_a_caption(recording):
    row = demo_playback.load_schedule(recording)[0]

    level = demo_playback.audio_level_for(row, random.Random(1))

    assert level.audio_type == "Speaking"
    assert level.level >= 45


def test_audio_level_reports_music_when_the_recording_did(tmp_path):
    path = make_recording(tmp_path / "s.db", [(1000, 0.0, 3.0, "singing", None)])
    conn = sqlite3.connect(path)
    conn.execute("UPDATE transcriptions SET music_prob = 0.91 WHERE ts_ms = 1000")
    conn.commit()
    conn.close()
    row = demo_playback.load_schedule(path)[0]

    assert demo_playback.audio_level_for(row, random.Random(1)).audio_type == "Music"


def test_audio_level_is_reproducible_for_a_seed(recording):
    row = demo_playback.load_schedule(recording)[0]

    first = demo_playback.audio_level_for(row, random.Random(7)).level
    second = demo_playback.audio_level_for(row, random.Random(7)).level

    assert first == second


def test_audio_level_maps_onto_the_state_keys_the_ui_reads(recording):
    row = demo_playback.load_schedule(recording)[0]

    state = demo_playback.audio_level_for(row, random.Random(1)).as_state()

    assert set(state) == {"audio_level", "audio_db", "audio_energy", "audio_type", "music_prob"}


# --- the player ------------------------------------------------------------


def test_a_session_starts_empty_and_fills_as_time_passes(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock)

    db_path = player.begin_session()
    assert session_rows(db_path) == []
    assert state["running"] is True
    assert state["db_name"] == db_path
    assert state["session_id"] == os.path.splitext(os.path.basename(db_path))[0]

    clock.advance(0.1)
    player.tick()
    assert [row[2] for row in session_rows(db_path)] == ["Peace be with you"]

    clock.advance(4.0)
    player.tick()
    assert [row[2] for row in session_rows(db_path)] == [
        "Peace be with you", "Let us pray together"]

    clock.advance(6.0)
    player.tick()
    assert len(session_rows(db_path)) == 3


def test_rows_land_with_sequential_ids_the_display_can_key_on(recording, tmp_path):
    clock = FakeClock()
    player, _ = player_for(recording, tmp_path, clock)
    db_path = player.begin_session()

    clock.advance(20.0)
    player.tick()

    rows = session_rows(db_path)
    assert [row[0] for row in rows] == [1, 2, 3]
    assert [row[4] for row in rows] == ["1", "2", "3"]


def test_translations_are_replayed_with_the_captions(recording, tmp_path):
    clock = FakeClock()
    player, _ = player_for(recording, tmp_path, clock)
    db_path = player.begin_session()

    clock.advance(20.0)
    player.tick()

    assert [row[3] for row in session_rows(db_path)] == [
        "Мир вам", "Помолимся вместе", "Садитесь пожалуйста"]


def test_stopping_halts_insertion(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock)
    db_path = player.begin_session()
    clock.advance(0.1)
    player.tick()

    player.end_session()
    clock.advance(60.0)
    player.tick()

    assert len(session_rows(db_path)) == 1
    assert state["running"] is False
    assert state["live_text"] == ""


def test_speed_doubles_how_much_of_the_service_plays(recording, tmp_path):
    clock = FakeClock()
    player, _ = player_for(recording, tmp_path, clock, speed=2.0)
    db_path = player.begin_session()

    clock.advance(2.1)  # 4.2 virtual seconds
    player.tick()

    assert len(session_rows(db_path)) == 2


def test_changing_speed_keeps_the_current_position(recording, tmp_path):
    clock = FakeClock()
    player, _ = player_for(recording, tmp_path, clock)
    player.begin_session()
    clock.advance(4.0)

    player.set_speed(2.0)

    assert player.elapsed_s() == pytest.approx(4.0)
    clock.advance(1.0)
    assert player.elapsed_s() == pytest.approx(6.0)


def test_the_live_preview_reveals_a_caption_before_it_commits(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock)
    player.begin_session()

    # Second caption speaks from 2.0 to 4.0; sample partway through.
    clock.advance(3.0)
    player.tick()

    assert state["live_text"]
    assert "Let us pray together".startswith(state["live_text"])
    assert state["live_text"] != "Let us pray together"


def test_the_preview_clears_once_the_caption_has_landed(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock)
    player.begin_session()
    clock.advance(3.0)
    player.tick()

    clock.advance(1.5)
    player.tick()

    assert state["live_text"] == ""


def test_the_meter_moves_while_the_session_runs(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock)
    player.begin_session()

    clock.advance(3.0)
    player.tick()

    assert state["audio_type"] == "Speaking"
    assert state["audio_level"] >= 45


def test_reaching_the_end_rolls_over_into_a_new_session(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock, loop_gap_s=2.0)
    first = player.begin_session()

    clock.advance(20.0)
    player.tick()
    second = state["db_name"]

    assert second != first
    assert os.path.exists(second)
    assert session_rows(second) == []
    assert state["session_id"] == os.path.splitext(os.path.basename(second))[0]


def test_without_looping_the_session_simply_ends(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock, loop=False, loop_gap_s=1.0)
    player.begin_session()

    clock.advance(30.0)
    player.tick()

    assert state["running"] is False


def test_old_sessions_are_pruned_so_the_demo_does_not_grow(recording, tmp_path):
    clock = FakeClock()
    player, _ = player_for(recording, tmp_path, clock, keep_loops=2, loop_gap_s=0.5)

    player.begin_session()
    for _ in range(4):
        clock.advance(15.0)
        player.tick()

    on_disk = glob.glob(os.path.join(str(tmp_path / "sessions"), "**", "*.db"), recursive=True)
    assert len(on_disk) <= 2


def test_two_sessions_in_the_same_second_do_not_collide(recording, tmp_path):
    clock = FakeClock()
    player, _ = player_for(recording, tmp_path, clock)

    first = player.begin_session()
    second = player.begin_session()

    assert first != second
    assert os.path.exists(first) and os.path.exists(second)


def test_the_ui_is_told_a_real_model_name(recording, tmp_path):
    clock = FakeClock()
    player, state = player_for(recording, tmp_path, clock)

    player.begin_session()

    assert state["loaded_model"] == "large-v3"


# --- control ---------------------------------------------------------------


def test_start_and_stop_commands_drive_playback(recording, tmp_path):
    import queue

    clock = FakeClock()
    control = queue.Queue()
    config = demo_playback.PlaybackConfig(recording, str(tmp_path / "sessions"))
    state = {"start_time": 0}
    player = demo_playback.Player(config, state, control_queue=control, clock=clock)

    control.put({"command": "start"})
    player.tick()
    assert state["running"] is True

    control.put({"command": "stop"})
    player.tick()
    assert state["running"] is False


def test_a_second_start_while_running_is_ignored(recording, tmp_path):
    import queue

    clock = FakeClock()
    control = queue.Queue()
    config = demo_playback.PlaybackConfig(recording, str(tmp_path / "sessions"))
    state = {"start_time": 0}
    player = demo_playback.Player(config, state, control_queue=control, clock=clock)
    control.put({"command": "start"})
    player.tick()
    first = state["db_name"]

    control.put({"command": "start"})
    player.tick()

    assert state["db_name"] == first


def test_unknown_control_messages_are_ignored(recording, tmp_path):
    import queue

    control = queue.Queue()
    config = demo_playback.PlaybackConfig(recording, str(tmp_path / "sessions"))
    player = demo_playback.Player(config, {"start_time": 0}, control_queue=control,
                                  clock=FakeClock())

    control.put("not a dict")
    control.put({"command": "calibrate"})
    player.tick()  # must not raise


def test_the_player_stands_in_for_the_worker_process(recording, tmp_path):
    """start_transcription reuses a live worker instead of spawning one."""
    config = demo_playback.PlaybackConfig(recording, str(tmp_path / "sessions"))
    player = demo_playback.Player(config, {"start_time": 0}, clock=FakeClock())

    assert player.is_alive() is True
    assert player.pid == os.getpid()


def test_a_tick_before_anything_started_does_nothing(recording, tmp_path):
    player, state = player_for(recording, tmp_path, FakeClock())

    player.tick()

    assert state.get("running") is not True


# --- against a real recorded service ---------------------------------------


def _real_sessions():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sessions")
    return [p for p in sorted(glob.glob(os.path.join(root, "*.db"))) if os.path.getsize(p) > 0]


@pytest.mark.skipif(not _real_sessions(), reason="no recorded sessions available")
def test_a_real_service_loads_and_replays(tmp_path):
    """A migration-scarred database from a real service still schedules and replays."""
    source = _real_sessions()[0]
    schedule = demo_playback.load_schedule(source)
    assert len(schedule) > 100
    assert schedule[0].offset_s == 0.0
    assert all(a.offset_s <= b.offset_s for a, b in zip(schedule, schedule[1:]))

    clock = FakeClock()
    config = demo_playback.PlaybackConfig(source, str(tmp_path / "sessions"))
    player = demo_playback.Player(config, {"start_time": 0}, clock=clock)
    db_path = player.begin_session()

    clock.advance(schedule[10].offset_s)
    player.tick()

    assert len(session_rows(db_path)) == 11


@pytest.mark.skipif(not _real_sessions(), reason="no recorded sessions available")
def test_replaying_a_real_service_leaves_it_byte_identical(tmp_path):
    import hashlib

    source = _real_sessions()[0]
    digest = hashlib.sha256(open(source, "rb").read()).hexdigest()

    config = demo_playback.PlaybackConfig(source, str(tmp_path / "sessions"))
    player = demo_playback.Player(config, {"start_time": 0}, clock=FakeClock())
    player.begin_session()
    player.tick()

    assert hashlib.sha256(open(source, "rb").read()).hexdigest() == digest

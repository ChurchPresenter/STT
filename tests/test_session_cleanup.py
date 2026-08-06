"""Telling finished services apart from test runs (stt/session_cleanup.py).

Only one verdict is deletable, so the tests that matter most are the ones
proving a real service — including a service that was restarted halfway, which
leaves a short second file — never reaches it.
"""

import os
import sqlite3
from datetime import datetime

import pytest

from stt.session_cleanup import (
    Refusal,
    plan_deletion,
    DEFAULT_SLOTS,
    UNDELETABLE,
    Session,
    classify,
    classify_live,
    idle_minutes,
    in_window,
    last_written,
    nearest_db,
    parse_slots,
    parse_started_at,
    read_session_span,
    scan,
    wav_seconds,
)

SUNDAY = datetime(2026, 1, 11)      # a Sunday
WEDNESDAY = datetime(2026, 1, 14)


def session_at(when, minutes, rows=100):
    return Session("/x/" + when.strftime("%Y-%m-%d_%H%M%S") + ".db",
                   when, minutes, rows, (), 0)


class TestParseStartedAt:
    def test_current_naming(self):
        assert parse_started_at("2026-07-05_093218.db") == datetime(2026, 7, 5, 9, 32, 18)

    def test_older_naming_with_suffix(self):
        assert parse_started_at("2026-01-11__10-01-48_Transcriptions.db") == \
            datetime(2026, 1, 11, 10, 1, 48)

    def test_recording_companion_of_the_older_naming(self):
        assert parse_started_at("2026-01-11__10-01-43_Recording.ts") == \
            datetime(2026, 1, 11, 10, 1, 43)

    def test_full_path_is_accepted(self):
        assert parse_started_at("/a/b/2026-07-05_093218.srt") == datetime(2026, 7, 5, 9, 32, 18)

    @pytest.mark.parametrize("name", ["notes.txt", "2026-13-99_000000.db", "", "README.md"])
    def test_anything_else_is_not_a_session_file(self, name):
        assert parse_started_at(name) is None


class TestNearestDb:
    """A recording is named for when capture began, its database for the session."""

    def test_picks_the_closer_of_two_databases_seconds_apart(self):
        dbs = [("/a/first.db", datetime(2026, 1, 11, 10, 35, 30)),
               ("/a/second.db", datetime(2026, 1, 11, 10, 35, 52))]
        # The archive really does hold two sessions 22 seconds apart.
        assert nearest_db(datetime(2026, 1, 11, 10, 35, 29), dbs) == "/a/first.db"
        assert nearest_db(datetime(2026, 1, 11, 10, 35, 50), dbs) == "/a/second.db"

    def test_a_recording_starting_before_its_database_still_matches(self):
        dbs = [("/a/s.db", datetime(2026, 7, 5, 9, 32, 18))]
        assert nearest_db(datetime(2026, 7, 5, 9, 31, 47), dbs) == "/a/s.db"

    def test_nothing_within_the_limit_is_an_orphan(self):
        dbs = [("/a/s.db", datetime(2026, 7, 5, 9, 32, 18))]
        assert nearest_db(datetime(2026, 7, 5, 14, 0, 0), dbs) is None


class TestInWindow:
    def test_a_service_starting_on_time(self):
        assert in_window(SUNDAY.replace(hour=10), DEFAULT_SLOTS, 60, 150) is not None

    def test_setting_up_an_hour_early_is_still_the_service_window(self):
        assert in_window(SUNDAY.replace(hour=9, minute=5), DEFAULT_SLOTS, 60, 150) is not None

    def test_a_restart_partway_through_is_still_the_service_window(self):
        assert in_window(SUNDAY.replace(hour=11, minute=20), DEFAULT_SLOTS, 60, 150) is not None

    def test_the_same_time_on_the_wrong_day_is_not(self):
        tuesday = datetime(2026, 1, 13, 10, 0)
        assert in_window(tuesday, DEFAULT_SLOTS, 60, 150) is None

    def test_wednesday_evening_is_covered(self):
        assert in_window(WEDNESDAY.replace(hour=19), DEFAULT_SLOTS, 60, 150) is not None

    def test_wednesday_afternoon_is_not(self):
        assert in_window(WEDNESDAY.replace(hour=14), DEFAULT_SLOTS, 60, 150) is None


class TestClassify:
    def test_a_full_sunday_morning_service_is_kept(self):
        verdict, _ = classify(session_at(SUNDAY.replace(hour=10, minute=2), 95))
        assert verdict == "service"

    def test_a_full_sunday_evening_service_is_kept(self):
        verdict, _ = classify(session_at(SUNDAY.replace(hour=17, minute=58), 100))
        assert verdict == "service"

    def test_a_short_session_inside_a_service_window_is_never_a_test(self):
        """A service whose operator restarted leaves a short second file.

        Deleting by length would throw away half a service, which is why this
        verdict exists and why nothing deletes it.
        """
        verdict, _ = classify(session_at(SUNDAY.replace(hour=10, minute=35), 27))
        assert verdict == "fragment"

    def test_a_two_minute_run_during_the_window_is_still_only_a_fragment(self):
        verdict, _ = classify(session_at(SUNDAY.replace(hour=9, minute=40), 2))
        assert verdict == "fragment"

    def test_a_weekday_run_is_a_test(self):
        verdict, why = classify(session_at(datetime(2026, 1, 13, 14, 30), 40))
        assert verdict == "test"
        assert "outside" in why

    def test_a_sunday_run_hours_from_any_service_is_a_test(self):
        verdict, _ = classify(session_at(SUNDAY.replace(hour=14), 45))
        assert verdict == "test"

    def test_a_long_run_outside_the_window_is_still_a_test(self):
        # Length alone must not rescue it — an afternoon of model testing can
        # easily run longer than a service. This is what set the window span:
        # at three hours the 10:00 service still claimed 13:00.
        verdict, _ = classify(session_at(SUNDAY.replace(hour=13), 200))
        assert verdict == "test"

    def test_custom_slots_are_honoured(self):
        slots = parse_slots("fri@19:30")
        friday = datetime(2026, 1, 16, 19, 35)
        assert classify(session_at(friday, 95), slots)[0] == "service"
        assert classify(session_at(SUNDAY.replace(hour=10), 95), slots)[0] == "test"


class TestParseSlots:
    def test_reads_a_schedule(self):
        assert parse_slots("sun@10:00,wed@19:00") == ((6, 10, 0), (2, 19, 0))

    def test_tolerates_spacing_and_case(self):
        assert parse_slots(" SUN@10:00 , Wed@19:00 ") == ((6, 10, 0), (2, 19, 0))

    @pytest.mark.parametrize("spec", ["", "sunday@10", "sun 10:00", "xyz@10:00"])
    def test_rejects_nonsense(self, spec):
        with pytest.raises(ValueError):
            parse_slots(spec)


def make_db(path, started, rows):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "timestamp TEXT, text TEXT, start_time REAL, end_time REAL)")
    for i, minutes in enumerate(rows):
        stamp = "" if minutes is None else \
            (started.timestamp() and datetime.fromtimestamp(
                started.timestamp() + minutes * 60).strftime("%Y-%m-%d %H:%M:%S"))
        conn.execute("INSERT INTO transcriptions (timestamp, text, start_time, end_time) "
                     "VALUES (?, ?, ?, ?)", (stamp, f"line {i}", 0.0, 0.0))
    conn.commit()
    conn.close()


class TestReadSessionSpan:
    def test_measures_to_the_last_row(self, tmp_path):
        started = datetime(2026, 1, 11, 10, 0, 0)
        db = tmp_path / "2026-01-11_100000.db"
        make_db(db, started, [0, 30, 95])
        minutes, rows = read_session_span(str(db), started)
        assert rows == 3
        assert minutes == pytest.approx(95, abs=0.5)

    def test_ignores_unusable_timestamps(self, tmp_path):
        """Real archives hold rows whose timestamp is blank."""
        started = datetime(2026, 1, 11, 10, 0, 0)
        db = tmp_path / "2026-01-11_100000.db"
        make_db(db, started, [0, 60, None])
        minutes, _ = read_session_span(str(db), started)
        assert minutes == pytest.approx(60, abs=0.5)

    def test_a_database_with_no_rows_has_no_span(self, tmp_path):
        started = datetime(2026, 1, 11, 10, 0, 0)
        db = tmp_path / "2026-01-11_100000.db"
        make_db(db, started, [])
        assert read_session_span(str(db), started) == (0.0, 0)

    def test_an_unreadable_file_is_not_an_error(self, tmp_path):
        broken = tmp_path / "2026-01-11_100000.db"
        broken.write_text("not a database")
        assert read_session_span(str(broken), datetime(2026, 1, 11, 10, 0)) == (0.0, 0)


class TestScan:
    def test_groups_a_session_with_its_recordings_and_exports(self, tmp_path):
        started = datetime(2026, 1, 11, 10, 0, 0)
        make_db(tmp_path / "2026-01-11_100000.db", started, [0, 95])
        for name in ("2026-01-11_095930.ts", "2026-01-11_095932.wav",
                     "2026-01-11_100000.srt", "2026-01-11_100000.html",
                     "2026-01-11_100000.db-wal"):
            (tmp_path / name).write_bytes(b"x" * 100)

        sessions, orphans = scan(str(tmp_path))
        assert len(sessions) == 1
        assert len(sessions[0].files) == 6, "database plus its five companions"
        assert orphans == []
        assert sessions[0].total_bytes > 0

    def test_two_sessions_seconds_apart_keep_their_own_files(self, tmp_path):
        first, second = datetime(2026, 1, 11, 10, 35, 30), datetime(2026, 1, 11, 10, 35, 52)
        make_db(tmp_path / "2026-01-11_103530.db", first, [0, 5])
        make_db(tmp_path / "2026-01-11_103552.db", second, [0, 5])
        (tmp_path / "2026-01-11_103529.ts").write_bytes(b"x")
        (tmp_path / "2026-01-11_103550.ts").write_bytes(b"x")

        sessions, _ = scan(str(tmp_path))
        by_name = {s.name: s for s in sessions}
        assert any(f.endswith("103529.ts") for f in by_name["2026-01-11_103530.db"].files)
        assert any(f.endswith("103550.ts") for f in by_name["2026-01-11_103552.db"].files)

    def test_a_stray_file_far_from_any_session_is_an_orphan(self, tmp_path):
        make_db(tmp_path / "2026-01-11_100000.db", datetime(2026, 1, 11, 10, 0), [0])
        (tmp_path / "2026-01-11_180000.ts").write_bytes(b"x")
        _sessions, orphans = scan(str(tmp_path))
        assert [o.split("/")[-1] for o in orphans] == ["2026-01-11_180000.ts"]

    def test_unrelated_files_are_ignored_entirely(self, tmp_path):
        make_db(tmp_path / "2026-01-11_100000.db", datetime(2026, 1, 11, 10, 0), [0])
        (tmp_path / "notes.txt").write_text("hello")
        sessions, orphans = scan(str(tmp_path))
        assert orphans == []
        assert all(not f.endswith("notes.txt") for f in sessions[0].files)

    def test_walks_the_year_month_layout(self, tmp_path):
        month = tmp_path / "2026" / "01"
        month.mkdir(parents=True)
        make_db(month / "2026-01-11_100000.db", datetime(2026, 1, 11, 10, 0), [0, 95])
        sessions, _ = scan(str(tmp_path))
        assert len(sessions) == 1


class TestSubstantialOffScheduleSessions:
    """The schedule describes the ordinary week, not every week.

    Measured against a real archive, schedule-only classification put a
    154-minute Saturday recording holding 2650 transcribed lines in the delete
    pile. Length with substance is kept for a human to look at; length without
    it — an afternoon of model testing — is still a test.
    """

    def test_a_recording_with_no_transcript_is_a_test(self):
        # 888 MB of audio and zero lines: nothing was said, or nothing was heard.
        friday = datetime(2026, 3, 27, 12, 16)
        assert classify(Session("/x.db", friday, 0.0, 0, (), 0))[0] == "test"

    def test_a_long_dense_saturday_is_kept_for_review(self):
        saturday = datetime(2026, 4, 11, 14, 44)
        verdict, why = classify(Session("/x.db", saturday, 153.7, 2650, (), 0))
        assert verdict == "unusual"
        assert "2650" in why

    def test_a_long_but_sparse_run_is_still_a_test(self):
        # 99 lines over 83 minutes is a microphone left open, not a service.
        thursday = datetime(2026, 4, 23, 19, 9)
        assert classify(Session("/x.db", thursday, 82.7, 99, (), 0))[0] == "test"

    def test_a_short_but_dense_run_is_kept_too(self):
        """43 minutes and 460 lines, on a Friday, was in the delete pile.

        Hundreds of transcribed lines mean someone was speaking for a while,
        which is not what a test looks like however short the file is.
        """
        friday = datetime(2026, 4, 10, 19, 7)
        assert classify(Session("/x.db", friday, 43.3, 460, (), 0))[0] == "unusual"

    def test_the_thresholds_can_be_moved(self):
        saturday = datetime(2026, 4, 11, 14, 44)
        session = Session("/x.db", saturday, 153.7, 2650, (), 0)
        assert classify(session, substantial_rows=5000)[0] == "test"


class TestEarlyStarts:
    """Operators start capture well before the service begins."""

    def test_an_hour_and_a_half_early_still_belongs_to_the_service(self):
        # Sun 08:53 for a 10:00 service: real, and a 60-minute lead missed it.
        verdict, _ = classify(session_at(datetime(2026, 4, 12, 8, 53), 193))
        assert verdict == "service"

    def test_nearly_two_hours_early_still_belongs(self):
        verdict, _ = classify(session_at(datetime(2026, 6, 7, 8, 10), 277))
        assert verdict == "service"

    def test_an_hour_early_for_the_evening_service_belongs(self):
        verdict, _ = classify(session_at(datetime(2026, 5, 3, 16, 57), 166))
        assert verdict == "service"

    def test_but_the_middle_of_the_night_does_not(self):
        assert classify(session_at(datetime(2026, 4, 12, 3, 0), 200))[0] in ("test", "unusual")


class TestReadingDoesNotModify:
    """Inspecting an archive must not change it.

    A read-only SQLite open still takes locks, and on a WAL database that
    touches the -shm sidecar — scanning .62 rewrote the modification time of all
    86 sessions there, which is both untrue and destroys the signal a caller
    might use to decide what is safe to delete.
    """

    def test_scanning_leaves_every_file_untouched(self, tmp_path):
        started = datetime(2026, 1, 11, 10, 0, 0)
        db = tmp_path / "2026-01-11_100000.db"
        make_db(db, started, [0, 95])
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA journal_mode=WAL")   # what a live session leaves behind
        conn.close()

        before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
        names_before = sorted(p.name for p in tmp_path.iterdir())

        read_session_span(str(db), started)

        after = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
        assert sorted(after) == names_before, "no sidecar may be created"
        assert after == before, "no file may be rewritten"


def write_wav(path, seconds, rate=16000, declare=None):
    """A RIFF header plus PCM. ``declare`` overrides the data-size field."""
    import struct
    data = b"\0\0" * int(rate * seconds)
    declared = len(data) if declare is None else declare
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
              + b"data" + struct.pack("<I", declared))
    with open(path, "wb") as fh:
        fh.write(header + data)


class TestWavLength:
    def test_reads_a_well_formed_header(self, tmp_path):
        p = tmp_path / "a.wav"
        write_wav(str(p), 60)
        assert wav_seconds(str(p)) == pytest.approx(60, abs=0.1)

    def test_a_stale_header_does_not_shrink_the_file(self, tmp_path):
        """These recordings are streamed and the size field is never patched.

        A 476 MB archive file declares 32000 bytes — one second — so anything
        that trusts the header (ffprobe, most players) sees a second of audio.
        The bytes on disk are what was actually recorded.
        """
        p = tmp_path / "streamed.wav"
        write_wav(str(p), 300, declare=32000)
        assert wav_seconds(str(p)) == pytest.approx(300, abs=1)

    def test_trailing_chunks_do_not_count_as_stale(self, tmp_path):
        p = tmp_path / "b.wav"
        write_wav(str(p), 60)
        with open(p, "ab") as fh:              # a small LIST/INFO chunk after the audio
            fh.write(b"LIST" + b"\0" * 40)
        assert wav_seconds(str(p)) == pytest.approx(60, abs=0.2)

    def test_not_a_wav_at_all(self, tmp_path):
        p = tmp_path / "c.wav"
        p.write_bytes(b"not a riff file at all, honestly")
        assert wav_seconds(str(p)) is None


# --- The guard that was only in the CLI wrapper -------------------------------


def session_with_files(tmp_path, when, minutes, rows, names=("s.db", "s.wav")):
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        paths.append(str(p))
    return Session(paths[0], when, minutes, rows, tuple(paths), 0)


class TestLastWritten:
    """Sidecar mtimes say when a reader looked, not when the session was written."""

    def test_ignores_wal_and_shm(self, tmp_path):
        real = tmp_path / "s.db"
        real.write_bytes(b"x")
        os.utime(str(real), (1000, 1000))
        for side in ("s.db-wal", "s.db-shm", "s.db-journal"):
            p = tmp_path / side
            p.write_bytes(b"x")
            os.utime(str(p), (9_000_000, 9_000_000))
        paths = [str(real)] + [str(tmp_path / s) for s in ("s.db-wal", "s.db-shm", "s.db-journal")]
        assert last_written(paths) == 1000

    def test_missing_files_are_skipped(self, tmp_path):
        real = tmp_path / "s.db"
        real.write_bytes(b"x")
        os.utime(str(real), (2000, 2000))
        assert last_written([str(real), str(tmp_path / "gone.wav")]) == 2000

    def test_nothing_readable_is_zero(self, tmp_path):
        assert last_written([str(tmp_path / "gone")]) == 0.0


class TestIdleMinutes:
    def test_measured_against_the_supplied_now(self, tmp_path):
        s = session_with_files(tmp_path, SUNDAY, 5, 10)
        for f in s.files:
            os.utime(f, (10_000, 10_000))
        assert idle_minutes(s, now=10_600) == pytest.approx(10.0)

    def test_undateable_session_is_not_treated_as_live(self, tmp_path):
        # No readable file left: that is not evidence of being written right now.
        s = Session(str(tmp_path / "gone.db"), SUNDAY, 5, 10, (str(tmp_path / "gone.db"),), 0)
        assert idle_minutes(s, now=10_000) == float("inf")


class TestClassifyLive:
    """A session being written now must outrank every other verdict.

    Without this the recording in progress looks exactly like a test: it is
    off-schedule as often as not, and has only a handful of rows early on.
    """

    def test_in_progress_beats_test(self, tmp_path):
        friday_2am = datetime(2026, 1, 9, 2, 0, 0)          # outside every window
        s = session_with_files(tmp_path, friday_2am, 3, 12)
        for f in s.files:
            os.utime(f, (10_000, 10_000))
        assert classify(s)[0] == "test"                     # what the plain call says
        assert classify_live(s, now=10_120)[0] == "in progress"

    def test_idle_session_is_judged_normally(self, tmp_path):
        friday_2am = datetime(2026, 1, 9, 2, 0, 0)
        s = session_with_files(tmp_path, friday_2am, 3, 12)
        for f in s.files:
            os.utime(f, (10_000, 10_000))
        assert classify_live(s, now=10_000 + 3600)[0] == "test"

    def test_a_live_service_is_still_protected(self, tmp_path):
        s = session_with_files(tmp_path, SUNDAY.replace(hour=10), 120, 2000)
        for f in s.files:
            os.utime(f, (10_000, 10_000))
        assert classify_live(s, now=10_060)[0] == "in progress"

    def test_thresholds_are_passed_through(self, tmp_path):
        friday = datetime(2026, 1, 9, 2, 0, 0)
        s = session_with_files(tmp_path, friday, 13, 341)
        for f in s.files:
            os.utime(f, (10_000, 10_000))
        old = 10_000 + 86_400
        # 341 rows clears the 300 default, so it is held back as "unusual" ...
        assert classify_live(s, now=old)[0] == "unusual"
        # ... but an operator who knows their sample runs can say otherwise.
        assert classify_live(s, now=old, substantial_rows=500)[0] == "test"


class TestUndeletable:
    def test_service_and_in_progress_are_never_deletable(self):
        assert "service" in UNDELETABLE
        assert "in progress" in UNDELETABLE

    def test_test_and_fragment_are_not_in_the_guard(self):
        assert "test" not in UNDELETABLE
        assert "fragment" not in UNDELETABLE


# --- What a sweep is allowed to remove ---------------------------------------


def scanned(tmp_path, name, verdict, extra=()):
    """A scan-shaped dict, with its files actually on disk."""
    files = []
    for suffix in ("", *extra):
        p = tmp_path / (name + suffix)
        p.write_bytes(b"x")
        files.append(str(p))
    return {"db_path": files[0], "name": name, "verdict": verdict, "files": files}


class TestPlanDeletion:
    def test_a_test_session_goes_whole(self, tmp_path):
        s = scanned(tmp_path, "t.db", "test", (".wav", ".srt"))
        go, refused = plan_deletion([s], [s["db_path"]])
        assert sorted(go) == sorted(s["files"])
        assert refused == []

    def test_a_service_is_refused_however_it_is_asked_for(self, tmp_path):
        s = scanned(tmp_path, "s.db", "service")
        go, refused = plan_deletion([s], [s["db_path"]])
        assert go == []
        assert refused == [Refusal("s.db", "service")]

    def test_a_session_being_written_is_refused(self, tmp_path):
        s = scanned(tmp_path, "p.db", "in progress")
        go, refused = plan_deletion([s], [s["db_path"]])
        assert go == [] and refused[0].reason == "in progress"

    def test_the_recording_session_is_refused_even_if_classified_test(self, tmp_path):
        # The classifier may not have noticed yet; the live path is authoritative.
        s = scanned(tmp_path, "live.db", "test")
        go, refused = plan_deletion([s], [s["db_path"]], live_db=s["db_path"])
        assert go == []
        assert refused[0].reason == "this session is recording"

    def test_an_unknown_path_cannot_be_named(self, tmp_path):
        s = scanned(tmp_path, "t.db", "test")
        go, refused = plan_deletion([s], ["/etc/passwd"])
        assert go == []
        assert refused[0].reason == "not a session under this directory"

    def test_a_wal_with_bytes_in_it_is_never_removed(self, tmp_path):
        s = scanned(tmp_path, "t.db", "test", (".wav",))
        wal = tmp_path / "t.db-wal"
        wal.write_bytes(b"unmerged rows")
        s["files"].append(str(wal))
        go, refused = plan_deletion([s], [s["db_path"]])
        assert str(wal) not in go
        assert any(r.reason == "unmerged -wal" for r in refused)
        assert len(go) == 2          # the rest of the session still goes

    def test_an_empty_wal_goes_with_the_session(self, tmp_path):
        s = scanned(tmp_path, "t.db", "test")
        wal = tmp_path / "t.db-wal"
        wal.write_bytes(b"")
        s["files"].append(str(wal))
        go, refused = plan_deletion([s], [s["db_path"]])
        assert str(wal) in go and refused == []

    def test_fragments_and_unusual_are_allowed_when_asked_for(self, tmp_path):
        # "let dev choose" — the guard only ever protects service and in progress.
        for verdict in ("fragment", "unusual"):
            s = scanned(tmp_path, f"{verdict}.db", verdict)
            go, refused = plan_deletion([s], [s["db_path"]])
            assert go and refused == []

    def test_nothing_requested_deletes_nothing(self, tmp_path):
        s = scanned(tmp_path, "t.db", "test")
        assert plan_deletion([s], []) == ([], [])

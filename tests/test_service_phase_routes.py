"""The /api/service-phase routes.

Extracted from the monolith and run against a stub namespace (see tests/conftest.py) — the
module cannot be imported, and CI installs no Flask. jsonify is stubbed to return the
mapping unchanged, which is what these assertions care about.

These routes are live-only and take no path from the caller, so there is nothing to
confine — the behaviour worth guarding is that they read the running session and nothing
else, and that a session without the tables reads as empty rather than as an error.
"""

import sqlite3

import pytest

from conftest import extract_definitions
from stt.coercion import coerce_int
from stt.service_phase import (
    analyze,
    delete_correction,
    load_analysis,
    load_corrections,
    read_rows,
    save_analysis,
    save_correction,
)

MIN = 60_000
CFG = {"enabled": True, "sermon_min_minutes": 8, "songs_min_minutes": 3,
       "cue_phrases": {"amen": [r"амин[ья]"]}}


def session_db(tmp_path, spec="M" * 5 + "S" * 12, name="2026-03-01_093218.db", analyzed=True):
    """A session db shaped like a real one, optionally with detector output already saved."""
    path = str(tmp_path / name)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "ts_ms INTEGER, speech_type TEXT, music_prob REAL, text TEXT, is_final INTEGER)")
    conn.executemany(
        "INSERT INTO transcriptions (ts_ms, speech_type, music_prob, text, is_final) "
        "VALUES (?, ?, ?, ?, 1)",
        [(1_000_000 + i * MIN, {"M": "Music", "S": "Speaking"}[c], 0.9 if c == "M" else 0.0, "")
         for i, c in enumerate(spec)])
    conn.commit()
    if analyzed:
        save_analysis(conn, analyze(read_rows(conn), CFG))
    conn.close()
    return path


def make_ns(*, live_db=None, params=None, args=None):
    ns = extract_definitions(
        "speech_to_text.py",
        ["_service_phase_resolve_db", "get_service_phase", "save_service_phase_correction",
         "delete_service_phase_correction", "_service_phase_first_sunday",
         "_service_phase_config"],
        extra_globals={
            "config": {"service_phase": CFG},
            "request": type("R", (), {
                "remote_addr": "127.0.0.1",
                "args": args or {},
            })(),
            "jsonify": lambda payload: payload,
            "check_ip_whitelist": lambda: True,
            "sqlite3": sqlite3,
            "coerce_int": coerce_int,
            "_control_params": lambda keep_blank=False: params or {},
            "_service_phase_session_db": lambda: live_db,
            "_service_phase_analyze": analyze,
            "_service_phase_load": load_analysis,
            "_service_phase_rows": read_rows,
            "_service_phase_corrections": load_corrections,
            "_service_phase_save_correction": save_correction,
            "_service_phase_delete_correction": delete_correction,
            "app": type("A", (), {"route": staticmethod(lambda *a, **k: (lambda f: f))})(),
            "datetime": __import__("datetime").datetime,
        })
    return ns


class TestFirstSunday:
    """Communion's usual slot — read from the session's own date, not today's."""

    def test_a_first_sunday_is_recognised(self):
        ns = make_ns()
        assert ns["_service_phase_first_sunday"]("/x/2026-03-01_093218.db") is True

    def test_a_later_sunday_is_not(self):
        ns = make_ns()
        assert ns["_service_phase_first_sunday"]("/x/2026-03-15_090615.db") is False

    def test_a_weekday_is_not(self):
        ns = make_ns()
        assert ns["_service_phase_first_sunday"]("/x/2026-02-11_183702.db") is False

    @pytest.mark.parametrize("bad", ["", None, "/x/nonsense.db", "/x/not-a-date_1.db"])
    def test_an_unparseable_name_is_survivable(self, bad):
        ns = make_ns()
        assert ns["_service_phase_first_sunday"](bad) is False


class TestGetServicePhase:
    def test_returns_the_running_sessions_saved_timeline(self, tmp_path):
        db = session_db(tmp_path)
        body = make_ns(live_db=db)["get_service_phase"]()
        assert body["success"] is True
        assert [b["kind"] for b in body["blocks"]] == ["M", "S"]
        assert body["current"]["kind"] == "S"
        assert body["session_id"] == "2026-03-01_093218.db"

    def test_no_running_session_is_a_404_not_a_crash(self):
        body, status = make_ns(live_db=None)["get_service_phase"]()
        assert status == 404 and body["success"] is False

    def test_a_session_argument_is_ignored(self, tmp_path):
        # The routes are live-only: a caller-supplied path must not reach sqlite at all.
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        body = make_ns(live_db=db, args={"session": other})["get_service_phase"]()
        assert body["session_id"] == "2026-03-01_093218.db"

    def test_recompute_reruns_the_detector_without_saving(self, tmp_path):
        db = session_db(tmp_path, analyzed=False)
        body = make_ns(live_db=db, args={"recompute": "1"})["get_service_phase"]()
        assert body["recomputed"] is True
        assert [b["kind"] for b in body["blocks"]] == ["M", "S"]
        # Nothing was written: reading it back without recompute is still empty.
        assert make_ns(live_db=db)["get_service_phase"]()["blocks"] == []

    def test_a_session_without_the_tables_reads_as_empty(self, tmp_path):
        db = session_db(tmp_path, analyzed=False)
        body = make_ns(live_db=db)["get_service_phase"]()
        assert body["success"] is True and body["blocks"] == []

    def test_first_sunday_is_reported_from_the_session_name(self, tmp_path):
        db = session_db(tmp_path, name="2026-03-01_093218.db")
        assert make_ns(live_db=db)["get_service_phase"]()["first_sunday"] is True


class TestSaveCorrection:
    def test_saves_and_returns_the_new_list(self, tmp_path):
        db = session_db(tmp_path)
        body = make_ns(live_db=db, params={"block_index": 1, "kind": "S",
                                           "label": "Communion"})["save_service_phase_correction"]()
        assert body["success"] is True
        assert [c["label"] for c in body["corrections"]] == ["Communion"]

    def test_a_correction_survives_the_next_detector_run(self, tmp_path):
        db = session_db(tmp_path)
        make_ns(live_db=db, params={"block_index": 1, "kind": "S",
                                    "label": "Communion"})["save_service_phase_correction"]()
        conn = sqlite3.connect(db)
        save_analysis(conn, analyze(read_rows(conn), CFG))
        assert [c["label"] for c in load_corrections(conn)] == ["Communion"]
        conn.close()

    def test_an_empty_correction_is_rejected(self, tmp_path):
        db = session_db(tmp_path)
        body, status = make_ns(live_db=db,
                               params={"block_index": 1})["save_service_phase_correction"]()
        assert status == 400 and body["success"] is False

    def test_a_session_argument_cannot_redirect_the_write(self, tmp_path):
        # Live-only: a path in the body must not steer the correction to another file.
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        make_ns(live_db=db, params={"session": other, "block_index": 0,
                                    "label": "Songs"})["save_service_phase_correction"]()
        assert len(load_corrections(sqlite3.connect(db))) == 1
        assert load_corrections(sqlite3.connect(other)) == []

    def test_no_running_session_is_a_404(self):
        _, status = make_ns(live_db=None, params={"block_index": 0,
                                                  "label": "x"})["save_service_phase_correction"]()
        assert status == 404

    def test_long_free_text_is_truncated_not_stored_whole(self, tmp_path):
        db = session_db(tmp_path)
        body = make_ns(live_db=db, params={"block_index": 0, "kind": "M", "label": "x" * 500,
                                           "note": "y" * 5000})["save_service_phase_correction"]()
        assert len(body["corrections"][0]["label"]) <= 120
        assert len(body["corrections"][0]["note"]) <= 500

    def test_a_block_index_of_zero_is_a_block_not_a_missing_one(self, tmp_path):
        # Falsy-but-present: block 0 is the first block of every service.
        db = session_db(tmp_path)
        body = make_ns(live_db=db, params={"block_index": 0, "kind": "M",
                                           "label": "Songs"})["save_service_phase_correction"]()
        assert body["corrections"][0]["block_index"] == 0


class TestDeleteCorrection:
    """Undo hands the block back to the detector, rather than overwriting one guess with another."""

    def correct(self, db, **params):
        return make_ns(live_db=db, params=params)["save_service_phase_correction"]()

    def test_removes_the_correction_and_returns_the_new_list(self, tmp_path):
        db = session_db(tmp_path)
        self.correct(db, block_index=1, kind="S", label="Communion")
        body = make_ns(live_db=db,
                       params={"block_index": 1})["delete_service_phase_correction"]()
        assert body["success"] is True and body["removed"] == 1
        assert body["corrections"] == []

    def test_it_leaves_the_other_blocks_corrections_alone(self, tmp_path):
        db = session_db(tmp_path)
        self.correct(db, block_index=0, kind="M", label="Other")
        self.correct(db, block_index=1, kind="S", label="Opening")
        body = make_ns(live_db=db,
                       params={"block_index": 0})["delete_service_phase_correction"]()
        assert [c["block_index"] for c in body["corrections"]] == [1]

    def test_undoing_nothing_is_a_success_with_no_rows_removed(self, tmp_path):
        # The page can race a re-render against the click; a no-op beats a 500.
        db = session_db(tmp_path)
        body = make_ns(live_db=db,
                       params={"block_index": 4})["delete_service_phase_correction"]()
        assert body["success"] is True and body["removed"] == 0

    def test_a_missing_block_index_is_rejected(self, tmp_path):
        # Without one the delete has no subject; it must not fall through to a wider wipe.
        db = session_db(tmp_path)
        self.correct(db, block_index=1, kind="S", label="Communion")
        body, status = make_ns(live_db=db, params={})["delete_service_phase_correction"]()
        assert status == 400 and body["success"] is False
        assert len(load_corrections(sqlite3.connect(db))) == 1

    def test_block_zero_is_undoable(self, tmp_path):
        # Falsy-but-present, the same trap the save route has.
        db = session_db(tmp_path)
        self.correct(db, block_index=0, kind="M", label="Other")
        body = make_ns(live_db=db,
                       params={"block_index": 0})["delete_service_phase_correction"]()
        assert body["removed"] == 1 and body["corrections"] == []

    def test_a_session_argument_cannot_redirect_the_delete(self, tmp_path):
        # Live-only, as with the save route: a path in the body must not reach another file.
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        conn = sqlite3.connect(other)
        save_correction(conn, 0, kind="M", label="Songs")
        conn.close()
        make_ns(live_db=db, params={"session": other, "block_index": 0}
                )["delete_service_phase_correction"]()
        assert len(load_corrections(sqlite3.connect(other))) == 1

    def test_no_running_session_is_a_404(self):
        _, status = make_ns(live_db=None,
                            params={"block_index": 0})["delete_service_phase_correction"]()
        assert status == 404

    def test_the_detector_reclaims_the_block_on_its_next_run(self, tmp_path):
        db = session_db(tmp_path)
        self.correct(db, block_index=0, kind="M", label="Other")
        make_ns(live_db=db, params={"block_index": 0})["delete_service_phase_correction"]()
        conn = sqlite3.connect(db)
        save_analysis(conn, analyze(read_rows(conn), CFG))
        assert load_corrections(conn) == []
        conn.close()


class TestAccessLogPollingSkip:
    """Polling endpoints are filtered at read time, so writing them is pure cost."""

    def ns(self, cfg=None):
        return extract_definitions(
            "speech_to_text.py", ["_access_log_skip_polling", "_access_log_enabled"],
            extra_globals={"config": {"access_log": cfg} if cfg is not None else {}})

    def test_on_by_default(self):
        assert self.ns()["_access_log_skip_polling"]() is True

    def test_can_be_turned_off_to_diagnose_the_polling_itself(self):
        assert self.ns({"skip_polling_paths": False})["_access_log_skip_polling"]() is False

    def test_a_broken_config_still_skips(self):
        # Failing open would resume thousands of fsyncs per service silently.
        ns = extract_definitions("speech_to_text.py", ["_access_log_skip_polling"],
                                 extra_globals={"config": None})
        assert ns["_access_log_skip_polling"]() is True

    def test_it_is_independent_of_the_enabled_flag(self):
        cfg = {"enabled": True, "skip_polling_paths": True}
        ns = self.ns(cfg)
        assert ns["_access_log_enabled"]() is True and ns["_access_log_skip_polling"]() is True

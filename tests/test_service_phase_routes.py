"""The /api/service-phase routes.

Extracted from the monolith and run against a stub namespace (see tests/conftest.py) — the
module cannot be imported, and CI installs no Flask. jsonify is stubbed to return the
mapping unchanged, which is what these assertions care about.

These routes now take a ``session`` from the caller so a finished service can be reviewed
and corrected, which makes one property load-bearing: the name is only ever *matched against
an enumeration the server built*, never joined onto a directory. A path, a traversal, or an
unknown name resolves to nothing rather than to a file. TestSessionSelection pins that, and
TestSaveCorrection keeps the older guarantee that a stray path cannot steer a write.
"""

import os
import sqlite3

import pytest

from conftest import extract_definitions
from stt.phase_rules import load_rules
from stt.coercion import coerce_int
from stt.db_maintenance import checkpoint_and_release, open_readonly
from stt.session_index import describe, index, resolve_session
from stt.service_phase import (
    analyze,
    delete_correction,
    delete_correction_by_id,
    load_analysis,
    load_corrections,
    read_rows,
    save_analysis,
    save_correction,
    save_group_correction,
)

MIN = 60_000
RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "config", "service_phases.default.json")
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


def make_ns(*, live_db=None, params=None, args=None, archive=()):
    """The routes under test, with the archive enumeration supplied by the caller.

    ``archive`` is what _archive_session_paths would have swept: the whitelist a ``session``
    name is matched against. Passing it explicitly is the point — it is the only thing
    standing between a caller-supplied string and a database.
    """
    ns = extract_definitions(
        "speech_to_text.py",
        ["_service_phase_resolve_db", "get_service_phase", "save_service_phase_correction",
         "delete_service_phase_correction", "group_service_phase_blocks",
         "_service_phase_first_sunday", "_service_phase_config",
         "_archive_session_paths", "_archive_resolve_db", "_archive_write_done",
         "_archive_open_ro", "rerun_service_phase", "list_service_phase_sessions"],
        extra_globals={
            "os": os,
            "_db_iter_databases": lambda dirs: list(archive),
            "_sidecar_sweep_dirs": lambda: ["/archive"],
            "_db_checkpoint_and_release": checkpoint_and_release,
            "_db_open_readonly": open_readonly,
            "_session_resolve": resolve_session,
            "_session_describe": describe,
            "_session_index": index,
            "_sermon_summary_config": lambda: {},
            "_service_phase_save": save_analysis,
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
            # The real shipped rules, so the routes are exercised the way they run.
            "_service_phase_rules": lambda: load_rules("", RULES_FILE),
            "_service_phase_load": load_analysis,
            "_service_phase_rows": read_rows,
            "_service_phase_corrections": load_corrections,
            "_service_phase_save_correction": save_correction,
            "_service_phase_delete_correction": delete_correction,
            "_service_phase_delete_correction_by_id": delete_correction_by_id,
            "_service_phase_save_group": save_group_correction,
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

    def test_it_reports_the_songs_threshold_the_page_renumbers_with(self, tmp_path):
        # The page redoes the song numbering when a correction moves the opening, so it
        # needs the detector's own threshold rather than a hardcoded guess.
        body = make_ns(live_db=session_db(tmp_path))["get_service_phase"]()
        assert body["songs_min_minutes"] == CFG["songs_min_minutes"]

    def test_no_running_session_is_a_404_not_a_crash(self):
        body, status = make_ns(live_db=None)["get_service_phase"]()
        assert status == 404 and body["success"] is False

    def test_a_session_path_does_not_reach_sqlite(self, tmp_path):
        # A caller-supplied *path* is never a session name: it is matched against the
        # enumeration, and an archive that does not contain it resolves to nothing.
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        body, status = make_ns(live_db=db, args={"session": other})["get_service_phase"]()
        assert status == 404 and body["success"] is False

    def test_a_known_session_is_read_instead_of_the_live_one(self, tmp_path):
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        body = make_ns(live_db=db, archive=[other],
                       args={"session": "2026-01-01_000000.db"})["get_service_phase"]()
        assert body["session_id"] == "2026-01-01_000000.db"
        assert body["live"] is False

    def test_the_live_session_reads_as_live(self, tmp_path):
        db = session_db(tmp_path)
        body = make_ns(live_db=db, archive=[db])["get_service_phase"]()
        assert body["session_id"] == "2026-03-01_093218.db" and body["live"] is True

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

    def test_a_session_path_cannot_redirect_the_write(self, tmp_path):
        # A path in the body is not a session name. It matches nothing in the enumeration,
        # so the write is refused outright rather than landing on either database.
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        _, status = make_ns(live_db=db, params={"session": other, "block_index": 0,
                                                "label": "Songs"})["save_service_phase_correction"]()
        assert status == 404
        assert load_corrections(sqlite3.connect(db)) == []
        assert load_corrections(sqlite3.connect(other)) == []

    def test_a_known_session_is_corrected_instead_of_the_live_one(self, tmp_path):
        # The point of the change: last Sunday can be reviewed without it being Sunday.
        db = session_db(tmp_path)
        other = session_db(tmp_path, name="2026-01-01_000000.db")
        make_ns(live_db=db, archive=[other],
                params={"session": "2026-01-01_000000.db", "block_index": 0,
                        "label": "Songs"})["save_service_phase_correction"]()
        assert len(load_corrections(sqlite3.connect(other))) == 1
        assert load_corrections(sqlite3.connect(db)) == []

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


class TestGroupBlocks:
    """Several detected blocks recorded as one phase, stored as a span with no block index."""

    def group(self, db, **params):
        return make_ns(live_db=db, params=params)["group_service_phase_blocks"]()

    def test_stores_the_span_and_returns_the_new_list(self, tmp_path):
        db = session_db(tmp_path)
        body = self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="Worship set")
        assert body["success"] is True
        stored = body["corrections"][0]
        assert stored["block_index"] is None and stored["label"] == "Worship set"
        assert (stored["start_ms"], stored["end_ms"]) == (1_000, 5_000)

    def test_a_group_needs_a_name(self, tmp_path):
        db = session_db(tmp_path)
        body, status = self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="  ")
        assert status == 400 and body["success"] is False
        assert load_corrections(sqlite3.connect(db)) == []

    @pytest.mark.parametrize("span", [
        {"start_ms": 0, "end_ms": 5_000},        # no start
        {"start_ms": 5_000, "end_ms": 5_000},    # zero length
        {"start_ms": 9_000, "end_ms": 5_000},    # backwards
    ])
    def test_an_unusable_span_is_rejected(self, tmp_path, span):
        db = session_db(tmp_path)
        body, status = self.group(db, label="Worship set", **span)
        assert status == 400 and body["success"] is False

    def test_regrouping_the_same_span_does_not_stack(self, tmp_path):
        db = session_db(tmp_path)
        self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="Worship set")
        body = self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="Opening songs")
        assert [c["label"] for c in body["corrections"]] == ["Opening songs"]

    def test_long_free_text_is_truncated(self, tmp_path):
        db = session_db(tmp_path)
        body = self.group(db, start_ms=1_000, end_ms=5_000, label="x" * 500, note="y" * 5000)
        assert len(body["corrections"][0]["label"]) <= 120
        assert len(body["corrections"][0]["note"]) <= 500

    def test_no_running_session_is_a_404(self):
        _, status = make_ns(live_db=None, params={"start_ms": 1_000, "end_ms": 5_000,
                                                  "label": "x"})["group_service_phase_blocks"]()
        assert status == 404

    def test_a_group_is_undone_by_id(self, tmp_path):
        db = session_db(tmp_path)
        made = self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="Worship set")
        body = make_ns(live_db=db,
                       params={"id": made["id"]})["delete_service_phase_correction"]()
        assert body["success"] is True and body["removed"] == 1
        assert body["corrections"] == []

    def test_an_undo_with_neither_index_nor_id_is_rejected(self, tmp_path):
        db = session_db(tmp_path)
        self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="Worship set")
        body, status = make_ns(live_db=db, params={})["delete_service_phase_correction"]()
        assert status == 400 and body["success"] is False
        assert len(load_corrections(sqlite3.connect(db))) == 1

    def test_an_id_undo_does_not_also_delete_by_block_index(self, tmp_path):
        # Both fields present: the id wins and block 0's own correction must survive.
        db = session_db(tmp_path)
        made = self.group(db, start_ms=1_000, end_ms=5_000, kind="M", label="Worship set")
        make_ns(live_db=db, params={"block_index": 0, "kind": "M",
                                    "label": "Other"})["save_service_phase_correction"]()
        body = make_ns(live_db=db, params={"id": made["id"], "block_index": 0}
                       )["delete_service_phase_correction"]()
        assert [c["label"] for c in body["corrections"]] == ["Other"]


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


class TestSessionSelection:
    """A ``session`` names a service; it never names a file.

    The whole safety story is that the string is matched against an enumeration the server
    built. These are the cases where getting that wrong would let a request reach a database
    nobody offered it.
    """

    @pytest.mark.parametrize("hostile", [
        "../../../etc/passwd",
        "/etc/passwd",
        "../2026-01-01_000000.db",
        "2026-01-01_000000.db/../../secret.db",
        "..",
        "",
    ])
    def test_a_hostile_name_never_reaches_a_database_outside_the_archive(self, tmp_path, hostile):
        db = session_db(tmp_path)
        ns = make_ns(live_db=db, archive=[db], args={"session": hostile})
        path, is_live, err = ns["_archive_resolve_db"](hostile)
        # Either refused, or resolved to something the archive actually offered.
        assert err is not None or path in (db,)
        if path == db:
            assert is_live is True   # "" means the live session, which is db here

    def test_an_unknown_session_is_a_404_not_a_fallback_to_live(self, tmp_path):
        # Silently writing to the live service because a name was unrecognised is the one
        # failure mode worse than an error.
        db = session_db(tmp_path)
        ns = make_ns(live_db=db, archive=[db])
        path, _, err = ns["_archive_resolve_db"]("2099-01-01_000000.db")
        assert path is None and err is not None

    def test_no_session_and_nothing_running_is_a_404(self):
        ns = make_ns(live_db=None, archive=[])
        path, _, err = ns["_archive_resolve_db"]("")
        assert path is None and err is not None

    def test_the_live_session_is_resolved_without_sweeping_the_archive(self, tmp_path):
        """The sweep walks the whole backup tree; the live case never needs it.

        Doing it anyway made every poll, correction and re-run pay for a filesystem walk to
        build an enumeration that resolve_session then does not look at — on a real archive
        that walk *was* the request.
        """
        class CountingArchive(list):
            sweeps = 0

            def __iter__(self):
                type(self).sweeps += 1
                return list.__iter__(self)

        db = session_db(tmp_path)
        archive = CountingArchive([db])
        ns = make_ns(live_db=db, archive=archive)

        path, is_live, err = ns["_archive_resolve_db"]("")
        assert (path, is_live, err) == (db, True, None)
        assert CountingArchive.sweeps == 0, "resolving the live session swept the archive"

        # Naming one is the case that genuinely needs the enumeration.
        ns["_archive_resolve_db"]("2026-03-01_093218.db")
        assert CountingArchive.sweeps > 0


class TestRerunAndSave:
    def test_it_saves_where_recompute_does_not(self, tmp_path):
        db = session_db(tmp_path, analyzed=False)
        assert load_analysis(sqlite3.connect(db))["blocks"] == []

        # ?recompute=1 deliberately writes nothing...
        make_ns(live_db=db, args={"recompute": "1"})["get_service_phase"]()
        assert load_analysis(sqlite3.connect(db))["blocks"] == []

        # ...and the rerun route is the one that does.
        body = make_ns(live_db=db, params={})["rerun_service_phase"]()
        assert body["success"] is True and body["blocks"] > 0
        assert [b["kind"] for b in load_analysis(sqlite3.connect(db))["blocks"]] == ["M", "S"]

    def test_it_runs_against_a_named_archived_service(self, tmp_path):
        live = session_db(tmp_path)
        old = session_db(tmp_path, name="2026-01-01_000000.db", analyzed=False)
        body = make_ns(live_db=live, archive=[old],
                       params={"session": "2026-01-01_000000.db"})["rerun_service_phase"]()
        assert body["session_id"] == "2026-01-01_000000.db" and body["live"] is False
        assert load_analysis(sqlite3.connect(old))["blocks"]

    def test_a_rerun_leaves_no_sidecars_on_an_archived_service(self, tmp_path):
        # A finished session may already have been delivered; writing to it must not leave
        # -wal/-shm beside a file this process no longer owns.
        old = session_db(tmp_path, name="2026-01-01_000000.db", analyzed=False)
        live = session_db(tmp_path)
        make_ns(live_db=live, archive=[old],
                params={"session": "2026-01-01_000000.db"})["rerun_service_phase"]()
        assert not os.path.exists(old + "-wal")
        assert not os.path.exists(old + "-shm")

    def test_a_rerun_keeps_the_corrections(self, tmp_path):
        # Corrections live in their own table precisely so a re-run cannot destroy them.
        old = session_db(tmp_path, name="2026-01-01_000000.db")
        conn = sqlite3.connect(old)
        save_correction(conn, 0, kind="M", label="Songs")
        conn.close()
        make_ns(live_db=None, archive=[old],
                params={"session": "2026-01-01_000000.db"})["rerun_service_phase"]()
        assert len(load_corrections(sqlite3.connect(old))) == 1

    def test_it_reports_how_long_it_took(self, tmp_path):
        # "It feels slow" is only actionable as a number.
        db = session_db(tmp_path, analyzed=False)
        body = make_ns(live_db=db, params={})["rerun_service_phase"]()
        assert isinstance(body["elapsed_ms"], int) and body["elapsed_ms"] >= 0

    def test_an_unknown_session_is_refused(self, tmp_path):
        db = session_db(tmp_path)
        _, status = make_ns(live_db=db, archive=[db],
                            params={"session": "nope.db"})["rerun_service_phase"]()
        assert status == 404


class TestListSessions:
    def test_lists_recorded_services_newest_first_with_their_shape(self, tmp_path):
        a = session_db(tmp_path, name="2026-03-01_093218.db")
        b = session_db(tmp_path, name="2026-01-01_000000.db")
        body = make_ns(live_db=None, archive=[a, b])["list_service_phase_sessions"]()
        assert body["success"] is True
        by_id = {r["session_id"]: r for r in body["sessions"]}
        assert set(by_id) == {"2026-03-01_093218.db", "2026-01-01_000000.db"}
        assert by_id["2026-03-01_093218.db"]["date"] == "2026-03-01"
        assert by_id["2026-03-01_093218.db"]["rows"] == 17
        assert by_id["2026-03-01_093218.db"]["has_phase"] is True

    def test_the_running_service_is_flagged(self, tmp_path):
        db = session_db(tmp_path)
        body = make_ns(live_db=db, archive=[db])["list_service_phase_sessions"]()
        assert [r["live"] for r in body["sessions"]] == [True]
        assert body["live_session_id"] == "2026-03-01_093218.db"

    def test_a_just_started_service_still_appears(self, tmp_path):
        # It has no rows yet, so the sweep drops it — but it is the one an operator
        # mid-service is looking for.
        empty = session_db(tmp_path, spec="", name="2026-03-01_093218.db", analyzed=False)
        body = make_ns(live_db=empty, archive=[empty])["list_service_phase_sessions"]()
        assert [r["session_id"] for r in body["sessions"]] == ["2026-03-01_093218.db"]
        assert body["sessions"][0]["live"] is True

    def test_an_unreadable_session_does_not_break_the_listing(self, tmp_path):
        good = session_db(tmp_path)
        junk = str(tmp_path / "2026-02-02_000000.db")
        with open(junk, "wb") as fh:
            fh.write(b"not a database at all")
        body = make_ns(live_db=None, archive=[junk, good])["list_service_phase_sessions"]()
        assert [r["session_id"] for r in body["sessions"]] == ["2026-03-01_093218.db"]

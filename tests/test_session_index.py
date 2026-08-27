"""Session enumeration and resolution (stt/session_index.py).

The property that matters is in TestResolveSession: a request names a service by basename
and that name is only ever compared against an enumeration the server built. Nothing the
caller sends is joined onto a directory, so the traversal tests below are not defending a
filter — they pin that a hostile name simply fails to match anything.
"""

import os
import sqlite3

import pytest

from stt.session_index import describe, index, resolve_session, session_date, unreadable

BASE = 1_700_000_000_000
MIN = 60_000

PATHS = [
    "/archive/2026/08/2026-08-16_101502.db",
    "/archive/2026/08/2026-08-09_100133.db",
    "/archive/2026/07/2026-07-26_095900.db",
]


class TestSessionDate:
    def test_reads_the_date_out_of_the_filename(self):
        assert session_date("2026-08-16_101502.db") == "2026-08-16"

    def test_accepts_a_full_path(self):
        assert session_date("/archive/2026/08/2026-08-16_101502.db") == "2026-08-16"

    def test_tolerates_a_filename_prefix_suffix(self):
        assert session_date("2026-08-16_101502_main.db") == "2026-08-16"

    @pytest.mark.parametrize("name", ["", "session.db", "notadate_1015.db", "26-08-16.db"])
    def test_an_undated_name_yields_empty(self, name):
        assert session_date(name) == ""


class TestResolveSession:
    def test_a_known_basename_resolves(self):
        got = resolve_session(PATHS, "2026-08-09_100133.db", None)
        assert got == ("/archive/2026/08/2026-08-09_100133.db", False)

    def test_an_empty_name_means_the_live_session(self, tmp_path):
        live = tmp_path / "live.db"
        live.write_text("")
        assert resolve_session(PATHS, "", str(live)) == (str(live), True)
        assert resolve_session(PATHS, None, str(live)) == (str(live), True)

    def test_an_empty_name_with_nothing_running_resolves_to_nothing(self):
        assert resolve_session(PATHS, "", None) is None
        assert resolve_session(PATHS, "", "/no/such/session.db") is None

    def test_a_name_not_in_the_enumeration_is_refused(self):
        assert resolve_session(PATHS, "2026-08-17_101502.db", None) is None

    @pytest.mark.parametrize("hostile", [
        "../../../etc/passwd",
        "/etc/passwd",
        "../2026-08-09_100133.db",
        "/archive/2026/08/2026-08-09_100133.db",   # a real path, but paths are not accepted
        "2026-08-09_100133.db/../../secret.db",
        "..",
        ".",
    ])
    def test_a_path_is_never_accepted_only_a_known_basename(self, hostile):
        # basename() reduces these to something that either matches an enumerated session
        # or does not; nothing is ever joined onto a directory.
        got = resolve_session(PATHS, hostile, None)
        assert got is None or got[0] in PATHS

    def test_traversal_cannot_reach_outside_the_enumeration(self):
        for hostile in ("../../../etc/passwd", "/etc/passwd", ".."):
            assert resolve_session(PATHS, hostile, None) is None

    def test_the_live_session_is_flagged_when_the_sweep_includes_it(self, tmp_path):
        live = tmp_path / "2026-08-20_101502.db"
        live.write_text("")
        got = resolve_session([str(live)], "2026-08-20_101502.db", str(live))
        assert got == (str(live), True)

    def test_a_relative_and_absolute_form_of_the_live_path_still_match(self, tmp_path, monkeypatch):
        live = tmp_path / "2026-08-20_101502.db"
        live.write_text("")
        monkeypatch.chdir(tmp_path)
        got = resolve_session([str(live)], "2026-08-20_101502.db", "2026-08-20_101502.db")
        assert got is not None and got[1] is True

    def test_an_archived_session_is_not_flagged_live(self, tmp_path):
        live = tmp_path / "live.db"
        live.write_text("")
        archived = tmp_path / "2026-08-09_100133.db"
        archived.write_text("")
        got = resolve_session([str(archived)], "2026-08-09_100133.db", str(live))
        assert got == (str(archived), False)

    def test_an_empty_enumeration_resolves_nothing(self):
        assert resolve_session([], "2026-08-09_100133.db", None) is None

    def test_an_unstattable_path_falls_back_to_path_comparison(self, tmp_path, monkeypatch):
        # samefile() needs both files to stat; a path on a dismounted NAS raises instead.
        live = tmp_path / "2026-08-20_101502.db"
        live.write_text("")

        def boom(_a, _b):
            raise OSError("stale NFS handle")

        monkeypatch.setattr(os.path, "samefile", boom)
        got = resolve_session([str(live)], "2026-08-20_101502.db", str(live))
        assert got == (str(live), True)


class TestDescribe:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "session.db")
        c.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, ts_ms INTEGER, "
                  "text TEXT, is_final INTEGER)")
        yield c
        c.close()

    def fill(self, conn, count, *, start=BASE, step=MIN, is_final=1):
        conn.executemany(
            "INSERT INTO transcriptions (ts_ms, text, is_final) VALUES (?, ?, ?)",
            [(start + i * step, f"caption {i}", is_final) for i in range(count)])
        conn.commit()

    def test_reports_rows_and_span(self, conn):
        self.fill(conn, 31)  # 30 minutes end to end
        got = describe(conn)
        assert got["rows"] == 31
        assert got["start_ms"] == BASE and got["end_ms"] == BASE + 30 * MIN
        assert got["minutes"] == 30

    def test_ignores_partial_rows(self, conn):
        self.fill(conn, 5)
        self.fill(conn, 5, start=BASE + 100 * MIN, is_final=0)
        assert describe(conn)["rows"] == 5

    def test_an_empty_session_reports_nothing(self, conn):
        got = describe(conn)
        assert got["rows"] == 0 and got["minutes"] == 0
        assert got["has_phase"] is False and got["has_summaries"] is False

    def test_feature_flags_follow_the_tables(self, conn):
        self.fill(conn, 3)
        assert describe(conn)["has_phase"] is False
        conn.execute("CREATE TABLE service_phase_blocks (block_index INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE sermon_summaries (id INTEGER PRIMARY KEY)")
        got = describe(conn)
        assert got["has_phase"] is True and got["has_summaries"] is True

    def test_a_failing_table_probe_degrades_rather_than_raising(self):
        # Older archived databases have been seen to read their rows and then fail a
        # schema probe; a listing must survive one bad session.
        class HalfBroken:
            def __init__(self):
                self.calls = 0

            def execute(self, sql, params=()):
                self.calls += 1
                if "sqlite_master" in sql:
                    raise sqlite3.DatabaseError("probe failed")

                class R:
                    @staticmethod
                    def fetchone():
                        return (7, BASE, BASE + 5 * MIN)
                return R()

        got = describe(HalfBroken())
        assert got["rows"] == 7 and got["minutes"] == 5
        assert got["has_phase"] is False and got["has_summaries"] is False

    def test_a_database_without_the_transcript_table_degrades(self, tmp_path):
        # Older builds exist in the archive; one must still list, not raise.
        c = sqlite3.connect(tmp_path / "old.db")
        assert describe(c)["rows"] == 0
        c.close()


class TestIndex:
    def described(self, rows=10, **kw):
        base = {"rows": rows, "start_ms": BASE, "end_ms": BASE + 30 * MIN,
                "minutes": 30, "has_phase": True, "has_summaries": False}
        base.update(kw)
        return base

    def test_builds_a_listing_row_per_session(self):
        out = index([(PATHS[0], self.described()), (PATHS[1], self.described())])
        assert [r["session_id"] for r in out] == ["2026-08-16_101502.db", "2026-08-09_100133.db"]
        assert out[0]["date"] == "2026-08-16"
        assert out[0]["minutes"] == 30 and out[0]["has_phase"] is True

    def test_sessions_with_no_rows_are_dropped(self):
        # A start/stop, or a process that died before anyone spoke.
        out = index([(PATHS[0], self.described()), (PATHS[1], self.described(rows=0))])
        assert [r["session_id"] for r in out] == ["2026-08-16_101502.db"]

    def test_the_live_session_is_flagged(self, tmp_path):
        live = tmp_path / "2026-08-20_101502.db"
        live.write_text("")
        out = index([(str(live), self.described())], live_path=str(live))
        assert out[0]["live"] is True

    def test_nothing_is_flagged_live_when_nothing_is_running(self):
        out = index([(PATHS[0], self.described())], live_path=None)
        assert out[0]["live"] is False

    def test_preserves_the_order_it_was_given(self):
        out = index([(p, self.described()) for p in PATHS])
        assert [r["session_id"] for r in out] == [os.path.basename(p) for p in PATHS]


class TestUnreadable:
    """A session that could not be opened, carried out with the listing.

    Dropping it silently is what let a server that had stopped being able to open its own
    archive present itself as a server with an empty one.
    """

    def test_names_the_session_and_the_reason(self):
        row = unreadable(PATHS[0], sqlite3.OperationalError("disk I/O error"))
        assert row["session_id"] == "2026-08-16_101502.db"
        assert row["date"] == "2026-08-16"
        assert row["error"] == "OperationalError: disk I/O error"
        assert row["stage"] == "open"

    def test_the_stage_says_which_half_failed(self):
        row = unreadable(PATHS[0], sqlite3.DatabaseError("malformed"), "read")
        assert row["stage"] == "read"

    def test_an_undated_name_still_produces_a_row(self):
        row = unreadable("/archive/scratch.db", OSError("gone"))
        assert row["session_id"] == "scratch.db" and row["date"] == ""

    def test_an_empty_path_is_survivable(self):
        assert unreadable("", RuntimeError("x"))["session_id"] == ""

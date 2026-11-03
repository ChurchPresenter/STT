"""Clear-all-screens markers: what stays on screen, and what stays in the record."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator, List, Optional

import pytest

from stt import display_clear as dc

T0 = 1_700_000_000_000  # an arbitrary wall-clock ms for the marker


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(tmp_path / "session.db")
    c.execute(
        "CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, text TEXT,"
        " start_time REAL DEFAULT 0, end_time REAL DEFAULT 0, ts_ms INTEGER)"
    )
    yield c
    c.close()


def add(c: sqlite3.Connection, text: str, start: Optional[float], ts_ms: Optional[int]) -> int:
    cur = c.execute(
        "INSERT INTO transcriptions (timestamp, text, start_time, ts_ms) VALUES ('t', ?, ?, ?)",
        (text, start, ts_ms),
    )
    c.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def shown(c: sqlite3.Connection) -> List[str]:
    clause, params = dc.sql_filter(dc.latest_clear(c))
    rows = c.execute(f"SELECT text FROM transcriptions WHERE 1=1{clause} ORDER BY id", params).fetchall()
    return [r[0] for r in rows]


def test_no_table_means_no_marker_and_everything_shown(conn: sqlite3.Connection) -> None:
    add(conn, "a", 1.0, T0)
    assert dc.latest_clear(conn) is None
    assert dc.sql_filter(None) == ("", ())
    assert dc.display_epoch(None) == 0
    assert shown(conn) == ["a"]


def test_empty_table_means_no_marker(conn: sqlite3.Connection) -> None:
    dc.ensure_table(conn)
    assert dc.latest_clear(conn) is None


def test_saved_rows_are_hidden_and_kept(conn: sqlite3.Connection) -> None:
    add(conn, "before 1", 1.0, T0 - 5000)
    add(conn, "before 2", 4.0, T0 - 1000)
    marker = dc.record_clear(conn, "2026-11-03 10:00:00", T0, 6.0)
    assert marker.after_row_id == 2
    assert shown(conn) == []
    # The record itself is untouched.
    assert conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0] == 2


def test_in_progress_sentence_finalised_after_clear_is_hidden(conn: sqlite3.Connection) -> None:
    add(conn, "said before", 1.0, T0 - 2000)
    dc.record_clear(conn, "t", T0, 10.0)
    # Spoken from 8s, still in the buffer at the clear, saved 3s later.
    add(conn, "was in progress", 8.0, T0 + 3000)
    add(conn, "said after", 10.5, T0 + 6000)
    assert shown(conn) == ["said after"]


def test_cutoff_expires_so_a_restarted_clock_cannot_hide_everything(conn: sqlite3.Connection) -> None:
    dc.record_clear(conn, "t", T0, 500.0)
    add(conn, "fresh clock, soon", 2.0, T0 + 30_000)
    add(conn, "fresh clock, later", 40.0, T0 + dc.CUTOFF_HORIZON_MS + 1)
    assert shown(conn) == ["fresh clock, later"]


def test_untimed_rows_are_never_lost(conn: sqlite3.Connection) -> None:
    dc.record_clear(conn, "t", T0, 10.0)
    add(conn, "no start time", None, T0 + 1000)
    assert shown(conn) == ["no start time"]


def test_row_without_ts_ms_falls_under_the_cutoff(conn: sqlite3.Connection) -> None:
    dc.record_clear(conn, "t", T0, 10.0)
    add(conn, "early, no wall clock", 5.0, None)
    add(conn, "late, no wall clock", 11.0, None)
    assert shown(conn) == ["late, no wall clock"]


def test_a_second_clear_supersedes_the_first(conn: sqlite3.Connection) -> None:
    first = dc.record_clear(conn, "t", T0, 5.0)
    add(conn, "between", 6.0, T0 + 1000)
    second = dc.record_clear(conn, "t", T0 + 2000, 7.0)
    add(conn, "after both", 8.0, T0 + 4000)
    assert dc.latest_clear(conn) == second
    assert dc.display_epoch(second) > dc.display_epoch(first)
    assert shown(conn) == ["after both"]


def test_negative_cutoff_is_clamped(conn: sqlite3.Connection) -> None:
    assert dc.record_clear(conn, "t", T0, -3.0).audio_cutoff == 0.0


def test_sql_filter_and_is_visible_agree(conn: sqlite3.Connection) -> None:
    for start, ts in [(1.0, T0 - 10), (None, T0 - 10), (3.0, None)]:
        add(conn, "pre", start, ts)
    marker = dc.record_clear(conn, "t", T0, 10.0)
    cases = [
        (2.0, T0 + 1), (9.99, T0 + dc.CUTOFF_HORIZON_MS), (9.99, T0 + dc.CUTOFF_HORIZON_MS + 1),
        (10.0, T0 + 1), (None, T0 + 1), (4.0, None), (12.0, None), (0.0, T0 + 5),
    ]
    for i, (start, ts) in enumerate(cases):
        add(conn, f"post{i}", start, ts)
    clause, params = dc.sql_filter(marker)
    via_sql = {r[0] for r in conn.execute(f"SELECT id FROM transcriptions WHERE 1=1{clause}", params)}
    rows = conn.execute("SELECT id, start_time, ts_ms FROM transcriptions").fetchall()
    via_py = {rid for rid, st, ts in rows if dc.is_visible(rid, st, ts, marker)}
    assert via_sql == via_py
    assert via_sql  # the case list does include visible rows

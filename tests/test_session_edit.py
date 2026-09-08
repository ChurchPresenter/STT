"""Editing a session database: what may change, and what must not.

The rows are a record of a service. These cases are mostly about refusal — the
column that is not editable, the caption blanked instead of deleted, the batch that
half-applied — because the damage from getting those wrong is a transcript nobody
can reconstruct.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from stt import session_edit


SCHEMA = """
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, text TEXT, start_time REAL, end_time REAL, confidence REAL,
    original_text TEXT, corrected_by TEXT, needs_review INTEGER,
    translated_text TEXT, translation_language TEXT, speech_type TEXT,
    denied INTEGER, ts_ms INTEGER, words_json TEXT, is_final INTEGER DEFAULT 1,
    segment_id TEXT
);
"""


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "2026-09-08_090000.db")
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO transcriptions (text, translated_text, ts_ms, start_time, end_time, is_final, denied) "
        "VALUES (?, ?, ?, ?, ?, 1, 0)",
        [("Первая строка", "First line", 1000, 0.0, 2.0),
         ("Вторая строка", "Second line", 3000, 2.0, 4.0),
         ("Третья строка", None, 5000, 4.0, 6.0)])
    conn.commit()
    conn.close()
    return path


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT id, text, translated_text, ts_ms FROM transcriptions ORDER BY ts_ms, id"
        ).fetchall()
    finally:
        conn.close()


def _apply(path, edits):
    conn = sqlite3.connect(path)
    try:
        with conn:
            return session_edit.apply_edits(conn, session_edit.parse_edits(edits))
    finally:
        conn.close()


# --- which columns are on offer --------------------------------------------


@pytest.mark.parametrize("column", ["text", "translated_text", "start_time", "end_time", "ts_ms"])
def test_the_columns_a_person_can_correct_are_editable(column):
    assert column in session_edit.EDITABLE


@pytest.mark.parametrize("column", ["id", "segment_id", "words_json", "session_id",
                                    "confidence", "is_final", "denied"])
def test_everything_else_is_refused(column):
    """words_json in particular: a hand-edited timing blob desynchronises the live
    preview from the text it highlights, and nothing on screen would show it."""
    with pytest.raises(session_edit.EditError):
        session_edit.coerce(column, "anything")


def test_a_caption_cannot_be_blanked():
    """A row with no text is invisible to every reader, so blanking one is a delete
    wearing an edit's clothes — and it would not show up as a deletion anywhere."""
    with pytest.raises(session_edit.EditError):
        session_edit.coerce("text", "   ")


def test_a_translation_can_be_cleared_because_untranslated_is_a_real_state():
    assert session_edit.coerce("translated_text", "") is None


def test_timings_are_read_as_numbers():
    assert session_edit.coerce("start_time", "1.5") == 1.5
    assert session_edit.coerce("ts_ms", "1500") == 1500
    assert isinstance(session_edit.coerce("ts_ms", "1500.9"), int)


def test_a_timing_that_is_not_a_number_is_refused():
    with pytest.raises(session_edit.EditError):
        session_edit.coerce("start_time", "soon")


def test_a_segment_cannot_end_before_it_starts():
    with pytest.raises(session_edit.EditError):
        session_edit.parse_edit({"op": "update", "id": 1,
                                 "values": {"start_time": 5, "end_time": 2}})


# --- parsing a batch --------------------------------------------------------


def test_an_update_needs_a_row_id():
    with pytest.raises(session_edit.EditError):
        session_edit.parse_edit({"op": "update", "values": {"text": "x"}})


def test_an_unknown_operation_is_refused():
    with pytest.raises(session_edit.EditError):
        session_edit.parse_edit({"op": "truncate", "id": 1})


def test_an_empty_batch_is_refused():
    with pytest.raises(session_edit.EditError):
        session_edit.parse_edits([])


def test_a_batch_is_refused_whole_when_one_edit_is_bad():
    with pytest.raises(session_edit.EditError):
        session_edit.parse_edits([
            {"op": "update", "id": 1, "values": {"text": "fine"}},
            {"op": "update", "id": 2, "values": {"words_json": "[]"}},
        ])


# --- applying ---------------------------------------------------------------


def test_a_caption_and_its_translation_can_be_corrected(db):
    counts = _apply(db, [{"op": "update", "id": 2,
                          "values": {"text": "Исправлено", "translated_text": "Corrected"}}])

    assert counts["updated"] == 1
    assert _rows(db)[1][1:3] == ("Исправлено", "Corrected")


def test_a_row_can_be_deleted(db):
    counts = _apply(db, [{"op": "delete", "id": 1}])

    assert counts["deleted"] == 1
    assert [r[1] for r in _rows(db)] == ["Вторая строка", "Третья строка"]


def test_an_inserted_line_lands_between_the_row_it_follows_and_the_next(db):
    _apply(db, [{"op": "insert", "after_id": 1, "values": {"text": "Вставка"}}])

    assert [r[1] for r in _rows(db)] == [
        "Первая строка", "Вставка", "Вторая строка", "Третья строка"]


def test_an_inserted_line_after_the_last_row_goes_to_the_end(db):
    _apply(db, [{"op": "insert", "after_id": 3, "values": {"text": "Последняя"}}])

    assert _rows(db)[-1][1] == "Последняя"


def test_an_inserted_line_with_no_anchor_goes_to_the_front(db):
    _apply(db, [{"op": "insert", "values": {"text": "Самая первая"}}])

    assert _rows(db)[0][1] == "Самая первая"


def test_an_inserted_row_is_final_and_not_denied_so_the_readers_see_it(db):
    """load_schedule and the export path both filter on these; a row missing them
    is written, invisible, and impossible to explain."""
    _apply(db, [{"op": "insert", "after_id": 1, "values": {"text": "Вставка"}}])

    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT is_final, denied, segment_id, timestamp, id FROM transcriptions "
            "WHERE text = 'Вставка'").fetchone()
    finally:
        conn.close()
    assert row[0] == 1 and row[1] == 0
    assert row[2] == str(row[4])   # segment_id is the id, as the recorder writes it
    assert row[3]                  # a human-readable timestamp was stamped


def test_editing_a_row_that_is_gone_is_reported_rather_than_ignored(db):
    with pytest.raises(session_edit.EditError):
        _apply(db, [{"op": "update", "id": 999, "values": {"text": "x"}}])


def test_a_column_this_database_does_not_have_is_refused(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, text TEXT, ts_ms INTEGER);")
    conn.commit()
    conn.close()

    with pytest.raises(session_edit.EditError):
        _apply(path, [{"op": "update", "id": 1, "values": {"translated_text": "x"}}])


def test_a_failed_batch_leaves_the_database_as_it_was(db):
    """All or nothing: a half-applied transcript is one nobody can reason about."""
    before = _rows(db)

    with pytest.raises(session_edit.EditError):
        conn = sqlite3.connect(db)
        try:
            with conn:
                session_edit.apply_edits(conn, session_edit.parse_edits([
                    {"op": "update", "id": 1, "values": {"text": "Изменено"}},
                    {"op": "delete", "id": 999},
                ]))
        finally:
            conn.close()

    assert _rows(db) == before


# --- not the session being recorded ----------------------------------------


def test_the_database_being_recorded_is_recognised(db):
    assert session_edit.is_live_database(db, db) is True


def test_a_symlink_to_the_live_database_is_still_the_live_database(db, tmp_path):
    """Compared by realpath, or the file manager's resolved path and the recorder's
    own name would look like two different files and an edit would land mid-service."""
    link = str(tmp_path / "link.db")
    try:
        os.symlink(db, link)
    except (OSError, NotImplementedError):  # Windows without developer mode
        pytest.skip("this filesystem will not make a symlink")

    assert session_edit.is_live_database(link, db) is True


def test_nothing_is_the_live_database_when_nothing_is_recording(db):
    assert session_edit.is_live_database(db, None) is False
    assert session_edit.is_live_database(db, "") is False


# --- keeping the original ---------------------------------------------------


def test_the_backup_keeps_the_extension_so_it_can_be_opened_again(db):
    assert session_edit.backup_path(db).endswith(".db")


def test_the_backup_is_stamped_so_a_second_edit_keeps_the_first_original(db):
    first = session_edit.backup_path(db, now=1_700_000_000)
    second = session_edit.backup_path(db, now=1_700_003_600)

    assert first != second


def test_making_a_backup_copies_the_rows(db):
    copy = session_edit.make_backup(db)
    _apply(db, [{"op": "delete", "id": 1}])

    assert len(_rows(copy)) == 3
    assert len(_rows(db)) == 2

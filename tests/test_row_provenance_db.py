"""The ASR-model stamp on a transcription row (speech_to_text.py's trigger).

Two things are being pinned here.

That the stamp cannot be bypassed: the row inserts live in nineteen statements with
nineteen different column lists — segment batch, phrase timeout, stop flush, and a
denied variant of each — so stamping from any of them means the twentieth, whenever
it is written, silently produces rows that cannot be attributed. A trigger has no
such hole.

And that it stays silent when it has nothing to add. While the transcribing model is
the one the session recorded at start, rows are NULL and the value lives once in
session_meta; repeating it per row measured 160 KB on a 3,500-row service. The
trigger is installed only when a hot reload changes the model, so a value in the
column means "not what the session started with" — and the boundary between the two
is visible in the rows themselves.
"""

import sqlite3

import pytest

from stt.session_meta import asr_row_label

# The statement built in initialize_database(). Kept verbatim so a change there
# that this file does not follow shows up as a failure rather than as a session
# whose rows are unattributed.
TRIGGER_SQL = (
    "CREATE TRIGGER stamp_asr_model AFTER INSERT ON transcriptions"
    " WHEN NEW.asr_model IS NULL"
    " BEGIN UPDATE transcriptions SET asr_model = '%s' WHERE id = NEW.id; END"
)

SCHEMA = """
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, text TEXT, translated_text TEXT,
    denied INTEGER DEFAULT 0, is_final INTEGER DEFAULT 1,
    asr_model TEXT DEFAULT NULL, mt_engine TEXT DEFAULT NULL, mt_model TEXT DEFAULT NULL)
"""


def session_db(path, label=None):
    """A session database. ``label`` None = the baseline case (no stamp installed)."""
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    set_stamp(conn, label)
    return conn


def set_stamp(conn, label):
    """Mirror of _set_asr_row_stamp(): install, replace, or remove the trigger."""
    conn.execute("DROP TRIGGER IF EXISTS stamp_asr_model")
    if label:
        conn.execute(TRIGGER_SQL % label.replace("'", "''"))
    conn.commit()


@pytest.fixture()
def conn(tmp_path):
    c = session_db(tmp_path / "session.db", "faster-whisper/large-v3")
    yield c
    c.close()


class TestAsrStampAfterAChange:
    """Once a hot reload has changed the model, every insert shape carries it."""

    @pytest.mark.parametrize("columns,values", [
        ("timestamp, text", ("t", "Мир вам.")),
        ("text", ("Мир вам.",)),
        ("timestamp, text, denied, is_final", ("t", "Мир вам.", 0, 1)),
        # A denied row is still evidence about the model that produced it.
        ("timestamp, text, denied", ("t", "Субтитры", 1)),
        # A partial row, written every second while a segment is still forming.
        ("timestamp, text, is_final", ("t", "Мир", 0)),
    ])
    def test_every_insert_shape_is_stamped(self, conn, columns, values):
        placeholders = ", ".join("?" * len(values))
        conn.execute(f"INSERT INTO transcriptions ({columns}) VALUES ({placeholders})", values)
        conn.commit()
        stamped = conn.execute("SELECT asr_model FROM transcriptions ORDER BY id DESC LIMIT 1").fetchone()
        assert stamped[0] == "faster-whisper/large-v3"

    def test_an_explicit_value_is_not_overwritten(self, conn):
        # A writer that does know the model — a mid-session change, an import — keeps it.
        conn.execute("INSERT INTO transcriptions (text, asr_model) VALUES (?, ?)",
                     ("Мир вам.", "openai-whisper/small"))
        conn.commit()
        assert conn.execute("SELECT asr_model FROM transcriptions").fetchone()[0] == \
            "openai-whisper/small"

    def test_a_label_containing_a_quote_does_not_break_the_trigger(self, tmp_path):
        # The label is interpolated into DDL, so the escaping is load-bearing.
        conn = session_db(tmp_path / "quoted.db", "custom/o'brien-asr")
        conn.execute("INSERT INTO transcriptions (text) VALUES ('x')")
        conn.commit()
        assert conn.execute("SELECT asr_model FROM transcriptions").fetchone()[0] == \
            "custom/o'brien-asr"
        conn.close()

    def test_a_real_config_label_round_trips(self, tmp_path):
        label = asr_row_label({"model": {"type": "whisper", "backend": "faster-whisper",
                                         "whisper": {"model": "large-v3"}}})
        conn = session_db(tmp_path / "real.db", label)
        conn.execute("INSERT INTO transcriptions (text) VALUES ('x')")
        conn.commit()
        assert conn.execute("SELECT asr_model FROM transcriptions").fetchone()[0] == label
        conn.close()

    def test_without_a_label_rows_are_simply_unstamped(self, tmp_path):
        # A config that names no model must not stop transcription being recorded.
        conn = session_db(tmp_path / "nolabel.db", "")
        conn.execute("INSERT INTO transcriptions (text) VALUES ('x')")
        conn.commit()
        assert conn.execute("SELECT asr_model FROM transcriptions").fetchone()[0] is None
        conn.close()


class TestBaselineRowsStayNull:
    """The normal case: nothing to add, so nothing is written."""

    def test_a_session_running_its_own_model_writes_no_label(self, tmp_path):
        conn = session_db(tmp_path / "baseline.db")
        for _ in range(50):
            conn.execute("INSERT INTO transcriptions (text) VALUES ('Мир вам.')")
        conn.commit()
        stamped = conn.execute(
            "SELECT COUNT(*) FROM transcriptions WHERE asr_model IS NOT NULL").fetchone()[0]
        assert stamped == 0, "the session's own model must not be repeated on every row"
        conn.close()


class TestHotReloadBoundary:
    """A model swapped mid-service must be visible in the rows, not inferred."""

    def test_rows_before_and_after_a_change_are_distinguishable(self, tmp_path):
        conn = session_db(tmp_path / "reload.db")
        conn.execute("INSERT INTO transcriptions (text) VALUES ('before')")
        conn.commit()

        set_stamp(conn, "faster-whisper/medium")          # the hot reload
        conn.execute("INSERT INTO transcriptions (text) VALUES ('after')")
        conn.commit()

        rows = conn.execute("SELECT text, asr_model FROM transcriptions ORDER BY id").fetchall()
        assert rows == [("before", None), ("after", "faster-whisper/medium")]
        conn.close()

    def test_changing_back_stops_stamping_again(self, tmp_path):
        # Returning to the session's own model makes the label redundant once more.
        conn = session_db(tmp_path / "back.db")
        set_stamp(conn, "faster-whisper/medium")
        conn.execute("INSERT INTO transcriptions (text) VALUES ('changed')")
        set_stamp(conn, None)
        conn.execute("INSERT INTO transcriptions (text) VALUES ('restored')")
        conn.commit()
        rows = conn.execute("SELECT text, asr_model FROM transcriptions ORDER BY id").fetchall()
        assert rows == [("changed", "faster-whisper/medium"), ("restored", None)]
        conn.close()


class TestTranslationColumns:
    """mt_engine/mt_model are written by the UPDATE that lands the translation."""

    def test_engine_and_model_are_recorded_together(self, conn):
        conn.execute("INSERT INTO transcriptions (text) VALUES ('Мир вам.')")
        conn.execute("UPDATE transcriptions SET translated_text = ?, mt_engine = ?, mt_model = ?"
                     " WHERE id = 1", ("Peace be with you.", "llm", "gemma-3-4b-it-Q4_K_M.gguf"))
        conn.commit()
        row = conn.execute("SELECT mt_engine, mt_model FROM transcriptions WHERE id = 1").fetchone()
        assert row == ("llm", "gemma-3-4b-it-Q4_K_M.gguf")

    def test_two_engines_in_one_session_stay_distinguishable(self, conn):
        # The reason these columns exist: a session configured for the LLM contains
        # NMT rows wherever the LLM declined, and a replay must not compare its
        # output against the other model's on exactly those rows.
        conn.execute("INSERT INTO transcriptions (text) VALUES ('a')")
        conn.execute("INSERT INTO transcriptions (text) VALUES ('b')")
        conn.execute("UPDATE transcriptions SET mt_engine = 'llm' WHERE id = 1")
        conn.execute("UPDATE transcriptions SET mt_engine = 'nmt' WHERE id = 2")
        conn.commit()
        engines = [r[0] for r in conn.execute("SELECT mt_engine FROM transcriptions ORDER BY id")]
        assert engines == ["llm", "nmt"]

    def test_an_untranslated_row_keeps_null_columns(self, conn):
        conn.execute("INSERT INTO transcriptions (text) VALUES ('Мир вам.')")
        conn.commit()
        row = conn.execute("SELECT mt_engine, mt_model FROM transcriptions").fetchone()
        assert row == (None, None)

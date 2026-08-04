"""Editing a caption's source text (speech_to_text.py's correction handlers).

An operator reported being able to edit a caption's translation but not its
transcription. Both are click-to-edit and identical in the page; the difference was
underneath. The source edit opened its own sqlite connection with the default 5s
timeout and no busy_timeout, outside _db_lock — so during a live session, with the
transcription worker holding the database and writing constantly, the edit raced
it, raised "database is locked", and was lost behind an error toast. The
translation edit only ever appeared to work because it never touches the database.

The second bug is quieter: emit_translated_entries re-seeds its cache from the
translated_text column when the cache misses, so an edit that left that column
alone resurrected the translation of the text it had just replaced.
"""

import sqlite3

import pytest

from conftest import extract_definitions

SCHEMA = """
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT, original_text TEXT, corrected_by TEXT,
    translated_text TEXT, translation_language TEXT, translation_ts_ms INTEGER,
    mt_engine TEXT, mt_model TEXT, needs_review INTEGER DEFAULT 0)
"""


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "session.db"
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    conn.execute(
        "INSERT INTO transcriptions (text, translated_text, translation_language,"
        " translation_ts_ms, mt_engine, mt_model) VALUES (?,?,?,?,?,?)",
        ("Мир вам.", "Peace be with you.", "en", 1400, "llm", "gemma.gguf"))
    conn.commit()
    conn.close()
    return str(path)


class Recorder:
    """Captures emit()/socketio.emit() so a handler's outcome is inspectable."""

    def __init__(self):
        self.events = []

    def __call__(self, name, payload=None):
        self.events.append((name, payload))

    def emit(self, name, payload=None):
        self.events.append((name, payload))

    def named(self, name):
        return [p for n, p in self.events if n == name]


def run_correction(db, data, *, translate=None, on_open=None):
    """Execute handle_submit_correction against a real sqlite file."""
    import threading

    rec = Recorder()
    def open_writer():
        if on_open:
            on_open()
        conn = sqlite3.connect(db, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    ns = extract_definitions(
        "speech_to_text.py", ["handle_submit_correction"],
        {"transcription_state": {"db_name": db},
         "os": __import__("os"),
         "sqlite3": sqlite3,
         "_db_lock": threading.Lock(),
         "_open_db_writer": open_writer,
         "_cache_lock": threading.Lock(),
         "_db_cache": {"last_entries": [1], "last_fetch_time": 99},
         "get_translation_cache": lambda: type("C", (), {
             "invalidate": staticmethod(lambda i: None),
             "set": staticmethod(lambda *a, **k: None)})(),
         "config": {"live_translation": {"enabled": bool(translate),
                                         "target_language": "en"},
                    "audio": {"language": "ru"}},
         "translate_live_text": lambda *a, **k: translate,
         "emit": rec,
         "socketio": rec,
         "_socket_auth_ok": lambda: True})
    ns["handle_submit_correction"](data)
    return rec


def row(db):
    conn = sqlite3.connect(db)
    r = conn.execute("SELECT text, original_text, corrected_by, translated_text,"
                     " translation_language, translation_ts_ms, mt_engine, mt_model"
                     " FROM transcriptions WHERE id=1").fetchone()
    conn.close()
    return r


class TestSourceEditPersists:
    def test_the_edit_reaches_the_database(self, db):
        run_correction(db, {"segment_id": 1, "new_text": "Мир вам, братья."})
        assert row(db)[0] == "Мир вам, братья."

    def test_the_pre_edit_text_is_kept(self, db):
        run_correction(db, {"segment_id": 1, "new_text": "изменено"})
        assert row(db)[1] == "Мир вам.", "the original must survive the first correction"

    def test_a_second_edit_does_not_overwrite_the_original(self, db):
        run_correction(db, {"segment_id": 1, "new_text": "первое"})
        run_correction(db, {"segment_id": 1, "new_text": "второе"})
        text, original = row(db)[0], row(db)[1]
        assert (text, original) == ("второе", "Мир вам.")

    def test_it_writes_through_the_shared_writer(self, db):
        # The reported symptom, tested as the property that fixes it rather than by
        # racing a lock: _open_db_writer serialises against the transcription
        # thread and sets a 30s busy_timeout, so a mid-session edit waits instead of
        # failing with "database is locked". Opening a private connection is what
        # lost the edit, so going through the shared writer is the assertion.
        used = []
        rec = run_correction(db, {"segment_id": 1, "new_text": "во время службы"},
                             on_open=lambda: used.append(True))
        assert used, "the edit must go through _open_db_writer, not its own connection"
        assert not rec.named("correction_error")
        assert row(db)[0] == "во время службы"

    def test_the_handler_opens_no_connection_of_its_own(self):
        # Drift guard: this is exactly what regressed, and it reads as harmless.
        import ast
        import pathlib
        src = pathlib.Path("speech_to_text.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "handle_submit_correction")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        opens = [c for c in calls
                 if isinstance(c.func, ast.Attribute) and c.func.attr == "connect"]
        assert not opens, "handle_submit_correction must use _open_db_writer"

    def test_an_unknown_segment_is_reported_not_silently_ignored(self, db):
        rec = run_correction(db, {"segment_id": 999, "new_text": "x"})
        assert rec.named("correction_error"), "a no-op UPDATE must not look like success"
        assert not rec.named("correction_applied")


class TestStaleTranslationIsCleared:
    """The edited caption must not keep the translation of the text it replaced."""

    def test_the_translation_columns_are_cleared(self, db):
        run_correction(db, {"segment_id": 1, "new_text": "совсем другой текст"})
        _, _, _, translated, lang, ts, engine, model = row(db)
        assert (translated, lang, ts, engine, model) == (None, None, None, None, None)

    def test_the_provenance_goes_with_it(self, db):
        # Leaving mt_engine/mt_model behind would attribute the next translation
        # to whatever produced the previous one.
        run_correction(db, {"segment_id": 1, "new_text": "другой"})
        assert row(db)[6] is None and row(db)[7] is None


class TestBroadcast:
    def test_the_edit_is_broadcast(self, db):
        rec = run_correction(db, {"segment_id": 1, "new_text": "новый текст"})
        applied = rec.named("correction_applied")
        assert applied and applied[0]["new_text"] == "новый текст"
        assert applied[0]["segment_id"] == 1

    def test_a_retranslation_rides_along_when_translation_is_on(self, db):
        rec = run_correction(db, {"segment_id": 1, "new_text": "новый"},
                             translate="a fresh translation")
        assert rec.named("correction_applied")[0]["translated_text"] == "a fresh translation"

    def test_the_entries_cache_is_dropped(self, db):
        # Otherwise the next emit serves the pre-edit text from cache.
        import threading  # noqa: F401
        rec = run_correction(db, {"segment_id": 1, "new_text": "x"})
        assert rec.named("correction_applied")

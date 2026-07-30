"""Session provenance sqlite storage (stt/session_meta.py write/append/read)."""

import os
import shutil
import sqlite3

from stt.session_meta import (
    append_changes,
    build_session_meta,
    changed_keys,
    load_session_meta,
    read_history,
    read_session_meta,
    remote_provenance,
    write_missing,
    write_session_meta,
)

MADLAD_DEFAULT = "google/madlad400-3b-mt"


def session_db(tmp_path, name="2026-05-20_183919.db"):
    """A session db shaped like a real one: transcriptions table, no session_meta."""
    path = str(tmp_path / name)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT, text TEXT, translated_text TEXT, translation_language TEXT)"
        )
        conn.commit()
    return path


def config_at(target_language="en", method="madlad", model=MADLAD_DEFAULT, **overrides):
    lt = {"enabled": True, "translation_method": method, "translation_model": model,
          "target_language": target_language, "context_window": 1}
    lt.update(overrides)
    return {
        "model": {"type": "whisper", "backend": "whisper", "whisper": {"model": "small"}},
        "whisper_decoding": {"live_transcription": {"beam_size": 3, "logprob_threshold": -0.5}},
        "live_translation": lt,
    }


def meta_for(config, started_at="2026-05-20T18:39:19"):
    return build_session_meta(config, "26.1.168", "abc1234", "26.1.168", "stt-box",
                              MADLAD_DEFAULT, started_at=started_at)


class TestWrite:
    def test_creates_table_and_stores_values(self, tmp_path):
        db = session_db(tmp_path)
        assert write_session_meta(db, meta_for(config_at())) is True

        stored = read_session_meta(db)
        assert stored["asr.model"] == "small"
        assert stored["asr.implementation"] == "openai-whisper"
        assert stored["mt.model"] == MADLAD_DEFAULT
        assert stored["mt.target_language"] == "en"
        assert stored["mt.context_window"] == "1"

    def test_leaves_the_transcriptions_table_alone(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at()))
        with sqlite3.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"transcriptions", "session_meta"} <= tables

    def test_second_call_is_idempotent(self, tmp_path):
        db = session_db(tmp_path)
        meta = meta_for(config_at())
        write_session_meta(db, meta)
        write_session_meta(db, meta)
        assert read_session_meta(db) == meta

    def test_empty_meta_is_a_noop(self, tmp_path):
        db = session_db(tmp_path)
        assert write_session_meta(db, {}) is False
        assert read_session_meta(db) == {}

    def test_unwritable_path_does_not_raise(self, tmp_path):
        # Provenance must never take down a session that is about to start.
        assert write_session_meta(str(tmp_path / "no" / "such" / "dir.db"),
                                  {"a": "b"}) is False


class TestAppendChanges:
    def test_hot_language_switch_appends_and_preserves_the_base_key(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at(target_language="en")))

        append_changes(db, {"mt.target_language": "es"}, changed_at="2026-05-20T19:10:00")

        stored = read_session_meta(db)
        assert stored["mt.target_language"] == "en", "session-start value must survive"
        assert stored["mt.target_language@2026-05-20T19:10:00"] == "es"

    def test_repeated_switches_build_a_timeline(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at(target_language="en")))
        append_changes(db, {"mt.target_language": "es"}, changed_at="2026-05-20T19:10:00")
        append_changes(db, {"mt.target_language": "de"}, changed_at="2026-05-20T20:05:00")
        append_changes(db, {"mt.target_language": "en"}, changed_at="2026-05-20T20:30:00")

        assert read_history(read_session_meta(db), "mt.target_language") == [
            ("", "en"),
            ("2026-05-20T19:10:00", "es"),
            ("2026-05-20T20:05:00", "de"),
            ("2026-05-20T20:30:00", "en"),
        ]

    def test_engine_swap_appends_method_and_model_together(self, tmp_path):
        db = session_db(tmp_path)
        before = meta_for(config_at(method="madlad", model=MADLAD_DEFAULT))
        write_session_meta(db, before)

        after = meta_for(config_at(method="nllb", model="facebook/nllb-200-distilled-600M"))
        append_changes(db, changed_keys(before, after), changed_at="2026-05-20T19:30:00")

        stored = read_session_meta(db)
        assert stored["mt.method"] == "madlad"
        assert stored["mt.model"] == MADLAD_DEFAULT
        assert stored["mt.method@2026-05-20T19:30:00"] == "nllb"
        assert stored["mt.model@2026-05-20T19:30:00"] == "facebook/nllb-200-distilled-600M"

    def test_unchanged_settings_are_not_appended(self, tmp_path):
        db = session_db(tmp_path)
        before = meta_for(config_at())
        write_session_meta(db, before)
        after = meta_for(config_at())

        assert append_changes(db, changed_keys(before, after)) is False
        assert read_session_meta(db) == before

    def test_watched_config_change_records_context_window(self, tmp_path):
        db = session_db(tmp_path)
        before = meta_for(config_at(context_window=1))
        write_session_meta(db, before)
        after = meta_for(config_at(context_window=3))

        append_changes(db, changed_keys(before, after), changed_at="2026-05-20T19:45:00")
        stored = read_session_meta(db)
        assert stored["mt.context_window"] == "1"
        assert stored["mt.context_window@2026-05-20T19:45:00"] == "3"

    def test_creates_the_table_when_write_never_ran(self, tmp_path):
        # A language switch can land before/without the initial write; it must
        # still record rather than silently drop the change.
        db = session_db(tmp_path)
        assert append_changes(db, {"mt.target_language": "es"},
                              changed_at="2026-05-20T19:10:00") is True
        assert read_session_meta(db)["mt.target_language@2026-05-20T19:10:00"] == "es"

    def test_failure_does_not_propagate(self, tmp_path):
        # A failed append must never break a live language switch.
        assert append_changes(str(tmp_path / "nope" / "x.db"), {"a": "b"}) is False

    def test_default_timestamp_is_used_when_omitted(self, tmp_path):
        db = session_db(tmp_path)
        append_changes(db, {"mt.target_language": "es"})
        keys = [k for k in read_session_meta(db) if k.startswith("mt.target_language@")]
        assert len(keys) == 1
        assert "T" in keys[0]


class TestLiveSession:
    """The real condition: the session's own connection is open, db is in WAL mode."""

    def open_wal_session(self, tmp_path):
        path = str(tmp_path / "2026-05-20_183919.db")
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "timestamp TEXT, text TEXT)")
        conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES (' ', ' ')")
        conn.commit()
        return path, conn

    def test_writes_while_the_session_connection_is_open(self, tmp_path):
        db, conn = self.open_wal_session(tmp_path)
        try:
            assert write_session_meta(db, meta_for(config_at())) is True
            assert append_changes(db, {"mt.target_language": "es"},
                                  changed_at="2026-05-20T19:10:00") is True

            # The session must keep recording afterwards, still in WAL.
            before = conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
            conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES ('19:00', 'hi')")
            conn.commit()
            after = conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
            assert after == before + 1
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            conn.close()

    def test_session_connection_sees_the_provenance(self, tmp_path):
        db, conn = self.open_wal_session(tmp_path)
        try:
            write_session_meta(db, meta_for(config_at()))
            row = conn.execute(
                "SELECT value FROM session_meta WHERE key='asr.model'").fetchone()
            assert row is not None and row[0] == "small"
        finally:
            conn.close()


class TestDirectionFlip:
    """A bilingual service flips both languages at once; both must be recorded.

    Reproduces the real 23:09 flip on .62 — transcription ru->en and translation
    en->ru together. Recording only the translation half leaves asr.language
    reading as the session's starting value while the per-row source_language
    column says otherwise, which is worse than silence: it looks authoritative.
    """

    def test_both_halves_of_a_flip_are_recorded(self, tmp_path):
        db = session_db(tmp_path)
        cfg = config_at(target_language="en")
        cfg["audio"] = {"language": "ru"}
        write_session_meta(db, meta_for(cfg))

        # 23:09:16 - flip: ASR told the audio is English, translate into Russian.
        append_changes(db, {"asr.language": "en"}, changed_at="2026-05-20T23:09:16")
        append_changes(db, {"mt.target_language": "ru"}, changed_at="2026-05-20T23:09:16")
        # 23:12:07 - flipped back.
        append_changes(db, {"asr.language": "ru"}, changed_at="2026-05-20T23:12:07")
        append_changes(db, {"mt.target_language": "en"}, changed_at="2026-05-20T23:12:07")

        stored = read_session_meta(db)
        assert read_history(stored, "asr.language") == [
            ("", "ru"), ("2026-05-20T23:09:16", "en"), ("2026-05-20T23:12:07", "ru")]
        assert read_history(stored, "mt.target_language") == [
            ("", "en"), ("2026-05-20T23:09:16", "ru"), ("2026-05-20T23:12:07", "en")]

    def test_flip_timeline_agrees_with_per_row_source_language(self, tmp_path):
        db = session_db(tmp_path)
        cfg = config_at(target_language="en")
        cfg["audio"] = {"language": "ru"}
        write_session_meta(db, meta_for(cfg))
        with sqlite3.connect(db) as conn:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN source_language TEXT")
            conn.executemany(
                "INSERT INTO transcriptions (timestamp, text, translated_text, "
                "translation_language, source_language) VALUES (?, ?, ?, ?, ?)",
                [("23:05:03", "Мир вам", "Peace be with you", "en", "ru"),
                 ("23:09:26", "A short line goes here.", "Короткая строка здесь.", "ru", "en"),
                 ("23:12:18", "Вы знаете", "You know", "en", "ru")],
            )
            conn.commit()
        append_changes(db, {"asr.language": "en"}, changed_at="2026-05-20T23:09:16")
        append_changes(db, {"asr.language": "ru"}, changed_at="2026-05-20T23:12:07")

        stored = read_session_meta(db)
        recorded = {v for _, v in read_history(stored, "asr.language")}
        with sqlite3.connect(db) as conn:
            per_row = {r[0] for r in conn.execute(
                "SELECT DISTINCT source_language FROM transcriptions "
                "WHERE source_language IS NOT NULL")}
        assert recorded == per_row, "provenance must not contradict the rows"


class TestWriteMissing:
    """Late-arriving session-start facts: the remote's model needs a network call."""

    def test_fills_absent_keys_as_base_keys(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at()))

        assert write_missing(db, {"mt.remote.effective.model": "google/madlad400-3b-mt"}) is True
        stored = read_session_meta(db)
        # A base key, not a timestamped change - nothing changed, we found out.
        assert stored["mt.remote.effective.model"] == "google/madlad400-3b-mt"
        assert read_history(stored, "mt.remote.effective.model") == [
            ("", "google/madlad400-3b-mt")]

    def test_never_overwrites_an_existing_value(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at()))
        before = read_session_meta(db)["mt.model"]

        assert write_missing(db, {"mt.model": "something/else"}) is False
        assert read_session_meta(db)["mt.model"] == before

    def test_empty_mapping_is_a_noop(self, tmp_path):
        db = session_db(tmp_path)
        assert write_missing(db, {}) is False

    def test_failure_does_not_propagate(self, tmp_path):
        assert write_missing(str(tmp_path / "nope" / "x.db"), {"a": "b"}) is False

    def test_a_later_genuine_change_still_appends(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at()))
        write_missing(db, {"mt.remote.effective.model": "google/madlad400-3b-mt"})

        # Remote later switched model: that IS a change and belongs in the timeline.
        append_changes(db, {"mt.remote.effective.model": "facebook/nllb-200-distilled-600M"},
                       changed_at="2026-05-20T23:00:00")
        assert read_history(read_session_meta(db), "mt.remote.effective.model") == [
            ("", "google/madlad400-3b-mt"),
            ("2026-05-20T23:00:00", "facebook/nllb-200-distilled-600M"),
        ]


class TestOffloadedSessionEndToEnd:
    def test_offloaded_session_records_the_remote_not_the_local_model(self, tmp_path):
        """Reproduces the .62 -> .52 topology that exposed the original gap."""
        db = session_db(tmp_path)
        cfg = config_at(method="nllb", model="facebook/nllb-200-distilled-600M")
        cfg["live_translation"]["remote"] = {"enabled": True,
                                             "endpoint": "192.168.2.52:8080",
                                             "model": ""}
        write_session_meta(db, meta_for(cfg))

        # Before the probe: no model is claimed at all.
        stored = read_session_meta(db)
        assert stored["mt.offloaded"] == "true"
        assert stored["mt.model"] == ""

        # The probe lands what actually translated.
        write_missing(db, remote_provenance({
            "success": True, "translation_model": "google/madlad400-3b-mt",
            "translation_method": "madlad", "model_device": "mps"}))

        stored = read_session_meta(db)
        assert stored["mt.remote.effective.model"] == "google/madlad400-3b-mt"
        assert stored["mt.remote.effective.device"] == "mps"
        # The local config stays visible without ever being mistaken for the truth.
        assert stored["mt.model_configured"] == "facebook/nllb-200-distilled-600M"
        assert stored["mt.model"] == ""

    def test_unreachable_remote_leaves_no_model_claim(self, tmp_path):
        db = session_db(tmp_path)
        cfg = config_at()
        cfg["live_translation"]["remote"] = {"enabled": True,
                                            "endpoint": "192.168.2.52:8080", "model": ""}
        write_session_meta(db, meta_for(cfg))
        write_missing(db, remote_provenance(None))  # probe failed

        stored = read_session_meta(db)
        assert stored["mt.model"] == ""
        assert not any(k.startswith("mt.remote.effective.") for k in stored)


class TestReadDetachedFromSidecars:
    """A WAL database copied without its -wal/-shm sidecars must still be readable.

    This is the normal state of any delivered or archived session: file_mover
    sends the .db to a NAS, and a download hands over the .db alone. Plain
    read-only cannot open such a file because SQLite wants to create the -shm.
    """

    def wal_db_without_sidecars(self, tmp_path):
        source = str(tmp_path / "source.db")
        conn = sqlite3.connect(source)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, text TEXT)")
        conn.commit()
        write_session_meta(source, meta_for(config_at()))
        # Checkpoint explicitly, exactly as ending a session does
        # (speech_to_text.py runs PRAGMA wal_checkpoint(TRUNCATE) before closing).
        # Relying on close() to checkpoint is not portable: macOS does, Linux
        # leaves the -wal in place, so a copy of the .db alone would silently
        # carry no rows and this fixture would model nothing.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        detached = str(tmp_path / "delivered.db")
        shutil.copyfile(source, detached)
        assert not os.path.exists(detached + "-shm")
        assert not os.path.exists(detached + "-wal")
        return detached

    def test_reads_a_detached_wal_database(self, tmp_path):
        db = self.wal_db_without_sidecars(tmp_path)
        meta, error = load_session_meta(db)
        assert error is None, f"detached WAL db should read cleanly, got {error}"
        assert meta["asr.model"] == "small"
        assert meta["mt.model"] == MADLAD_DEFAULT

    def test_reading_creates_no_sidecars(self, tmp_path):
        db = self.wal_db_without_sidecars(tmp_path)
        read_session_meta(db)
        assert not os.path.exists(db + "-shm"), "read must not write beside the file"
        assert not os.path.exists(db + "-wal")


class TestReadErrorsAreDistinguishable:
    """"Nothing recorded" and "couldn't read it" must never look the same."""

    def test_pre_provenance_session_is_not_an_error(self, tmp_path):
        meta, error = load_session_meta(session_db(tmp_path))
        assert (meta, error) == ({}, None)

    def test_missing_file_is_an_error_not_an_empty_session(self, tmp_path):
        meta, error = load_session_meta(str(tmp_path / "absent.db"))
        assert meta == {}
        assert error, "a file that isn't there must not read as 'nothing recorded'"

    def test_non_database_file_is_an_error(self, tmp_path):
        junk = tmp_path / "notadb.db"
        junk.write_text("this is not sqlite")
        meta, error = load_session_meta(str(junk))
        assert meta == {}
        assert error

    def test_unusable_path_is_reported_not_raised(self):
        # A path sqlite can't even be handed (null byte) fails outside sqlite3's
        # own error hierarchy; it must still be reported rather than propagate.
        meta, error = load_session_meta("bad\x00path.db")
        assert meta == {}
        assert error

    def test_recorded_session_has_no_error(self, tmp_path):
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at()))
        meta, error = load_session_meta(db)
        assert error is None
        assert meta["asr.model"] == "small"


class TestRead:
    def test_pre_provenance_session_reads_as_empty(self, tmp_path):
        # Sessions recorded before this feature have no session_meta table; that
        # is normal and must not surface as an error.
        assert read_session_meta(session_db(tmp_path)) == {}

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_session_meta(str(tmp_path / "absent.db")) == {}

    def test_non_database_file_reads_as_empty(self, tmp_path):
        junk = tmp_path / "notadb.db"
        junk.write_text("this is not sqlite")
        assert read_session_meta(str(junk)) == {}

    def test_null_value_reads_as_empty_string(self, tmp_path):
        db = session_db(tmp_path)
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE session_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO session_meta (key, value) VALUES ('mt.model', NULL)")
            conn.commit()
        assert read_session_meta(db)["mt.model"] == ""


class TestAgreementWithRowData:
    def test_recorded_timeline_matches_per_row_translation_language(self, tmp_path):
        """session_meta history and the per-row column must tell the same story.

        translation_language already records what each row was translated to;
        session_meta explains when and why it changed. If they disagree, one of
        them is lying about the session.
        """
        db = session_db(tmp_path)
        write_session_meta(db, meta_for(config_at(target_language="en")))
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO transcriptions (timestamp, text, translated_text, "
                "translation_language) VALUES (?, ?, ?, ?)",
                [("18:40:00", "привет", "hello", "en"),
                 ("18:50:00", "мир", "peace", "en"),
                 ("19:20:00", "привет", "hola", "es")],
            )
            conn.commit()
        append_changes(db, {"mt.target_language": "es"}, changed_at="2026-05-20T19:10:00")

        timeline = read_history(read_session_meta(db), "mt.target_language")
        with sqlite3.connect(db) as conn:
            row_langs = [r[0] for r in conn.execute(
                "SELECT DISTINCT translation_language FROM transcriptions "
                "WHERE translation_language IS NOT NULL ORDER BY translation_language")]

        assert sorted({value for _, value in timeline}) == row_langs

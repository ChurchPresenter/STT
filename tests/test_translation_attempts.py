"""Retrying a caption nothing translated, and never storing the source as its translation.

The defect these rules replace: a caption whose remote translation timed out was written
to the database as its own source text, which took it out of every set that would have
retried or repaired it. It was then Russian in an English SRT, permanently.
"""

import sqlite3

from stt.translation_attempts import (
    BACKFILL_SUCCESS_SET,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    LIVE_SUCCESS_SET,
    LiveTranslationAttempts,
    missing_attempt_columns,
    persist_decision,
    record_failed_attempt,
)

NONE = "none"
NOW = 1_000.0


class TestPersistDecision:
    def test_a_real_translation_is_shown_and_stored(self):
        assert persist_decision("remote", NONE, model_ready=True) == (True, True)
        assert persist_decision("llm", NONE, model_ready=True) == (True, True)

    def test_an_untranslated_caption_is_shown_but_never_stored(self):
        assert persist_decision(NONE, NONE, model_ready=True) == (True, False)

    def test_a_loading_model_produces_neither(self):
        # The warmup echo: not a translation and not an attempt, so the row stays NULL
        # and the caption is picked up again once the model is up.
        assert persist_decision("remote", NONE, model_ready=False) == (False, False)
        assert persist_decision(NONE, NONE, model_ready=False) == (False, False)


class TestShouldAttempt:
    def test_a_caption_never_seen_is_attempted(self):
        attempts = LiveTranslationAttempts()
        assert attempts.should_attempt(1, now=NOW) is True

    def test_a_failed_caption_waits_out_the_cooldown(self):
        attempts = LiveTranslationAttempts(cooldown_seconds=20.0)
        attempts.record_failure(1, "исходный текст", now=NOW)
        assert attempts.should_attempt(1, now=NOW + 1.0) is False
        assert attempts.should_attempt(1, now=NOW + 19.9) is False
        assert attempts.should_attempt(1, now=NOW + 20.0) is True

    def test_retries_end_at_the_cap(self):
        attempts = LiveTranslationAttempts(max_attempts=3, cooldown_seconds=1.0)
        now = NOW
        for _ in range(3):
            assert attempts.should_attempt(1, now=now) is True
            attempts.record_failure(1, "исходный текст", now=now)
            now += 10.0
        assert attempts.should_attempt(1, now=now) is False
        assert attempts.exhausted(1) is True

    def test_a_dead_peer_costs_one_timeout_per_cooldown_not_one_per_cycle(self):
        # The pump cycles every 0.5s; without the cooldown each cycle would spend another
        # 15s timeout on the same caption.
        attempts = LiveTranslationAttempts(cooldown_seconds=20.0)
        attempts.record_failure(7, "исходный текст", now=NOW)
        cycles = [NOW + 0.5 * n for n in range(1, 40)]
        assert not any(attempts.should_attempt(7, now=t) for t in cycles)

    def test_captions_are_tracked_independently(self):
        attempts = LiveTranslationAttempts(max_attempts=1)
        attempts.record_failure(1, "один", now=NOW)
        assert attempts.should_attempt(1, now=NOW + 600.0) is False
        assert attempts.should_attempt(2, now=NOW + 600.0) is True


class TestDisplayText:
    def test_the_source_is_held_for_the_display(self):
        attempts = LiveTranslationAttempts()
        attempts.record_failure(1, "исходный текст", now=NOW)
        assert attempts.display_text(1) == "исходный текст"

    def test_nothing_is_held_for_a_caption_that_never_failed(self):
        assert LiveTranslationAttempts().display_text(1) is None

    def test_a_caption_that_recovers_is_dropped(self):
        attempts = LiveTranslationAttempts()
        attempts.record_failure(1, "исходный текст", now=NOW)
        attempts.record_success(1)
        assert attempts.display_text(1) is None
        assert attempts.size() == 0

    def test_recovery_also_clears_the_attempt_count(self):
        attempts = LiveTranslationAttempts(max_attempts=2, cooldown_seconds=0.0)
        attempts.record_failure(1, "исходный текст", now=NOW)
        attempts.record_failure(1, "исходный текст", now=NOW + 1.0)
        assert attempts.exhausted(1) is True
        attempts.record_success(1)
        assert attempts.exhausted(1) is False
        assert attempts.should_attempt(1, now=NOW + 2.0) is True


class TestReset:
    def test_a_new_session_starts_clean(self):
        # Ids restart low in a new session database, so a carried count would land on an
        # unrelated caption.
        attempts = LiveTranslationAttempts(max_attempts=1)
        attempts.record_failure(1, "исходный текст", now=NOW)
        assert attempts.should_attempt(1, now=NOW + 600.0) is False
        attempts.reset()
        assert attempts.should_attempt(1, now=NOW + 600.0) is True
        assert attempts.display_text(1) is None
        assert attempts.size() == 0


class TestDefaults:
    def test_the_shipped_defaults_are_the_measured_ones(self):
        assert DEFAULT_MAX_ATTEMPTS == 3
        assert DEFAULT_COOLDOWN_SECONDS == 20.0

    def test_the_default_cap_applies_when_unconfigured(self):
        attempts = LiveTranslationAttempts(cooldown_seconds=0.0)
        for n in range(DEFAULT_MAX_ATTEMPTS):
            attempts.record_failure(1, "исходный текст", now=NOW + n)
        assert attempts.should_attempt(1, now=NOW + 100.0) is False


# --- the trace in the session database --------------------------------------

def _db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "s.db"))
    conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, text TEXT, translated_text TEXT)")
    for name, kind in missing_attempt_columns({"id", "text", "translated_text"}):
        conn.execute("ALTER TABLE transcriptions ADD COLUMN %s %s" % (name, kind))
    conn.execute("INSERT INTO transcriptions (id, text) VALUES (1, 'Мир вам')")
    return conn


def _row(conn):
    return conn.execute("SELECT translated_text, mt_attempts, mt_via FROM transcriptions WHERE id = 1").fetchone()


def _store(conn, set_clause):
    conn.execute("UPDATE transcriptions SET translated_text = 'Peace', %s WHERE id = 1" % set_clause)


def test_a_first_try_success_is_live(tmp_path):
    conn = _db(tmp_path)
    _store(conn, LIVE_SUCCESS_SET)
    assert _row(conn) == ("Peace", 1, "live")


def test_a_success_after_failures_is_a_retry_and_counts_every_try(tmp_path):
    conn = _db(tmp_path)
    record_failed_attempt(conn, 1)
    record_failed_attempt(conn, 1)
    _store(conn, LIVE_SUCCESS_SET)
    assert _row(conn) == ("Peace", 3, "retry")


def test_a_backfill_repair_says_so(tmp_path):
    conn = _db(tmp_path)
    for _ in range(3):
        record_failed_attempt(conn, 1)
    _store(conn, BACKFILL_SUCCESS_SET)
    assert _row(conn) == ("Peace", 4, "backfill")


def test_a_caption_that_never_translated_still_shows_how_hard_it_was_tried(tmp_path):
    conn = _db(tmp_path)
    record_failed_attempt(conn, 1)
    assert _row(conn) == (None, 1, None)


def test_a_late_failure_does_not_touch_a_translated_caption(tmp_path):
    conn = _db(tmp_path)
    _store(conn, LIVE_SUCCESS_SET)
    record_failed_attempt(conn, 1)
    assert _row(conn) == ("Peace", 1, "live")


def test_only_missing_columns_are_added():
    assert missing_attempt_columns({"mt_attempts", "mt_via"}) == ()
    assert [c[0] for c in missing_attempt_columns({"mt_via"})] == ["mt_attempts"]


def test_the_server_counts_a_failed_try_in_the_session_database(tmp_path):
    import os

    from conftest import extract_definitions

    conn = _db(tmp_path)
    conn.commit()
    conn.close()
    db = str(tmp_path / "s.db")
    ns = extract_definitions("speech_to_text.py", ["_note_failed_translation"], {
        "_ts_get": lambda key: db, "os": os, "sqlite3": sqlite3,
        "_record_failed_attempt": record_failed_attempt})
    ns["_note_failed_translation"](1)
    ns["_note_failed_translation"](1)
    assert _row(sqlite3.connect(db)) == (None, 2, None)


def test_a_missing_database_is_not_an_error(tmp_path):
    import os

    from conftest import extract_definitions

    ns = extract_definitions("speech_to_text.py", ["_note_failed_translation"], {
        "_ts_get": lambda key: str(tmp_path / "gone.db"), "os": os, "sqlite3": sqlite3,
        "_record_failed_attempt": record_failed_attempt})
    ns["_note_failed_translation"](1)


# --- waiting on a paired server that says whether it is working -----------------


class TestWaitingOnAPeer:
    def test_a_server_that_is_not_working_gets_one_try(self):
        a = LiveTranslationAttempts()
        a.record_failure(1, "Мир вам", NOW, peer_working=False)
        assert a.exhausted(1)
        assert not a.should_attempt(1, NOW + 10_000)
        assert a.display_text(1) == "Мир вам", "the transcription stays on screen"

    def test_a_busy_server_making_progress_keeps_the_caption_waiting(self):
        a = LiveTranslationAttempts()
        for i in range(DEFAULT_MAX_ATTEMPTS * 3):
            a.record_failure(1, "Мир вам", NOW + i * DEFAULT_COOLDOWN_SECONDS, peer_working=True)
        assert not a.exhausted(1), "a slow server's timeouts are not counted"

    def test_waiting_on_a_busy_server_is_capped(self):
        a = LiveTranslationAttempts(max_wait_seconds=180)
        a.record_failure(1, "Мир вам", NOW, peer_working=True)
        a.record_failure(1, "Мир вам", NOW + 179, peer_working=True)
        assert not a.exhausted(1)
        a.record_failure(1, "Мир вам", NOW + 180, peer_working=True)
        assert a.exhausted(1)

    def test_an_older_server_that_cannot_say_keeps_the_attempt_cap(self):
        a = LiveTranslationAttempts()
        for i in range(DEFAULT_MAX_ATTEMPTS - 1):
            a.record_failure(1, "Мир вам", NOW + i * 60, peer_working=None)
        assert not a.exhausted(1)
        a.record_failure(1, "Мир вам", NOW + 999, peer_working=None)
        assert a.exhausted(1)

    def test_a_server_that_stops_working_ends_the_wait(self):
        a = LiveTranslationAttempts()
        a.record_failure(1, "Мир вам", NOW, peer_working=True)
        a.record_failure(1, "Мир вам", NOW + 20, peer_working=False)
        assert a.exhausted(1)

    def test_success_forgets_the_wait(self):
        a = LiveTranslationAttempts(max_wait_seconds=60)
        a.record_failure(1, "Мир вам", NOW, peer_working=True)
        a.record_success(1)
        a.record_failure(1, "Мир вам", NOW + 100, peer_working=True)
        assert not a.exhausted(1), "a fresh failure starts a fresh wait"

    def test_reset_forgets_given_up_captions(self):
        a = LiveTranslationAttempts()
        a.record_failure(1, "x", NOW, peer_working=False)
        a.reset()
        assert not a.exhausted(1)

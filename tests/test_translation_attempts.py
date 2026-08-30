"""Retrying a caption nothing translated, and never storing the source as its translation.

The defect these rules replace: a caption whose remote translation timed out was written
to the database as its own source text, which took it out of every set that would have
retried or repaired it. It was then Russian in an English SRT, permanently.
"""

from stt.translation_attempts import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    LiveTranslationAttempts,
    persist_decision,
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

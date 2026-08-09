"""Detecting a session rollover (stt/session_reset.py)."""

import threading

from stt.session_reset import SessionTracker


class TestChanged:
    def test_first_sighting_is_not_a_change(self):
        # Starting up is not a rollover; reporting one would clear every
        # per-session cache on each boot for nothing.
        assert SessionTracker().changed("2026-08-08_101500") is False

    def test_same_session_repeated_is_not_a_change(self):
        t = SessionTracker()
        t.changed("s1")
        assert t.changed("s1") is False
        assert t.changed("s1") is False

    def test_a_new_session_is_a_change(self):
        t = SessionTracker()
        t.changed("2026-08-08_101500")
        assert t.changed("2026-08-08_113000") is True

    def test_a_change_is_reported_once_not_repeatedly(self):
        # The caller resets caches on a True, so a second True for the same
        # rollover would clear caches the new session has already filled.
        t = SessionTracker()
        t.changed("s1")
        assert t.changed("s2") is True
        assert t.changed("s2") is False

    def test_returning_to_a_previous_session_still_counts(self):
        t = SessionTracker()
        t.changed("s1")
        t.changed("s2")
        assert t.changed("s1") is True


class TestNoneHandling:
    def test_the_gap_between_sessions_is_not_a_change(self):
        # The worker publishes None while it retires one database before
        # opening the next. Treating that as a rollover fires the reset twice:
        # once into the gap, once on the real new id.
        t = SessionTracker()
        t.changed("s1")
        assert t.changed(None) is False

    def test_coming_back_from_the_gap_is_not_a_change_either(self):
        # ...and the reset must not fire on the way out of the gap for the
        # SAME session, which is what a plain != would do.
        t = SessionTracker()
        t.changed("s1")
        t.changed(None)
        assert t.changed("s1") is False

    def test_a_real_rollover_through_the_gap_is_caught_on_the_new_id(self):
        t = SessionTracker()
        t.changed("s1")
        t.changed(None)
        assert t.changed("s2") is True

    def test_none_before_any_session_is_ignored(self):
        t = SessionTracker()
        assert t.changed(None) is False
        assert t.current is None          # not recorded — it carries no id
        assert t.changed("s1") is False   # s1 is still the first real sighting

    def test_never_started_reports_nothing(self):
        assert SessionTracker().changed(None) is False


class TestState:
    def test_current_exposes_the_last_seen(self):
        t = SessionTracker()
        assert t.current is None
        t.changed("s1")
        assert t.current == "s1"

    def test_initial_value_primes_it(self):
        t = SessionTracker(initial="s1")
        assert t.current == "s1"
        assert t.changed("s2") is True

    def test_reset_makes_the_next_call_record_again(self):
        t = SessionTracker()
        t.changed("s1")
        t.reset()
        assert t.current is None
        assert t.changed("s2") is False  # recorded, not reported


class TestThreadSafety:
    def test_only_one_caller_sees_a_given_rollover(self):
        # Two emit loops consult the same tracker concurrently. `changed` is
        # the trigger for a cache reset, so exactly one of them must be told.
        t = SessionTracker()
        t.changed("s1")

        results = []
        results_lock = threading.Lock()
        start = threading.Event()

        def observe():
            start.wait()
            got = t.changed("s2")
            with results_lock:
                results.append(got)

        threads = [threading.Thread(target=observe) for _ in range(24)]
        for th in threads:
            th.start()
        start.set()
        for th in threads:
            th.join(5)

        assert len(results) == 24
        assert sum(1 for r in results if r) == 1

"""Backing off the music override when only credits come back (stt/music_gate.py)."""

from stt.music_gate import MusicGate


class TestOpen:
    def test_open_at_the_start(self):
        assert MusicGate().allow(0.0)

    def test_a_few_credits_do_not_shut_it(self):
        g = MusicGate(trip_after=3)
        assert not g.record(True, 1.0) and not g.record(True, 2.0)
        assert g.allow(2.0)

    def test_real_text_resets_the_streak(self):
        g = MusicGate(trip_after=3)
        g.record(True, 1.0)
        g.record(True, 2.0)
        g.record(False, 3.0)      # sung lyrics, denied as music but not a hallucination
        assert not g.record(True, 4.0)
        assert g.allow(4.0)


class TestTrip:
    def test_consecutive_credits_shut_it_for_the_cooldown(self):
        g = MusicGate(trip_after=3, cooldown=30.0)
        g.record(True, 1.0)
        g.record(True, 2.0)
        assert g.record(True, 3.0) is True
        assert not g.allow(10.0) and not g.allow(32.9)
        assert g.allow(33.0)

    def test_a_probe_that_returns_a_credit_shuts_it_for_longer(self):
        g = MusicGate(trip_after=3, cooldown=30.0, max_cooldown=240.0)
        for t in (1.0, 2.0, 3.0):
            g.record(True, t)
        assert g.allow(33.0)                 # cooldown over: one probe allowed
        assert g.record(True, 34.0) is True  # probe came back a credit: no new streak needed
        assert not g.allow(34.0 + 59.0) and g.allow(34.0 + 60.0)

    def test_cooldown_stops_doubling_at_the_cap(self):
        g = MusicGate(trip_after=1, cooldown=30.0, max_cooldown=100.0)
        now = 0.0
        for _ in range(6):
            g.record(True, now)
            now = g.closed_until
        assert g.cooldown == 100.0

    def test_a_probe_that_returns_real_text_opens_it_for_good(self):
        g = MusicGate(trip_after=3, cooldown=30.0)
        for t in (1.0, 2.0, 3.0):
            g.record(True, t)
        g.record(False, 40.0)
        assert g.allow(40.0) and g.cooldown == 30.0

    def test_credits_while_shut_do_not_extend_it(self):
        # Sentences already in flight when it shut must not push the reopening back.
        g = MusicGate(trip_after=3, cooldown=30.0)
        for t in (1.0, 2.0, 3.0):
            g.record(True, t)
        until = g.closed_until
        for t in (4.0, 5.0, 6.0, 7.0, 8.0):
            assert g.record(True, t) is False
        assert g.closed_until == until and g.trips == 1

    def test_reset_opens_a_new_session(self):
        g = MusicGate(trip_after=1)
        g.record(True, 1.0)
        g.reset()
        assert g.allow(1.0) and g.trips == 0


def test_configure_takes_new_settings_and_starts_over():
    g = MusicGate(trip_after=1)
    g.record(True, 1.0)
    g.configure(trip_after=5, cooldown=10.0)
    assert g.allow(1.0) and g.trip_after == 5 and g.cooldown == 10.0 and g.trips == 0

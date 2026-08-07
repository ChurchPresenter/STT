"""Unit tests for stt.repeat_filter — thinning a flood of identical log rows."""

from stt.repeat_filter import RepeatSuppressor


class FakeClock:
    """Hand-advanced clock, so the cooldown is tested without waiting."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_the_first_few_are_logged_in_full():
    s = RepeatSuppressor(first_n=3, clock=FakeClock())
    assert [s.decide("k").log for _ in range(3)] == [True, True, True]


def test_and_then_the_repeats_are_dropped():
    s = RepeatSuppressor(first_n=3, clock=FakeClock())
    for _ in range(3):
        s.decide("k")
    assert [s.decide("k").log for _ in range(50)] == [False] * 50


def test_one_row_gets_through_per_cooldown():
    clock = FakeClock()
    s = RepeatSuppressor(first_n=1, cooldown_seconds=600, clock=clock)
    assert s.decide("k").log is True
    for _ in range(60):
        assert s.decide("k").log is False
    clock.advance(600)
    assert s.decide("k").log is True


def test_the_row_that_gets_through_carries_what_was_dropped():
    # Silently thinning would make a hammering client look quiet — the count is
    # the whole point of still writing a row.
    clock = FakeClock()
    s = RepeatSuppressor(first_n=1, cooldown_seconds=600, clock=clock)
    s.decide("k")
    for _ in range(59):
        s.decide("k")
    clock.advance(600)
    verdict = s.decide("k")
    assert verdict.log is True
    assert verdict.suppressed == 59


def test_the_count_resets_after_each_reported_row():
    clock = FakeClock()
    s = RepeatSuppressor(first_n=1, cooldown_seconds=60, clock=clock)
    s.decide("k")
    for _ in range(10):
        s.decide("k")
    clock.advance(60)
    assert s.decide("k").suppressed == 10
    for _ in range(4):
        s.decide("k")
    clock.advance(60)
    assert s.decide("k").suppressed == 4


def test_different_keys_are_independent():
    s = RepeatSuppressor(first_n=1, clock=FakeClock())
    s.decide("a")
    assert s.decide("a").log is False
    assert s.decide("b").log is True


def test_forgetting_a_key_starts_it_over():
    s = RepeatSuppressor(first_n=1, clock=FakeClock())
    s.decide("k")
    assert s.decide("k").log is False
    s.forget("k")
    assert s.decide("k").log is True


def test_the_key_map_is_bounded():
    # Keyed by client address among other things, so it must not be a way for
    # an outsider to grow this process.
    s = RepeatSuppressor(first_n=1, max_keys=10, clock=FakeClock())
    for i in range(200):
        s.decide(f"k{i}")
    assert len(s._state) <= 10


def test_eviction_drops_the_least_recently_seen():
    clock = FakeClock()
    s = RepeatSuppressor(first_n=1, max_keys=2, clock=clock)
    s.decide("old")
    clock.advance(1)
    s.decide("recent")
    clock.advance(1)
    s.decide("recent")          # keeps "recent" fresh
    clock.advance(1)
    s.decide("new")             # evicts "old"
    assert s.decide("recent").log is False   # still remembered

"""Tests for the TTS spoken-segment tracker.

The regression these exist for: the live translation loop drains its per-cycle
budget newest-first during a backlog, so translations land out of order. A
monotonic high-water mark treated the late-arriving lower id as already spoken
and silently dropped it from the audio output.
"""

from stt.tts_queue import SpokenTracker


class TestOutOfOrderArrival:
    """The behaviour a high-water mark got wrong."""

    def test_late_lower_id_is_still_spoken(self):
        t = SpokenTracker()
        # Backlog drain: 12 and 13 are translated before 10 catches up.
        assert t.select_unspoken([12, 13]) == [12, 13]
        t.mark_spoken([12, 13])
        # 10 finally lands — it is still owed.
        assert t.select_unspoken([10, 12, 13]) == [10]

    def test_each_id_is_spoken_exactly_once(self):
        t = SpokenTracker()
        spoken = []
        # Ids arrive in a scrambled order across several cycles.
        for cycle in ([3, 4], [1, 3, 4, 5], [1, 2, 3, 4, 5]):
            picked = t.select_unspoken(cycle)
            spoken.extend(picked)
            t.mark_spoken(picked)
        assert sorted(spoken) == [1, 2, 3, 4, 5]
        assert len(spoken) == len(set(spoken))

    def test_selection_is_ascending_regardless_of_input_order(self):
        t = SpokenTracker()
        assert t.select_unspoken([9, 2, 7, 4]) == [2, 4, 7, 9]

    def test_nothing_available_yields_nothing(self):
        t = SpokenTracker()
        assert t.select_unspoken([]) == []


class TestPrime:
    """Enabling TTS mid-session skips the backlog rather than replaying it."""

    def test_prime_suppresses_existing_ids(self):
        t = SpokenTracker()
        t.prime(100)
        assert t.select_unspoken([98, 99, 100]) == []
        assert t.select_unspoken([100, 101, 102]) == [101, 102]

    def test_prime_does_not_move_backwards(self):
        t = SpokenTracker()
        t.prime(100)
        t.prime(50)
        assert t.select_unspoken([75]) == []

    def test_speaking_never_moves_the_baseline(self):
        # This is the actual bug: voicing a high id must not disqualify lower ones.
        t = SpokenTracker()
        t.mark_spoken([500])
        assert t.select_unspoken([497, 498, 500]) == [497, 498]


class TestReset:
    def test_reset_clears_baseline_and_spoken(self):
        t = SpokenTracker()
        t.prime(100)
        t.mark_spoken([101, 102])
        t.reset()
        assert t.select_unspoken([1, 101, 102]) == [1, 101, 102]


class TestPrune:
    """Bookkeeping stays bounded without ever re-speaking a segment."""

    def test_prune_drops_ids_below_floor(self):
        t = SpokenTracker()
        t.mark_spoken(range(1, 101))
        assert t.size() == 100
        t.prune(90)
        assert t.size() == 11  # 90..100 retained

    def test_pruned_id_is_not_resurrected(self):
        t = SpokenTracker()
        t.mark_spoken([5, 6, 7])
        t.prune(7)
        # 5 and 6 were forgotten from the set but must still count as spoken.
        assert t.select_unspoken([5, 6, 7, 8]) == [8]

    def test_prune_below_baseline_is_a_noop(self):
        t = SpokenTracker()
        t.prime(100)
        t.mark_spoken([101])
        t.prune(50)
        assert t.select_unspoken([101, 102]) == [102]

    def test_prune_at_zero_floor_keeps_everything(self):
        # min_segment_id() returns 0 for an empty cache; that must not
        # fold live ids into the baseline.
        t = SpokenTracker()
        t.mark_spoken([1, 2, 3])
        t.prune(0)
        assert t.select_unspoken([1, 2, 3, 4]) == [4]

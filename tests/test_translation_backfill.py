"""Selection and give-up rules for repairing captions the live window lost.

The scenario: a backlog outlives the max_entries_to_send window, an untranslated
row scrolls out of it, and nothing ever looks at it again — the caption stays
NULL in the session database and is missing from the SRT and HTML exports.
"""

from stt.translation_backfill import (
    DEFAULT_MAX_ATTEMPTS,
    BackfillAttempts,
    select_backfill_ids,
)


class TestSelectBackfillIds:
    def test_picks_the_oldest_orphan_first(self):
        assert select_backfill_ids([40, 12, 33], visible_ids=[], limit=1) == [12]

    def test_ordering_is_ascending(self):
        assert select_backfill_ids([40, 12, 33], visible_ids=[], limit=3) == [12, 33, 40]

    def test_visible_ids_are_left_to_the_main_loop(self):
        # Racing the live loop would spend the remote model twice on one caption.
        assert select_backfill_ids([12, 33, 40], visible_ids=[33, 40], limit=3) == [12]

    def test_limit_is_respected(self):
        assert select_backfill_ids([1, 2, 3, 4], visible_ids=[], limit=2) == [1, 2]

    def test_zero_or_negative_limit_yields_nothing(self):
        assert select_backfill_ids([1, 2], visible_ids=[], limit=0) == []
        assert select_backfill_ids([1, 2], visible_ids=[], limit=-1) == []

    def test_no_orphans_yields_nothing(self):
        assert select_backfill_ids([], visible_ids=[5, 6], limit=1) == []

    def test_all_orphans_visible_yields_nothing(self):
        assert select_backfill_ids([5, 6], visible_ids=[5, 6], limit=1) == []


class TestBackfillAttempts:
    """Bounded retries: a caption that can never translate must not cost forever."""

    def test_gives_up_after_max_attempts(self):
        a = BackfillAttempts(max_attempts=3)
        for _ in range(3):
            assert not a.exhausted(7)
            a.record(7)
        assert a.exhausted(7)

    def test_eligible_filters_out_exhausted_ids(self):
        a = BackfillAttempts(max_attempts=1)
        a.record(7)
        assert a.eligible([7, 8, 9]) == [8, 9]

    def test_success_clears_the_record(self):
        a = BackfillAttempts(max_attempts=1)
        a.record(7)
        assert a.exhausted(7)
        a.succeeded(7)
        assert not a.exhausted(7)
        assert a.size() == 0

    def test_reset_clears_every_id(self):
        a = BackfillAttempts(max_attempts=1)
        a.record(1)
        a.record(2)
        a.reset()
        assert a.eligible([1, 2]) == [1, 2]
        assert a.size() == 0

    def test_untried_id_is_never_exhausted(self):
        assert not BackfillAttempts().exhausted(999)

    def test_default_max_attempts_is_applied(self):
        a = BackfillAttempts()
        for _ in range(DEFAULT_MAX_ATTEMPTS):
            a.record(1)
        assert a.exhausted(1)


class TestSelectionWithAttempts:
    """The two pieces as the caller composes them."""

    def test_exhausted_orphan_yields_to_the_next_one(self):
        a = BackfillAttempts(max_attempts=2)
        orphans = [12, 33, 40]
        # 12 fails twice and is given up on.
        a.record(12)
        a.record(12)
        picked = select_backfill_ids(a.eligible(orphans), visible_ids=[], limit=1)
        assert picked == [33]

    def test_all_exhausted_stops_the_work_entirely(self):
        a = BackfillAttempts(max_attempts=1)
        for sid in (12, 33):
            a.record(sid)
        assert select_backfill_ids(a.eligible([12, 33]), visible_ids=[], limit=1) == []

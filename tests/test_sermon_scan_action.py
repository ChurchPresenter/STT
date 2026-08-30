"""One sermon, one automatic summary — and a boundary change reported rather than re-run.

The scan used to key everything on the transcript fingerprint, so any boundary move past a
single caption made a new row and queued a fresh run. A real service (2026-08-30) ended up
with two full summaries of one sermon whose start had shifted by 30 seconds, from three
queued runs in ten minutes — and those runs are what held the shared model away from the
live captions. The numbers in ``TestTheServiceThatPromptedThis`` are that service's.

Every case here is the rule as the operator meets it: the first summary happens by itself,
and every later one is something they asked for.
"""

from stt.sermon_summary import (
    ACTION_NOTE,
    ACTION_QUEUE,
    ACTION_SKIP,
    covering_summary,
    range_change,
    scan_action,
)

MIN = 60_000
BASE = 1_700_000_000_000
FP_NEW = "ffff"


def stored(status="done", *, label="Sermon 1", start_ms=BASE, end_ms=BASE + 30 * MIN,
           fingerprint="aaaa", **kw):
    row = {"label": label, "start_ms": start_ms, "end_ms": end_ms, "status": status,
           "fingerprint": fingerprint}
    row.update(kw)
    return row


class TestCoveringSummary:
    def test_nothing_stored_covers_nothing(self):
        assert covering_summary([], start_ms=BASE, end_ms=BASE + 30 * MIN) is None

    def test_an_overlapping_finished_summary_covers_the_range(self):
        assert covering_summary([stored()], start_ms=BASE - MIN,
                                end_ms=BASE + 30 * MIN) is not None

    def test_a_range_that_only_touches_does_not_overlap(self):
        # The next block starting exactly where this one ended is a different phase.
        after = stored(start_ms=BASE + 30 * MIN, end_ms=BASE + 60 * MIN)
        assert covering_summary([after], start_ms=BASE, end_ms=BASE + 30 * MIN) is None

    def test_a_renumbered_label_still_covers(self):
        # Merging blocks renumbers them, and merging is when boundaries move.
        assert covering_summary([stored(label="Sermon 2")], start_ms=BASE,
                                end_ms=BASE + 30 * MIN) is not None

    def test_a_different_phase_does_not_cover(self):
        # A Communion block that grew over the sermon must not claim its summary.
        assert covering_summary([stored(label="Communion")], start_ms=BASE,
                                end_ms=BASE + 30 * MIN) is None

    def test_only_a_finished_summary_covers(self):
        for status in ("pending", "running", "error"):
            assert covering_summary([stored(status)], start_ms=BASE,
                                    end_ms=BASE + 30 * MIN) is None, status

    def test_a_row_with_no_end_falls_back_to_its_start(self):
        partial = stored(end_ms=0)
        assert covering_summary([partial], start_ms=BASE - MIN,
                                end_ms=BASE + 30 * MIN) is not None
        assert covering_summary([partial], start_ms=BASE + 60 * MIN,
                                end_ms=BASE + 90 * MIN) is None


class TestScanAction:
    def test_a_sermon_with_no_summary_is_summarised(self):
        assert scan_action([], start_ms=BASE, end_ms=BASE + 30 * MIN,
                           fingerprint=FP_NEW) == ACTION_QUEUE

    def test_the_same_text_again_is_left_alone(self):
        # Already done, running or queued under this exact fingerprint.
        assert scan_action([stored(fingerprint=FP_NEW)], start_ms=BASE,
                           end_ms=BASE + 30 * MIN, fingerprint=FP_NEW) == ACTION_SKIP

    def test_a_moved_boundary_is_noted_not_re_run(self):
        assert scan_action([stored()], start_ms=BASE - 30_000, end_ms=BASE + 30 * MIN,
                           fingerprint=FP_NEW) == ACTION_NOTE

    def test_a_merge_that_renumbers_is_noted_not_re_run(self):
        assert scan_action([stored(label="Sermon 1")], start_ms=BASE,
                           end_ms=BASE + 45 * MIN, fingerprint=FP_NEW) == ACTION_NOTE

    def test_both_halves_of_a_split_are_noted(self):
        rows = [stored()]
        first = scan_action(rows, start_ms=BASE, end_ms=BASE + 14 * MIN, fingerprint="b1")
        second = scan_action(rows, start_ms=BASE + 16 * MIN, end_ms=BASE + 30 * MIN,
                             fingerprint="b2")
        assert (first, second) == (ACTION_NOTE, ACTION_NOTE)

    def test_the_operator_asking_always_wins(self):
        assert scan_action([stored()], start_ms=BASE - 30_000, end_ms=BASE + 30 * MIN,
                           fingerprint=FP_NEW, manual=True) == ACTION_QUEUE

    def test_a_failed_summary_is_retried_automatically(self):
        # Usually a model that was unreachable then and is not now; nothing to preserve.
        assert scan_action([stored("error")], start_ms=BASE, end_ms=BASE + 30 * MIN,
                           fingerprint=FP_NEW) == ACTION_QUEUE

    def test_an_unfinished_summary_does_not_block_the_catch_up(self):
        for status in ("pending", "running"):
            assert scan_action([stored(status)], start_ms=BASE, end_ms=BASE + 30 * MIN,
                               fingerprint=FP_NEW) == ACTION_QUEUE, status

    def test_a_second_sermon_is_still_summarised(self):
        # The rule must not swallow the rest of the service.
        assert scan_action([stored()], start_ms=BASE + 60 * MIN, end_ms=BASE + 90 * MIN,
                           fingerprint=FP_NEW) == ACTION_QUEUE


class TestRangeChange:
    def test_an_unchanged_range_reports_nothing(self):
        assert range_change(stored(), start_ms=BASE, end_ms=BASE + 30 * MIN) is None

    def test_a_moved_start_is_reported_with_its_delta(self):
        change = range_change(stored(), start_ms=BASE - 30_000, end_ms=BASE + 30 * MIN)
        assert change["start_delta_ms"] == -30_000
        assert change["end_delta_ms"] == 0
        assert change["was_start_ms"] == BASE

    def test_a_moved_end_is_reported(self):
        change = range_change(stored(), start_ms=BASE, end_ms=BASE + 37 * MIN)
        assert change["end_delta_ms"] == 7 * MIN

    def test_a_relabel_alone_is_still_a_change(self):
        # How a merge shows up when it renumbers without moving an edge.
        change = range_change(stored(), start_ms=BASE, end_ms=BASE + 30 * MIN,
                              label="Sermon 2")
        assert change is not None
        assert (change["was_label"], change["label"]) == ("Sermon 1", "Sermon 2")


class TestTheServiceThatPromptedThis:
    """The 2026-08-30 service, to the second.

    Sermon 1 was summarised over 10:08:37-10:37:37, the detector then moved its start back
    30 seconds, and the whole sermon was written a second time. Under this rule the second
    run never happens.
    """

    SUMMARISED_START = 1_788_098_917_493   # 10:08:37
    SUMMARISED_END = 1_788_100_657_493     # 10:37:37
    MOVED_START = 1_788_098_887_660        # 10:08:07

    def rows(self):
        return [stored(label="Sermon 1", start_ms=self.SUMMARISED_START,
                       end_ms=self.SUMMARISED_END, fingerprint="01f8fdcd")]

    def test_the_second_run_is_a_note(self):
        assert scan_action(self.rows(), start_ms=self.MOVED_START,
                           end_ms=self.SUMMARISED_END,
                           fingerprint="c7b05c81") == ACTION_NOTE

    def test_the_note_says_the_start_moved_thirty_seconds(self):
        change = range_change(self.rows()[0], start_ms=self.MOVED_START,
                              end_ms=self.SUMMARISED_END)
        assert change["start_delta_ms"] == -29_833
        assert change["end_delta_ms"] == 0

    def test_the_operator_can_still_ask_for_it(self):
        assert scan_action(self.rows(), start_ms=self.MOVED_START,
                           end_ms=self.SUMMARISED_END, fingerprint="c7b05c81",
                           manual=True) == ACTION_QUEUE

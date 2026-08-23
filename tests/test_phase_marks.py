"""stt.phase_marks: the operator's live marks, combined with what the detector found.

The requirement these pin down is a division of labour, not a preference. An operator in
the room knows the exact moment a sermon starts and can name it before it is over; an
operator watching a service will also mark the start and then never mark the end. So a mark
decides the edge and the name, and the detector decides everything nobody marked —
including the end of a marked phase, which is what makes spacing out cost nothing.

Times are constructed, and so is every label (see AGENTS.md).
"""

import pytest

from stt.phase_marks import END_LABEL, describe, is_mark, marks, resolve

MIN = 60_000
BASE = 1_700_000_000_000       # a service start, in epoch ms


def mark(at_min, label="Sermon 1", kind="S", row_id=1):
    """A corrections row as the mark route writes it: a start, and no end."""
    return {"id": row_id, "block_index": None, "start_ms": BASE + at_min * MIN,
            "end_ms": None, "kind": kind, "label": label, "note": "live mark"}


def block(from_min, to_min, kind="S", ongoing=False, index=0):
    return {"index": index, "kind": kind, "start_ms": BASE + from_min * MIN,
            "end_ms": BASE + to_min * MIN, "ongoing": ongoing}


class TestWhatCountsAsAMark:
    def test_a_start_without_an_end_is_a_mark(self):
        assert is_mark(mark(10)) is True

    def test_a_drawn_span_is_not(self):
        """The grouping control writes both edges; that is a claim, not a moment."""
        assert is_mark({"block_index": None, "start_ms": BASE, "end_ms": BASE + MIN}) is False

    def test_a_relabel_is_not(self):
        assert is_mark({"block_index": 2, "start_ms": None, "end_ms": None}) is False

    def test_pressing_again_at_the_same_moment_replaces_the_first(self):
        """Picking the wrong label and pressing again is how a mark is corrected."""
        rows = [mark(10, "Songs 1", row_id=1), mark(10, "Sermon 1", row_id=2)]
        assert [m["label"] for m in marks(rows)] == ["Sermon 1"]

    def test_marks_come_back_in_time_order_whatever_order_they_were_written(self):
        rows = [mark(30, "Closing", row_id=2), mark(10, "Sermon 1", row_id=1)]
        assert [m["label"] for m in marks(rows)] == ["Sermon 1", "Closing"]


class TestTheDetectorEndsWhatTheOperatorStarted:
    """The half that matters: a start with no end still produces a finished phase."""

    def test_the_phase_ends_where_the_detectors_run_ends(self):
        blocks = [block(8, 50, "S", index=0), block(50, 60, "M", index=1)]
        spans = resolve([mark(10)], blocks, now_ms=BASE + 70 * MIN)
        assert len(spans) == 1
        assert spans[0]["start_ms"] == BASE + 10 * MIN      # the operator's moment, exactly
        assert spans[0]["end_ms"] == BASE + 50 * MIN        # the detector's
        assert spans[0]["open"] is False

    def test_consecutive_blocks_of_one_kind_are_one_run(self):
        """A cough splits a sermon into two speaking blocks; it does not end the sermon."""
        blocks = [block(8, 30, "S", index=0), block(30, 55, "S", index=1),
                  block(55, 60, "M", index=2)]
        spans = resolve([mark(10)], blocks, now_ms=BASE + 70 * MIN)
        assert spans[0]["end_ms"] == BASE + 55 * MIN

    def test_an_ongoing_run_leaves_the_phase_open_at_now(self):
        blocks = [block(8, 25, "S", ongoing=True, index=0)]
        spans = resolve([mark(10)], blocks, now_ms=BASE + 25 * MIN)
        assert spans[0]["end_ms"] == BASE + 25 * MIN
        assert spans[0]["open"] is True

    def test_a_mark_ahead_of_the_detector_is_normal_and_not_dropped(self):
        """Live, a block does not exist until dwell has passed — the mark comes first."""
        blocks = [block(0, 9, "M", index=0)]
        spans = resolve([mark(10)], blocks, now_ms=BASE + 12 * MIN)
        assert spans[0]["start_ms"] == BASE + 10 * MIN
        assert spans[0]["open"] is True

    def test_no_blocks_at_all_still_gives_a_running_phase(self):
        spans = resolve([mark(10)], [], now_ms=BASE + 20 * MIN)
        assert spans[0]["end_ms"] == BASE + 20 * MIN
        assert spans[0]["open"] is True


class TestOneMarkClosesTheOneBefore:
    def test_the_next_mark_ends_the_previous_phase(self):
        blocks = [block(0, 60, "S", ongoing=True, index=0)]
        spans = resolve([mark(10, "Sermon 1", row_id=1), mark(40, "Closing", row_id=2)],
                        blocks, now_ms=BASE + 50 * MIN)
        assert [s["label"] for s in spans] == ["Sermon 1", "Closing"]
        assert spans[0]["end_ms"] == BASE + 40 * MIN
        assert spans[0]["open"] is False

    def test_an_end_press_closes_without_naming_anything_new(self):
        blocks = [block(0, 60, "S", ongoing=True, index=0)]
        spans = resolve([mark(10, "Sermon 1", row_id=1),
                         mark(35, END_LABEL, row_id=2)], blocks, now_ms=BASE + 50 * MIN)
        assert [s["label"] for s in spans] == ["Sermon 1"]
        assert spans[0]["end_ms"] == BASE + 35 * MIN

    def test_the_earlier_of_the_detector_and_the_next_mark_wins(self):
        """The operator marked the next phase late; the audio had already changed."""
        blocks = [block(0, 30, "S", index=0), block(30, 60, "M", index=1)]
        spans = resolve([mark(10, "Sermon 1", row_id=1), mark(45, "Closing", row_id=2)],
                        blocks, now_ms=BASE + 60 * MIN)
        assert spans[0]["end_ms"] == BASE + 30 * MIN

    def test_a_press_a_second_later_is_a_correction_not_a_phase(self):
        blocks = [block(0, 60, "S", ongoing=True, index=0)]
        rows = [mark(10, "Songs 1", row_id=1),
                {"id": 2, "block_index": None, "start_ms": BASE + 10 * MIN + 2000,
                 "end_ms": None, "kind": "S", "label": "Sermon 1"}]
        spans = resolve(rows, blocks, now_ms=BASE + 30 * MIN)
        assert [s["label"] for s in spans] == ["Sermon 1"]


class TestNothingMarkedChangesNothing:
    def test_no_marks_produce_no_spans(self):
        blocks = [block(0, 40, "S", index=0)]
        assert resolve([], blocks, now_ms=BASE + 40 * MIN) == []

    def test_spans_and_relabels_are_left_to_the_code_that_owns_them(self):
        corrections = [{"id": 1, "block_index": 2, "start_ms": None, "end_ms": None,
                        "label": "Sermon 1"},
                       {"id": 2, "block_index": None, "start_ms": BASE,
                        "end_ms": BASE + 20 * MIN, "label": "Songs 1"}]
        assert resolve(corrections, [], now_ms=BASE + 30 * MIN) == []


class TestTheOperatorsOwnRow:
    def test_it_says_what_is_marked_and_how_long_ago(self):
        assert describe([mark(10)], now_ms=BASE + 13 * MIN) == "Sermon 1 marked 3 min ago"

    def test_a_fresh_press_reads_as_just_now(self):
        assert describe([mark(10)], now_ms=BASE + 10 * MIN) == "Sermon 1 marked just now"

    def test_an_end_says_so(self):
        assert describe([mark(10, END_LABEL)], now_ms=BASE + 12 * MIN) == "ended 2 min ago"

    def test_nothing_marked_says_nothing(self):
        assert describe([], now_ms=BASE) == ""


@pytest.mark.parametrize("kind", ["S", "M", "_"])
def test_the_kind_pressed_is_the_kind_stored(kind):
    blocks = [block(0, 40, "S", ongoing=True, index=0)]
    spans = resolve([mark(5, "Other", kind=kind)], blocks, now_ms=BASE + 20 * MIN)
    assert spans[0]["kind"] == kind

import re

from stt import diagnostics
from stt.caption_order import DEFAULT_HOLD_SECONDS, EmissionOrder

HOLD = 10.0


def _seg(seg_id, text=None):
    return {"id": seg_id, "translated_text": text or f"line {seg_id}"}


def _ids(segs):
    return [s["id"] for s in segs]


def _order(state, ready_ids, pending, now, hold=HOLD):
    return _ids(state.order([_seg(i) for i in ready_ids], pending, now, hold))


def _fresh():
    state = EmissionOrder()
    state.begin_cycle("s1")
    return state


def test_a_caption_finished_early_waits_for_the_one_before_it():
    # The 2026-10-25 shape: 11 is translated while 10 is still with the paired server.
    state = _fresh()
    assert _order(state, [9], [], 0.0) == [9]
    assert _order(state, [9, 11], [10], 1.0) == [9]
    assert _order(state, [9, 10, 11], [], 2.5) == [9, 10, 11]


def test_the_stream_only_ever_grows_at_its_end():
    # Translations landing out of order, one far too late: every emit must extend the last.
    state = _fresh()
    landed = {1: 0.0, 3: 1.0, 2: 2.0, 5: 2.5, 4: 4.0, 6: 4.5, 7: 30.0, 8: 6.0}
    previous = []
    for tick in range(0, 80):
        now = tick * 0.5
        ready = sorted(i for i, t in landed.items() if t <= now)
        pending = sorted(i for i, t in landed.items() if t > now)
        shown = _order(state, ready, pending, now)
        assert shown[:len(previous)] == previous
        previous = shown
    assert previous == [1, 2, 3, 4, 5, 6, 8]


def test_a_caption_stuck_past_the_hold_is_never_shown_live():
    state = _fresh()
    assert _order(state, [1, 3], [2], 0.0) == [1]
    assert _order(state, [1, 3], [2], HOLD - 0.1) == [1]
    assert _order(state, [1, 3], [2], HOLD) == [1, 3]
    assert _order(state, [1, 3, 4], [2], HOLD + 1) == [1, 3, 4]
    # Its translation lands, but the room has moved on: inserting it now is the jump
    assert _order(state, [1, 2, 3, 4], [], HOLD + 2) == [1, 3, 4]


def test_a_failed_caption_is_skipped_rather_than_shown_in_the_source_language():
    state = _fresh()
    state.skip(2, "translation failed (none)", 0.0)
    # It no longer holds anything back
    assert _order(state, [1, 3], [2], 0.5) == [1, 3]
    # A retry that succeeds later never puts it on screen
    assert _order(state, [1, 2, 3], [], 25.0) == [1, 3]


def test_skipping_a_caption_already_on_screen_leaves_it_there():
    state = _fresh()
    assert _order(state, [1], [], 0.0) == [1]
    state.skip(1, "translation failed (none)", 1.0)
    assert _order(state, [1], [], 1.5) == [1]
    assert state.drain_events() == []


def test_a_caption_already_sent_is_never_withdrawn():
    state = _fresh()
    assert _order(state, [1, 2], [3], 0.0) == [1, 2]
    assert _order(state, [1, 2, 4], [3], HOLD) == [1, 2, 4]
    assert _order(state, [1, 2, 4], [3, 5], HOLD + 1) == [1, 2, 4]


def test_a_gap_that_opens_below_a_sent_caption_does_not_pull_it_back():
    state = _fresh()
    assert _order(state, [1, 2, 4], [], 0.0) == [1, 2, 4]
    assert _order(state, [1, 2, 4, 5], [3], 1.0) == [1, 2, 4]


def test_the_newest_caption_pending_holds_nothing_back():
    state = _fresh()
    assert _order(state, [1, 2], [3], 0.0) == [1, 2]


def test_zero_hold_turns_ordering_off():
    state = _fresh()
    assert _order(state, [1, 3], [2], 0.0, hold=0) == [1, 3]


def test_every_decision_is_traceable():
    state = _fresh()
    _order(state, [1, 3], [2], 0.0)
    _order(state, [1, 2, 3], [], 2.0)
    _order(state, [1, 2, 3, 5], [4], 3.0)
    _order(state, [1, 2, 3, 5], [4], 3.0 + HOLD)
    _order(state, [1, 2, 3, 4, 5], [], 3.0 + HOLD + 4)
    state.skip(6, "translation failed (none)", 20.0)
    assert state.drain_events() == [
        "held 3 behind 2",
        "sent 3 after holding 2.0s",
        "held 5 behind 4",
        "gave up on 4 after 10.0s untranslated; later captions released, it will not be shown live",
        "sent 5 after holding 10.0s",
        "kept 4 off screen: translated 4.0s after it was given up",
        "skipped 6: translation failed (none)",
    ]
    assert state.drain_events() == []


def test_events_carry_no_caption_text():
    # The tag is in the diagnostic report's allowlist only because of this.
    state = _fresh()
    secret = "words somebody said"
    state.order([_seg(1, secret), _seg(3, secret)], [2], 0.0, HOLD)
    state.order([_seg(1, secret), _seg(3, secret)], [2], HOLD, HOLD)
    state.order([_seg(i, secret) for i in (1, 2, 3)], [], HOLD + 1, HOLD)
    state.skip(4, "translation failed (none)", HOLD + 2)
    events = state.drain_events()
    assert events
    for event in events:
        assert secret not in event
        assert re.fullmatch(r"[a-z ]+\d+[a-z0-9 .;,():]*", event), event
    assert "CAPTION-ORDER" in diagnostics.LOG_TAGS


def test_a_new_session_starts_clean():
    state = _fresh()
    state.skip(5, "translation failed (none)", 0.0)
    _order(state, [1, 3], [2], 0.0)
    state.begin_cycle("s2")
    # Ids restart with a new database; nothing carries over
    assert _order(state, [1, 3], [2], 0.0) == [1]
    assert _order(state, [5], [], 1.0) == [5]


def test_the_same_session_keeps_its_state():
    state = _fresh()
    _order(state, [1, 3], [2], 0.0)
    state.begin_cycle("s1")
    assert _order(state, [1, 3], [2], HOLD) == [1, 3]


def test_state_for_captions_that_left_the_window_is_dropped():
    state = _fresh()
    state.skip(1, "translation failed (none)", 0.0)
    _order(state, [50, 51], [], 1.0)
    assert state._given_up == {}


def test_the_default_hold_covers_sundays_slowest_translations():
    # 10s left 6 translated captions off the live display on 2026-10-25; 15s left none.
    assert DEFAULT_HOLD_SECONDS >= 15

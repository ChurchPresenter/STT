"""The end-of-session gate that holds a session back until its summaries exist.

A finished session is copied to the NAS and deleted locally about ten seconds after export,
so a summary still being written is lost rather than late. The gate makes the file mover wait
for it — and every test here is really about the same worry: a flag that holds up delivery is
a promise something will clear it, and a crashed summariser makes a liar of it.

Hence the deadline. The worst case has to be a late delivery, never a machine that will not
start again.
"""

import time

import pytest

from conftest import extract_definitions
from stt.sermon_summary import wait_while


def make_ns(state=None, cfg=None, sleep=None, clock=None):
    state = {} if state is None else state
    return extract_definitions(
        "speech_to_text.py",
        ["sermon_finalising", "_sermon_set_finalising", "_sermon_finalise_deadline",
         "_sermon_wait_for_finalise"],
        extra_globals={
            "time": clock or time,
            "sleep": sleep or (clock.sleep if clock else (lambda s: None)),
            "transcription_state": state,
            "_sermon_summary_config": lambda: cfg or {},
            "_sermon_wait_while": wait_while,
            "coerce_int": __import__("stt.coercion", fromlist=["x"]).coerce_int,
        }), state


class TestDeadline:
    def test_it_defaults_to_half_an_hour(self):
        # Chosen against a measured ~6-7 minutes per sermon, not guessed: two sermons on a
        # busy peer must fit inside it.
        ns, _ = make_ns()
        assert ns["_sermon_finalise_deadline"]() == 1800

    def test_it_is_configurable(self):
        ns, _ = make_ns(cfg={"finalise_max_seconds": 600})
        assert ns["_sermon_finalise_deadline"]() == 600

    @pytest.mark.parametrize("raw,expected", [(5, 30), (99999, 14400), ("abc", 1800)])
    def test_it_is_clamped_to_something_survivable(self, raw, expected):
        ns, _ = make_ns(cfg={"finalise_max_seconds": raw})
        assert ns["_sermon_finalise_deadline"]() == expected


class TestFlag:
    def test_nothing_set_is_not_finalising(self):
        ns, _ = make_ns()
        assert ns["sermon_finalising"] is not None
        assert ns["sermon_finalising"]() is False

    def test_setting_it_stamps_the_time(self):
        ns, state = make_ns()
        ns["_sermon_set_finalising"](True)
        assert state["finalising"] is True
        assert state["finalising_since"] > 0
        assert ns["sermon_finalising"]() is True

    def test_clearing_it_clears_the_stamp(self):
        ns, state = make_ns()
        ns["_sermon_set_finalising"](True)
        ns["_sermon_set_finalising"](False)
        assert state["finalising"] is False and state["finalising_since"] == 0
        assert ns["sermon_finalising"]() is False

    def test_a_flag_past_its_deadline_reads_as_clear(self):
        """The one that matters: a crashed summariser must not wedge the machine.

        Nothing clears the flag if the process holding it died, so it has to expire on its
        own — otherwise the file mover waits forever and no service can be started again.
        """
        ns, state = make_ns(cfg={"finalise_max_seconds": 60})
        state["finalising"] = True
        state["finalising_since"] = time.time() - 3600
        assert ns["sermon_finalising"]() is False

    def test_a_flag_inside_its_deadline_still_holds(self):
        ns, state = make_ns(cfg={"finalise_max_seconds": 600})
        state["finalising"] = True
        state["finalising_since"] = time.time() - 30
        assert ns["sermon_finalising"]() is True

    def test_a_flag_with_no_stamp_at_all_still_holds(self):
        # An older build, or a write that half-landed: honour the flag rather than
        # silently ignoring it, since delivering early is the destructive direction.
        ns, state = make_ns()
        state["finalising"] = True
        assert ns["sermon_finalising"]() is True

    def test_an_unusable_stamp_does_not_raise(self):
        ns, state = make_ns()
        state["finalising"] = True
        state["finalising_since"] = "not a number"
        assert ns["sermon_finalising"]() is False

    def test_a_broken_shared_state_reads_as_clear(self):
        """The proxy dies during a restart; the stop path must not raise into it.

        Reading as clear is the safe answer: it releases delivery, and the alternative is an
        exception in the middle of tearing a session down.
        """
        class Dead:
            def get(self, *a, **k):
                raise EOFError("manager gone")

        ns, _ = make_ns(state=Dead())
        assert ns["sermon_finalising"]() is False

    def test_setting_it_on_a_broken_state_does_not_raise(self):
        class Dead(dict):
            def __setitem__(self, *a):
                raise EOFError("manager gone")

        ns, _ = make_ns(state=Dead())
        ns["_sermon_set_finalising"](True)  # must not propagate


class FakeClock:
    """A clock that only moves when something sleeps on it."""

    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


class TestWaitWhile:
    """The one wait shared by the database close, the sidecar retirement and delivery.

    Each of those must not run while the end-of-session summariser still holds the session
    database open — checkpointing it underneath a live holder is what left a server unable
    to open its own archive until it was restarted.
    """

    def test_it_does_not_wait_when_nothing_is_finalising(self):
        clock = FakeClock()
        assert wait_while(lambda: False, 1800, clock.sleep, clock.now) == 0
        assert clock.slept == []

    def test_it_returns_as_soon_as_the_flag_clears(self):
        clock = FakeClock()
        calls = iter([True, True, False, False])
        waited = wait_while(lambda: next(calls), 1800, clock.sleep, clock.now, step_s=5)
        assert waited == 10 and clock.slept == [5, 5]

    def test_a_summariser_that_never_finishes_is_bounded_by_the_deadline(self):
        clock = FakeClock()
        waited = wait_while(lambda: True, 30, clock.sleep, clock.now, step_s=5)
        assert waited == 30, "a stuck summariser delays a stop; it must never prevent one"
        assert sum(clock.slept) == 30

    def test_a_deadline_of_zero_waits_for_nothing(self):
        clock = FakeClock()
        assert wait_while(lambda: True, 0, clock.sleep, clock.now) == 0
        assert clock.slept == []

    def test_the_step_never_overshoots_far_past_the_deadline(self):
        clock = FakeClock()
        waited = wait_while(lambda: True, 12, clock.sleep, clock.now, step_s=5)
        assert 12 <= waited < 12 + 5


class FakeTime:
    """A stand-in for the time module whose clock only moves when something sleeps."""

    def __init__(self, start=1000.0):
        self.t = start
        self.slept = []

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


class TestTheStopWaitsForIt:
    """What the stop path does with the flag, not just what the flag says.

    The database close checkpoints the session, and a checkpoint under a process that still
    holds the database open is what left a server unable to read its own archive. So the
    close has to wait, and — because the flag is a promise a background task might break —
    the wait has to end.
    """

    def test_it_returns_at_once_when_nothing_is_owed(self):
        ns, _ = make_ns()
        assert ns["_sermon_wait_for_finalise"]("[DB-CLEANUP]") == 0

    def test_it_waits_while_a_summary_is_still_being_written(self, capsys):
        clock = FakeTime()
        state = {}
        ns, _ = make_ns(state=state, clock=clock)
        ns["_sermon_set_finalising"](True)

        # The summariser clears the flag from its finally, part-way through the wait.
        def sleeping(seconds):
            clock.sleep(seconds)
            if len(clock.slept) == 2:
                ns["_sermon_set_finalising"](False)

        ns, _ = make_ns(state=state, clock=clock, sleep=sleeping)
        waited = ns["_sermon_wait_for_finalise"]("[DB-CLEANUP]")
        assert waited == sum(clock.slept) and len(clock.slept) == 2
        assert "[DB-CLEANUP] Waiting for end-of-session summaries" in capsys.readouterr().out

    def test_a_summariser_that_never_clears_the_flag_still_lets_the_stop_finish(self):
        # The deadline is what makes this safe to put in front of the database close.
        clock = FakeTime()
        ns, _ = make_ns(cfg={"finalise_max_seconds": 30}, clock=clock)
        ns["_sermon_set_finalising"](True)
        assert ns["_sermon_wait_for_finalise"]("[DB-CLEANUP]") <= 30
        assert clock.slept, "it did wait; it simply did not wait forever"

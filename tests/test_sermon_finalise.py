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


def make_ns(state=None, cfg=None):
    state = {} if state is None else state
    return extract_definitions(
        "speech_to_text.py",
        ["sermon_finalising", "_sermon_set_finalising", "_sermon_finalise_deadline"],
        extra_globals={
            "time": time,
            "transcription_state": state,
            "_sermon_summary_config": lambda: cfg or {},
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

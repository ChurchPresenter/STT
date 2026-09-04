"""The status poll's reaper: the route glue around stt.start_watch.

speech_to_text.py cannot be imported, so the two functions are extracted and run
against a stub namespace — see tests/conftest.py:extract_definitions. The
decision itself is tested in test_start_watch.py; what matters here is the
plumbing: that the correction is *written* to the shared state rather than only
decorating one response, and that a dying Manager does not take the poll with it.
"""

import threading
import time

import pytest

from conftest import extract_definitions
from stt import start_watch as _start_watch


class FakeProcess:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class DeadProxy(dict):
    """A Manager dict whose owning process has gone away mid-shutdown.

    Reads and writes both fail on a real one; _ts_snapshot already absorbs that
    and answers "restarting", which is what the stub below reproduces.
    """

    def __setitem__(self, key, value):
        raise BrokenPipeError("Manager is gone")


def build(state, process, stall=600, config=None):
    shutting_down = threading.Event()
    ns = extract_definitions(
        "speech_to_text.py", ["_start_stall_seconds", "_reap_stuck_start"],
        {
            "_start_watch": _start_watch,
            "transcription_state": state,
            "transcription_process": process,
            "_transcription_state_lock": threading.Lock(),
            "_TS_PROXY_ERRORS": (BrokenPipeError, EOFError, ConnectionError,
                                 FileNotFoundError, AttributeError, TypeError),
            "_server_shutting_down": shutting_down,
            "_ts_snapshot": (
                (lambda: {"running": False, "status": "restarting",
                          "message": "Server is restarting"})
                if isinstance(state, DeadProxy) else (lambda: dict(state))
            ),
            "config": config if config is not None else {
                "transcription": {"start_stall_seconds": stall}},
        })
    ns["_shutting_down"] = shutting_down
    return ns


def starting(stage="3/5 loading the faster-whisper model", age=0.0):
    return {
        "running": False,
        "status": "starting",
        "error": None,
        "message": "Initializing...",
        _start_watch.STAGE_KEY: stage,
        _start_watch.STAGE_AT_KEY: time.time() - age,
    }


class TestStallSeconds:
    def test_default_when_unconfigured(self):
        assert build({}, None, config={})["_start_stall_seconds"]() == 600.0

    def test_reads_the_configured_value(self):
        ns = build({}, None, config={"transcription": {"start_stall_seconds": 90}})
        assert ns["_start_stall_seconds"]() == 90.0

    @pytest.mark.parametrize("bad", ["soon", None, {}])
    def test_a_nonsense_value_falls_back_rather_than_raising(self, bad):
        """A typo in config.json must not 500 the poll every viewer makes."""
        ns = build({}, None, config={"transcription": {"start_stall_seconds": bad}})
        assert ns["_start_stall_seconds"]() == 600.0


class TestReaper:
    def test_a_healthy_start_is_returned_untouched(self):
        state = starting()
        ns = build(state, FakeProcess(alive=True))
        assert ns["_reap_stuck_start"](dict(state))["status"] == "starting"
        assert state["status"] == "starting", "nothing should have been written"

    def test_a_dead_worker_is_reported(self):
        state = starting()
        ns = build(state, FakeProcess(alive=False))
        out = ns["_reap_stuck_start"](dict(state))
        assert out["status"] == "error"
        assert out["running"] is False

    def test_the_correction_is_written_to_the_shared_state(self):
        """A reaper that only decorated the response would leave Stop and Start
        looking at a phantom 'starting' and refusing on it."""
        state = starting()
        ns = build(state, FakeProcess(alive=False))
        ns["_reap_stuck_start"](dict(state))
        assert state["status"] == "error"
        assert state["running"] is False
        assert state[_start_watch.STAGE_KEY] == ""
        assert state[_start_watch.STAGE_AT_KEY] is None

    def test_a_missing_process_counts_as_dead(self):
        """The worker is spawned lazily; None means nothing is coming."""
        state = starting()
        ns = build(state, None)
        assert ns["_reap_stuck_start"](dict(state))["status"] == "error"

    def test_a_stalled_stage_is_reported_with_its_stage(self):
        state = starting(age=700)
        ns = build(state, FakeProcess(alive=True), stall=600)
        out = ns["_reap_stuck_start"](dict(state))
        assert out["status"] == "error"
        assert "3/5 loading the faster-whisper model" in out["error"]

    def test_a_slow_but_moving_start_is_left_alone(self):
        state = starting(age=500)
        ns = build(state, FakeProcess(alive=True), stall=600)
        assert ns["_reap_stuck_start"](dict(state))["status"] == "starting"

    def test_a_running_transcription_is_never_touched(self):
        """Even with the worker handle gone — a live run is not this code's
        business, and killing one mid-service would be far worse than the bug."""
        state = {"running": True, "status": "running"}
        ns = build(state, None)
        assert ns["_reap_stuck_start"](dict(state))["status"] == "running"
        assert state["status"] == "running"

    def test_a_torn_down_manager_does_not_raise(self):
        """The poll runs during shutdown and restart too; a dead proxy there is
        expected, not an error to surface."""
        state = DeadProxy(starting())
        ns = build(state, FakeProcess(alive=False))
        out = ns["_reap_stuck_start"](dict(state))
        # The write could not land, so the poll answers with what _ts_snapshot
        # gives during a teardown rather than raising out of the request.
        assert out["status"] == "restarting"
        assert ns["_shutting_down"].is_set()

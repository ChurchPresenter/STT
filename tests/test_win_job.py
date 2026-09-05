"""Binding the server's process tree to a Windows job object.

POSIX gets this from ``start_new_session`` plus one ``killpg``. Windows has no
equivalent signal, so terminating the server left its multiprocessing children
attached to nothing — measured in the field at four strays holding ~1.1 GB, one
with a Whisper model still resident.

The Win32 calls are injected, so the sequence and every failure branch are
exercised here rather than only on the machines we cannot debug.
"""

from __future__ import annotations

import pytest

from stt import win_job


class FakeApi:
    """Records the call sequence and fails wherever it is told to."""

    def __init__(self, fail_at=None, raise_at=None):
        self.fail_at = fail_at
        self.raise_at = raise_at
        self.calls = []
        self.closed = []

    def _step(self, name, ok_value):
        self.calls.append(name)
        if self.raise_at == name:
            raise OSError(f"{name} exploded")
        return 0 if self.fail_at == name else ok_value

    def create_job(self):
        return self._step("create_job", 111)

    def set_kill_on_close(self, job):
        return bool(self._step("set_kill_on_close", 1))

    def open_process(self, pid):
        return self._step("open_process", 222)

    def assign(self, job, process):
        return bool(self._step("assign", 1))

    def close(self, handle):
        self.calls.append("close")
        self.closed.append(handle)


# --- the happy path --------------------------------------------------------


def test_the_tree_is_bound_and_the_handle_returned():
    api = FakeApi()
    result = win_job.bind_process_tree(4242, platform="win32", api=api)

    assert result.bound is True
    assert result.handle == 111
    assert api.calls == ["create_job", "set_kill_on_close", "open_process",
                         "assign", "close"]


def test_the_process_handle_is_closed_but_the_job_handle_is_kept():
    """Closing the job is what kills the tree; it must outlive binding."""
    api = FakeApi()
    result = win_job.bind_process_tree(4242, platform="win32", api=api)

    assert api.closed == [222], "only the process handle is closed here"
    assert result.handle not in api.closed


def test_releasing_closes_the_job_and_so_kills_the_children():
    api = FakeApi()
    result = win_job.bind_process_tree(4242, platform="win32", api=api)
    api.closed.clear()

    assert win_job.release(result, api=api) is True
    assert api.closed == [111]


# --- failing is allowed, leaking a handle is not ---------------------------


@pytest.mark.parametrize("step", ["create_job", "set_kill_on_close",
                                  "open_process", "assign"])
def test_a_failure_at_any_step_leaves_the_caller_running(step):
    """A leak beats a server that will not start."""
    api = FakeApi(fail_at=step)
    result = win_job.bind_process_tree(4242, platform="win32", api=api)

    assert result.bound is False
    assert result.handle is None
    assert result.error and step.split("_")[0] in result.error.lower()


@pytest.mark.parametrize("step,expected", [
    ("set_kill_on_close", [111]),
    ("open_process", [111]),
    ("assign", [222, 111]),
])
def test_every_handle_opened_before_a_failure_is_closed(step, expected):
    api = FakeApi(fail_at=step)
    win_job.bind_process_tree(4242, platform="win32", api=api)
    assert api.closed == expected


@pytest.mark.parametrize("step", ["create_job", "set_kill_on_close",
                                  "open_process", "assign"])
def test_a_raising_api_does_not_escape(step):
    api = FakeApi(raise_at=step)
    result = win_job.bind_process_tree(4242, platform="win32", api=api)

    assert result.bound is False
    assert "exploded" in (result.error or "")


def test_a_raise_still_closes_what_was_opened():
    api = FakeApi(raise_at="assign")
    win_job.bind_process_tree(4242, platform="win32", api=api)
    assert sorted(api.closed) == [111, 222]


# --- other platforms -------------------------------------------------------


def test_posix_does_nothing_because_killpg_already_covers_it():
    api = FakeApi()
    result = win_job.bind_process_tree(4242, platform="linux", api=api)

    assert result.bound is False
    assert api.calls == [], "no Win32 call may be attempted off Windows"


def test_releasing_an_unbound_result_is_a_no_op():
    """So the caller needs no second condition around its own teardown."""
    api = FakeApi()
    assert win_job.release(win_job.JobResult(None, False), api=api) is False
    assert api.closed == []


def test_releasing_survives_a_close_that_raises():
    class Boom(FakeApi):
        def close(self, handle):
            raise OSError("already gone")

    assert win_job.release(win_job.JobResult(111, True), api=Boom()) is False


def test_the_kill_on_close_flag_is_the_one_that_matters():
    """Without this constant the job exists and guarantees nothing."""
    assert win_job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x2000


def test_cleanup_after_a_raise_survives_a_close_that_also_raises():
    """The machine that fails one Win32 call is the machine that fails two."""
    class Hostile(FakeApi):
        def close(self, handle):
            raise OSError("close failed too")

    result = win_job.bind_process_tree(
        4242, platform="win32", api=Hostile(raise_at="assign"))

    assert result.bound is False
    assert "exploded" in (result.error or ""), "the original failure is reported"

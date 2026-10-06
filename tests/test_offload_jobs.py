"""Offloaded caption calls beside the live loop: cap, session keying, failures."""

from __future__ import annotations

import threading

import pytest

from stt.offload_jobs import DEFAULT_MAX_IN_FLIGHT, OffloadJobs


class Deferred:
    """A thread starter that holds jobs until the test runs them."""

    def __init__(self):
        self.jobs = []

    def __call__(self, run):
        self.jobs.append(run)

    def run_all(self):
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            job()


def test_an_answer_is_picked_up_once():
    start = Deferred()
    jobs = OffloadJobs(start=start)
    assert jobs.submit("s1", 7, lambda: "Peace be with you")
    assert jobs.take("s1", 7) == (False, None), "nothing until the call finishes"
    start.run_all()
    assert jobs.take("s1", 7) == (True, "Peace be with you")
    assert jobs.take("s1", 7) == (False, None)


def test_calls_in_flight_are_capped():
    start = Deferred()
    jobs = OffloadJobs(max_in_flight=2, start=start)
    assert jobs.submit("s", 1, lambda: 1) and jobs.submit("s", 2, lambda: 2)
    assert jobs.slots() == 0
    assert not jobs.submit("s", 3, lambda: 3)
    start.run_all()
    assert jobs.slots() == 2


def test_a_caption_is_never_sent_twice():
    start = Deferred()
    jobs = OffloadJobs(start=start)
    assert jobs.submit("s", 1, lambda: "a")
    assert not jobs.submit("s", 1, lambda: "b"), "while in flight"
    start.run_all()
    assert not jobs.submit("s", 1, lambda: "b"), "while its answer waits to be applied"
    assert jobs.busy("s", 1)


def test_a_late_answer_never_reaches_the_next_session():
    # Caption ids restart in every session database.
    start = Deferred()
    jobs = OffloadJobs(start=start)
    jobs.submit("old", 5, lambda: "old answer")
    start.run_all()
    assert jobs.take("new", 5) == (False, None)
    assert jobs.drop_other_sessions("new") == 1
    assert jobs.take("old", 5) == (False, None)


def test_a_failing_call_hands_back_its_exception():
    start = Deferred()
    jobs = OffloadJobs(start=start)

    def boom():
        raise ConnectionError("peer gone")

    jobs.submit("s", 1, boom)
    start.run_all()
    found, answer = jobs.take("s", 1)
    assert found and isinstance(answer, ConnectionError)
    assert jobs.in_flight() == 0


def test_a_thread_that_cannot_start_frees_its_slot():
    def refuse(_run):
        raise RuntimeError("can't start new thread")

    jobs = OffloadJobs(start=refuse)
    with pytest.raises(RuntimeError):
        jobs.submit("s", 1, lambda: 1)
    assert jobs.slots() == DEFAULT_MAX_IN_FLIGHT
    assert not jobs.busy("s", 1)


def test_real_threads_run_side_by_side():
    # The point of the module: three slow calls take one call's time, not three.
    gate = threading.Barrier(3, timeout=5)
    jobs = OffloadJobs(max_in_flight=3)
    for i in range(3):
        jobs.submit("s", i, lambda i=i: (gate.wait(), i)[1])
    deadline = threading.Event()
    for _ in range(500):
        if jobs.in_flight() == 0:
            break
        deadline.wait(0.01)
    assert sorted(jobs.take("s", i)[1] for i in range(3)) == [0, 1, 2]

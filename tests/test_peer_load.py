"""Telling a busy translation server from one that is not answering."""

from __future__ import annotations

import threading

import pytest

from stt.peer_load import (
    FAILURE_DOWN,
    FAILURE_ERROR,
    FAILURE_TIMEOUT,
    LOAD_KEY,
    WorkTracker,
    classify_failure,
    parse_load,
    peer_is_working,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_an_idle_server_reports_nothing_in_flight():
    snap = WorkTracker(Clock()).snapshot()
    assert snap == {"in_flight": 0, "oldest_started_age_s": None, "last_done_age_s": None}


def test_ages_are_relative_so_the_two_clocks_never_meet():
    clock = Clock()
    tracker = WorkTracker(clock)
    done = tracker.begin()
    clock.now += 2
    tracker.end(done)
    tracker.begin()
    clock.now += 10
    snap = tracker.snapshot()
    assert snap == {"in_flight": 1, "oldest_started_age_s": 10.0, "last_done_age_s": 10.0}


def test_a_failed_request_still_counts_as_finished():
    clock = Clock()
    tracker = WorkTracker(clock)
    with pytest.raises(RuntimeError):
        with tracker.working():
            raise RuntimeError("model blew up")
    assert tracker.snapshot()["in_flight"] == 0
    assert tracker.snapshot()["last_done_age_s"] == 0.0


def test_ending_an_unknown_token_changes_nothing():
    tracker = WorkTracker(Clock())
    tracker.end(99)
    assert tracker.snapshot()["last_done_age_s"] is None


def test_concurrent_requests_are_all_counted():
    tracker = WorkTracker()
    tokens = []
    threads = [threading.Thread(target=lambda: tokens.append(tracker.begin())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.snapshot()["in_flight"] == 8
    for token in tokens:
        tracker.end(token)
    assert tracker.snapshot()["in_flight"] == 0


# --- reading it on the other machine -------------------------------------------


def reply(**load):
    return {"success": True, LOAD_KEY: load}


def test_busy_and_finishing_things_is_working():
    assert peer_is_working(parse_load(reply(in_flight=2, oldest_started_age_s=20, last_done_age_s=3))) is True


def test_stuck_requests_with_no_recent_finish_is_not_working():
    # A hung server: requests pile up and nothing completes.
    assert peer_is_working(parse_load(reply(in_flight=5, oldest_started_age_s=200, last_done_age_s=190))) is False


def test_in_flight_but_never_finished_anything_is_not_working():
    assert peer_is_working(parse_load(reply(in_flight=1, oldest_started_age_s=60, last_done_age_s=None))) is False


def test_nothing_in_flight_means_the_caption_never_arrived():
    assert peer_is_working(parse_load(reply(in_flight=0, last_done_age_s=1))) is False


@pytest.mark.parametrize("raw", [
    {"success": True}, None, "ok", {LOAD_KEY: "busy"}, {LOAD_KEY: {"in_flight": "2"}},
    {LOAD_KEY: {"in_flight": -1}}, {LOAD_KEY: {"in_flight": True}},
])
def test_an_older_or_malformed_server_is_unknown(raw):
    assert parse_load(raw) is None
    assert peer_is_working(parse_load(raw)) is None


def test_garbage_ages_are_ignored_not_trusted():
    load = parse_load(reply(in_flight=1, last_done_age_s="soon", oldest_started_age_s=-4))
    assert load == {"in_flight": 1.0, "oldest_started_age_s": None, "last_done_age_s": None}
    assert peer_is_working(load) is False


# --- what kind of failure it was --------------------------------------------------



class RequestException(OSError):
    pass


class Timeout(RequestException):
    pass


class ConnectionError_(RequestException):  # requests.ConnectionError, renamed below
    pass


ConnectionError_.__name__ = "ConnectionError"


class ReadTimeout(Timeout):
    pass


class ConnectTimeout(ConnectionError_, Timeout):
    pass


class HTTPError(RequestException):
    pass


def wrapped(cause):
    """As _translate_via_remote raises it: its own error, from the transport's."""
    try:
        raise RuntimeError("remote failed") from cause
    except RuntimeError as exc:
        return exc


@pytest.mark.parametrize("cause,kind", [
    (ConnectTimeout(), FAILURE_DOWN),
    (ConnectionError_(), FAILURE_DOWN),
    (ConnectionRefusedError(), FAILURE_DOWN),
    (ReadTimeout(), FAILURE_TIMEOUT),
    (TimeoutError(), FAILURE_TIMEOUT),
    (HTTPError(), FAILURE_ERROR),
    (ValueError("bad json"), FAILURE_ERROR),
])
def test_the_failure_is_read_through_the_wrapper(cause, kind):
    assert classify_failure(wrapped(cause)) == kind


def test_no_exception_is_an_error_not_a_guess():
    assert classify_failure(None) == FAILURE_ERROR

"""The monolith's half of "wait only for a paired server that is working".

stt/peer_load.py decides; these pin how speech_to_text.py feeds it: which failures
ask the server, and what a heartbeat reply carries.
"""

from __future__ import annotations

import pytest

from conftest import extract_definitions
from stt.coercion import coerce_int
from stt.peer_load import (
    FAILURE_DOWN,
    FAILURE_ERROR,
    FAILURE_TIMEOUT,
    LOAD_KEY,
    WorkTracker,
    parse_load,
    peer_is_working,
)


class Local:
    pass


class Reply:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def ns(failure, reply=None, raises=None):
    local = Local()
    local.remote_failure = failure
    calls = []

    def peer_request(method, endpoint, path, **kw):
        calls.append((method, endpoint, path, kw.get("timeout")))
        if raises:
            raise raises
        return Reply(reply)

    space = extract_definitions("speech_to_text.py", ["_peer_working_after_failure"], {
        "_mt_provenance": local, "_peer_request": peer_request,
        "_PEER_DOWN": FAILURE_DOWN, "_PEER_TIMEOUT": FAILURE_TIMEOUT,
        "_parse_peer_load": parse_load, "_peer_is_working": peer_is_working,
        "coerce_int": coerce_int, "config": {}, "_DEFAULT_PORT": 80})
    return space["_peer_working_after_failure"], calls


def test_a_failure_with_no_server_involved_keeps_the_plain_cap():
    fn, calls = ns(None)
    assert fn() is None and calls == []


def test_an_unreachable_server_is_not_working_without_asking_it():
    fn, calls = ns((FAILURE_DOWN, "http://peer"))
    assert fn() is False and calls == []


def test_an_error_reply_keeps_the_plain_cap():
    fn, calls = ns((FAILURE_ERROR, "http://peer"))
    assert fn() is None and calls == []


def test_a_timeout_asks_the_server_quickly():
    fn, calls = ns((FAILURE_TIMEOUT, "http://peer"),
                   reply={"success": True, LOAD_KEY: {"in_flight": 2, "last_done_age_s": 4}})
    assert fn() is True
    assert calls == [("POST", "http://peer", "/api/translate/heartbeat", 3)]


def test_a_timeout_on_a_server_making_no_progress_gives_up():
    fn, _ = ns((FAILURE_TIMEOUT, "http://peer"),
               reply={"success": True, LOAD_KEY: {"in_flight": 4, "last_done_age_s": 300}})
    assert fn() is False


def test_an_older_server_that_reports_nothing_keeps_the_plain_cap():
    fn, _ = ns((FAILURE_TIMEOUT, "http://peer"), reply={"success": True})
    assert fn() is None


def test_a_server_too_busy_to_answer_the_check_is_not_working():
    fn, _ = ns((FAILURE_TIMEOUT, "http://peer"), raises=TimeoutError())
    assert fn() is False


def test_the_heartbeat_reply_carries_the_load():
    class Request:
        remote_addr = "192.168.2.62"

        def get_json(self, silent=False):
            return {}

    tracker = WorkTracker()
    tracker.begin()
    space = extract_definitions("speech_to_text.py", ["translate_remote_heartbeat"], {
        "request": Request(), "jsonify": lambda obj: obj, "_paired_client_ok": lambda: True,
        "coerce_int": coerce_int, "_register_translation_client": lambda *a: None,
        "_peer_activity": type("A", (), {"record": staticmethod(lambda *a: None)})(),
        "_ACT_HEARTBEAT": "heartbeat", "time": __import__("time"),
        "_remember_client_port": lambda *a: None, "_peer_work": tracker,
        "_PEER_LOAD_KEY": LOAD_KEY,
        "app": type("A", (), {"route": staticmethod(lambda *a, **k: (lambda f: f))})()})
    body = space["translate_remote_heartbeat"]()
    assert parse_load(body) is not None and body[LOAD_KEY]["in_flight"] == 1


@pytest.mark.parametrize("name", ["translate_remote", "summarize_remote"])
def test_both_peer_routes_count_as_work(name):
    import re
    source = open("speech_to_text.py", encoding="utf-8").read()
    assert re.search(r"@_tracks_peer_work\ndef %s\(" % name, source), \
        "%s must be counted, or a busy server looks idle" % name


def test_the_route_wrapper_counts_while_running_and_after_a_crash():
    import functools

    tracker = WorkTracker()
    space = extract_definitions("speech_to_text.py", ["_tracks_peer_work"],
                                {"functools": functools, "_peer_work": tracker})
    seen = []

    @space["_tracks_peer_work"]
    def view():
        seen.append(tracker.snapshot()["in_flight"])
        raise RuntimeError("model blew up")

    with pytest.raises(RuntimeError):
        view()
    assert seen == [1]
    assert tracker.snapshot()["in_flight"] == 0
    assert view.__name__ == "view", "Flask routes by function name"

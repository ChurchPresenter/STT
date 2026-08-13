"""Monolith wiring for the live-map pings, exercised without importing it.

speech_to_text.py cannot be imported (ML libraries, Flask app, background threads at
import time), so these extract the individual functions and exec them against a stub
namespace — see tests/conftest.py:extract_definitions.

What is pinned here is everything the pure module cannot see: that an opted-out install
makes no request at all, that a dead collector cannot raise into a service start, and
that one machine cannot register as two installs.
"""

import threading

import pytest

from conftest import extract_definitions
from stt import livemap as _livemap


class _Recorder:
    """Stands in for the requests module, remembering one call."""

    def __init__(self, raises=None):
        self.calls = []
        self._raises = raises

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "timeout": timeout})
        if self._raises is not None:
            raise self._raises
        return object()


def _ns(config, requests_stub, *, saves=None, ids=("generated-id",)):
    """_send_livemap_ping and _get_install_id over a controlled config."""
    remaining = list(ids)

    def _save_config(cfg):
        if saves is not None:
            saves.append(dict(cfg.get("analytics", {})))
        return True

    ns = extract_definitions(
        "speech_to_text.py", ["_send_livemap_ping", "_get_install_id"],
        {"config": config,
         "_livemap": _livemap,
         "_install_id_lock": threading.Lock(),
         "save_config": _save_config,
         "uuid": type("U", (), {"uuid4": staticmethod(lambda: remaining.pop(0))}),
         "sys": type("S", (), {"platform": "darwin"}),
         "SERVER_DISPLAY_VERSION": "26.1.22-gc588d29",
         "SERVER_VERSION": "26.1.22",
         "SERVER_COMMIT": "c588d29"})
    # The function imports requests lazily inside its body, so the stub is installed
    # under the name the import binds.
    import sys as _real_sys
    _real_sys.modules["requests"] = requests_stub
    return ns


@pytest.fixture(autouse=True)
def _drop_requests_stub():
    yield
    import sys as _real_sys
    _real_sys.modules.pop("requests", None)


class TestOptedOut:
    def test_a_blank_endpoint_makes_no_request_at_all(self):
        # The kill switch has to stop the call, not just blank the URL — an opted-out
        # install must put nothing on the wire.
        req = _Recorder()
        ns = _ns({"analytics": {"endpoint": "", "install_id": "x"}}, req)
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is False
        assert req.calls == []

    def test_a_missing_analytics_section_is_opted_out(self):
        req = _Recorder()
        ns = _ns({}, req)
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is False
        assert req.calls == []


class TestPingSent:
    def test_the_app_start_ping_carries_its_event_and_the_install_id(self):
        req = _Recorder()
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"}}, req)
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is True
        call = req.calls[0]
        assert call["url"].endswith("&event=app_start")
        assert "os=macos" in call["url"] and "version=26.1.22" in call["url"]
        assert call["headers"]["X-Install-Id"] == "abc"
        assert call["timeout"] == 10

    def test_the_transcription_ping_carries_the_session_fields(self):
        req = _Recorder()
        config = {"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"},
                  "audio": {"language": "ru"},
                  "live_translation": {"enabled": True, "target_language": "en"}}
        ns = _ns(config, req)
        ns["_send_livemap_ping"](_livemap.EVENT_TRANSCRIPTION_START,
                                 **_livemap.ping_fields_from_config(config))
        url = req.calls[0]["url"]
        assert "transcribe_lang=ru" in url and "translate_lang=en" in url
        assert url.endswith("&event=transcription_start")

    def test_the_two_events_are_distinguishable(self):
        req = _Recorder()
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"}}, req)
        ns["_send_livemap_ping"](_livemap.EVENT_APP_START)
        ns["_send_livemap_ping"](_livemap.EVENT_TRANSCRIPTION_START)
        events = [c["url"].rsplit("event=", 1)[1] for c in req.calls]
        assert events == ["app_start", "transcription_start"]


class TestFailureIsSwallowed:
    def test_a_dead_collector_cannot_raise_into_a_start(self):
        # This runs on a daemon thread spawned from the Start path and from boot; an
        # exception escaping here would be an unhandled thread exception on every
        # service start while the collector is down.
        req = _Recorder(raises=OSError("connection refused"))
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"}}, req)
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is False
        assert len(req.calls) == 1, "it tried once and gave up"


class TestInstallId:
    def test_an_id_is_generated_and_persisted_once(self):
        saves = []
        config = {"analytics": {"endpoint": "https://c/api/ping", "install_id": ""}}
        ns = _ns(config, _Recorder(), saves=saves, ids=("generated-id",))
        assert ns["_get_install_id"]() == "generated-id"
        assert len(saves) == 1, "the config is written exactly once"
        assert ns["_get_install_id"]() == "generated-id"
        assert len(saves) == 1, "a boot that already has an id must not rewrite it"

    def test_concurrent_callers_produce_one_id(self):
        # With audio.autostart on, the app-start and transcription-start pings fire
        # within a second of each other at boot. Two ids would make the map count one
        # machine as two installs.
        saves = []
        config = {"analytics": {"endpoint": "https://c/api/ping", "install_id": ""}}
        ns = _ns(config, _Recorder(), saves=saves, ids=("first-id", "second-id"))
        seen = []
        ready = threading.Barrier(4)

        def _call():
            ready.wait(timeout=5)
            seen.append(ns["_get_install_id"]())

        threads = [threading.Thread(target=_call) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert set(seen) == {"first-id"}
        assert config["analytics"]["install_id"] == "first-id"
        assert len(saves) == 1

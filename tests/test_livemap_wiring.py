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


def _ns(config, requests_stub, *, saves=None, ids=("generated-id",), gpu="Test GPU",
        remote=None, probes=None, is_dev=False):
    """_send_livemap_ping and _get_install_id over a controlled config.

    The machine description (OS release, arch, GPU) is stubbed rather than probed: the
    real values come from platform and nvidia-smi, and a test that asserted on them
    would assert on whichever machine happened to run it. ``remote`` stands in for the
    paired machine's provenance, and ``probes`` (a list) records each time the peer
    would have been asked. ``is_dev`` stands in for the dirty-worktree probe, which
    would otherwise answer for whichever checkout is running the suite.
    """
    remaining = list(ids)

    def _fetch_remote_provenance():
        if probes is not None:
            probes.append(True)
        return dict(remote or {})

    def _save_config(cfg):
        if saves is not None:
            saves.append(dict(cfg.get("analytics", {})))
        return True

    ns = extract_definitions(
        "speech_to_text.py",
        ["_send_livemap_ping", "_get_install_id", "_remote_ping_provenance"],
        {"config": config,
         "_livemap": _livemap,
         "_install_id_lock": threading.Lock(),
         "save_config": _save_config,
         "uuid": type("U", (), {"uuid4": staticmethod(lambda: remaining.pop(0))}),
         "sys": type("S", (), {"platform": "darwin"}),
         "SERVER_DISPLAY_VERSION": "26.1.22-gc588d29",
         "SERVER_VERSION": "26.1.22",
         "SERVER_COMMIT": "c588d29",
         "SERVER_OS_VERSION": "15.5",
         "SERVER_ARCH": "arm64",
         "SERVER_IS_DEV": is_dev,
         "_probe_hardware": lambda: {"gpu_name": gpu},
         "_remote_effective": {},
         "_fetch_remote_provenance": _fetch_remote_provenance})
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

    def test_a_clean_checkout_pings_as_a_real_install(self):
        # No src at all is the historical shape, and what the collector reads as
        # a real install — a production box self-updates over git, so simply
        # being a checkout must not mark it as a maintainer's machine.
        req = _Recorder()
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"}}, req)
        ns["_send_livemap_ping"](_livemap.EVENT_APP_START)
        assert "src=" not in req.calls[0]["url"]

    def test_a_dirty_checkout_pings_as_dev(self):
        req = _Recorder()
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"}}, req,
                 is_dev=True)
        ns["_send_livemap_ping"](_livemap.EVENT_APP_START)
        assert "src=dev" in req.calls[0]["url"]

    def test_both_events_describe_the_machine_and_its_models(self):
        # An install left running all week never reaches a transcription start, so
        # app_start has to carry the description too or the support case has nothing.
        req = _Recorder()
        config = {"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"},
                  "model": {"type": "whisper", "whisper": {"model": "large-v3"}},
                  "live_translation": {"enabled": True, "translation_method": "nllb",
                                       "translation_model": "facebook/nllb-200"}}
        ns = _ns(config, req, gpu="NVIDIA GeForce RTX 4060")
        ns["_send_livemap_ping"](_livemap.EVENT_APP_START)
        ns["_send_livemap_ping"](_livemap.EVENT_TRANSCRIPTION_START)
        for call in req.calls:
            url = call["url"]
            assert "os_version=15.5" in url and "arch=arm64" in url
            assert "gpu=NVIDIA%20GeForce%20RTX%204060" in url
            assert "stt_model=large-v3" in url
            assert "mt_model=nllb%3Afacebook%2Fnllb-200" in url

    def test_an_offloading_box_reports_its_peers_model_on_both_events(self):
        # The local translation config on such a box is a standby that never runs.
        req = _Recorder()
        config = {"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"},
                  "live_translation": {
                      "enabled": True, "translation_method": "nllb",
                      "translation_model": "facebook/nllb-200",
                      "remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080"}}}
        ns = _ns(config, req,
                 remote={"mt.remote.effective.model": "gemma-4-12b-it-Q4_K_M.gguf",
                         "mt.remote.effective.method": "llm",
                         "mt.remote.effective.llm_endpoint": "http://192.168.2.52:11434"})
        ns["_send_livemap_ping"](_livemap.EVENT_APP_START)
        ns["_send_livemap_ping"](_livemap.EVENT_TRANSCRIPTION_START)
        for call in req.calls:
            assert "mt_model=remote%3Allm%3Agemma-4-12b-it-Q4_K_M.gguf" in call["url"]
            assert "192.168.2.52" not in call["url"], "the peer's endpoint is not ours to report"
            assert "11434" not in call["url"]

    def test_an_unreachable_peer_still_pings_as_offloaded(self):
        req = _Recorder()
        config = {"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"},
                  "live_translation": {
                      "enabled": True,
                      "remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080"}}}
        ns = _ns(config, req, remote={})
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is True
        assert "mt_model=remote&" in req.calls[0]["url"]

    def test_an_opted_out_install_does_not_even_probe_its_peer(self):
        # The kill switch means no analytics work at all, not just no collector call.
        probes = []
        config = {"analytics": {"endpoint": "", "install_id": "abc"},
                  "live_translation": {
                      "enabled": True,
                      "remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080"}}}
        ns = _ns(config, _Recorder(), probes=probes)
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is False
        assert probes == []

    def test_a_box_that_does_not_offload_never_probes(self):
        probes = []
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"},
                  "live_translation": {"enabled": True, "translation_method": "nllb"}},
                 _Recorder(), probes=probes)
        ns["_send_livemap_ping"](_livemap.EVENT_APP_START)
        assert probes == []

    def test_an_unprobeable_machine_still_pings(self):
        # Every probe fails open: a box where nvidia-smi is absent and platform tells
        # us nothing must still register as an install.
        req = _Recorder()
        ns = _ns({"analytics": {"endpoint": "https://c/api/ping", "install_id": "abc"}},
                 req, gpu=None)
        assert ns["_send_livemap_ping"](_livemap.EVENT_APP_START) is True
        assert "gpu=" not in req.calls[0]["url"]


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


class TestEveryStartPathPings:
    """Which paths that begin a session tell the map about it.

    Four places put {"command": "start"} on the control queue, and for a long time
    only the /api/transcription/start route pinged. An install running with
    audio.autostart therefore reported itself alive at every boot and never once
    reported a service — inverted for exactly the unattended installs that caption
    every service they are used for.

    Asserted against the source because the call sites live in a Flask route, a
    restart route and the __main__ block, none of which can be imported. The unit of
    meaning is "this start path pings", and that is what a future edit would drop.
    """

    import pathlib
    SOURCE = (pathlib.Path(__file__).resolve().parent.parent / "speech_to_text.py").read_text(
        encoding="utf-8")

    def _block_after(self, marker, lines=25):
        """The lines following a start command, where the ping belongs.

        Wide enough to span the state updates and the comment block the route
        carries between the queue put and its ping, and still tight enough that the
        calibration case cannot pass by picking up an unrelated call site.
        """
        assert marker in self.SOURCE, f"anchor moved: {marker!r}"
        return self.SOURCE.split(marker, 1)[1].split("\n", lines)[0:lines]

    def test_the_helper_exists_and_sends_the_transcription_event(self):
        assert "def _ping_transcription_started():" in self.SOURCE
        body = self.SOURCE.split("def _ping_transcription_started():", 1)[1][:800]
        assert "EVENT_TRANSCRIPTION_START" in body
        assert "daemon=True" in body, "a ping must never hold up a start"

    def test_the_start_route_pings(self):
        block = self._block_after('        # Send start command through queue\n'
                                  '        control_queue.put({"command": "start"})')
        assert any("_ping_transcription_started()" in line for line in block)

    def test_autostart_pings(self):
        block = self._block_after('            print("[AUTOSTART] audio.autostart enabled; '
                                  'starting transcription")')
        assert any("_ping_transcription_started()" in line for line in block)

    def test_restart_pings(self):
        block = self._block_after('            # CRITICAL: Update global reference for '
                                  'signal handler\n'
                                  '            globals()["thread1"] = transcription_process')
        assert any("_ping_transcription_started()" in line for line in block)

    def test_calibration_does_not_ping(self):
        # Calibration starts a real session, but to set levels. The map counts
        # services, so this one is excluded on purpose.
        block = self._block_after('            print("[CALIBRATION] Auto-starting '
                                  'transcription for calibration...", flush=True)')
        assert not any("_ping_transcription_started()" in line for line in block)

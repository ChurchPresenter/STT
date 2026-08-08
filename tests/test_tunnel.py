"""Cloudflare quick-tunnel lifecycle and URL scraping (stt/tunnel.py)."""

import os
import shutil
import threading

import pytest

from stt.tunnel import (
    DEFAULT_AUTO_STOP_SECONDS,
    STATUS_ERROR,
    STATUS_RUNNING,
    STATUS_STOPPED,
    CloudflareTunnel,
    build_command,
    cloudflared_asset,
    cloudflared_download_url,
    install_cloudflared,
    managed_binary_path,
    parse_quick_tunnel_url,
    resolve_binary,
    should_auto_stop,
)

# The line cloudflared actually prints, box drawing and all.
URL_LINE = (
    "2026-08-07T21:41:48Z INF |  https://sox-commission-fun-dylan.trycloudflare.com"
    "                                        |"
)
# The startup banner, which mentions two other cloudflare URLs. A looser
# "first https link wins" parser returns the terms-of-use page from this.
BANNER_LINE = (
    "2026-08-07T21:41:45Z INF Thank you for trying Cloudflare Tunnel. ... subject to the "
    "Cloudflare Online Services Terms of Use (https://www.cloudflare.com/website-terms/), ... "
    "see https://developers.cloudflare.com/cloudflare-one/connections/connect-apps"
)


class TestParseUrl:
    def test_extracts_the_boxed_url(self):
        assert parse_quick_tunnel_url(URL_LINE) == "https://sox-commission-fun-dylan.trycloudflare.com"

    def test_ignores_other_cloudflare_urls_in_the_banner(self):
        assert parse_quick_tunnel_url(BANNER_LINE) is None

    def test_no_url_returns_none(self):
        assert parse_quick_tunnel_url("2026-08-07T21:41:48Z INF Initial protocol quic") is None

    def test_matches_a_single_label_host(self):
        assert parse_quick_tunnel_url("INF |  https://abc.trycloudflare.com |") == "https://abc.trycloudflare.com"

    def test_does_not_match_a_lookalike_domain(self):
        assert parse_quick_tunnel_url("https://evil-trycloudflare.com.example.net") is None


class TestBuildCommand:
    def test_points_at_the_local_server(self):
        assert build_command("cloudflared", 8080) == [
            "cloudflared", "tunnel", "--url", "http://127.0.0.1:8080",
        ]

    def test_honours_an_explicit_binary_and_host(self):
        cmd = build_command("/opt/homebrew/bin/cloudflared", 80, host="0.0.0.0")
        assert cmd[0] == "/opt/homebrew/bin/cloudflared"
        assert cmd[-1] == "http://0.0.0.0:80"


class TestResolveBinary:
    def test_bare_name_found_on_path(self, monkeypatch):
        monkeypatch.setattr("stt.tunnel.shutil.which", lambda name: "/usr/bin/" + name)
        assert resolve_binary("cloudflared") == "/usr/bin/cloudflared"

    def test_falls_back_to_known_install_locations(self, tmp_path, monkeypatch):
        # A supervisor-started server has a minimal PATH, so PATH alone finds
        # nothing on a machine where cloudflared came from Homebrew.
        monkeypatch.setattr("stt.tunnel.shutil.which", lambda name: None)
        installed = tmp_path / "cloudflared"
        installed.write_text("#!/bin/sh\n")
        installed.chmod(0o755)
        assert resolve_binary("cloudflared", candidates=(str(installed),)) == str(installed)

    def test_none_when_nowhere_to_be_found(self, monkeypatch):
        monkeypatch.setattr("stt.tunnel.shutil.which", lambda name: None)
        assert resolve_binary("cloudflared", candidates=()) is None

    def test_explicit_path_is_used_verbatim(self, tmp_path):
        binary = tmp_path / "cf"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        assert resolve_binary(str(binary)) == str(binary)

    def test_explicit_path_that_does_not_exist_is_rejected(self, tmp_path):
        # Better a clear "not found" than an exec failure five seconds later.
        assert resolve_binary(str(tmp_path / "nope")) is None

    def test_blank_config_means_the_default_name(self, monkeypatch):
        monkeypatch.setattr("stt.tunnel.shutil.which", lambda name: "/usr/bin/" + name)
        assert resolve_binary("") == "/usr/bin/cloudflared"


class TestShouldAutoStop:
    def test_stops_once_the_delay_has_passed(self):
        assert should_auto_stop(True, False, idle_since=100.0, now=161.0) is True

    def test_waits_out_the_delay(self):
        assert should_auto_stop(True, False, idle_since=100.0, now=159.0) is False

    def test_boundary_is_inclusive(self):
        assert should_auto_stop(True, False, idle_since=100.0, now=100.0 + DEFAULT_AUTO_STOP_SECONDS) is True

    def test_never_stops_while_transcription_runs(self):
        # Restarting within the window has to cancel a pending stop, not fire
        # it mid-service.
        assert should_auto_stop(True, True, idle_since=100.0, now=9999.0) is False

    def test_tunnel_opened_before_any_transcription_stays_up(self):
        assert should_auto_stop(True, False, idle_since=None, now=9999.0) is False

    def test_nothing_to_stop_when_the_tunnel_is_down(self):
        assert should_auto_stop(False, False, idle_since=100.0, now=9999.0) is False

    def test_delay_is_configurable(self):
        assert should_auto_stop(True, False, 100.0, 130.0, delay_seconds=120) is False
        assert should_auto_stop(True, False, 100.0, 230.0, delay_seconds=120) is True

    def test_manual_mode_never_stops_it(self):
        # Operator closes it by hand; no elapsed time should ever fire the stop.
        assert should_auto_stop(True, False, 100.0, 99999.0, auto_stop_enabled=False) is False

    def test_manual_mode_is_opt_in(self):
        # The safe behaviour is the default: an omitted flag still auto-stops.
        assert should_auto_stop(True, False, idle_since=100.0, now=200.0) is True


class FakeProcess:
    """A cloudflared stand-in whose log stream the test drives line by line.

    After emitting its lines the stream *stays open*, as a live cloudflared's
    does — it only ends when the process is terminated or the test says the
    process died. A fake whose iterator returns immediately would look like a
    crash to the manager, which is a different scenario (covered separately by
    ``exits_after`` below).
    """

    def __init__(self, lines=(), exits_after=False):
        self._lines = list(lines)
        self._exits_after = exits_after
        self._exit_code = None
        self.terminated = False
        self.killed = False
        self._emitted = threading.Event()
        self._closed = threading.Event()
        self.stderr = self

    def __iter__(self):
        for line in self._lines:
            yield line
        self._emitted.set()
        if not self._exits_after:
            self._closed.wait(10)  # a live process keeps its stream open

    def wait_until_drained(self, timeout=5):
        assert self._emitted.wait(timeout), "reader thread never drained the stream"

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15
        self._closed.set()

    def wait(self, timeout=None):
        return self._exit_code

    def kill(self):  # pragma: no cover - only on a terminate timeout
        self.killed = True
        self._closed.set()


class TestLifecycle:
    def test_start_scrapes_the_url_and_reports_running(self):
        proc = FakeProcess([BANNER_LINE, "INF Requesting new quick Tunnel...", URL_LINE])
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: proc)

        tunnel.start(8080)
        assert tunnel.wait_for_url(timeout=5) == "https://sox-commission-fun-dylan.trycloudflare.com"

        status = tunnel.status()
        assert status["status"] == STATUS_RUNNING
        assert status["url"].endswith(".trycloudflare.com")
        assert status["error"] == ""

    def test_spawn_receives_the_configured_port(self):
        seen = {}

        def spawn(cmd):
            seen["cmd"] = cmd
            return FakeProcess([URL_LINE])

        CloudflareTunnel(binary="/usr/local/bin/cloudflared", resolve=lambda b: b, spawn=spawn).start(9999)
        assert seen["cmd"] == ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:9999"]

    def test_starting_twice_does_not_spawn_a_second_process(self):
        spawns = []

        def spawn(cmd):
            proc = FakeProcess([URL_LINE])
            spawns.append(proc)
            return proc

        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=spawn)
        tunnel.start(8080)
        tunnel.wait_for_url(timeout=5)
        tunnel.start(8080)
        assert len(spawns) == 1

    def test_stop_terminates_and_clears_the_url(self):
        proc = FakeProcess([URL_LINE])
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: proc)
        tunnel.start(8080)
        tunnel.wait_for_url(timeout=5)

        status = tunnel.stop(reason="transcription ended")
        assert proc.terminated is True
        assert status["status"] == STATUS_STOPPED
        assert status["url"] is None
        assert tunnel.is_running() is False

    def test_stop_when_never_started_is_harmless(self):
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: FakeProcess())
        assert tunnel.stop()["status"] == STATUS_STOPPED

    def test_missing_binary_reports_an_actionable_error(self):
        spawned = []
        tunnel = CloudflareTunnel(
            binary="cloudflared",
            resolve=lambda b: None,  # not installed anywhere we look
            spawn=lambda cmd: spawned.append(cmd),
        )
        status = tunnel.start(8080)

        assert status["status"] == STATUS_ERROR
        assert "cloudflared not found" in status["error"]
        assert "brew install cloudflared" in status["error"]
        assert status["url"] is None
        assert spawned == []  # never even tried to launch

    def test_failed_start_does_not_block_wait_for_url(self):
        # A caller blocked on wait_for_url must not sit out the whole startup
        # timeout for a tunnel that failed before it launched.
        tunnel = CloudflareTunnel(resolve=lambda b: None, spawn=lambda cmd: None)
        tunnel.start(8080)
        assert tunnel.wait_for_url(timeout=30) is None
        assert tunnel.status()["uptime_seconds"] is None  # nothing was ever up

    def test_start_uses_the_resolved_path_not_the_configured_name(self):
        seen = {}

        def spawn(cmd):
            seen["cmd"] = cmd
            return FakeProcess([URL_LINE])

        CloudflareTunnel(
            binary="cloudflared", resolve=lambda b: "/opt/homebrew/bin/cloudflared", spawn=spawn
        ).start(8080)
        assert seen["cmd"][0] == "/opt/homebrew/bin/cloudflared"

    def test_process_exiting_on_its_own_surfaces_as_an_error(self):
        # Network loss or a cloudflared crash must not leave the UI showing a
        # tunnel that is no longer there.
        proc = FakeProcess(["INF Requesting new quick Tunnel..."], exits_after=True)
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: proc)
        tunnel.start(8080)
        proc.wait_until_drained()
        tunnel.wait_for_url(timeout=5)

        status = tunnel.status()
        assert status["status"] == STATUS_ERROR
        assert "exited unexpectedly" in status["error"]
        assert tunnel.is_running() is False

    def test_wait_for_url_gives_up_rather_than_hanging(self):
        proc = FakeProcess(["INF still connecting"])
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: proc)
        tunnel.start(8080)
        assert tunnel.wait_for_url(timeout=2) is None

    def test_status_keeps_a_log_tail_for_diagnosis(self):
        proc = FakeProcess(["INF one", "INF two", URL_LINE])
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: proc)
        tunnel.start(8080)
        tunnel.wait_for_url(timeout=5)
        proc.wait_until_drained()
        assert any("INF one" in line for line in tunnel.status()["log_tail"])

    def test_uptime_is_reported_from_the_injected_clock(self):
        ticks = iter([1000.0, 1000.0, 1042.0, 1042.0])
        proc = FakeProcess([URL_LINE])
        tunnel = CloudflareTunnel(resolve=lambda b: b, spawn=lambda cmd: proc, clock=lambda: next(ticks))
        tunnel.start(8080)
        tunnel.wait_for_url(timeout=5)
        assert tunnel.status()["uptime_seconds"] == pytest.approx(42.0)


class TestCloudflaredAsset:
    def test_the_platforms_we_actually_ship_on(self):
        assert cloudflared_asset("Linux", "x86_64") == "cloudflared-linux-amd64"
        assert cloudflared_asset("Linux", "aarch64") == "cloudflared-linux-arm64"
        assert cloudflared_asset("Darwin", "arm64") == "cloudflared-darwin-arm64.tgz"
        assert cloudflared_asset("Darwin", "x86_64") == "cloudflared-darwin-amd64.tgz"
        assert cloudflared_asset("Windows", "AMD64") == "cloudflared-windows-amd64.exe"

    def test_unsupported_platform_returns_none(self):
        assert cloudflared_asset("freebsd", "x86_64") is None
        assert cloudflared_asset("linux", "s390x") is None

    def test_only_macos_needs_unpacking(self):
        # The install path branches on this, so a change in Cloudflare's
        # packaging should break a test rather than a production Start press.
        for system, machine in (("linux", "x86_64"), ("windows", "amd64")):
            assert not cloudflared_asset(system, machine).endswith(".tgz")
        assert cloudflared_asset("darwin", "arm64").endswith(".tgz")

    def test_download_url_points_at_the_latest_release(self):
        url = cloudflared_download_url("cloudflared-linux-amd64")
        assert url == (
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        )

    def test_managed_path_gets_an_exe_suffix_on_windows_only(self):
        assert managed_binary_path("/opt/stt/bin", "windows").endswith("cloudflared.exe")
        assert managed_binary_path("/opt/stt/bin", "linux").endswith("cloudflared")


class TestInstallCloudflared:
    def _fake_fetch(self, payload=b"#!/bin/sh\nexit 0\n"):
        def fetch(url, dest):
            with open(dest, "wb") as f:
                f.write(payload)
        return fetch

    def test_installs_a_bare_binary_and_makes_it_executable(self, tmp_path):
        path = install_cloudflared(
            str(tmp_path), self._fake_fetch(), system="linux", machine="x86_64"
        )
        assert path == str(tmp_path / "cloudflared")
        assert os.access(path, os.X_OK)

    def test_unpacks_the_macos_archive(self, tmp_path):
        archive = self._make_tgz(tmp_path, "cloudflared", b"mach-o-ish")

        def fetch(url, dest):
            shutil.copyfile(archive, dest)

        path = install_cloudflared(str(tmp_path / "bin"), fetch, system="darwin", machine="arm64")
        assert open(path, "rb").read() == b"mach-o-ish"
        assert os.access(path, os.X_OK)

    def test_leaves_no_partial_file_when_the_download_fails(self, tmp_path):
        # A half-written file at the real path would look installed and fail to
        # exec on every later press.
        def fetch(url, dest):
            with open(dest, "wb") as f:
                f.write(b"partial")
            raise OSError("connection reset")

        with pytest.raises(RuntimeError, match="Could not install cloudflared"):
            install_cloudflared(str(tmp_path), fetch, system="linux", machine="x86_64")
        assert list(tmp_path.iterdir()) == []

    def test_unsupported_platform_says_what_to_do_instead(self, tmp_path):
        with pytest.raises(RuntimeError, match="No cloudflared build published"):
            install_cloudflared(str(tmp_path), self._fake_fetch(), system="freebsd", machine="x86_64")

    def test_archive_without_the_binary_is_an_error_not_a_silent_success(self, tmp_path):
        archive = self._make_tgz(tmp_path, "README.md", b"nope")

        def fetch(url, dest):
            shutil.copyfile(archive, dest)

        with pytest.raises(RuntimeError, match="Could not install cloudflared"):
            install_cloudflared(str(tmp_path / "bin"), fetch, system="darwin", machine="arm64")

    def test_downloaded_copy_is_preferred_over_a_system_one(self, tmp_path, monkeypatch):
        # A later apt/brew install must not silently take over from the binary
        # this box already downloaded and is known to work.
        monkeypatch.setattr("stt.tunnel.shutil.which", lambda name: "/usr/bin/cloudflared")
        install_cloudflared(str(tmp_path), self._fake_fetch(), system="linux", machine="x86_64")
        assert resolve_binary("cloudflared", managed_dir=str(tmp_path)) == str(tmp_path / "cloudflared")

    def test_managed_dir_is_skipped_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("stt.tunnel.shutil.which", lambda name: "/usr/bin/cloudflared")
        assert resolve_binary("cloudflared", managed_dir=str(tmp_path)) == "/usr/bin/cloudflared"

    @staticmethod
    def _make_tgz(tmp_path, member_name, payload):
        import tarfile

        inner = tmp_path / member_name
        inner.write_bytes(payload)
        archive = tmp_path / "asset.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(str(inner), arcname=member_name)
        inner.unlink()
        return str(archive)

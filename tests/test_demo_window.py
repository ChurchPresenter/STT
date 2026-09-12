"""The demo's control window: the decisions, not the widgets.

The demo ships windowed on macOS and Windows, so before this existed the only way
to quit it was the task manager and there was no way to move it off whatever port
the search happened to land on. Everything the window *decides* lives in pure
functions so it can be tested without a display; the Tk shell around them is kept
deliberately thin.
"""

from __future__ import annotations

import os
import sys

import pytest

from stt import demo_mode, demo_window


# --- what the window says --------------------------------------------------


def test_a_running_server_is_reported_with_the_address_to_open():
    text = demo_window.status_text(True, 8099)
    assert "http://127.0.0.1:8099/" in text


def test_a_stopped_server_says_so_and_names_no_address():
    assert "http" not in demo_window.status_text(False, 8099)


def test_the_browser_url_is_loopback_even_though_the_server_binds_the_network():
    """The person reading the window is at the machine; the LAN address is the
    banner's job, not this one's."""
    assert demo_window.browser_url(8099) == "http://127.0.0.1:8099/"


# --- the port field --------------------------------------------------------


@pytest.mark.parametrize("raw", ["8099", " 8099 ", "65535", "1024"])
def test_a_usable_port_is_accepted(raw):
    port, reason = demo_window.validate_port(raw)
    assert port == int(raw.strip())
    assert reason == ""


@pytest.mark.parametrize("raw", ["", "   ", "eighty", "80.5"])
def test_an_unparseable_port_is_refused_with_a_reason(raw):
    port, reason = demo_window.validate_port(raw)
    assert port is None
    assert reason


def test_a_privileged_port_is_refused_because_a_demo_must_never_need_a_password():
    port, reason = demo_window.validate_port("80")
    assert port is None
    assert "administrator" in reason


def test_a_port_above_the_protocol_maximum_is_refused():
    port, reason = demo_window.validate_port("70000")
    assert port is None
    assert "65535" in reason


# --- relaunching on a new port ---------------------------------------------


def test_a_frozen_demo_re_execs_its_own_binary_once():
    """sys.executable and argv[0] are the same file in a frozen build, so repeating
    the script argument would hand the binary its own path to open."""
    command, _ = demo_window.relaunch_command(
        9000, ["/Apps/STT-Demo"], "/Apps/STT-Demo", frozen=True)
    assert command == ["/Apps/STT-Demo"]


def test_a_source_run_re_execs_the_interpreter_with_the_script():
    command, _ = demo_window.relaunch_command(
        9000, ["speech_to_text.py"], "/venv/bin/python3", frozen=False)
    assert command == ["/venv/bin/python3", "speech_to_text.py"]


def test_the_new_port_travels_in_the_environment():
    _, env = demo_window.relaunch_command(
        9000, ["/Apps/STT-Demo"], "/Apps/STT-Demo", frozen=True, environ={})
    assert env[demo_mode.ENV_PORT] == "9000"


def test_the_rest_of_the_environment_survives_the_relaunch():
    _, env = demo_window.relaunch_command(
        9000, ["/Apps/STT-Demo"], "/Apps/STT-Demo", frozen=True,
        environ={"STT_DEMO": "1", "HOME": "/Users/someone"})
    assert env["STT_DEMO"] == "1"
    assert env["HOME"] == "/Users/someone"


@pytest.mark.parametrize("stale", [["--port", "8099"], ["--port=8099"]])
def test_the_old_port_flag_is_dropped_so_it_cannot_outvote_the_new_one(stale):
    """The flag wins over the variable in requested_port, so leaving it in place
    would silently relaunch on the port the visitor just changed away from."""
    command, env = demo_window.relaunch_command(
        9000, ["/Apps/STT-Demo", *stale], "/Apps/STT-Demo", frozen=True, environ={})
    assert "--port" not in " ".join(command)
    assert env[demo_mode.ENV_PORT] == "9000"
    assert demo_mode.requested_port(env, command[1:]) == 9000


def test_other_arguments_are_carried_across_unchanged():
    command, _ = demo_window.relaunch_command(
        9000, ["/Apps/STT-Demo", "--session", "/tmp/a.db", "--port", "8099"],
        "/Apps/STT-Demo", frozen=True, environ={})
    assert command == ["/Apps/STT-Demo", "--session", "/tmp/a.db"]


def test_strip_port_flag_leaves_a_bare_trailing_flag_from_taking_the_next_argument():
    assert demo_window.strip_port_flag(["--session", "a.db", "--port"]) == [
        "--session", "a.db"]


# --- deciding whether to open a window at all -------------------------------


def test_no_preference_means_probe():
    assert demo_window.forced_window({}) is None


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", " 1 "])
def test_the_window_can_be_forced_on(value):
    assert demo_window.forced_window({demo_window.ENV_WINDOW: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_the_window_can_be_forced_off(value):
    """What a smoke test, a CI run or a service-managed demo wants: those have no
    window server, and asking for one there does not raise, it kills the process."""
    assert demo_window.forced_window({demo_window.ENV_WINDOW: value}) is False


def test_an_unrecognised_preference_falls_back_to_probing():
    assert demo_window.forced_window({demo_window.ENV_WINDOW: "maybe"}) is None


def test_forcing_the_window_off_skips_the_probe_entirely(monkeypatch):
    def explode():
        raise AssertionError("the probe must not run when the answer is settled")

    monkeypatch.setenv(demo_window.ENV_WINDOW, "0")
    monkeypatch.setattr(demo_window, "_forked_probe", explode)
    monkeypatch.setattr(demo_window, "_probe_window", explode)

    assert demo_window.display_available() is False


def test_forcing_the_window_on_skips_the_probe_entirely(monkeypatch):
    monkeypatch.setenv(demo_window.ENV_WINDOW, "1")
    monkeypatch.setattr(demo_window, "_forked_probe", lambda: False)

    assert demo_window.display_available() is True


def test_a_probe_exception_is_recorded_as_the_reason(monkeypatch):
    """print() is invisible on the windowed builds, so the reason a display was
    not found has to survive somewhere a log file can pick it up."""
    monkeypatch.delenv(demo_window.ENV_WINDOW, raising=False)

    def boom():
        raise RuntimeError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(demo_window, "_forked_probe", boom)
    monkeypatch.setattr(sys, "platform", "linux")

    assert demo_window.display_available() is False
    assert "no display name" in demo_window.last_probe_error()


def test_a_successful_probe_clears_the_reason(monkeypatch):
    monkeypatch.delenv(demo_window.ENV_WINDOW, raising=False)
    monkeypatch.setattr(demo_window, "_forked_probe", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")

    assert demo_window.display_available() is True
    assert demo_window.last_probe_error() is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_a_probe_that_crashes_the_child_reports_no_display(monkeypatch):
    """The reason the probe is forked at all: Tk failing to reach a window server
    aborts the process instead of raising, so no except clause in the parent can
    see it coming. Measured as SIGSEGV when the demo was started detached."""
    import signal

    def crash():
        # SIGKILL rather than SIGSEGV: the same WIFSIGNALED path, without
        # faulthandler dumping the child's stack across the test output.
        os.kill(os.getpid(), signal.SIGKILL)

    monkeypatch.setattr(demo_window, "_probe_window", crash)

    assert demo_window._forked_probe() is False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_a_probe_that_raises_in_the_child_reports_no_display(monkeypatch):
    def boom():
        raise RuntimeError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(demo_window, "_probe_window", boom)

    assert demo_window._forked_probe() is False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_a_probe_that_succeeds_in_the_child_reports_a_display(monkeypatch):
    monkeypatch.setattr(demo_window, "_probe_window", lambda: True)

    assert demo_window._forked_probe() is True


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_the_probe_leaves_no_child_behind(monkeypatch):
    """An unreaped child would sit as a zombie for the life of the demo."""
    monkeypatch.setattr(demo_window, "_probe_window", lambda: True)
    demo_window._forked_probe()

    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)

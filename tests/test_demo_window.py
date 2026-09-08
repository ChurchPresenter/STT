"""The demo's control window: the decisions, not the widgets.

The demo ships windowed on macOS and Windows, so before this existed the only way
to quit it was the task manager and there was no way to move it off whatever port
the search happened to land on. Everything the window *decides* lives in pure
functions so it can be tested without a display; the Tk shell around them is kept
deliberately thin.
"""

from __future__ import annotations

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

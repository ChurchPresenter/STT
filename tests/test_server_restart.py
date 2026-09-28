"""The web UI's Restart button (stt/server_restart.py and the watchdog's side of it).

On a fresh Windows PC the button stopped the server and never started it again: the
restart script was looked up in the data dir, and the fallback relaunched a relative
``speech_to_text.py`` from there.
"""

import os

import pytest

from stt import watchdog
from stt.server_restart import (
    RESTART_EXIT_CODE,
    is_watchdog_managed,
    relaunch_argv,
    restart_script,
)


class TestRestartScript:
    @pytest.mark.parametrize("windows, name", [(True, "restart_server.bat"), (False, "restart_server.sh")])
    def test_found_in_the_checkout(self, tmp_path, windows, name):
        (tmp_path / name).write_text("x")
        assert restart_script(str(tmp_path), windows=windows) == str(tmp_path / name)

    def test_absent_is_none_so_the_caller_falls_back(self, tmp_path):
        assert restart_script(str(tmp_path), windows=True) is None

    def test_the_repo_ships_both(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert restart_script(repo, windows=True) and restart_script(repo, windows=False)


class TestRelaunchArgv:
    def test_a_relative_script_becomes_absolute(self, tmp_path):
        script = str(tmp_path / "speech_to_text.py")
        argv = relaunch_argv("/venv/python", script, ["speech_to_text.py", "--headless"], frozen=False)
        assert argv == ["/venv/python", script, "--headless"]
        assert os.path.isabs(argv[1])

    def test_a_frozen_build_is_its_own_executable(self):
        assert relaunch_argv("C:/STT/stt.exe", "ignored.py", ["C:/STT/stt.exe", "-x"],
                             frozen=True) == ["C:/STT/stt.exe", "-x"]


class TestIsWatchdogManaged:
    @pytest.mark.parametrize("env, managed", [
        ({}, False),
        ({"STT_MANAGED": "1", "STT_DATA_DIR": "/d"}, True),       # today's watchdog
        ({"STT_DATA_DIR": "/d"}, True),                            # a pre-2026-07-16 watchdog
        ({"STT_MANAGED": "0", "STT_DATA_DIR": "/home/ai/.stt"}, False),  # a launch script
        ({"STT_MANAGED": ""}, False),
    ])
    def test_cases(self, env, managed):
        assert is_watchdog_managed(env) is managed


class StopsWhenHandled(watchdog.WatchdogState):
    """Ends the recovery loop once the exit has been handled, whichever branch took it."""

    def set(self, **kw):
        super().set(**kw)
        if kw.get("status") in ("stopped", "crashed"):
            self.stop_requested = True


class FakeProc:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code

    def poll(self):
        return self.code


class Recorder:
    def __init__(self, state):
        self.state = state
        self.started = 0
        self.reports = []

    def start(self):
        self.started += 1
        self.state.stop_requested = True

    def report(self, code, n):
        self.reports.append(code)


def run_once(code):
    state = StopsWhenHandled()
    state.process = FakeProc(code)
    rec = Recorder(state)
    cr = watchdog.CrashRecoveryThread(state, rec, watchdog.threading.Event(), crash_reporter=rec)
    cr.run()  # returns once the exit is handled
    return state, rec


class TestWatchdogHonoursTheRestartCode:
    def test_restarts_at_once_without_counting_a_crash(self):
        state, rec = run_once(RESTART_EXIT_CODE)
        assert rec.started == 1
        assert rec.reports == [] and state.consecutive_crashes == 0
        assert state.status != "crashed"

    def test_a_clean_exit_is_still_a_stop(self):
        state, rec = run_once(0)
        assert rec.started == 0 and state.status == "stopped"

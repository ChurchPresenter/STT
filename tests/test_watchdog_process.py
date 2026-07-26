"""ProcessManager start/stop process-group handling: the worker is launched in
its own POSIX session and stop() sweeps that group so multiprocessing children
can't orphan to init (the restart-time memory leak)."""

import signal
import threading
from unittest import mock

import pytest

from stt import watchdog


class _FakeStdin:
    def __init__(self, proc):
        self._proc = proc

    def write(self, data):
        if b"shutdown" in data:
            self._proc._graceful_shutdown()

    def flush(self):
        return None


class _FakeProc:
    """Stand-in for subprocess.Popen that dies at a configurable escalation
    step: 'graceful' (stdin shutdown), 'terminate' (SIGTERM), or 'kill'."""

    def __init__(self, pid=4242, dies_on="graceful"):
        self.pid = pid
        self._alive = True
        self._dies_on = dies_on
        self.stdin = _FakeStdin(self)
        self.terminated = False
        self.killed = False

    def _graceful_shutdown(self):
        if self._dies_on == "graceful":
            self._alive = False

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        if self._alive:
            raise watchdog.subprocess.TimeoutExpired(cmd="stt", timeout=timeout)
        return 0

    def terminate(self):
        self.terminated = True
        if self._dies_on in ("graceful", "terminate"):
            self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False


def _make_pm(proc):
    pm = watchdog.ProcessManager(watchdog.WatchdogState(), threading.Event())
    pm.state.set(process=proc, status="running")
    return pm


def _patch_pgids(monkeypatch, worker_pgid, self_pgid=999):
    """os.getpgid(worker.pid) -> worker_pgid; os.getpgid(0) -> self_pgid."""
    monkeypatch.setattr(watchdog.os, "getpgid",
                        lambda pid: worker_pgid if pid else self_pgid)
    monkeypatch.setattr(watchdog.subprocess, "run", lambda *a, **k: None)
    killed = []
    monkeypatch.setattr(watchdog.os, "killpg",
                        lambda pgid, sig: killed.append((pgid, sig)))
    return killed


@pytest.mark.skipif(watchdog.IS_WINDOWS, reason="process-group sweep is POSIX-only")
def test_start_launches_worker_in_new_session(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(watchdog, "_rotate_if_large", lambda *a, **k: None)
    monkeypatch.setattr(watchdog, "get_python_bin", lambda: "python3")
    monkeypatch.setattr(watchdog.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(watchdog, "LOG_DIR", str(tmp_path))

    pm = watchdog.ProcessManager(watchdog.WatchdogState(), threading.Event())
    with mock.patch("builtins.open", mock.mock_open()):
        assert pm.start() is True

    # setsid via start_new_session so the whole tree can be swept on stop()
    assert captured.get("start_new_session") is True


@pytest.mark.skipif(watchdog.IS_WINDOWS, reason="process-group sweep is POSIX-only")
def test_stop_sweeps_the_worker_process_group(monkeypatch):
    proc = _FakeProc(pid=4242, dies_on="graceful")
    pm = _make_pm(proc)
    killed = _patch_pgids(monkeypatch, worker_pgid=4242)

    assert pm.stop() is True
    assert killed == [(4242, signal.SIGKILL)]


@pytest.mark.skipif(watchdog.IS_WINDOWS, reason="process-group sweep is POSIX-only")
def test_stop_never_signals_its_own_group(monkeypatch):
    # If start_new_session somehow didn't take effect, the worker shares the
    # watchdog's group — the guard must prevent us from killing ourselves.
    proc = _FakeProc(pid=4242, dies_on="graceful")
    pm = _make_pm(proc)
    killed = _patch_pgids(monkeypatch, worker_pgid=777, self_pgid=777)

    assert pm.stop() is True
    assert killed == []  # guard tripped: never killpg our own group


@pytest.mark.skipif(watchdog.IS_WINDOWS, reason="process-group sweep is POSIX-only")
def test_stop_sweeps_group_even_after_forced_kill(monkeypatch):
    # Worker ignores graceful shutdown and SIGTERM -> kill(); the group sweep
    # must still run to reap surviving children.
    proc = _FakeProc(pid=4242, dies_on="kill")
    pm = _make_pm(proc)
    killed = _patch_pgids(monkeypatch, worker_pgid=4242)

    assert pm.stop() is True
    assert proc.killed is True
    assert killed == [(4242, signal.SIGKILL)]

"""ProcessManager.stop(): graceful stdin-channel shutdown with escalation.

Windows cannot deliver SIGTERM across processes, so the watchdog asks the
worker to exit by writing 'shutdown' to its stdin pipe (the worker mirrors
this in speech_to_text.py's __main__ watcher) and only then escalates to
terminate/kill. These tests run real child processes to prove both rungs.
"""

import subprocess
import sys
import threading
import time

from stt import watchdog

# Reads stdin like the managed worker's watcher thread; exits cleanly on the
# shutdown command, exit code 0 proves the graceful path was taken.
COOPERATIVE = (
    "import sys\n"
    "for line in sys.stdin:\n"
    "    if line.strip() == 'shutdown':\n"
    "        sys.exit(0)\n"
)

# Never reads stdin — the graceful request must time out and stop() must
# escalate to terminate/kill.
STUBBORN = "import time\ntime.sleep(60)\n"


def _manager_with_child(monkeypatch, child_code):
    # The ffmpeg-orphan cleanup would `taskkill /F /IM ffmpeg.exe` for real —
    # stub it so tests can't touch unrelated processes on the host.
    monkeypatch.setattr(watchdog.subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 0))
    state = watchdog.WatchdogState()
    pm = watchdog.ProcessManager(state, threading.Event())
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.PIPE,
        creationflags=watchdog._CREATE_NO_WINDOW,
    )
    state.set(process=proc, status="running")
    return pm, state, proc


def test_cooperative_worker_exits_cleanly(monkeypatch):
    pm, state, proc = _manager_with_child(monkeypatch, COOPERATIVE)
    started = time.monotonic()
    assert pm.stop(timeout=10, graceful_timeout=10)
    assert proc.returncode == 0            # graceful: its own sys.exit(0)
    assert time.monotonic() - started < 8  # answered the request, no timeout ladder
    assert state.get("status") == "stopped"


def test_stubborn_worker_is_escalated(monkeypatch):
    pm, state, proc = _manager_with_child(monkeypatch, STUBBORN)
    assert pm.stop(timeout=3, graceful_timeout=1)
    assert proc.returncode is not None and proc.returncode != 0  # killed, not graceful
    assert state.get("status") == "stopped"


def test_stop_when_nothing_runs(monkeypatch):
    monkeypatch.setattr(watchdog.subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 0))
    state = watchdog.WatchdogState()
    pm = watchdog.ProcessManager(state, threading.Event())
    assert pm.stop()
    assert state.get("status") == "stopped"

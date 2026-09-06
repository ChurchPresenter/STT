"""Can a spawned child start while the parent holds a blocking read on stdin?

From the reproduction in ChurchPresenter/STT#13: on the reporter's Windows 11
machine, a `multiprocessing` child spawned *after* a thread began
`for line in sys.stdin` — with the parent's stdin an open, unwritten pipe — never
ran a line of Python. Zero CPU, one thread, no Python frames.

Nothing in our suite could see that. `tests/test_watchdog_process.py` uses a fake
`Popen` with a fake stdin, so no test creates a real pipe or a real child. This
does both, and is deliberately the reporter's shape rather than a tidier one.

**It wedged on a GitHub `windows-2022` runner**, so the mechanism is general
rather than anything about the reporting machine — and the objection raised
during review, that `popen_spawn_win32` passes `bInheritHandles=False` and so the
handle cannot reach the child, was simply wrong about the outcome whatever the
route turns out to be.

That platform behaviour is not ours to fix, so it is recorded as an expected
failure rather than asserted. What *is* ours is the mitigation, and the third
case here is the one that earns its keep: with the pipe moved to a private
descriptor and fd 0 left on `NUL` — exactly what `stt.shutdown_channel` does in
the server — the child starts normally. That is the proof the fix works on the
platform we cannot otherwise test.

Windows-only: the reported failure is a Win32 handle-inheritance effect, and on
POSIX the child gets its own fd table anyway.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
import threading

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="reported failure is Windows-only")

#: The child-spawning server, as the report wrote it.
SERVER = textwrap.dedent('''
    import sys, os, threading, time, multiprocessing as mp

    def child(q):
        q.put("child-alive")

    def watcher(stream):
        for line in stream:      # the pending blocking read
            pass

    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)
        mode = os.environ.get("USE_STDIN_WATCHER", "0")
        if mode in ("1", "detached"):
            stream = sys.stdin
            if mode == "detached":
                sys.path.insert(0, os.environ["STT_ROOT"])
                from stt.shutdown_channel import detach_stdin
                stream = detach_stdin().stream
            threading.Thread(target=watcher, args=(stream,), daemon=True).start()
            time.sleep(1.0)      # let the read block
        q = mp.Queue()
        p = mp.Process(target=child, args=(q,))
        p.start()
        try:
            print("RESULT:OK:" + q.get(timeout=30), flush=True)
        except Exception:
            print("RESULT:WEDGED", flush=True)
        finally:
            if p.is_alive():
                p.terminate()
''')


def _run(tmp_path, *, mode: str) -> str:
    """Spawn the server with its stdin pipe held open and unwritten.

    `communicate()` is deliberately not used: it closes stdin, the watcher sees
    EOF at once, and the condition under test disappears.
    """
    script = tmp_path / "srv.py"
    script.write_text(SERVER, encoding="utf-8")

    env = dict(os.environ)
    env["USE_STDIN_WATCHER"] = mode
    env["STT_ROOT"] = str(pathlib.Path(__file__).resolve().parent.parent)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, close_fds=False, text=True, cwd=str(tmp_path))
    lines: list[str] = []
    reader = threading.Thread(
        target=lambda: lines.extend(line.strip() for line in proc.stdout))
    reader.start()
    reader.join(timeout=70)
    try:
        proc.kill()
    except OSError:
        pass
    return next((line for line in lines if line.startswith("RESULT:")), "RESULT:NONE")


def test_a_child_spawns_when_nothing_is_reading_stdin(tmp_path):
    """The control: without the pending read the child is immediate."""
    assert _run(tmp_path, mode="0") == "RESULT:OK:child-alive"


@pytest.mark.xfail(strict=True, reason="the platform bug this exists to record: a "
                                       "spawned child wedges behind a pending read "
                                       "on an inherited stdin pipe (STT#13)")
def test_a_child_wedges_behind_a_pending_read_on_inherited_stdin(tmp_path):
    """Reproduced on a GitHub windows-2022 runner, so it is not machine-specific.

    strict=True on purpose: if this ever starts passing, the platform behaviour
    has changed and the mitigation below can be reconsidered — which is worth
    being told about rather than discovering by accident.
    """
    assert _run(tmp_path, mode="1") == "RESULT:OK:child-alive"


def test_the_detached_channel_lets_the_child_start(tmp_path):
    """The fix, on the platform that actually breaks without it.

    Same pending read, same spawn — but the pipe is on a private descriptor and
    fd 0 is the null device, which is what the server now does.
    """
    assert _run(tmp_path, mode="detached") == "RESULT:OK:child-alive", (
        "detaching the pipe from fd 0 did not unblock the spawn — the mitigation "
        "in stt/shutdown_channel.py does not work on this platform"
    )

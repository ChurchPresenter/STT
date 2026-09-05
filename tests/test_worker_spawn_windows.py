"""Can a spawned child start while the parent holds a blocking read on stdin?

From the reproduction in ChurchPresenter/STT#13: on the reporter's Windows 11
machine, a `multiprocessing` child spawned *after* a thread began
`for line in sys.stdin` — with the parent's stdin an open, unwritten pipe — never
ran a line of Python. Zero CPU, one thread, no Python frames.

Nothing in our suite could see that. `tests/test_watchdog_process.py` uses a fake
`Popen` with a fake stdin, so no test creates a real pipe or a real child. This
does both, and is deliberately the reporter's shape rather than a tidier one.

It is worth running for two opposite reasons. If it wedges on a GitHub runner,
the mechanism is real and general and we have it pinned. If it passes there while
their machine still fails, the cause is local to that machine and our fix is
treating a symptom — which is just as useful to know, and is why this asserts the
behaviour rather than skipping when it cannot explain it.

Windows-only: the reported failure is a Win32 handle-inheritance effect, and on
POSIX the child gets its own fd table anyway.
"""

from __future__ import annotations

import os
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

    def watcher():
        for line in sys.stdin:   # the pending blocking read
            pass

    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)
        if os.environ.get("USE_STDIN_WATCHER") == "1":
            threading.Thread(target=watcher, daemon=True).start()
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


def _run(tmp_path, *, watcher: bool, detach: bool = False) -> str:
    """Spawn the server with its stdin pipe held open and unwritten.

    `communicate()` is deliberately not used: it closes stdin, the watcher sees
    EOF at once, and the condition under test disappears.
    """
    script = tmp_path / "srv.py"
    script.write_text(SERVER, encoding="utf-8")

    env = dict(os.environ)
    env["USE_STDIN_WATCHER"] = "1" if watcher else "0"
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
    assert _run(tmp_path, watcher=False) == "RESULT:OK:child-alive"


def test_a_child_spawns_even_while_a_thread_blocks_on_stdin(tmp_path):
    """The reported failure. A wedge here is the bug, reproduced on CI."""
    result = _run(tmp_path, watcher=True)
    assert result == "RESULT:OK:child-alive", (
        "a spawned child did not start while the parent held a blocking read on "
        "an inherited stdin pipe — see ChurchPresenter/STT#13"
    )

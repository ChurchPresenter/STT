"""A multiprocessing child exits with its parent (stt/parent_watch.py).

Found on a Windows test PC: a killed server left its transcription worker and Manager
running, invisible to the stop scripts, and seven had built up across restarts.
"""

import multiprocessing
import os
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from stt.parent_watch import watch_parent

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_callback_runs_when_the_parent_goes():
    reader, writer = multiprocessing.Pipe(duplex=False)
    gone = threading.Event()
    thread = watch_parent(gone.set, parent=SimpleNamespace(sentinel=reader))
    assert thread is not None and thread.daemon
    assert not gone.wait(0.2)  # parent still alive: nothing happens
    writer.close()  # the parent's end closing is what a dying parent looks like
    assert gone.wait(10)


def test_the_main_process_has_no_parent_to_watch():
    assert watch_parent(lambda: pytest.fail("called")) is None


def test_a_parent_without_a_sentinel_is_left_alone():
    assert watch_parent(lambda: None, parent=SimpleNamespace()) is None


CHILD = textwrap.dedent("""
    import multiprocessing, os, sys, time
    sys.path.insert(0, {repo!r})

    def child(path):
        from stt.parent_watch import watch_parent
        watch_parent()
        with open(path, "w") as f:
            f.write(str(os.getpid()))
        time.sleep(120)  # would outlive the test without the watch

    if __name__ == "__main__":
        multiprocessing.set_start_method("spawn")
        p = multiprocessing.Process(target=child, args=(sys.argv[1],))
        p.start()
        while not os.path.exists(sys.argv[1]) or not open(sys.argv[1]).read():
            time.sleep(0.05)
        os._exit(0)  # the server being killed: no cleanup, no join
""")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX liveness check")
def test_a_real_child_exits_when_its_parent_is_killed(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(CHILD.format(repo=REPO))
    pid_file = tmp_path / "child.pid"
    subprocess.run([sys.executable, str(script), str(pid_file)], check=True, timeout=60)
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 20
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = _alive(child)
    if alive:
        os.kill(child, 9)
    assert not alive, "the child outlived its parent"

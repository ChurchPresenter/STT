"""Make a multiprocessing child exit when the server that started it is gone.

The server has two such children: the multiprocessing Manager that holds the shared
state, and the transcription worker. Neither noticed its parent dying. A server that
exits normally shuts them down itself, but one that is killed — Task Manager, a stop
script, ``schtasks /End``, a watchdog killed along with it — left them running. On a
Windows test PC, seven had built up across restarts, some alive for over an hour and
holding the old "STT Server" console window open. The stop scripts could not find them
either: a worker's command line is ``-c "from multiprocessing.spawn import spawn_main..."``
and never mentions speech_to_text.

Python gives every multiprocessing child a sentinel for its parent (3.8+): a handle to
the parent process on Windows, a pipe that closes with it elsewhere. A daemon thread
waits on it and ends the process. That covers every way the parent can die, which no
script can.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import os
import threading
from typing import Any, Callable, Optional


def _exit_now() -> None:
    # os._exit, not sys.exit: this runs on a side thread, and nothing is left to report
    # to. SQLite's WAL keeps every committed row; an open ffmpeg pipe breaks and ends too.
    os._exit(0)


def watch_parent(on_parent_exit: Optional[Callable[[], None]] = None, *,
                 parent: Any = None) -> Optional[threading.Thread]:
    """Start a daemon thread that calls ``on_parent_exit`` once the parent process ends.

    Returns the thread, or None when there is no parent to watch (the main process, or
    a caller outside multiprocessing). Safe to call from a Manager ``initializer``,
    which is why it takes no required arguments.
    """
    who = parent if parent is not None else multiprocessing.parent_process()
    sentinel = getattr(who, "sentinel", None)
    if sentinel is None:
        return None
    act = on_parent_exit or _exit_now

    def run() -> None:
        try:
            multiprocessing.connection.wait([sentinel])
        except (OSError, ValueError):
            return  # a sentinel we cannot wait on: keep the old behaviour
        act()

    thread = threading.Thread(target=run, name="parent-watch", daemon=True)
    thread.start()
    return thread

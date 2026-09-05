"""Reading the watchdog's shutdown pipe without leaving it on the stdin slot.

Windows cannot deliver SIGTERM across processes, so the watchdog stops the server
by writing ``shutdown`` to a pipe it holds as the server's standard input
(``stt/watchdog.py``). The server reads that pipe for the whole of its life.

The pipe being *on fd 0* is incidental to that job, and costs something: fd 0 is
what every child inherits. A transcription worker has no business holding the
watchdog's shutdown channel, and on Windows a report from the field
(ChurchPresenter/STT#13) has the worker wedging at interpreter startup, before a
line of Python runs, apparently behind the server's own outstanding read on that
same pipe.

That mechanism is not settled — the server-to-worker hop uses
``multiprocessing.popen_spawn_win32``, which passes ``bInheritHandles=False``, so
how the handle reaches the child is unexplained. This module is worth having
either way: the shutdown channel keeps working, and children get ``NUL`` for
standard input, which is what a service's children should have.

**The channel matters more than the detachment.** Losing graceful shutdown means
every stop becomes a kill, with mid-write cuts to the transcription database and
backup files — far worse than an inherited handle. So every failure here falls
back to reading the pipe where it is, and the caller is told which happened.

Windows needs two moves rather than one: the CRT's fd 0 and the Win32
``STD_INPUT_HANDLE`` that children actually inherit are separate, and ``dup2``
updates only the former.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, NamedTuple, Optional, TextIO

#: ``GetStdHandle``/``SetStdHandle`` identifier for standard input.
STD_INPUT_HANDLE = -10


class DetachResult(NamedTuple):
    """What became of the shutdown channel.

    ``stream`` is always readable — the point of the exercise is the channel, so a
    failed detach still returns something to read from. ``detached`` says whether
    fd 0 was actually freed, and ``error`` explains why not, for the log.
    """

    stream: TextIO
    detached: bool
    error: Optional[str] = None


#: Sentinel for "use sys.stdin", so an explicit ``None`` stays distinguishable
#: from an omitted argument — a caller whose stdin really is None (a GUI build,
#: pythonw) must not be silently redirected to the interpreter's.
_USE_SYS_STDIN: Any = object()


def detach_stdin(
    stream: Any = _USE_SYS_STDIN,
    *,
    platform: str = sys.platform,
    set_std_handle: Optional[Callable[[int, int], Any]] = None,
    get_osfhandle: Optional[Callable[[int], int]] = None,
) -> DetachResult:
    """Move the shutdown pipe to a private fd and point fd 0 at the null device.

    Returns the private stream to read the pipe from. The original ``stream`` is
    deliberately **not** closed: closing it would close the underlying fd, and the
    duplicate is what keeps the pipe open.

    ``platform``, ``set_std_handle`` and ``get_osfhandle`` are injected so the
    Windows branch can be exercised on any platform; the defaults resolve
    ``kernel32`` and ``msvcrt`` lazily, and only when actually running on Windows.
    A branch that can only be tested where it runs is a branch that ships untested,
    and this one has no second chance: it is reached once, at startup, on the
    machines we cannot debug.
    """
    original = sys.stdin if stream is _USE_SYS_STDIN else stream
    if original is None:
        return DetachResult(_null_stream(), True, "stdin was None")

    try:
        fd = original.fileno()
    except (AttributeError, OSError, ValueError) as exc:
        # A stream with no file descriptor (pytest's capture, a StringIO) has
        # nothing to detach and nothing to inherit. Read it where it is.
        return DetachResult(original, False, f"stdin has no file descriptor: {exc}")

    if fd != 0:
        # Already off the slot — nothing to do, and moving it would be wrong.
        return DetachResult(original, True)

    private_fd = None
    null_fd = None
    try:
        private_fd = os.dup(fd)
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)
        null_fd = None
        if platform == "win32":
            _point_win32_handle_at_fd0(set_std_handle, get_osfhandle)
        return DetachResult(
            os.fdopen(private_fd, "r", encoding="utf-8", errors="replace"), True)
    except Exception as exc:  # noqa: BLE001 - the channel outranks the detachment
        # Put fd 0 back first. If dup2 had already succeeded, fd 0 is the null
        # device and the caller's fallback would read EOF for ever — silently
        # losing the shutdown channel, which is the one outcome worse than not
        # detaching at all.
        if private_fd is not None:
            try:
                os.dup2(private_fd, 0)
            except OSError:
                pass
        for leaked in (null_fd, private_fd):
            if leaked is not None:
                try:
                    os.close(leaked)
                except OSError:
                    pass
        return DetachResult(original, False, str(exc))


def _point_win32_handle_at_fd0(
    set_std_handle: Optional[Callable[[int, int], Any]] = None,
    get_osfhandle: Optional[Callable[[int], int]] = None,
) -> None:
    """Make ``STD_INPUT_HANDLE`` agree with whatever fd 0 now refers to.

    Without this the CRT reads the null device while children still inherit the
    pipe, which is the half of the problem that actually matters.
    """
    if set_std_handle is None:  # pragma: no cover - needs Windows
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        set_std_handle = lambda which, handle: kernel32.SetStdHandle(  # noqa: E731
            which, ctypes.c_void_p(handle))
    if get_osfhandle is None:  # pragma: no cover - needs Windows
        import msvcrt  # type: ignore[import-not-found]

        get_osfhandle = msvcrt.get_osfhandle  # type: ignore[attr-defined]

    set_std_handle(STD_INPUT_HANDLE, get_osfhandle(0))


def _null_stream() -> TextIO:
    """A readable stream at EOF, for when there is no stdin to read."""
    return open(os.devnull, encoding="utf-8")


def is_shutdown_request(line: str) -> bool:
    """Whether a line from the channel asks the server to stop.

    EOF is deliberately not a shutdown request, and is handled by the caller
    running out of lines: a dead watchdog must not stop the service, because its
    replacement re-attaches.
    """
    return line.strip() == "shutdown"

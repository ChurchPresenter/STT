"""Detaching the watchdog's shutdown pipe from the stdin slot.

The channel is the thing that matters: without it every stop on Windows becomes a
TerminateProcess, with mid-write cuts to the transcription database. So the rule
these enforce is that a failure to detach never costs us the channel — it falls
back to reading the pipe where it is.
"""

from __future__ import annotations

import io
import os

import pytest

from stt import shutdown_channel


@pytest.fixture
def pipe_on_fd0():
    """Put a real pipe on fd 0, as the watchdog does, and yield the writer.

    The real ``os.dup2`` is captured up front: one test replaces it to force a
    mid-swap failure, and teardown must still be able to put fd 0 back.
    """
    real_dup2 = os.dup2
    read_fd, write_fd = os.pipe()
    saved = os.dup(0)
    real_dup2(read_fd, 0)
    os.close(read_fd)
    writer = os.fdopen(write_fd, "w")
    try:
        yield writer
    finally:
        try:
            writer.close()
        except OSError:
            pass
        real_dup2(saved, 0)
        os.close(saved)


# --- the detach itself -----------------------------------------------------


def test_the_pipe_is_still_readable_after_being_moved(pipe_on_fd0):
    writer = pipe_on_fd0
    result = shutdown_channel.detach_stdin(os.fdopen(0, "r", closefd=False))

    assert result.detached is True
    writer.write("shutdown\n")
    writer.flush()
    assert result.stream.readline().strip() == "shutdown"


def test_fd_zero_becomes_the_null_device(pipe_on_fd0):
    """What children inherit is the whole point of the exercise."""
    writer = pipe_on_fd0
    # Held: this owns the only remaining read end of the pipe, and letting it be
    # collected would close it and break the writer below.
    result = shutdown_channel.detach_stdin(os.fdopen(0, "r", closefd=False))
    assert result.detached is True

    writer.write("shutdown\n")
    writer.flush()
    with os.fdopen(os.dup(0), "r") as inherited:
        assert inherited.read() == "", "fd 0 must be at EOF, not holding the pipe"


def test_the_moved_pipe_is_off_fd_zero(pipe_on_fd0):
    result = shutdown_channel.detach_stdin(os.fdopen(0, "r", closefd=False))
    assert result.stream.fileno() != 0


# --- the channel outranks the detachment -----------------------------------


def test_a_stream_with_no_descriptor_is_read_where_it_is():
    """pytest's own capture has no fileno; losing shutdown would be worse."""
    original = io.StringIO("shutdown\n")
    result = shutdown_channel.detach_stdin(original)

    assert result.detached is False
    assert result.error is not None
    assert result.stream is original, "the channel must survive a failed detach"


def test_an_explicit_none_stdin_is_not_silently_swapped_for_the_interpreters():
    """A GUI build (pythonw) really can have sys.stdin be None."""
    result = shutdown_channel.detach_stdin(None)
    assert result.stream.read() == ""
    result.stream.close()


def test_a_stream_already_off_fd_zero_is_left_alone(tmp_path):
    path = tmp_path / "chan"
    path.write_text("shutdown\n", encoding="utf-8")
    with open(path, encoding="utf-8") as handle:
        result = shutdown_channel.detach_stdin(handle)
        assert result.stream is handle
        assert result.detached is True


# --- the Windows half, exercised off Windows -------------------------------
#
# dup2 moves the CRT's fd 0; children inherit the Win32 STD_INPUT_HANDLE, which
# is a separate thing. Missing the second call would leave the CRT reading NUL
# while children still got the pipe — the half that actually matters, and the
# half no test on this platform would otherwise reach.


def test_windows_also_repoints_the_inherited_handle(pipe_on_fd0):
    calls = []

    shutdown_channel.detach_stdin(
        os.fdopen(0, "r", closefd=False),
        platform="win32",
        set_std_handle=lambda which, handle: calls.append((which, handle)),
        get_osfhandle=lambda fd: 4242 + fd,
    )

    assert calls == [(shutdown_channel.STD_INPUT_HANDLE, 4242)], (
        "children inherit STD_INPUT_HANDLE, which dup2 does not touch"
    )


def test_other_platforms_do_not_call_the_windows_api(pipe_on_fd0):
    calls = []

    shutdown_channel.detach_stdin(
        os.fdopen(0, "r", closefd=False), platform="linux",
        set_std_handle=lambda *a: calls.append(a))

    assert calls == []


def test_a_failing_windows_call_falls_back_to_reading_in_place(pipe_on_fd0):
    writer = pipe_on_fd0

    def boom(*_a):
        raise OSError("SetStdHandle failed")

    original = os.fdopen(0, "r", closefd=False)
    result = shutdown_channel.detach_stdin(
        original, platform="win32", set_std_handle=boom, get_osfhandle=lambda fd: 1)

    assert result.detached is False
    assert result.stream is original
    # The dangerous case: dup2 had already succeeded, so fd 0 was the null device
    # when the Windows call blew up. Unless it is put back, this "fallback" reads
    # EOF for ever and the shutdown channel is silently gone.
    writer.write("shutdown\n")
    writer.flush()
    assert result.stream.readline().strip() == "shutdown"


# --- what counts as a shutdown --------------------------------------------


@pytest.mark.parametrize("line", ["shutdown", "shutdown\n", "  shutdown  \n"])
def test_a_shutdown_request_is_recognised(line):
    assert shutdown_channel.is_shutdown_request(line) is True


@pytest.mark.parametrize("line", ["", "\n", "restart\n", "shut down\n", "SHUTDOWN\n"])
def test_anything_else_is_not(line):
    """EOF in particular: a dead watchdog must not stop the service."""
    assert shutdown_channel.is_shutdown_request(line) is False

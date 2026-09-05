"""Killing a process tree on Windows, the way ``killpg`` does on POSIX.

``ProcessManager.start()`` passes ``start_new_session=not IS_WINDOWS`` so that
``stop()`` can sweep the server and everything it spawned — the multiprocessing
workers and their resource tracker — with one ``killpg``. The comment there says
plainly that this is a no-op on Windows, and the consequence was measured in the
field (ChurchPresenter/STT#13): four orphaned manager and worker processes after a
handful of restarts, holding ~1.1 GB between them, one still with a Whisper model
resident. They survive because a Windows child has no parent-linked lifetime —
terminating the server leaves its children attached to nothing.

The Windows primitive for this is a **Job Object**. A process assigned to a job
with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` dies when the last handle to that job
closes, and so does everything it spawns, because children inherit job membership.
Holding the handle for as long as the server should live, and closing it to stop,
gives the same guarantee ``start_new_session`` + ``killpg`` gives on POSIX.

The Win32 calls are injected rather than imported at module scope, so the sequence
and its failure handling can be tested away from Windows. That matters here more
than usual: this code runs once, at startup, on machines we cannot attach a
debugger to, and every branch in it is a branch that only ever executes there.

**Failing is allowed.** A job cannot be created inside some sandboxes, and a
process already in a job that forbids breakaway cannot be assigned to another. In
every such case the caller keeps the behaviour it has today — a server that runs,
and children that may outlive it — rather than a server that will not start.
"""

from __future__ import annotations

import sys
from typing import NamedTuple, Optional, Protocol

#: ``JOBOBJECTINFOCLASS`` value for extended limit information.
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

#: Kill every process in the job when the last handle to it is closed.
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

#: Rights needed to put an already-running process into a job.
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001


class JobApi(Protocol):
    """The four Win32 calls this needs, so a test can supply its own."""

    def create_job(self) -> int: ...
    def set_kill_on_close(self, job: int) -> bool: ...
    def open_process(self, pid: int) -> int: ...
    def assign(self, job: int, process: int) -> bool: ...
    def close(self, handle: int) -> None: ...


class JobResult(NamedTuple):
    """Whether the tree is now bound to a job, and why not when it is not.

    ``handle`` must be kept by the caller for as long as the tree should live:
    letting it be garbage collected is what kills the processes, which is the
    whole mechanism and also the easiest way to shoot yourself.
    """

    handle: Optional[int]
    bound: bool
    error: Optional[str] = None


def bind_process_tree(
    pid: int,
    *,
    platform: str = sys.platform,
    api: Optional[JobApi] = None,
) -> JobResult:
    """Put ``pid`` (and everything it later spawns) into a kill-on-close job.

    Returns the job handle to hold. On any failure the result is unbound and the
    caller carries on exactly as before — orphaned children are a leak, refusing
    to start the server is an outage.
    """
    if platform != "win32":
        return JobResult(None, False, "not Windows")

    calls = api if api is not None else _ctypes_api()
    job = None
    process = None
    try:
        job = calls.create_job()
        if not job:
            return JobResult(None, False, "CreateJobObject failed")
        if not calls.set_kill_on_close(job):
            calls.close(job)
            return JobResult(None, False, "SetInformationJobObject failed")
        process = calls.open_process(pid)
        if not process:
            calls.close(job)
            return JobResult(None, False, f"OpenProcess failed for pid {pid}")
        if not calls.assign(job, process):
            calls.close(process)
            calls.close(job)
            return JobResult(None, False, "AssignProcessToJobObject failed")
        calls.close(process)
        return JobResult(job, True)
    except Exception as exc:  # noqa: BLE001 - a leak beats a failed start
        for leaked in (process, job):
            if leaked:
                try:
                    calls.close(leaked)
                except Exception:
                    pass
        return JobResult(None, False, str(exc))


def release(result: JobResult, *, api: Optional[JobApi] = None) -> bool:
    """Close the job handle, killing every process still in it.

    Safe to call on an unbound result, so the caller needs no second condition
    around its own teardown.
    """
    if not result.bound or result.handle is None:
        return False
    calls = api if api is not None else _ctypes_api()
    try:
        calls.close(result.handle)
        return True
    except Exception:  # noqa: BLE001 - teardown must not raise
        return False


def _ctypes_api() -> JobApi:  # pragma: no cover - needs Windows
    """The real Win32 calls, resolved lazily so importing this is safe anywhere."""
    import ctypes
    from ctypes import wintypes  # type: ignore[attr-defined]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _Api:
        def create_job(self) -> int:
            return int(kernel32.CreateJobObjectW(None, None) or 0)

        def set_kill_on_close(self, job: int) -> bool:
            info = _ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            return bool(kernel32.SetInformationJobObject(
                wintypes.HANDLE(job), JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)))

        def open_process(self, pid: int) -> int:
            return int(kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid) or 0)

        def assign(self, job: int, process: int) -> bool:
            return bool(kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(job), wintypes.HANDLE(process)))

        def close(self, handle: int) -> None:
            kernel32.CloseHandle(wintypes.HANDLE(handle))

    return _Api()

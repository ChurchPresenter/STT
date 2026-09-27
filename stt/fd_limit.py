"""Raising the open-file limit a launchd agent starts with.

A LaunchAgent with no ``SoftResourceLimits`` inherits launchd's default soft
``maxfiles`` of **256**, and the watchdog, the server and its multiprocessing
worker all run under it. That budget is shared between pipes, POSIX semaphores,
the Manager's per-thread sockets, every open session database with its WAL and
SHM files, and each HTTP connection — and it ran out in the field (STT-2D,
STT-2E): ``/api/translation/status`` failed opening the Manager proxy socket and
werkzeug failed creating a selector, both with ``EMFILE``.

The hard limit is normally far higher, and an unprivileged process may raise its
soft limit up to it, so the fix is one ``setrlimit`` at startup. Children inherit
it, so calling this in the watchdog covers the server and worker too; the server
calls it again for launches that do not go through the watchdog.

macOS refuses a soft value above ``OPEN_MAX`` (10240) with ``EINVAL`` even when
the hard limit reports ``RLIM_INFINITY``, hence the default target.

**Failing is allowed.** A limit we cannot raise leaves the process exactly as it
was; it must never stop the server starting.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

#: macOS OPEN_MAX: the largest soft limit the kernel accepts from setrlimit.
DEFAULT_TARGET = 10240


def _load_resource() -> Any:
    try:
        import resource
    except ImportError:  # Windows
        return None
    return resource


def raise_nofile_limit(target: int = DEFAULT_TARGET, *, resource_mod: Any = None) -> Optional[Tuple[int, int]]:
    """Raise the soft ``RLIMIT_NOFILE`` towards ``target``; never lower it.

    Returns ``(before, after)`` soft limits, or ``None`` when nothing could be
    read (no ``resource`` module, or getrlimit failed). ``before == after`` means
    the limit was already high enough or could not be raised.
    """
    res = resource_mod if resource_mod is not None else _load_resource()
    if res is None:
        return None
    try:
        soft, hard = res.getrlimit(res.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None

    infinity = res.RLIM_INFINITY
    want = target if hard == infinity else min(hard, target)
    if soft == infinity or soft >= want:
        return soft, soft

    # One retry at the kernel's historical ceiling: some kernels cap the soft
    # limit below what the hard limit advertises.
    for candidate in (want, min(want, 4096)):
        try:
            res.setrlimit(res.RLIMIT_NOFILE, (candidate, hard))
        except (OSError, ValueError):
            continue
        return soft, candidate
    return soft, soft


def current_nofile_limit(*, resource_mod: Any = None) -> Optional[int]:
    """The soft ``RLIMIT_NOFILE``, or ``None`` where it cannot be read."""
    res = resource_mod if resource_mod is not None else _load_resource()
    if res is None:
        return None
    try:
        soft = res.getrlimit(res.RLIMIT_NOFILE)[0]
    except (OSError, ValueError):
        return None
    return None if soft == res.RLIM_INFINITY else int(soft)


def open_fd_count(fd_dir: Optional[str] = None) -> Optional[int]:
    """How many descriptors this process holds, or ``None`` where unknowable.

    Reads ``/proc/self/fd`` (Linux) or ``/dev/fd`` (macOS). Listing the
    directory opens one descriptor itself, which appears in the listing, so the
    count is reduced by one.
    """
    candidates = [fd_dir] if fd_dir else ["/proc/self/fd", "/dev/fd"]
    for path in candidates:
        try:
            entries = os.listdir(path)
        except OSError:
            continue
        return max(len(entries) - 1, 0)
    return None

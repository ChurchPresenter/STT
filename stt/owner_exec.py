"""Run git and uv as the user who owns the checkout, even when the server runs as root.

Every Linux install made by install.sh before 2026-09-27 runs its service as
``User=root`` with the installing user's HOME, from a checkout that user owns. Two
things follow, and both are silent:

* git (2.35.2+) refuses to work in a repository owned by another user — "detected
  dubious ownership" — unless it was started through sudo, which a systemd unit is not.
  The in-app self-update then fails on every check and is only a log line.
* where git does proceed (an older git, or a ``safe.directory`` override), it writes
  root-owned refs and objects, and uv writes root-owned packages into ``.venv``. The
  user's own ``update_server.sh`` — which runs git and uv as the checkout's owner — then
  fails on them. .62 had 1,831 such files before it was cleaned up.

So when this process is root and the directory belongs to someone else, the command
runs as that someone. **If that fails, it runs again as root, exactly as before**: an
install whose update works today must not stop updating because of this module.

Everything here is a no-op off POSIX, for a non-root process, and for a root-owned
directory — which is every Windows and macOS install and every Linux install whose
service already runs as its user.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Sequence, Set

log = logging.getLogger(__name__)


class Owner(NamedTuple):
    uid: int
    gid: int
    name: str
    home: str


def _euid() -> int:
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else -1


def _lookup(uid: int) -> Optional[Owner]:
    try:
        import pwd
    except ImportError:  # Windows
        return None
    try:
        entry = pwd.getpwuid(uid)
    except KeyError:
        return None
    return Owner(entry.pw_uid, entry.pw_gid, entry.pw_name, entry.pw_dir)


def foreign_owner(path: str, *, euid: Optional[int] = None, via_parent: bool = False,
                  lookup: Callable[[int], Optional[Owner]] = _lookup) -> Optional[Owner]:
    """Who ``path`` belongs to, when that is somebody other than this root process.

    None means "run as you are": not root, not POSIX, the path is root's, or its owner
    has no passwd entry to take a home and groups from.

    ``via_parent`` is for a data dir. A root service with the user's HOME creates
    ``~/.stt`` itself, so the directory is root's while the home it sits in says whose
    it really is. Not used for the checkout: a checkout root cloned into a user's home is
    root's on purpose.
    """
    if (euid if euid is not None else _euid()) != 0:
        return None
    candidates = [path]
    if via_parent:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and parent != os.path.dirname(parent):  # never reach up to /
            candidates.append(parent)
    for candidate in candidates:
        try:
            uid = os.stat(candidate).st_uid
        except OSError:
            continue
        if uid != 0:
            return lookup(uid)
    return None


def owner_env(owner: Owner, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment a command sees when run as ``owner``: their HOME, user and logname."""
    env = dict(os.environ if base is None else base)
    env.update(HOME=owner.home, USER=owner.name, LOGNAME=owner.name)
    return env


def owner_popen_kwargs(owner: Owner, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """``subprocess`` arguments that drop a child to ``owner`` (POSIX, Python 3.9+)."""
    kwargs: Dict[str, Any] = {"user": owner.uid, "group": owner.gid,
                              "env": owner_env(owner, env)}
    getgrouplist = getattr(os, "getgrouplist", None)
    if getgrouplist is not None:
        try:
            kwargs["extra_groups"] = list(getgrouplist(owner.name, owner.gid))
        except OSError:
            pass
    return kwargs


_reclaimed: Set[str] = set()


def reclaim(path: str, owner: Owner, *, once: bool = True) -> int:
    """Give everything under ``path`` that is not ``owner``'s back to them. Returns the count.

    Root-created directories are the part that matters: a root-owned ``.git/objects/ab``
    stops the owner adding any object whose hash starts ``ab``. Symlinks are changed, not
    followed. Best-effort — a file that cannot be changed is left for the root fallback.
    ``once`` walks each path a single time per process, since ``_git`` runs many times
    per update.
    """
    key = os.path.abspath(path)
    if once and key in _reclaimed:
        return 0
    _reclaimed.add(key)
    if not os.path.lexists(path):
        return 0
    changed = 0

    def fix(p: str) -> None:
        nonlocal changed
        try:
            st = os.lstat(p)
            if st.st_uid != owner.uid:
                os.lchown(p, owner.uid, owner.gid)
                changed += 1
        except OSError:
            pass

    fix(path)
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            fix(os.path.join(root, name))
    if changed:
        log.info("[owner] gave %d root-owned entries under %s back to %s", changed, path, owner.name)
    return changed


def run(cmd: Sequence[str], *, owned_by: str, reclaim_dirs: Iterable[str] = (),
        runner: Optional[Callable[..., "subprocess.CompletedProcess[Any]"]] = None,
        owner: Optional[Owner] = None, **kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    """``subprocess.run(cmd, **kwargs)`` as the owner of ``owned_by``, falling back to root.

    With no foreign owner this is exactly ``runner(cmd, **kwargs)``. Otherwise
    ``reclaim_dirs`` are handed back first, the command runs as the owner, and a failure —
    a non-zero exit, or a child that cannot be started as them — runs it again unchanged
    as root. A timeout is not retried: waiting twice for a hung network is worse than
    reporting it.
    """
    # Looked up per call, not bound as a default, so a caller's monkeypatched
    # subprocess.run is the one that runs.
    runner = runner if runner is not None else subprocess.run
    who = owner if owner is not None else foreign_owner(owned_by)
    if who is None:
        return runner(list(cmd), **kwargs)
    for d in reclaim_dirs:
        reclaim(d, who)
    as_owner = dict(kwargs)
    as_owner.update(owner_popen_kwargs(who, kwargs.get("env")))
    first: Optional["subprocess.CompletedProcess[Any]"] = None
    try:
        first = runner(list(cmd), **as_owner)
    except subprocess.TimeoutExpired:
        raise
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        log.warning("[owner] could not run %s as %s (%s); running as root", cmd[0], who.name, e)
    if first is not None and first.returncode == 0:
        return first
    second = runner(list(cmd), **kwargs)
    if first is None:
        return second
    if second.returncode == 0:
        # Root succeeded where the owner could not: an ownership problem reclaim did not
        # reach. Warning, not error — the update worked, and this is the old behaviour.
        log.warning("[owner] %s failed as %s but succeeded as root; files it wrote are root's",
                    " ".join(str(c) for c in list(cmd)[:3]), who.name)
        return second
    # Both failed: the owner's error is the real one. Root's is usually just git refusing
    # a repository it does not own, which would hide why a pull was actually rejected.
    return first


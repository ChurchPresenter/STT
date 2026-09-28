"""How the server restarts itself: which script, which command line, which exit code.

The web UI's Restart button reaches ``perform_server_restart``. On Windows it looked for
``restart_server.bat`` in the data dir (``~\\.stt``) rather than the checkout, so it was
never found; the fallback then relaunched ``sys.argv`` — the relative
``speech_to_text.py`` — from that same data dir, which has no such file. The server
stopped and never came back. The POSIX branch had the same wrong directory and only
worked because its last resort, an in-place execv, did not care.

Under the watchdog a restart is the watchdog's job. It used to see the server exit 0,
read that as a deliberate stop, and leave it down. ``RESTART_EXIT_CODE`` asks it to start
the server again at once; a watchdog that predates the code reads it as a crash, which
still restarts the server — after a backoff, with a crash report — so no install is left
worse off than before.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence

#: EX_TEMPFAIL. Not 0 (a stop), not 1 (the generic crash code the server already uses).
RESTART_EXIT_CODE = 75


def is_watchdog_managed(env: Mapping[str, str]) -> bool:
    """Whether a watchdog started this server and so owns its restarts and updates.

    The watchdog sets ``STT_MANAGED=1`` and ``STT_DATA_DIR``; one from before 2026-07-16
    sets only the second, so ``STT_DATA_DIR`` alone still means managed. But the launch
    scripts pass ``STT_DATA_DIR`` too — a server they start as root must read the
    invoking user's data dir — and to them it must *not* mean managed: that switched off
    the server's own self-update and made Restart exit for a watchdog that was not there.
    So an explicit ``STT_MANAGED=0`` wins.
    """
    managed = env.get("STT_MANAGED")
    if managed is not None:
        return managed.strip() not in ("", "0")
    return bool(env.get("STT_DATA_DIR"))


def restart_script(bundle_dir: str, *, windows: bool) -> Optional[str]:
    """The restart script in the checkout, or None if this install has none.

    ``bundle_dir`` is where the code is — never the data dir, which is where the bug put
    it. A frozen build has no scripts, and None sends the caller to its fallback.
    """
    path = os.path.join(bundle_dir, "restart_server.bat" if windows else "restart_server.sh")
    return path if os.path.isfile(path) else None


def relaunch_argv(executable: str, script: str, argv: Sequence[str], *,
                  frozen: bool) -> List[str]:
    """The command line that starts this same server again, independent of the cwd.

    ``argv[0]`` is whatever the server was started with — often a bare
    ``speech_to_text.py`` — so it is replaced by the absolute ``script`` path, and the
    arguments after it are kept. A frozen build is its own executable.
    """
    rest = list(argv[1:])
    if frozen:
        return [executable, *rest]
    return [executable, os.path.abspath(script), *rest]

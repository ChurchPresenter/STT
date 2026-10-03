"""Which config the server reads, and which port that config binds.

The launch scripts print "Server started on port N" and kill whatever holds port N, so
they must read the same config.json the server does. Each used to answer that with its own
one-liner, ``os.path.expanduser('~')/.stt``, which is right for the user who runs the
server and wrong under sudo: ``restart_server.sh`` refuses to run on Linux unless it is
root, so ``~`` was ``/root``, the file did not exist, and a box serving on port 80 was
reported — and port-killed — as 8080.

Stdlib only: the scripts run this with whatever Python they find, before a venv may exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Mapping, Optional, Sequence

from stt.coercion import coerce_int

#: What config.default.json ships. The server patches missing keys in from that template
#: on every load, so this — not 80 — is the port a config without one ends up bound to.
DEFAULT_PORT = 8080


def _home_of(user: str) -> Optional[str]:
    if sys.platform == "win32":  # no sudo, no pwd
        return None
    import pwd
    try:
        return pwd.getpwnam(user).pw_dir
    except KeyError:
        return None


def _euid() -> int:
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else -1


def data_dir(env: Mapping[str, str], *, euid: int, home: str,
             home_of: Callable[[str], Optional[str]] = _home_of) -> str:
    """The data dir the server uses: STT_DATA_DIR, else the invoking user's ~/.stt.

    "Invoking user" is the point. Under sudo the process is root and ``home`` is root's,
    but the server being restarted belongs to whoever typed sudo — the systemd unit runs
    as that user, and it is their config the operator edited.
    """
    override = env.get("STT_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    sudo_user = env.get("SUDO_USER")
    if euid == 0 and sudo_user and sudo_user != "root":
        user_home = home_of(sudo_user)
        if user_home:
            return os.path.join(user_home, ".stt")
    return os.path.join(home, ".stt")


def configured_port(data_dir_path: str) -> int:
    """``web_server.port`` from the live config, or DEFAULT_PORT if it cannot be read."""
    path = os.path.join(data_dir_path, "config", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return DEFAULT_PORT
    web = cfg.get("web_server") if isinstance(cfg, dict) else None
    raw = web.get("port") if isinstance(web, dict) else None
    port = coerce_int(raw, DEFAULT_PORT)
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", action="store_true",
                        help="print the resolved data dir instead of the port")
    args = parser.parse_args(argv)
    directory = data_dir(os.environ, euid=_euid(), home=os.path.expanduser("~"))
    print(directory if args.data_dir else configured_port(directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())

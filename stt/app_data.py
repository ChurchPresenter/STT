"""One-time relocation of the app data directory (APP_DIR).

Historically a run-from-repo server kept its data (config, models, per-session
databases, logs) inside the checkout directory, while a frozen install kept it
under ``~/.stt``. Unifying every non-override run on ``~/.stt`` means existing
installs have data stranded in the old repo location. This module copies that
data across once so nothing is lost on upgrade.

Copy, never move: the operation must be safe on a live production box, so the
originals are always left in place as a recoverable backup. Idempotent via a
marker file the caller writes after a successful pass. Stdlib-only, paths passed
in explicitly, so it imports clean and is unit-testable without the monolith.
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, Iterable, List, Tuple

# Top-level entries under the old APP_DIR that hold user data worth carrying
# over. Caches (models/.hf_cache, models/tts) ride along inside models/.
DEFAULT_MIGRATION_ITEMS = (
    "config",
    "models",
    "_AUTOMATIC_BACKUP",
    "logs",
    "panns_data",
    "download_progress.json",
    "server.log",
)


def migrate_app_data(
    old_root: str,
    new_root: str,
    items: Iterable[str] = DEFAULT_MIGRATION_ITEMS,
    *,
    log: Callable[[str], None] = print,
) -> List[Tuple[str, str]]:
    """Copy each named item from ``old_root`` into ``new_root`` if not present.

    Returns a list of ``(name, status)`` where status is one of ``"copied"``,
    ``"skipped"`` (source missing, or target already exists), or
    ``"error: <msg>"``. Never deletes or overwrites anything in either root, so
    a partial or repeated run is safe. Directories are copied recursively; files
    are copied with metadata preserved.
    """
    results: List[Tuple[str, str]] = []
    for name in items:
        src = os.path.join(old_root, name)
        dst = os.path.join(new_root, name)
        if not os.path.exists(src) or os.path.exists(dst):
            results.append((name, "skipped"))
            continue
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst) or new_root, exist_ok=True)
                shutil.copy2(src, dst)
            log(f"migrated {name} -> {dst}")
            results.append((name, "copied"))
        except OSError as e:
            # e.g. a root-owned source unreadable by a non-root run; keep going
            # so the rest still migrates and the run stays non-fatal.
            log(f"could not migrate {name}: {e}")
            results.append((name, f"error: {e}"))
    return results

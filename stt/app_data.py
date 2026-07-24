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


def tree_stats(path: str) -> Tuple[int, int]:
    """Return ``(file_count, total_bytes)`` for a file or directory tree.

    Unreadable entries are silently skipped (counted as absent), so the result
    is a lower bound for a tree with permission holes.
    """
    if os.path.isfile(path):
        try:
            return 1, os.path.getsize(path)
        except OSError:
            return 0, 0
    count = 0
    size = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(root, f))
                count += 1
            except OSError:
                pass
    return count, size


def removable_originals(
    old_root: str,
    new_root: str,
    items: Iterable[str] = DEFAULT_MIGRATION_ITEMS,
) -> List[str]:
    """Names whose ``new_root`` copy is verified complete enough to delete the
    ``old_root`` original and reclaim its space.

    "Complete" means the target exists and its file count and total byte size
    are **at least** the source's (equal for a clean copy; the migration never
    shrinks data). Anything missing from either root, or whose target has fewer
    files or bytes than the source (a partial/failed copy), is omitted — the
    caller must never delete an original that has not been fully mirrored.
    """
    removable: List[str] = []
    for name in items:
        old = os.path.join(old_root, name)
        new = os.path.join(new_root, name)
        if not os.path.exists(old) or not os.path.exists(new):
            continue
        old_count, old_bytes = tree_stats(old)
        new_count, new_bytes = tree_stats(new)
        if new_count >= old_count and new_bytes >= old_bytes:
            removable.append(name)
    return removable

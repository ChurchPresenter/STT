"""Keep the last few versions of a small config file that gets overwritten.

The glossary is one textarea and one Save button, and saving writes the whole
file — so a save made with the box empty replaces the operator's terms with
nothing, in place, with no undo. That happened, and there was nothing anywhere
to restore from: no snapshot, no copy, no previous version. The content was
reconstructed from a default seed and an access log, which is luck rather than
recovery.

Config files here are a couple of hundred bytes and are written by hand a few
times a year, so keeping the last handful of versions costs nothing measurable
and turns that class of accident back into an inconvenience.

Deliberately not a general backup system: no compression, no manifest, no
restore command. A timestamped sibling that an operator can read and copy back
with the tools already on the machine is the whole point.

Stdlib-only, with injectable clock and directory listing so the rotation can be
tested without touching a real filesystem clock.
"""

import os
import shutil
import time
from typing import Callable, List, Optional

#: How many previous versions to keep. Enough to survive a bad save that is not
#: noticed until the next one, without turning the config directory into a list.
DEFAULT_KEEP = 5

#: Marks a file as one of ours, so rotation never considers anything else.
BACKUP_INFIX = ".backup-"


def backup_name(path: str, stamp: str) -> str:
    """Where the copy of ``path`` taken at ``stamp`` lives.

    A sibling rather than a subdirectory: an operator looking for the file they
    just destroyed finds it next to the one they were editing.
    """
    return f"{path}{BACKUP_INFIX}{stamp}"


def existing_backups(path: str, listdir: Optional[Callable[[str], List[str]]] = None) -> List[str]:
    """Every backup of ``path``, oldest first.

    The stamp sorts lexicographically in time order, so plain sorting is enough
    and no file has to be stat-ed to place it.
    """
    directory = os.path.dirname(path) or "."
    prefix = os.path.basename(path) + BACKUP_INFIX
    lister = listdir or os.listdir
    try:
        names = lister(directory)
    except OSError:
        return []
    return sorted(os.path.join(directory, n) for n in names if n.startswith(prefix))


def prune(path: str, keep: int = DEFAULT_KEEP,
          listdir: Optional[Callable[[str], List[str]]] = None) -> List[str]:
    """Delete all but the newest ``keep`` backups. Returns what was deleted.

    ``keep <= 0`` removes every backup, which is what "stop keeping these"
    should mean rather than a silent no-op.
    """
    backups = existing_backups(path, listdir=listdir)
    doomed = backups[:-keep] if keep > 0 else backups
    removed = []
    for old in doomed:
        try:
            os.remove(old)
            removed.append(old)
        except OSError:
            pass  # a backup we cannot delete is not a reason to fail the save
    return removed


def backup_file(path: str, keep: int = DEFAULT_KEEP,
                clock: Optional[Callable[[], float]] = None) -> Optional[str]:
    """Copy ``path`` aside before it is overwritten, then prune old copies.

    Returns the backup's path, or None when there was nothing to copy (a file
    being created for the first time) or the copy failed. Never raises: a save
    must not fail because its safety net could not be strung up.
    """
    if not os.path.exists(path):
        return None
    now = (clock or time.time)()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    destination = backup_name(path, stamp)
    if os.path.exists(destination):
        # Two saves inside one second share a stamp. Keep the first, which is
        # the one still holding the pre-mistake content — overwriting it would
        # lose exactly the version worth having when someone saves twice in a
        # panic.
        return destination
    try:
        shutil.copy2(path, destination)
    except OSError:
        return None
    prune(path, keep=keep)
    return destination

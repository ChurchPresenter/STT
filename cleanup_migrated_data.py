#!/usr/bin/env python3
"""Reclaim disk space after the one-time ~/.stt data migration.

speech_to_text.py COPIES a run-from-repo install's data into ~/.stt on first
start (config, models, session DBs, logs, ...), deliberately leaving the
originals behind as a recoverable backup. Once you've confirmed the new ~/.stt
location works, run this to delete the old copies in the checkout.

Safety:
  * Dry-run by default — it only prints what it *would* remove. Pass --confirm
    to actually delete.
  * It deletes an original ONLY when its ~/.stt copy is verified complete
    (same-or-more files and bytes). A partial/failed copy is never cleaned up.
  * It refuses to run unless the migration marker (~/.stt/.migrated_from_repo)
    is present, so it can't run before a migration happened. Override with
    --force only if you know the data is safely elsewhere.

Usage:
    python3 cleanup_migrated_data.py            # dry run (shows the plan)
    python3 cleanup_migrated_data.py --confirm  # actually delete the originals

Run it as the SAME user that owns the old data. Root-owned files need sudo:
    sudo python3 cleanup_migrated_data.py --confirm
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from stt.app_data import DEFAULT_MIGRATION_ITEMS, removable_originals, tree_stats

MARKER_NAME = ".migrated_from_repo"


def _human(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true",
                        help="actually delete the originals (default is a dry run)")
    parser.add_argument("--force", action="store_true",
                        help="proceed even if the migration marker is absent")
    parser.add_argument("--old-root", default=os.path.dirname(os.path.abspath(__file__)),
                        help="the old data dir to clean up (default: this checkout)")
    parser.add_argument("--new-root",
                        default=os.path.abspath(os.path.expanduser(
                            os.environ.get("STT_DATA_DIR")
                            or os.path.join("~", ".stt"))),
                        help="the migrated-to data dir (default: $STT_DATA_DIR or ~/.stt)")
    args = parser.parse_args(argv)

    old_root, new_root = args.old_root, args.new_root
    print(f"Old data dir (clean up): {old_root}")
    print(f"New data dir (keep)    : {new_root}")

    if os.path.abspath(old_root) == os.path.abspath(new_root):
        print("Old and new data dirs are the same — nothing to clean up.")
        return 0

    marker = os.path.join(new_root, MARKER_NAME)
    if not os.path.exists(marker) and not args.force:
        print(f"\nRefusing to run: migration marker not found at {marker}.\n"
              "The migration may not have happened yet. If you are certain the "
              "data is safely in the new location, re-run with --force.")
        return 1

    removable = removable_originals(old_root, new_root, DEFAULT_MIGRATION_ITEMS)
    if not removable:
        print("\nNothing to clean up: no originals have a verified complete copy "
              "in the new location.")
        return 0

    total = 0
    print("\nOriginals with a verified copy in the new location:")
    for name in removable:
        _count, nbytes = tree_stats(os.path.join(old_root, name))
        total += nbytes
        print(f"  {name:<24} {_human(nbytes):>10}")
    print(f"  {'-' * 34}\n  {'reclaimable':<24} {_human(total):>10}")

    if not args.confirm:
        print("\nDry run — nothing deleted. Re-run with --confirm to delete these.")
        return 0

    print("\nDeleting originals...")
    errors = 0
    for name in removable:
        path = os.path.join(old_root, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  removed {name}")
        except OSError as e:
            errors += 1
            print(f"  [ERROR] could not remove {name}: {e}")
    if errors:
        print(f"\n{errors} item(s) could not be removed (usually root-owned). "
              "Re-run with sudo:\n    sudo python3 cleanup_migrated_data.py --confirm")
        return 1
    print(f"\nDone — reclaimed ~{_human(total)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

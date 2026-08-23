"""
Local build script — wraps the PyInstaller specs for convenience.

Two targets:

  (default)  The thin STT bootstrapper (stt/watchdog.py only, ~10-20 MB). It
             provisions a local venv + downloads dependencies + models on first run;
             nothing heavy is bundled.

  --demo     The self-contained demo (~25 MB): the whole app, no ML libraries, no
             models, replaying a recorded service. Runs on a machine with no Python.

Usage (from anywhere):
    python packaging/build.py [--platform NAME]
    python packaging/build.py --demo --synthetic
    python packaging/build.py --demo --session /path/to/session.db

Output (in the repo root):
    dist/STT/            — tiny bootstrapper application directory (one-dir)
    dist/STT-Demo/       — self-contained demo (one-dir); "STT Demo.app" on macOS
"""

import os
import sys
import shutil
import platform
import subprocess
import argparse

# Everything (icons, PyInstaller build/dist dirs) is produced in the repo root,
# regardless of where this script is invoked from.
PACKAGING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGING_DIR)


def run(cmd):
    print(f"\n> {' '.join(cmd)}\n")
    ret = subprocess.call(cmd, cwd=ROOT)
    if ret != 0:
        print(f"ERROR: command exited with code {ret}")
        sys.exit(ret)


def get_platform_name():
    s = platform.system().lower()
    return "windows" if s == "windows" else "macos" if s == "darwin" else "linux"


def _resolve_demo_session(args, parser):
    """Which recording the demo bundles, refusing to guess."""
    if args.session and args.synthetic:
        parser.error("choose either --session or --synthetic, not both")
    if args.session:
        if not os.path.isfile(args.session):
            parser.error(f"--session does not exist: {args.session}")
        return os.path.abspath(args.session)
    if args.synthetic:
        target = os.path.join(ROOT, "build", "demo.db")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        run([sys.executable, os.path.join(ROOT, "scripts", "make_demo_session.py"),
             "--synthetic", "-o", target])
        return target
    existing = os.environ.get("STT_DEMO_DB", "").strip()
    if existing:
        return existing
    parser.error(
        "--demo needs a recording to bundle. Use --synthetic for a written service "
        "(safe to publish), or --session <db> for a specific one.")


SESSIONS_README = """\
Drop a session database here
============================

Any STT session .db placed in this folder is replayed by the demo instead of the
service it ships with. The newest file wins, matched on its filename date.

Session databases live under _AUTOMATIC_BACKUP in an STT install, named like
2026-08-02_090000.db.

A recording is verbatim speech from whoever was in the room. Think about that before
copying one onto a machine you do not control.
"""


def _write_sessions_readme(out_dir):
    sessions = os.path.join(out_dir, "sessions")
    os.makedirs(sessions, exist_ok=True)
    with open(os.path.join(sessions, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(SESSIONS_README)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=get_platform_name(),
                        help="Platform name for output directory label")
    parser.add_argument("--demo", action="store_true",
                        help="Build the self-contained demo instead of the bootstrapper")
    parser.add_argument("--session", metavar="DB",
                        help="--demo: the recorded service to bundle")
    parser.add_argument("--synthetic", action="store_true",
                        help="--demo: generate a written service and bundle that")
    args = parser.parse_args()

    if (args.session or args.synthetic) and not args.demo:
        parser.error("--session and --synthetic only apply to --demo")

    spec = "demo.spec" if args.demo else "watchdog.spec"
    out_name = "STT-Demo" if args.demo else "STT"

    if args.demo:
        os.environ["STT_DEMO_DB"] = _resolve_demo_session(args, parser)

    # Clean previous artefacts — only this target's, so building the demo does not
    # silently delete a bootstrapper someone is about to sign.
    shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)
    for name in (out_name, "STT Demo.app") if args.demo else (out_name,):
        shutil.rmtree(os.path.join(ROOT, "dist", name), ignore_errors=True)

    # Generate application icon (requires Pillow)
    run([sys.executable, os.path.join(PACKAGING_DIR, "make_icon.py")])

    run([sys.executable, "-m", "PyInstaller",
         os.path.join(PACKAGING_DIR, spec), "--noconfirm"])

    # Clean intermediate build dir
    shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)

    out = os.path.join(ROOT, "dist", out_name)
    print(f"\nBuild complete: {out}")
    exe = f"{out_name}.exe" if sys.platform == "win32" else out_name
    print(f"Run: {os.path.join(out, exe)}")
    if args.demo:
        _write_sessions_readme(out)
        print("\nDrop any STT session .db into the sessions/ folder beside the "
              "executable to replay that service instead.")


if __name__ == "__main__":
    main()

"""Optional runtime dependencies that the live config asks for but requirements.txt omits.

A few packages are deliberately left out of requirements.txt because most installs
will never use them and they are large: see the commented block at the end of that
file. ``llama-cpp-python`` is the current one — the in-process GGUF runtime behind
``live_translation.llm.provider = "local"``.

Leaving them to a manual ``pip install`` worked until the venv was rebuilt. Then the
package was gone, the config still asked for a local GGUF, and the translation path
degraded exactly as designed: ``get_local_llm()`` returned None and ``fallback:
"skip"`` handed each caption back untranslated, with HTTP 200. The paired machine
could not tell that apart from a working translation, so a service ran on passthrough
captions. Nothing was broken enough to notice.

The dependency sync in ``self_update`` cannot cover this. It is gated on the sha256
of requirements.txt and installs exactly what that file lists, so a package the file
deliberately omits is invisible to it no matter how often it runs.

What this module adds is the missing question: *does the venv have what the config is
asking it to do?* Asked at startup, when a restart is already happening anyway and an
install costs nothing that is not already being spent.

Stdlib-only and side-effect free at import. The pure decisions take config and probes
as parameters so they can be unit-tested without a venv; the CLI at the bottom is the
only part that touches the filesystem or runs uv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Sequence, Set

from stt.llm_translate import uses_local_llm

# The server runs windowless on Windows; without this the uv child would flash a
# console window. 0 off-Windows: safe to pass unconditionally.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Long enough for a source build on a slow box, bounded so a wedged install can
# never hold a restart open forever — a late server is recoverable, a hung one
# during a service is not.
DEFAULT_TIMEOUT = 900

# Specs that failed to install are recorded here so the next restart does not pay
# for the same failing build again. Inside .venv so it dies with the venv it
# describes, the same reasoning as .requirements-synced (see self_update).
SKIP_MARKER_NAME = ".optional-deps-skip"


class OptionalDep(NamedTuple):
    """A package requirements.txt omits, and the config that makes it necessary."""

    module: str                                  # import name, for probing
    spec: str                                    # requirement string, for uv
    setting: str                                 # the config that asks for it
    wanted: Callable[[Mapping[str, Any]], bool]  # given the whole config


def _wants_local_llm(config: Mapping[str, Any]) -> bool:
    """True when captions are meant to be translated by an in-process GGUF here.

    Delegates to the one definition of that question rather than re-reading
    provider == "local" — llm_translate.uses_local_llm documents what drifted the
    last time two copies of this test existed.
    """
    return uses_local_llm(config.get("live_translation") or {})


# The plain PyPI wheel is the right thing to install unattended on every platform:
# it is CPU-only except on macOS ARM, where it already includes Metal, and a caption
# is short enough for CPU to be viable (measured p50 19 output tokens). A CUDA build
# needs a specific extra index and is the installer's business, not a restart's.
OPTIONAL_DEPS: Sequence[OptionalDep] = (
    OptionalDep(
        module="llama_cpp",
        spec="llama-cpp-python>=0.3.0",
        setting='live_translation.llm.provider = "local"',
        wanted=_wants_local_llm,
    ),
)


def wanted_deps(config: Mapping[str, Any],
                deps: Sequence[OptionalDep] = OPTIONAL_DEPS) -> List[OptionalDep]:
    """The optional deps this config actually asks for.

    A config that offloads translation to a peer, or uses NMT, asks for none of
    them — the point is to install what this machine is configured to do, not
    every optional package that exists.
    """
    wanted = []
    for dep in deps:
        try:
            if dep.wanted(config):
                wanted.append(dep)
        except Exception:  # noqa: BLE001 - a malformed config must not block startup
            continue
    return wanted


def missing_deps(config: Mapping[str, Any],
                 is_installed: Callable[[str], bool],
                 deps: Sequence[OptionalDep] = OPTIONAL_DEPS) -> List[OptionalDep]:
    """Optional deps the config asks for that are not importable."""
    return [dep for dep in wanted_deps(config, deps) if not is_installed(dep.module)]


def parse_skip_marker(text: str) -> Set[str]:
    """Specs recorded as previously-failed. Blank lines and # comments ignored."""
    specs = set()
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            specs.add(stripped)
    return specs


def format_skip_marker(specs: Set[str]) -> str:
    """Serialize the failed-spec set, with a note for whoever finds the file."""
    header = ("# Optional dependencies whose automatic install failed.\n"
              "# Delete this file (or the offending line) to retry on next start.\n")
    return header + "".join(spec + "\n" for spec in sorted(specs))


def module_installed(name: str) -> bool:
    """Whether *name* is importable by the running interpreter.

    Probed rather than imported: importing llama_cpp loads a multi-hundred-MB
    native library, which is a lot to pay to answer a yes/no question during
    startup. Mirrors local_llm_available() in the monolith.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - a broken/partial install must read as absent
        return False


def venv_python(repo_dir: str) -> str:
    """Path to the venv interpreter, or '' when there is no venv to install into."""
    if os.name == "nt":
        candidate = os.path.join(repo_dir, ".venv", "Scripts", "python.exe")
    else:
        candidate = os.path.join(repo_dir, ".venv", "bin", "python3")
    return candidate if os.path.isfile(candidate) else ""


def data_dir() -> str:
    """The live data directory — where config.json actually lives.

    Not the checkout: config/ there holds only the shipped template. Same
    resolution the start/restart scripts use for the port.
    """
    return os.environ.get("STT_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".stt")


def read_config(directory: str) -> dict:
    """The live config, or {} when it cannot be read.

    An unreadable config means "ask for nothing" rather than an error: this runs
    before the server starts, and a fresh install has no config yet.
    """
    try:
        with open(os.path.join(directory, "config", "config.json"), encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def install_command(uv: str, python_bin: str, spec: str) -> List[str]:
    """The uv invocation that installs *spec* into *python_bin*'s environment.

    ``--python`` is explicit for the same reason update_server.sh passes it: bare
    ``uv`` commands can decide to build a project environment of their own, and
    the one that must be modified is the venv the server will be launched from.
    """
    return [uv, "pip", "install", "--python", python_bin, spec]


def ensure(repo_dir: str,
           directory: Optional[str] = None,
           timeout: int = DEFAULT_TIMEOUT,
           echo: Callable[[str], None] = print) -> int:
    """Install optional deps the live config asks for and the venv lacks.

    Returns the number installed. Best-effort throughout: every failure path
    reports the manual command and returns, because a missing optional package
    degrades one feature while a start script that refuses to start does not.
    """
    from stt.self_update import find_uv  # local: keeps import-time cost off startup

    config = read_config(directory if directory is not None else data_dir())
    missing = missing_deps(config, module_installed)
    if not missing:
        return 0

    python_bin = venv_python(repo_dir)
    if not python_bin:
        echo("[DEPS] No .venv found; skipping optional dependency check.")
        return 0

    marker = os.path.join(repo_dir, ".venv", SKIP_MARKER_NAME)
    try:
        with open(marker, encoding="utf-8") as f:
            failed = parse_skip_marker(f.read())
    except OSError:
        failed = set()

    uv = find_uv(repo_dir)
    installed = 0
    for dep in missing:
        manual = " ".join(install_command(uv or "uv", python_bin, dep.spec))
        if dep.spec in failed:
            echo(f"[DEPS] {dep.spec} is missing and a previous install failed; "
                 f"not retrying automatically. To retry: rm {marker} && {manual}")
            continue
        if not uv:
            echo(f"[DEPS] {dep.spec} is required by {dep.setting} but uv was not found; "
                 f"install it manually: {manual}")
            continue
        echo(f"[DEPS] {dep.setting} needs {dep.spec}, which is not installed. Installing...")
        try:
            r = subprocess.run(install_command(uv, python_bin, dep.spec),
                               cwd=repo_dir, capture_output=True, text=True,
                               timeout=timeout, check=False,
                               creationflags=_CREATE_NO_WINDOW)
            ok = r.returncode == 0
            detail = (r.stderr or r.stdout or "").strip().splitlines()
        except subprocess.TimeoutExpired:
            ok, detail = False, [f"timed out after {timeout}s"]
        except Exception as e:  # noqa: BLE001 - never let this stop the server starting
            ok, detail = False, [f"{type(e).__name__}: {e}"]

        if ok:
            installed += 1
            failed.discard(dep.spec)
            echo(f"[DEPS] Installed {dep.spec}.")
        else:
            failed.add(dep.spec)
            echo(f"[DEPS] Could not install {dep.spec}: {detail[-1] if detail else 'unknown error'}")
            echo(f"[DEPS] Starting without it — {dep.setting} will fall back. "
                 f"To retry manually: {manual}")

    try:
        if failed:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(format_skip_marker(failed))
        elif os.path.exists(marker):
            os.unlink(marker)
    except OSError:
        pass  # the marker is an optimisation; losing it only costs a retry

    return installed


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for the start/restart scripts.

    Always exits 0 in --ensure mode: this runs on the startup path, and a
    dependency problem must never be the reason the server does not come up.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--timeout", type=int,
                        default=int(os.environ.get("STT_OPTIONAL_DEPS_TIMEOUT") or DEFAULT_TIMEOUT))
    parser.add_argument("--check", action="store_true",
                        help="report what is missing without installing (exit 1 if any)")
    args = parser.parse_args(argv)

    if args.check:
        config = read_config(args.data_dir if args.data_dir is not None else data_dir())
        missing = missing_deps(config, module_installed)
        for dep in missing:
            print(f"{dep.spec}\t{dep.setting}")
        return 1 if missing else 0

    try:
        ensure(args.repo_dir, args.data_dir, timeout=args.timeout)
    except Exception as e:  # noqa: BLE001 - belt and braces on the startup path
        print(f"[DEPS] Optional dependency check failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

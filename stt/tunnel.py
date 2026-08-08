"""Cloudflare quick-tunnel process management.

Gives the operator a public URL for this server without a Cloudflare account:
``cloudflared tunnel --url http://127.0.0.1:<port>`` prints an ephemeral
``*.trycloudflare.com`` address, which we scrape from its log stream.

The tunnel is started only on an explicit press and stops itself a configurable
delay after transcription ends, so the public window is bounded by the service
rather than left open indefinitely.

**The URL is the only secret.** Requests arrive at the server from 127.0.0.1,
which ``check_ip_whitelist`` treats as trusted, so anyone holding the URL has
the same access as someone sitting at this machine. That is a deliberate
choice by the operator; the short auto-stop window is what limits it.

Stdlib-only, as required of every ``stt/`` logic module. All IO is injected
(the spawn callable, the clock) so the lifecycle is testable without launching
a real tunnel.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

#: cloudflared prints the address inside an ASCII box, e.g.
#: ``INF |  https://sox-commission-fun-dylan.trycloudflare.com  |``.
#: Anchored to the trycloudflare host on purpose: the startup banner also
#: contains cloudflare.com and developers.cloudflare.com links, and a looser
#: "first https URL" match would happily return one of those.
_QUICK_TUNNEL_URL = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com", re.IGNORECASE)

#: How many recent cloudflared log lines to keep for the UI's error display.
_LOG_TAIL = 40

STATUS_STOPPED = "stopped"
STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"

DEFAULT_AUTO_STOP_SECONDS = 60


def parse_quick_tunnel_url(line: str) -> Optional[str]:
    """The trycloudflare address in a cloudflared log line, or None.

    Returns None for the banner lines that mention other cloudflare.com URLs.
    """
    match = _QUICK_TUNNEL_URL.search(line)
    return match.group(0) if match else None


#: Where cloudflared lands when installed by the usual package managers. The
#: server is started by a supervisor (launchd, systemd, the watchdog) whose PATH
#: is far shorter than a login shell's, so "it works in my terminal" is not
#: enough — /opt/homebrew/bin in particular is absent from launchd's default.
_BINARY_CANDIDATES = (
    "/opt/homebrew/bin/cloudflared",   # macOS, Apple silicon Homebrew
    "/usr/local/bin/cloudflared",      # macOS Intel Homebrew, manual installs
    "/usr/bin/cloudflared",            # Linux .deb / .rpm
    "/snap/bin/cloudflared",           # Linux snap
    r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    r"C:\Program Files\cloudflared\cloudflared.exe",
)


def resolve_binary(
    configured: str,
    candidates: Any = _BINARY_CANDIDATES,
    managed_dir: Optional[str] = None,
) -> Optional[str]:
    """An executable path for cloudflared, or None if it cannot be found.

    An explicit path in config is honoured as given. A bare name is looked up in
    the copy we manage ourselves first, then on PATH, then in the known install
    locations — a supervisor-started server inherits a minimal PATH, so PATH
    alone finds nothing on a machine where the operator installed it with
    Homebrew.

    Ours wins over a system copy so a box that downloaded a working binary keeps
    using it, rather than silently switching to whatever a later apt or brew
    install dropped on PATH.
    """
    configured = (configured or "cloudflared").strip()

    if os.path.sep in configured or (os.path.altsep and os.path.altsep in configured):
        return configured if _is_executable(configured) else None

    if managed_dir:
        managed = managed_binary_path(managed_dir)
        if _is_executable(managed):
            return managed

    found = shutil.which(configured)
    if found:
        return found

    for candidate in candidates:
        if _is_executable(candidate):
            return candidate
    return None


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


# ─── self-install ───────────────────────────────────────────────────────────
#
# cloudflared is a system binary, not a pip package, so the server's dependency
# sync (which only runs `uv pip install -r requirements.txt`) can never bring it
# in. Rather than make every box need a manual apt/brew step, fetch the official
# release build on first use.
#
# Cloudflare publishes no checksum files alongside these assets, so HTTPS to
# github.com is the whole trust anchor. That is the same basis as `brew install`
# and the .deb repo, but it is worth being explicit about.

CLOUDFLARED_RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"

#: (system, machine) -> release asset name. Machine names are what
#: platform.machine() reports, which differs per OS for the same silicon.
_ASSETS = {
    ("linux", "x86_64"): "cloudflared-linux-amd64",
    ("linux", "amd64"): "cloudflared-linux-amd64",
    ("linux", "aarch64"): "cloudflared-linux-arm64",
    ("linux", "arm64"): "cloudflared-linux-arm64",
    ("linux", "armv7l"): "cloudflared-linux-armhf",
    ("linux", "armv6l"): "cloudflared-linux-arm",
    ("linux", "i386"): "cloudflared-linux-386",
    ("linux", "i686"): "cloudflared-linux-386",
    ("darwin", "x86_64"): "cloudflared-darwin-amd64.tgz",
    ("darwin", "arm64"): "cloudflared-darwin-arm64.tgz",
    ("windows", "amd64"): "cloudflared-windows-amd64.exe",
    ("windows", "x86_64"): "cloudflared-windows-amd64.exe",
    ("windows", "i386"): "cloudflared-windows-386.exe",
    ("windows", "x86"): "cloudflared-windows-386.exe",
}


def cloudflared_asset(system: str, machine: str) -> Optional[str]:
    """Release asset for a platform, or None where Cloudflare ships no build."""
    return _ASSETS.get((system.strip().lower(), machine.strip().lower()))


def cloudflared_download_url(asset: str) -> str:
    """Download URL for a release asset, always the latest published build."""
    return f"{CLOUDFLARED_RELEASE_BASE}/{asset}"


def managed_binary_path(bin_dir: str, system: Optional[str] = None) -> str:
    """Where our own copy of cloudflared lives."""
    name = "cloudflared.exe" if (system or _current_system()) == "windows" else "cloudflared"
    return os.path.join(bin_dir, name)


def _current_system() -> str:
    import platform

    return platform.system().strip().lower()


def _current_machine() -> str:
    import platform

    return platform.machine().strip().lower()


def install_cloudflared(
    bin_dir: str,
    fetch: Callable[[str, str], Any],
    system: Optional[str] = None,
    machine: Optional[str] = None,
) -> str:
    """Download the official cloudflared build into ``bin_dir`` and return its path.

    ``fetch(url, dest)`` does the transfer (injected so this is testable, and so
    the caller can reuse the retrying downloader the rest of the app uses).
    macOS ships a .tgz rather than a bare executable, so that one is unpacked.

    Raises RuntimeError with an actionable message on any failure — the caller
    turns it into the error shown on the settings page.
    """
    system = (system or _current_system()).lower()
    machine = (machine or _current_machine()).lower()

    asset = cloudflared_asset(system, machine)
    if asset is None:
        raise RuntimeError(
            f"No cloudflared build published for {system}/{machine}. "
            "Install it by hand and set web_server.cloudflare_tunnel.binary to its path."
        )

    os.makedirs(bin_dir, exist_ok=True)
    target = managed_binary_path(bin_dir, system)
    # Download beside the target, then move: a half-written file left at the
    # real path would look installed and fail to exec on every later press.
    staging = target + ".part"

    try:
        fetch(cloudflared_download_url(asset), staging)

        if asset.endswith(".tgz"):
            _extract_tgz_binary(staging, target)
            os.remove(staging)
        else:
            os.replace(staging, target)

        os.chmod(target, 0o755)
    except Exception as exc:
        for leftover in (staging, target + ".tmpdir"):
            _remove_quietly(leftover)
        raise RuntimeError(f"Could not install cloudflared: {exc}") from exc

    if not _is_executable(target):
        raise RuntimeError(f"Downloaded cloudflared is not executable at {target}")
    return target


def _extract_tgz_binary(archive_path: str, target: str) -> None:
    """Pull the cloudflared executable out of the macOS .tgz onto ``target``."""
    import tarfile

    with tarfile.open(archive_path, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) == "cloudflared"),
            None,
        )
        if member is None:
            raise RuntimeError("archive contained no cloudflared executable")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError("could not read cloudflared from the archive")
        with open(target, "wb") as out:
            shutil.copyfileobj(extracted, out)


def _remove_quietly(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def build_command(binary: str, port: int, host: str = "127.0.0.1") -> List[str]:
    """The cloudflared argv for a quick tunnel pointing at this server."""
    return [binary, "tunnel", "--url", f"http://{host}:{port}"]


def should_auto_stop(
    tunnel_running: bool,
    transcription_running: bool,
    idle_since: Optional[float],
    now: float,
    delay_seconds: float = DEFAULT_AUTO_STOP_SECONDS,
    auto_stop_enabled: bool = True,
) -> bool:
    """Whether a running tunnel has been idle long enough to shut itself down.

    ``auto_stop_enabled`` False means the operator closes it by hand — useful
    when the tunnel is wanted for something other than the service it happens
    to sit alongside.

    ``idle_since`` is when transcription last stopped, or None if it has not
    run (or is running now). A tunnel started before any transcription stays
    up: there is no "end of transcription" to measure from yet, and closing it
    out from under the operator who just opened it would be worse than leaving
    it. Restarting transcription clears ``idle_since``, which cancels a pending
    stop rather than letting it fire mid-service.
    """
    if not auto_stop_enabled:
        return False
    if not tunnel_running or transcription_running or idle_since is None:
        return False
    return (now - idle_since) >= delay_seconds


class CloudflareTunnel:
    """A single cloudflared quick tunnel, started and stopped on demand.

    Thread-safe: the web routes, the auto-stop watcher and the log reader all
    touch it concurrently.
    """

    def __init__(
        self,
        binary: str = "cloudflared",
        spawn: Optional[Callable[[List[str]], Any]] = None,
        clock: Callable[[], float] = time.time,
        startup_timeout: float = 30.0,
        resolve: Callable[[str], Optional[str]] = resolve_binary,
    ) -> None:
        self._binary = binary
        self._spawn = spawn if spawn is not None else _default_spawn
        self._clock = clock
        self._resolve = resolve
        self._startup_timeout = startup_timeout

        self._lock = threading.Lock()
        self._process: Any = None
        self._status = STATUS_STOPPED
        self._url: Optional[str] = None
        self._error: str = ""
        self._started_at: Optional[float] = None
        self._log_tail: List[str] = []
        self._url_ready = threading.Event()

    # ─── lifecycle ──────────────────────────────────────────────────────────

    def start(self, port: int, host: str = "127.0.0.1") -> Dict[str, Any]:
        """Launch a tunnel to ``host:port``. A no-op if one is already up."""
        with self._lock:
            if self._status in (STATUS_STARTING, STATUS_RUNNING):
                return self._status_locked()

            self._status = STATUS_STARTING
            self._url = None
            self._error = ""
            self._log_tail = []
            self._url_ready.clear()
            self._started_at = self._clock()

            executable = self._resolve(self._binary)
            if executable is None:
                return self._fail_locked(
                    f"cloudflared not found (looked for '{self._binary}' on PATH and in the "
                    "usual install locations). Install it — 'brew install cloudflared' on macOS, "
                    "the .deb/.rpm from Cloudflare on Linux — or set web_server.cloudflare_tunnel.binary "
                    "to its full path."
                )

            try:
                self._process = self._spawn(build_command(executable, port, host))
            except FileNotFoundError:
                return self._fail_locked(f"cloudflared could not be executed at '{executable}'.")
            except Exception as exc:  # pragma: no cover - defensive
                return self._fail_locked(f"Failed to launch cloudflared: {exc}")

            process = self._process

        threading.Thread(
            target=self._read_output, args=(process,), daemon=True, name="cloudflared-reader"
        ).start()
        return self.status()

    def wait_for_url(self, timeout: Optional[float] = None) -> Optional[str]:
        """Block until the URL appears, or the timeout elapses. Returns it or None."""
        self._url_ready.wait(self._startup_timeout if timeout is None else timeout)
        with self._lock:
            return self._url

    def stop(self, reason: str = "") -> Dict[str, Any]:
        """Terminate the tunnel. A no-op when nothing is running."""
        with self._lock:
            process = self._process
            self._process = None
            self._status = STATUS_STOPPED
            self._url = None
            self._started_at = None
            self._url_ready.clear()
            if reason:
                self._error = ""
                self._log_tail.append(f"stopped: {reason}")
                self._log_tail = self._log_tail[-_LOG_TAIL:]

        if process is not None:
            _terminate(process)
        return self.status()

    def status(self) -> Dict[str, Any]:
        """A JSON-ready snapshot for the API and the settings page."""
        with self._lock:
            # A process that died on its own (network loss, cloudflared crash)
            # would otherwise keep reporting "running" until someone pressed stop.
            if self._process is not None and self._process.poll() is not None:
                self._status = STATUS_ERROR if not self._error else self._status
                if not self._error:
                    self._error = "cloudflared exited unexpectedly"
                self._process = None
                self._url = None
                self._url_ready.clear()
            return self._status_locked()

    def is_running(self) -> bool:
        return self.status()["status"] in (STATUS_STARTING, STATUS_RUNNING)

    # ─── internals ──────────────────────────────────────────────────────────

    def _fail_locked(self, message: str) -> Dict[str, Any]:
        """Record a start failure. Caller holds the lock.

        Releases ``_url_ready`` as well: a caller blocked in ``wait_for_url``
        must not sit out the full startup timeout for a tunnel that already
        failed to launch.
        """
        self._status = STATUS_ERROR
        self._error = message
        self._process = None
        self._started_at = None
        self._url_ready.set()
        return self._status_locked()

    def _status_locked(self) -> Dict[str, Any]:
        """Snapshot. Caller holds the lock."""
        return {
            "status": self._status,
            "url": self._url,
            "error": self._error,
            "started_at": self._started_at,
            "uptime_seconds": (
                round(self._clock() - self._started_at, 1) if self._started_at else None
            ),
            "log_tail": list(self._log_tail),
        }

    def _read_output(self, process: Any) -> None:
        """Scrape cloudflared's log stream for the URL; keep a tail for errors."""
        stream = getattr(process, "stderr", None) or getattr(process, "stdout", None)
        if stream is None:
            return
        try:
            for raw in stream:
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
                line = line.rstrip("\n")
                with self._lock:
                    if process is not self._process:
                        return  # superseded by a newer tunnel, or stopped
                    self._log_tail.append(line)
                    self._log_tail = self._log_tail[-_LOG_TAIL:]
                    if self._url is None:
                        url = parse_quick_tunnel_url(line)
                        if url:
                            self._url = url
                            self._status = STATUS_RUNNING
                            self._url_ready.set()
        except Exception:  # pragma: no cover - stream closed under us on stop
            pass
        finally:
            with self._lock:
                if process is self._process:
                    self._process = None
                    if self._status != STATUS_STOPPED:
                        self._status = STATUS_ERROR
                        if not self._error:
                            self._error = "cloudflared exited unexpectedly"
                    self._url = None
                self._url_ready.set()  # unblock any waiter rather than hang


def _default_spawn(command: List[str]) -> "subprocess.Popen[str]":
    """Launch cloudflared with its log stream piped back to us."""
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _terminate(process: Any, timeout: float = 10.0) -> None:
    """Ask the process to exit, then insist."""
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except Exception:
            process.kill()
    except Exception:  # pragma: no cover - already reaped
        pass

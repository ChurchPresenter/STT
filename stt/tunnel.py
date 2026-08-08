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


def resolve_binary(configured: str, candidates: Any = _BINARY_CANDIDATES) -> Optional[str]:
    """An executable path for cloudflared, or None if it cannot be found.

    An explicit path in config is honoured as given. A bare name is looked up on
    PATH first, then in the known install locations — a supervisor-started
    server inherits a minimal PATH, so PATH alone finds nothing on a machine
    where the operator installed it with Homebrew.
    """
    configured = (configured or "cloudflared").strip()

    if os.path.sep in configured or (os.path.altsep and os.path.altsep in configured):
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None

    found = shutil.which(configured)
    if found:
        return found

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def build_command(binary: str, port: int, host: str = "127.0.0.1") -> List[str]:
    """The cloudflared argv for a quick tunnel pointing at this server."""
    return [binary, "tunnel", "--url", f"http://{host}:{port}"]


def should_auto_stop(
    tunnel_running: bool,
    transcription_running: bool,
    idle_since: Optional[float],
    now: float,
    delay_seconds: float = DEFAULT_AUTO_STOP_SECONDS,
) -> bool:
    """Whether a running tunnel has been idle long enough to shut itself down.

    ``idle_since`` is when transcription last stopped, or None if it has not
    run (or is running now). A tunnel started before any transcription stays
    up: there is no "end of transcription" to measure from yet, and closing it
    out from under the operator who just opened it would be worse than leaving
    it. Restarting transcription clears ``idle_since``, which cancels a pending
    stop rather than letting it fire mid-service.
    """
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

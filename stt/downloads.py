"""Model-download tracking: registration state machine, progress persistence,
size monitoring, and the resumable URL downloader.

Extracted from speech_to_text.py so it can be imported (and unit-tested)
without the monolith's import-time side effects. This module OWNS the shared
state (active_downloads / lock / cancelled_downloads); the monolith re-imports
these names, so both sides mutate the same objects. Call configure() with the
progress-file path before persistence matters, then load_state() to restore
the previous run's entries in place.
"""

import fnmatch
import json
import os
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

# Path of the JSON progress file; set via configure(). While None, persistence
# is skipped (state-machine behavior is unaffected).
_progress_file: Optional[str] = None

# Global dictionary to track active downloads
active_downloads: Dict[str, dict] = {}
active_downloads_lock = threading.Lock()
cancelled_downloads: Set[str] = set()  # Track cancelled download IDs to prevent re-adding


def configure(progress_file: Optional[str]) -> None:
    """Set the on-disk location for download-progress persistence."""
    global _progress_file
    _progress_file = progress_file


def load_state() -> None:
    """Restore active_downloads from disk IN PLACE (the dict object is shared
    with importers, so it must never be rebound)."""
    data = load_download_progress()
    with active_downloads_lock:
        active_downloads.clear()
        active_downloads.update(data)


def load_download_progress() -> dict:
    """Load download progress from file"""
    try:
        if _progress_file and os.path.exists(_progress_file):
            with open(_progress_file, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load download progress: {e}")
    return {}


def save_download_progress() -> None:
    """Save download progress to file"""
    if _progress_file is None:
        return  # not configured (e.g. tests exercising only the state machine)
    try:
        with active_downloads_lock:
            with open(_progress_file, "w") as f:
                json.dump(active_downloads, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save download progress: {e}")


def cleanup_stale_downloads() -> None:
    """Remove downloads based on status and age"""
    import time

    current_time = time.time()

    # Different retention periods by status
    DOWNLOADING_STALE_THRESHOLD = 86400  # 24 hours for stuck downloads
    COMPLETED_GRACE_PERIOD = 7200  # 2 hours for completed downloads
    FAILED_GRACE_PERIOD = 3600  # 1 hour for failed downloads

    with active_downloads_lock:
        stale_keys = []
        for model_id, info in active_downloads.items():
            last_update = info.get("last_update", 0)
            status = info.get("status", "downloading")
            age = current_time - last_update

            # Determine if should be removed based on status
            should_remove = False

            if status == "downloading" and age > DOWNLOADING_STALE_THRESHOLD:
                # Stuck download, likely stale
                should_remove = True
                print(f"[CLEANUP] Removing stale downloading: {model_id} (age: {age/3600:.1f}h)")
            elif status == "completed" and age > COMPLETED_GRACE_PERIOD:
                # Completed downloads after grace period
                should_remove = True
                print(f"[CLEANUP] Removing old completed download: {model_id} (age: {age/3600:.1f}h)")
            elif status == "failed" and age > FAILED_GRACE_PERIOD:
                # Failed downloads after shorter grace period
                should_remove = True
                print(f"[CLEANUP] Removing old failed download: {model_id} (age: {age/3600:.1f}h)")

            if should_remove:
                stale_keys.append(model_id)

        for key in stale_keys:
            del active_downloads[key]

        if stale_keys:
            # Save while still holding the lock - don't call save_download_progress() which would deadlock
            if _progress_file is not None:
                try:
                    with open(_progress_file, "w") as f:
                        json.dump(active_downloads, f, indent=2)
                except Exception as e:
                    print(f"[ERROR] Failed to save download progress: {e}")
            print(f"[CLEANUP] Removed {len(stale_keys)} stale download record(s)")


def try_register_download(key: str, total: Optional[int] = None) -> bool:
    """Atomically register a download in active_downloads.

    Returns False if a download for this key is already in progress."""
    with active_downloads_lock:
        existing = active_downloads.get(key)
        if existing and existing.get("status") == "downloading":
            return False
        cancelled_downloads.discard(key)
        active_downloads[key] = {
            "downloaded": 0,
            "total": total,
            "percentage": 0 if total else None,
            "start_time": time.time(),
            "last_update": time.time(),
            "status": "downloading",
        }
    save_download_progress()
    return True


def set_download_total(key: str, total: Optional[int]) -> None:
    """Fill in a download's byte total once it is known.

    Registration happens before the repo is queried — the key is what makes a second
    request a 409 rather than a second download — so the total arrives a moment later.
    Without it a download has no denominator: percentage stays None and the UI can
    only say "starting", which is indistinguishable from a stall for as long as the
    file takes.
    """
    if not total or total <= 0:
        return
    with active_downloads_lock:
        entry = active_downloads.get(key)
        if entry is None or entry.get("status") != "downloading":
            return
        entry["total"] = int(total)
        downloaded = entry.get("downloaded") or 0
        entry["percentage"] = min(int((downloaded / total) * 100), 99)
        entry["last_update"] = time.time()
    save_download_progress()


def select_repo_files(files: Sequence[str], include: Optional[Any] = None) -> List[str]:
    """The repo files an ``include`` filter selects, in repo order.

    ``include`` is a filename or a list of them, matched exactly or as an fnmatch
    pattern; None selects everything. Shared by the downloader and by the size lookup
    that gives it a denominator — computing the total over a different set than the
    one being fetched is how a progress bar ends at 40% or at 300%.
    """
    names = list(files)
    if not include:
        return names
    patterns = [include] if isinstance(include, str) else list(include)
    return [f for f in names
            if any(f == p or fnmatch.fnmatch(f, p) for p in patterns)]


def finish_download(key: str, error: Optional[Any] = None, cancelled: bool = False) -> None:
    """Mark a download completed/failed and drop it from the cancelled set."""
    with active_downloads_lock:
        cancelled_downloads.discard(key)
        if not cancelled and key in active_downloads:
            entry = active_downloads[key]
            entry["last_update"] = time.time()
            if error is not None:
                entry["status"] = "failed"
                entry["error"] = str(error)
            else:
                entry["status"] = "completed"
                entry["percentage"] = 100
                entry["completion_time"] = time.time()
                if entry.get("total"):
                    entry["downloaded"] = entry["total"]
    save_download_progress()


def _path_size(path: str) -> int:
    """Size in bytes of a file, or recursive size of a directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def monitor_download_progress(key: str, path: str, total: Optional[int] = None, interval: float = 2) -> None:
    """Poll the size of `path` (file or directory) and update active_downloads[key].

    Runs until the entry leaves "downloading" state, disappears, or is cancelled.
    Percentage is capped at 99 — the download code sets 100 on completion."""
    import time as _time

    while True:
        with active_downloads_lock:
            entry = active_downloads.get(key)
            if entry is None or entry.get("status") != "downloading" or key in cancelled_downloads:
                return
        if os.path.exists(path):
            size = _path_size(path)
            with active_downloads_lock:
                entry = active_downloads.get(key)
                if entry is None or entry.get("status") != "downloading":
                    return
                entry["downloaded"] = size
                entry["last_update"] = _time.time()
                entry_total = entry.get("total") or total
                if entry_total and entry_total > 0:
                    entry["percentage"] = min(int((size / entry_total) * 100), 99)
            save_download_progress()
        _time.sleep(interval)


def start_download_monitor(key: str, path: str, total: Optional[int] = None, interval: float = 2) -> None:
    """Spawn the directory-size progress monitor as a daemon thread."""
    threading.Thread(
        target=monitor_download_progress,
        args=(key, path, total, interval),
        daemon=True,
        name=f"dl-monitor-{key}",
    ).start()


# --- Transient network failures -------------------------------------------------
#
# Every HuggingFace metadata lookup (list_repo_files, model_info) is a single
# HTTPS request with no retry of its own, while the file downloads that follow
# retry five times. On a connection that drops one handshake in twenty — which is
# the ordinary case outside of a datacentre — the download therefore dies at 0%
# on the *cheapest* request it makes, and the operator is shown the raw transport
# text ("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of
# protocol (_ssl.c:1016)"), which names neither the cause nor the remedy.
#
# Matching is by exception class name and message rather than by isinstance:
# these modules import stdlib-only (httpx/httpcore/requests are not importable
# here), and the same failure arrives wrapped in a different library's class
# depending on which huggingface_hub version made the call.

_MAX_CAUSE_DEPTH = 10

_TRANSIENT_EXC_NAMES = frozenset({
    "ChunkedEncodingError", "ConnectError", "ConnectTimeout", "ConnectionAbortedError",
    "ConnectionError", "ConnectionResetError", "IncompleteRead", "NetworkError",
    "PoolTimeout", "ProtocolError", "ProxyError", "ReadError", "ReadTimeout",
    "ReadTimeoutError", "RemoteDisconnected", "RemoteProtocolError", "SSLEOFError",
    "SSLError", "SSLZeroReturnError", "Timeout", "TimeoutError", "WriteError",
    "WriteTimeout", "gaierror", "herror", "timeout",
})

_TRANSIENT_MESSAGE_HINTS = (
    "bad handshake",
    "connection aborted",
    "connection refused",
    "connection reset",
    "connection timed out",
    "eof occurred in violation of protocol",
    "getaddrinfo failed",
    "handshake operation timed out",
    "name or service not known",
    "network is unreachable",
    "remote end closed connection",
    "temporary failure in name resolution",
    "timed out",
    "unexpected_eof",
)

# Not retried: a rejected certificate is a stable condition (an intercepting
# proxy, antivirus TLS inspection, or a wrong system clock), so retrying only
# delays the message that would let someone fix it.
_CERTIFICATE_HINTS = (
    "certificate verify failed",
    "self signed certificate",
    "self-signed certificate",
    "certificate_verify_failed",
    "unable to get local issuer certificate",
)

_DNS_HINTS = (
    "getaddrinfo",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
)

_TIMEOUT_HINTS = ("timed out", "timeout")


def _exception_chain(exc: BaseException) -> List[BaseException]:
    """`exc` followed by its ``__cause__``/``__context__`` ancestors.

    The transport error is usually two or three re-raises below the exception
    the caller sees, and huggingface_hub raises `from` in some paths and not in
    others, so both links are walked. Bounded in case a chain loops.
    """
    chain: List[BaseException] = []
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and len(chain) < _MAX_CAUSE_DEPTH:
        if id(current) in seen:
            break
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _chain_text(exc: BaseException) -> str:
    """Lower-cased class names and messages across the whole cause chain."""
    return " ".join(f"{type(e).__name__}: {e}" for e in _exception_chain(exc)).lower()


def is_transient_network_error(exc: BaseException) -> bool:
    """Whether `exc` is a transport failure that a retry could plausibly fix.

    A rejected certificate is deliberately excluded: it fails identically every
    time, so retrying it costs the operator half a minute and tells them nothing.
    """
    text = _chain_text(exc)
    if any(hint in text for hint in _CERTIFICATE_HINTS):
        return False
    if any(type(e).__name__ in _TRANSIENT_EXC_NAMES for e in _exception_chain(exc)):
        return True
    return any(hint in text for hint in _TRANSIENT_MESSAGE_HINTS)


def network_error_message(exc: BaseException, host: str = "huggingface.co") -> Optional[str]:
    """A sentence an operator can act on, or None if `exc` is not network-shaped.

    Returning None is what keeps this from swallowing real bugs: a KeyError in
    the download code must still surface as itself.
    """
    text = _chain_text(exc)
    if any(hint in text for hint in _CERTIFICATE_HINTS):
        return (f"Could not verify the secure connection to {host}. An antivirus, "
                "firewall or proxy that inspects HTTPS traffic will cause this, and so "
                "will a system clock that is days out of date.")
    if not is_transient_network_error(exc):
        return None
    if any(hint in text for hint in _DNS_HINTS):
        return (f"Could not look up {host}. Check the internet connection and the "
                "DNS settings on this machine.")
    if any(hint in text for hint in _TIMEOUT_HINTS):
        return (f"The connection to {host} timed out. The download can be started "
                "again once the connection is steady.")
    return (f"The connection to {host} was interrupted before the download could "
            "start. This is usually a temporary network problem — try again.")


def call_with_retry(
    func: Callable[[], Any],
    description: str = "request",
    max_attempts: int = 4,
    base_delay: float = 2.0,
    log: Callable[[str], Any] = print,
    sleep: Callable[[float], Any] = time.sleep,
) -> Any:
    """Call `func()`, retrying transient network failures with linear backoff.

    Non-network exceptions (and certificate rejections) propagate on the first
    attempt — retrying a bug or a stable misconfiguration only delays the report.
    The final failure is re-raised unchanged so the caller keeps the original
    traceback; `network_error_message` turns it into operator-facing text.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_network_error(exc):
                raise
            delay = base_delay * attempt
            log(f"[WARNING] {description} failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s (attempt {attempt}/{max_attempts})")
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def download_url_to_file(url: str, dest_path: str, cancel_check: Optional[Callable[[], bool]] = None, max_attempts: int = 5, log: Callable[[str], Any] = print) -> str:
    """Download a URL to a file with resume + retry, preferring wget/curl.

    Falls back to a pure-Python streaming download when neither tool exists
    (e.g. minimal Windows installs). `cancel_check` is polled during the
    download; returning True aborts it. Returns "ok" or "cancelled"; raises
    after all attempts fail."""
    # Every download in the application funnels through here, so this is the one
    # place that has to know a demo fetches nothing. Checked by env rather than by
    # a parameter because the alternative is threading a flag through a dozen
    # unrelated call sites, and a door that is only sometimes shut is not shut.
    from stt import demo_mode
    if demo_mode.enabled():
        from stt import demo_guard
        raise RuntimeError(demo_guard.blocked_message("downloads"))

    import subprocess
    import tempfile as _tempfile
    import time as _time
    import urllib.request

    if shutil.which("wget"):
        dl_cmd = ['wget', '-c', '-t', '3', '-T', '120', '--retry-connrefused',
                  '--waitretry', '5', '-O', dest_path, url]
    elif shutil.which("curl"):
        dl_cmd = ['curl', '-L', '-C', '-', '--retry', '3', '--retry-delay', '5',
                  '--retry-connrefused', '--connect-timeout', '30',
                  '--max-time', '600', '-o', dest_path, url]
    else:
        dl_cmd = None  # pure-Python fallback below

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        if dl_cmd:
            # Output goes to a temp file: a PIPE would fill up with progress
            # noise and block the process, since nothing drains it while we poll
            with _tempfile.TemporaryFile(mode="w+", errors="replace") as outf:
                # creationflags: windowless server — a console child would flash
                # a window on Windows (0 elsewhere).
                proc = subprocess.Popen(dl_cmd, stdout=outf, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while proc.poll() is None:
                    if cancel_check and cancel_check():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return "cancelled"
                    _time.sleep(0.5)
                if proc.returncode == 0:
                    return "ok"
                outf.seek(0)
                last_error = outf.read()[-500:]
            returncode = proc.returncode
        else:
            try:
                with urllib.request.urlopen(url, timeout=120) as src, open(dest_path, "wb") as out:
                    while True:
                        if cancel_check and cancel_check():
                            return "cancelled"
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                return "ok"
            except Exception as e:
                last_error = str(e)
                returncode = 1

        log(f"[WARNING] Download attempt {attempt}/{max_attempts} failed for "
            f"{os.path.basename(dest_path)} (exit code {returncode})")
        if attempt < max_attempts:
            if os.path.exists(dest_path):
                partial_size = os.path.getsize(dest_path)
                log(f"[INFO] Partial file exists ({partial_size / (1024*1024):.1f} MB), will resume")
            _time.sleep(5 * attempt)

    raise Exception(
        f"Failed to download {os.path.basename(dest_path)} after {max_attempts} attempts: {last_error[:300]}"
    )

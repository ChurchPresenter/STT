"""Reverse-DNS cache for access-log IP addresses.

The log viewer wants a name next to each client IP ("who is 192.168.2.62?"),
but ``socket.gethostbyaddr`` blocks — on a LAN address with no PTR record it can
sit for seconds waiting on the resolver, and doing that inside a request handler
would stall the logs page for every unknown address in the list.

So resolution happens off the request path: :meth:`HostnameCache.get` answers
from cache immediately and queues a lookup on a background worker thread for
anything it doesn't know. The first render of a fresh IP shows no name; the next
poll of the page (the viewer auto-refreshes) shows it. Failures are cached too,
with a shorter TTL, so an address with no PTR record isn't re-resolved on every
refresh.

Stdlib-only and dependency-injected (``resolver``) so it can be unit-tested
without touching DNS.
"""

import queue
import socket
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# A resolver takes an IP string and returns a hostname, or raises.
Resolver = Callable[[str], str]


def system_resolver(ip: str) -> str:
    """Reverse-resolve ``ip`` via the OS resolver. Raises if there is no name."""
    return socket.gethostbyaddr(ip)[0]


class HostnameCache:
    """Non-blocking reverse-DNS lookups with positive and negative caching."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        negative_ttl_seconds: float = 300.0,
        max_entries: int = 1024,
        resolver: Optional[Resolver] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        """Create a cache.

        ``ttl_seconds`` is how long a resolved name is trusted;
        ``negative_ttl_seconds`` how long a failure is remembered before the
        address is tried again. ``max_entries`` caps memory — the entries
        closest to expiry are dropped first. ``resolver`` defaults to the OS,
        and ``clock`` to :func:`time.time` (injectable so TTL behaviour can be
        tested without waiting).
        """
        self.ttl = max(1.0, float(ttl_seconds))
        self.negative_ttl = max(1.0, float(negative_ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._resolver: Resolver = resolver or system_resolver
        self._clock: Callable[[], float] = clock or time.time
        self._lock = threading.Lock()
        # ip -> (hostname or None, expires_at)
        self._entries: Dict[str, Tuple[Optional[str], float]] = {}
        self._pending: set = set()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    # -- lookups ------------------------------------------------------------

    def get(self, ip: Optional[str]) -> Optional[str]:
        """Return the cached hostname for ``ip``, queueing a lookup if unknown.

        Never blocks on DNS. Returns ``None`` when the name isn't known yet (or
        the address has no PTR record); a stale-but-present name is returned
        while the refresh runs, so a name never flickers away.
        """
        if not ip:
            return None
        ts = self._clock()
        with self._lock:
            entry = self._entries.get(ip)
            if entry is not None and ts < entry[1]:
                return entry[0]
            stale = entry[0] if entry is not None else None
            if ip not in self._pending:
                self._pending.add(ip)
                self._queue.put(ip)
                self._ensure_worker_locked()
        return stale

    def get_many(self, ips: Iterable[Optional[str]]) -> Dict[str, Optional[str]]:
        """:meth:`get` over a batch, as an ``{ip: hostname or None}`` mapping."""
        out: Dict[str, Optional[str]] = {}
        for ip in ips:
            if ip:
                out[ip] = self.get(ip)
        return out

    def resolve_now(self, ip: str) -> Optional[str]:
        """Resolve synchronously (blocking) and cache the result.

        Used by the background worker, and by callers that genuinely want to
        wait — not by request handlers.
        """
        try:
            raw = self._resolver(ip)
        except Exception:
            raw = ""
        name: Optional[str] = (raw or "").strip().rstrip(".") or None
        ts = self._clock()
        with self._lock:
            self._entries[ip] = (name, ts + (self.ttl if name else self.negative_ttl))
            self._pending.discard(ip)
            self._prune_locked()
        return name

    def prime(self, ip: str, hostname: Optional[str]) -> None:
        """Insert a known answer directly (tests, or a name from another source)."""
        ts = self._clock()
        with self._lock:
            self._entries[ip] = (hostname, ts + (self.ttl if hostname else self.negative_ttl))
            self._pending.discard(ip)
            self._prune_locked()

    # -- internals ----------------------------------------------------------

    def _ensure_worker_locked(self) -> None:
        """Start the single daemon resolver thread if it isn't running."""
        if self._worker is not None and self._worker.is_alive():
            return
        worker = threading.Thread(target=self._run, name="hostname-lookup", daemon=True)
        self._worker = worker
        worker.start()

    def _run(self) -> None:
        """Drain queued addresses until idle, then exit (restarted on demand)."""
        while True:
            try:
                ip = self._queue.get(timeout=30.0)
            except queue.Empty:
                return
            try:
                self.resolve_now(ip)
            except Exception:
                with self._lock:
                    self._pending.discard(ip)
            finally:
                self._queue.task_done()

    def _prune_locked(self) -> None:
        """Drop the soonest-to-expire entries once over ``max_entries``."""
        if len(self._entries) <= self.max_entries:
            return
        ordered: List[Tuple[str, float]] = sorted(
            ((ip, exp) for ip, (_name, exp) in self._entries.items()),
            key=lambda pair: pair[1],
        )
        for ip, _exp in ordered[: len(self._entries) - self.max_entries]:
            self._entries.pop(ip, None)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until queued lookups finish. For tests; returns False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.01)
        with self._lock:
            return not self._pending

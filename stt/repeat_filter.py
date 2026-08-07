"""Keep one client's refused request from filling the access log.

A page left open on a machine that is not whitelisted polls forever and is
refused every time. One display client produced 9,182 consecutive 403s in a
week — a fifth of the whole log — and because the log is capped at a row count,
every one of those rows pushed out a real request. The browser-side fix (see
static/poll-fetch.js) stops the traffic, but only once that page reloads, and a
tab that has been open for days does not.

So the log stops writing the same refusal over and over. The first few are
recorded in full, then one row per cooldown window, and that row carries how
many were suppressed behind it — the fact that a machine is hammering a refused
endpoint is exactly what an operator needs to see, but it takes one line to say
it, not six hundred.

Nothing here is a rate limit: the request is still served (refused) as before.
Only the logging is thinned.

Stdlib-only, with an injectable clock so the windows can be tested directly.
"""

import threading
from typing import Callable, Dict, Hashable, NamedTuple, Optional
import time as _time


class Verdict(NamedTuple):
    """Whether to write this row, and what was dropped since the last one."""

    log: bool
    #: How many identical events were suppressed since the last logged row.
    #: Non-zero only on a row that ends a suppressed run, so the count can be
    #: written into the row rather than lost.
    suppressed: int = 0


class RepeatSuppressor:
    """Thins repeats of an identical event down to a heartbeat."""

    def __init__(self, *, first_n: int = 3, cooldown_seconds: float = 600.0,
                 max_keys: int = 512, clock: Optional[Callable[[], float]] = None) -> None:
        """``first_n`` rows are always written; after that, one per cooldown.

        ``max_keys`` bounds memory — an unbounded map keyed by client address
        would be a way to grow this process from outside. The least recently
        seen key is dropped, which at worst lets one extra row through.
        """
        self.first_n = max(1, int(first_n))
        self.cooldown = max(0.0, float(cooldown_seconds))
        self.max_keys = max(1, int(max_keys))
        self._clock: Callable[[], float] = clock or _time.time
        self._lock = threading.Lock()
        # key -> [seen_in_burst, suppressed, next_allowed_ts, last_touch_ts]
        self._state: Dict[Hashable, list] = {}

    def decide(self, key: Hashable) -> Verdict:
        """Whether this occurrence of ``key`` should be written to the log."""
        now = self._clock()
        with self._lock:
            entry = self._state.get(key)
            if entry is None:
                self._evict_locked()
                self._state[key] = [1, 0, 0.0, now]
                return Verdict(True)

            entry[3] = now
            if entry[0] < self.first_n:
                entry[0] += 1
                return Verdict(True)

            if entry[2] == 0.0:
                # First one past the burst allowance: start the cooldown.
                entry[2] = now + self.cooldown
                entry[1] += 1
                return Verdict(False)

            if now >= entry[2]:
                suppressed = entry[1]
                entry[1] = 0
                entry[2] = now + self.cooldown
                return Verdict(True, suppressed)

            entry[1] += 1
            return Verdict(False)

    def forget(self, key: Hashable) -> None:
        """Drop the state for ``key`` — the next occurrence is logged in full."""
        with self._lock:
            self._state.pop(key, None)

    def _evict_locked(self) -> None:
        """Make room by dropping the least recently seen key."""
        if len(self._state) < self.max_keys:
            return
        oldest = min(self._state.items(), key=lambda kv: kv[1][3])[0]
        del self._state[oldest]

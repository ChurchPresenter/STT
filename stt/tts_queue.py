"""Which translated segments the TTS voice still owes the listener.

The live translation loop deliberately drains its per-cycle budget *newest-first*
during a backlog, reserving a single slot for the oldest pending segment, so the
caption a viewer is reading right now is translated before the tail catches up.
That means translations land out of order: segment N+3 can be cached seconds
before segment N.

A monotonic high-water mark cannot survive that. Speaking N+3 advances the mark
past N, and when N finally arrives it compares as "already spoken" and is
silently never voiced — the on-screen text recovers (the client sorts by id) but
the spoken output loses the segment permanently.

So track spoken ids as a *set* instead of a ceiling. A late-arriving lower id is
still owed and still gets spoken; each id is spoken exactly once. The set is
pruned against a caller-supplied floor (the oldest id the translation cache still
holds) so it stays bounded over a long service without ever resurrecting an id
that was already voiced.

Pure/stdlib so it can be imported and unit-tested without the monolith.
"""

from typing import Iterable, List, Set


class SpokenTracker:
    """Tracks which segment ids have been spoken, tolerating out-of-order arrival."""

    def __init__(self) -> None:
        self._spoken: Set[int] = set()
        # Ids at or below this are treated as spoken without being enumerated.
        # Only ever moved by prime() — never by speaking, which is precisely
        # the bug this class exists to avoid.
        self._baseline: int = 0

    def reset(self) -> None:
        """Forget everything (call when the session stops)."""
        self._spoken = set()
        self._baseline = 0

    def prime(self, through_id: int) -> None:
        """Treat every id up to and including ``through_id`` as already handled.

        Used when TTS is switched on mid-session: the backlog that accumulated
        while it was off is skipped rather than replayed from the top. This is a
        deliberate one-shot jump, unlike speaking, which never moves the
        baseline.
        """
        if through_id > self._baseline:
            self._baseline = through_id
            self._spoken = {sid for sid in self._spoken if sid > through_id}

    def mark_spoken(self, segment_ids: Iterable[int]) -> None:
        """Record that these ids have been voiced."""
        self._spoken.update(sid for sid in segment_ids if sid > self._baseline)

    def select_unspoken(self, available_ids: Iterable[int]) -> List[int]:
        """Ids that are available but not yet spoken, in ascending order.

        Ascending order is what makes a late arrival usable: a segment that
        shows up after newer ones were already voiced is still returned, and the
        caller speaks it in id order rather than in arrival order.
        """
        return sorted(
            sid for sid in available_ids
            if sid > self._baseline and sid not in self._spoken
        )

    def prune(self, floor_id: int) -> None:
        """Forget spoken ids below ``floor_id`` by folding them into the baseline.

        The floor must be an id the caller can no longer be offered — in
        practice the oldest id still in the translation cache. Folding rather
        than simply discarding is what keeps this safe: a pruned id stays
        "spoken" via the baseline, so an evicted-then-rediscovered segment is
        never voiced twice.
        """
        if floor_id > self._baseline:
            self._baseline = floor_id - 1
            self._spoken = {sid for sid in self._spoken if sid >= floor_id}

    def size(self) -> int:
        """How many ids are currently remembered (for tests and diagnostics)."""
        return len(self._spoken)

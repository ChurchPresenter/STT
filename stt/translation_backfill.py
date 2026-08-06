"""Recovering captions the live translation loop can no longer reach.

The live loop only ever sees the newest ``max_entries_to_send`` visible rows.
Under backlog it drains newest-first and reserves a single slot per cycle for the
oldest pending segment, so the tail clears slowly. If a backlog outlives that
window, an untranslated row scrolls out of it and is never looked at again — it
stays ``translated_text IS NULL`` in the session database forever, and the
archive (SRT/HTML export, replay) is missing a caption that was spoken.

This module decides *which* orphaned rows to retry and *when to stop* retrying.
The database query and the translation call stay with the caller; everything here
is pure so the give-up behaviour can be tested without a model or a network.

Retries are bounded on purpose. An orphan whose translation keeps coming back
empty — an unreachable remote with ``fallback: skip``, a caption the model always
declines — would otherwise be re-attempted every cycle for the rest of the
service, spending the idle slot forever and never succeeding.

Pure/stdlib so it can be imported and unit-tested without the monolith.
"""

from typing import Dict, Iterable, List, Set

# An orphan is retried at most this many times per session before being left
# alone. Three is enough to ride out a transient remote outage without turning a
# permanently-undecodable caption into a standing cost.
DEFAULT_MAX_ATTEMPTS = 3


def select_backfill_ids(
    orphan_ids: Iterable[int],
    visible_ids: Iterable[int],
    limit: int = 1,
) -> List[int]:
    """Orphaned row ids worth retrying, oldest first, at most ``limit`` of them.

    Ids the main loop can still see are excluded: it will translate them itself,
    and racing it would double-spend the remote model on one caption.
    """
    if limit <= 0:
        return []
    visible: Set[int] = set(visible_ids)
    candidates = sorted(sid for sid in orphan_ids if sid not in visible)
    return candidates[:limit]


class BackfillAttempts:
    """Per-session record of how often each orphan has been retried."""

    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
        self._max_attempts = max_attempts
        self._attempts: Dict[int, int] = {}

    def reset(self) -> None:
        """Forget all attempts (call when a new session starts)."""
        self._attempts = {}

    def record(self, segment_id: int) -> None:
        """Note that a backfill was attempted for this id."""
        self._attempts[segment_id] = self._attempts.get(segment_id, 0) + 1

    def exhausted(self, segment_id: int) -> bool:
        """Whether this id has used up its retries."""
        return self._attempts.get(segment_id, 0) >= self._max_attempts

    def succeeded(self, segment_id: int) -> None:
        """Drop bookkeeping for an id that no longer needs backfilling."""
        self._attempts.pop(segment_id, None)

    def eligible(self, segment_ids: Iterable[int]) -> List[int]:
        """Filter ids down to those that still have retries left."""
        return [sid for sid in segment_ids if not self.exhausted(sid)]

    def size(self) -> int:
        """How many ids are being tracked (for tests and diagnostics)."""
        return len(self._attempts)

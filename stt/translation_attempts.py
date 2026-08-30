"""What the live pump does with a caption that came back untranslated.

``translate_live_text`` returns the *source text* when it cannot translate — the remote
peer timed out and ``remote.fallback`` is "skip", or the LLM declined and
``llm.fallback`` is "skip". That is the right thing to put on screen: the operator needs
to see that the speaker said something. It is emphatically the wrong thing to write to
the database, and writing it was a real defect: the row stops matching
``translated_text IS NULL``, so the pump never retries it, the backfill cannot find it,
and the caption is Russian forever — in the live view, the session database, the SRT and
the HTML export. In one measured service that was 30 captions of 448.

So: display it, never persist it, retry it, and give up after a few tries. The give-up is
not a detail. Without it a peer that is merely slow costs one timeout per segment per
cycle, and the repair would stall the pump harder than the bug it fixes; the cooldown
spaces the retries and the cap ends them. When the cap is reached the row is left NULL
rather than filled with the source, because a missing caption is recoverable — the
backfill and the replay harness both key on NULL — and a poisoned one is not.
"""

from typing import Dict, Optional, Tuple

from stt.translation_backfill import BackfillAttempts

# Tries before a caption is left alone. Enough to ride out a chunk of contention or a
# brief peer stall, few enough that a genuinely dead peer stops costing timeouts.
DEFAULT_MAX_ATTEMPTS = 3

# Space between retries of one caption. A caption that failed on a 15s timeout will
# almost certainly fail again immediately; waiting a cycle or two costs nothing anyone is
# watching, because the source text is already on screen.
DEFAULT_COOLDOWN_SECONDS = 20.0


def persist_decision(mt_engine: str, none_engine: str,
                     model_ready: bool) -> Tuple[bool, bool]:
    """``(display_it, persist_it)`` for one freshly translated segment.

    Three cases, and the difference between the last two is the whole point:

    * the model is still loading — the returned text is an echo of the source and the
      caption has not really been attempted yet, so neither show nor store it;
    * nothing translated it (``none_engine``) — show the source, store nothing, so the
      row stays NULL and is tried again;
    * something did — show it and store it, as always.
    """
    if not model_ready:
        return (False, False)
    if mt_engine == none_engine:
        return (True, False)
    return (True, True)


class LiveTranslationAttempts:
    """Per-session record of captions the pump could not translate.

    Wraps :class:`~stt.translation_backfill.BackfillAttempts` for the counting — the same
    give-up rule, already tested — and adds the two things the live path needs that the
    archive-repair path does not: a cooldown, so retries are spaced rather than
    per-cycle, and the source text, so the caption stays on screen while its row stays
    NULL.
    """

    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._attempts = BackfillAttempts(max_attempts=max_attempts)
        self._cooldown = float(cooldown_seconds)
        self._last_try: Dict[int, float] = {}
        self._source: Dict[int, str] = {}

    def should_attempt(self, segment_id: int, now: float) -> bool:
        """Whether this caption may be sent to the model again on this cycle."""
        if self._attempts.exhausted(segment_id):
            return False
        last = self._last_try.get(segment_id)
        if last is None:
            return True
        return (now - last) >= self._cooldown

    def record_failure(self, segment_id: int, source_text: str, now: float) -> None:
        """Note that nothing translated this caption, keeping its text for the display."""
        self._attempts.record(segment_id)
        self._last_try[segment_id] = float(now)
        self._source[segment_id] = source_text

    def record_success(self, segment_id: int) -> None:
        """Forget a caption that came back translated, so it costs nothing to carry."""
        self._attempts.succeeded(segment_id)
        self._last_try.pop(segment_id, None)
        self._source.pop(segment_id, None)

    def exhausted(self, segment_id: int) -> bool:
        """Whether this caption has used up its retries and should be left alone."""
        return self._attempts.exhausted(segment_id)

    def display_text(self, segment_id: int) -> Optional[str]:
        """The untranslated text to keep showing, or None if there is nothing pending."""
        return self._source.get(segment_id)

    def reset(self) -> None:
        """Forget everything — call when the session changes.

        Segment ids restart low in a new session database, so a carried-over count would
        be applied to an unrelated caption.
        """
        self._attempts.reset()
        self._last_try = {}
        self._source = {}

    def size(self) -> int:
        """How many captions are being carried (for tests and diagnostics)."""
        return len(self._source)

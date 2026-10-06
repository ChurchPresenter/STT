"""Running counts of what the sentence filters did, so a loss is visible while it happens.

Sunday's missed words were found by reading a session database afterwards. Nothing in the
running server could say "speech is being thrown away". Every sentence now passes through
:func:`stt.segment_disposition.decide`, so counting its outcomes is one call per sentence.

Counts, never text: this feeds ``/api/health`` and is safe to share. The one derived signal is
``suspect``: a sentence denied as music while the audio tagger's dominant class was Speech.
That is the shape Sunday's loss had (speech over quiet background music), so a handful in a few
minutes is worth an operator's attention even when each denial looked reasonable alone. With
``speech_overrides_music`` on, the live detector no longer labels such a segment Music, so this
reads zero in normal running; it is the tripwire for the rule failing or being switched off.

Stdlib-only, no I/O; the clock is a parameter so tests need no sleeping.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

from stt.segment_disposition import REASON_CJK, REASON_HALLUCINATION, Disposition

# A sentence's reason, bucketed. Music reasons carry the threshold ("music:0.5").
KEPT = "kept"
HALLUCINATION = "hallucination"
MUSIC = "music"
CJK = "cjk"

#: How far back the alert looks, and how many suspect denials in it raise each level.
WINDOW_SECONDS = 300.0
DEGRADED_AT = 3
ERROR_AT = 10

SPEECH_TAG = "Speech"


def _bucket(disposition: Disposition) -> str:
    if not disposition.denied:
        return KEPT
    if disposition.reason == REASON_HALLUCINATION:
        return HALLUCINATION
    if disposition.reason == REASON_CJK:
        return CJK
    return MUSIC


class DispositionStats:
    """Counters for one session. Not thread-safe by itself; the worker records from one thread."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start a new session's counts."""
        self.counts: Dict[str, int] = {KEPT: 0, HALLUCINATION: 0, MUSIC: 0, CJK: 0}
        self.stripped = 0
        self.suspect_total = 0
        self._suspect_times: Deque[float] = deque()
        self.last_status = "healthy"

    def record(self, disposition: Disposition, audio_tag: Optional[str], now: float) -> None:
        """Count one sentence's outcome at time ``now`` (seconds, any monotonic epoch)."""
        bucket = _bucket(disposition)
        self.counts[bucket] += 1
        if disposition.stripped:
            self.stripped += 1
        if bucket == MUSIC and audio_tag == SPEECH_TAG:
            self.suspect_total += 1
            self._suspect_times.append(now)

    def _recent_suspect(self, now: float) -> int:
        cutoff = now - WINDOW_SECONDS
        while self._suspect_times and self._suspect_times[0] < cutoff:
            self._suspect_times.popleft()
        return len(self._suspect_times)

    def snapshot(self, now: float) -> Dict[str, object]:
        """The counters plus an alert status, as plain JSON-able values."""
        recent = self._recent_suspect(now)
        status = "error" if recent >= ERROR_AT else "degraded" if recent >= DEGRADED_AT else "healthy"
        return {
            "kept": self.counts[KEPT],
            "denied_hallucination": self.counts[HALLUCINATION],
            "denied_music": self.counts[MUSIC],
            "denied_cjk": self.counts[CJK],
            "stripped_credits": self.stripped,
            "suspect_total": self.suspect_total,
            "suspect_recent": recent,
            "window_seconds": int(WINDOW_SECONDS),
            "status": status,
        }

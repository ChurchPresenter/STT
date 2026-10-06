"""Back off the music override when Whisper only answers it with credits.

The live loop lets confident music through to Whisper even where VAD says there is no speech,
so sung lines can be captured (hidden, restorable on /corrections). An instrumental prelude
goes through the same door, and Whisper answers a bar of organ with its stock subtitle credit,
over and over: on one Sunday, 52 rows in five minutes before anyone spoke. Nothing on screen
showed it, but every pass was a wasted decode and the loop was still running when the first
words arrived.

The two cases separate on what comes back, not on the audio. Voice VAD scores for singing sat
below 0.3 where the instrumental peaked at 0.38, so no VAD rule can tell them apart; but sung
audio yields lyrics and an instrumental yields only credits. So the override stays open until
``trip_after`` hallucinations arrive in a row with no real text between them, then shuts for
``cooldown`` seconds. When it reopens, one probe is allowed: real text closes the loop's
suspicion for good, another credit shuts the override again for twice as long, up to
``max_cooldown``. Speech that VAD hears is never affected, because this gate only governs the
override.

Stdlib-only; the clock is a parameter.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_TRIP_AFTER = 3
DEFAULT_COOLDOWN = 30.0
DEFAULT_MAX_COOLDOWN = 240.0


class MusicGate:
    """Decides whether the music override may force a decode right now."""

    def __init__(self, trip_after: int = DEFAULT_TRIP_AFTER, cooldown: float = DEFAULT_COOLDOWN,
                 max_cooldown: float = DEFAULT_MAX_COOLDOWN) -> None:
        self.configure(trip_after, cooldown, max_cooldown)

    def configure(self, trip_after: int, cooldown: float, max_cooldown: float = DEFAULT_MAX_COOLDOWN) -> None:
        """Take new settings and start over; called at each session start."""
        self.trip_after = max(1, int(trip_after))
        self.base_cooldown = max(0.0, float(cooldown))
        self.max_cooldown = max(self.base_cooldown, float(max_cooldown))
        self.reset()

    def reset(self) -> None:
        """Start a new session with the override fully open."""
        self.streak = 0
        self.closed_until: Optional[float] = None
        self.cooldown = self.base_cooldown
        self.trips = 0

    def allow(self, now: float) -> bool:
        """Whether the override may force a decode at ``now``. True once the cooldown has passed."""
        return self.closed_until is None or now >= self.closed_until

    def record(self, hallucination: bool, now: float) -> bool:
        """Note one decoded sentence. Returns True when this call shut the override.

        ``hallucination`` is a sentence denied as a credit, with nothing real left. Anything
        else, sung lyrics included, counts as real and reopens the override for good.
        """
        if not hallucination:
            self.streak = 0
            self.closed_until = None
            self.cooldown = self.base_cooldown
            return False
        if self.closed_until is not None and now < self.closed_until:
            return False  # a sentence already in flight when it shut says nothing new
        self.streak += 1
        probing = self.closed_until is not None and now >= self.closed_until
        if self.streak < self.trip_after and not probing:
            return False
        if probing:
            self.cooldown = min(self.cooldown * 2.0, self.max_cooldown)
        self.closed_until = now + self.cooldown
        self.streak = 0
        self.trips += 1
        return True

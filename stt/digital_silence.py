"""Telling a microphone that is blocked from one in a quiet room.

macOS charges microphone access to the app, and when that access is refused it
does not fail the capture. Every input device still lists, ffmpeg opens the one
selected, and audio arrives at the normal rate: every sample exactly zero. The
level meter reads 0, which looks like a quiet room, so an operator presses
Start, sees nothing wrong, and records nothing. ``audio_stalled`` cannot catch
it, because data *is* arriving.

A live analogue input never produces exact zeros: preamp noise and room tone keep
the low bits moving even in silence. A buffer of pure zeros for seconds on end
is something upstream withholding the signal: the OS privacy gate, a device
muted in software, or a digital source with nothing routed to it. Each of those
needs an operator to act, and none of them needs a level threshold, so this
tests for exact zeros only and never guesses at "quiet".

**Only a session that has never had signal counts.** A digital source *can*
send exact zeros legitimately: a mixer's USB or Dante feed with the channel
gated or muted between speakers, a loopback device with nothing playing. A
refused permission is different in one way that is easy to test: it is zero
from the first sample and never anything else. So the first non-zero sample
of a session settles it: that input works, and later silence is just silence.
The cost is one case that still warns: a gated feed started long before anyone
speaks shows the warning until the first sound, then never again.

The delay matches the 10s of ``audio_stalled`` so the two warnings agree on what
"for a while" means.
"""

from __future__ import annotations

from typing import Optional

#: How long input must be exactly zero before it counts, in seconds.
DEFAULT_THRESHOLD_SECONDS = 10.0


def is_digital_silence(pcm: bytes) -> bool:
    """Whether every byte of a PCM buffer is zero.

    Byte-wise is exact for any signed integer sample format, where zero is all
    zero bits. An empty buffer is not silence: it carries no evidence at all.
    """
    return bool(pcm) and not pcm.strip(b"\x00")


class DigitalSilence:
    """Tracks how long the input has been exactly zero, one chunk at a time.

    The caller supplies the clock so the judgement is testable without waiting.
    One instance belongs to one recording session; a new session starts fresh,
    and so forgets that the previous one ever heard anything.
    """

    def __init__(self, threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS) -> None:
        self.threshold_seconds = threshold_seconds
        self._zero_since: Optional[float] = None
        self._heard_signal = False

    def feed(self, pcm: bytes, now: float) -> bool:
        """Account for one chunk received at ``now``; return whether the input is silent.

        A single non-zero sample clears it at once and for the rest of the
        session: the input has proved it works, so a later gate or mute
        between speakers is not reported.
        """
        if self._heard_signal:
            return False
        if not pcm:
            return self.silent(now)
        if not is_digital_silence(pcm):
            self._heard_signal = True
            self._zero_since = None
            return False
        if self._zero_since is None:
            self._zero_since = now
        return self.silent(now)

    def silent(self, now: float) -> bool:
        """Whether the input has only ever been exactly zero, for at least the threshold."""
        if self._heard_signal or self._zero_since is None:
            return False
        return now - self._zero_since >= self.threshold_seconds

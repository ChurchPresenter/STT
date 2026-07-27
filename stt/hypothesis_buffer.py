"""LocalAgreement stabilization for the live (in-progress) transcription line.

Whisper re-decodes the un-finalized audio tail on every pass, so the live
hypothesis text keeps revising itself as new audio arrives — the caption
visibly rewrites (jitter). LocalAgreement-2 fixes the *displayed* text: a word
is only revealed once two consecutive hypotheses agree on it, so the shown
(confirmed) prefix never rewrites. The cost is a small, self-correcting lag —
the newest still-volatile word(s) are held back until they stabilize.

This only stabilizes the in-progress DISPLAY line; finalized/committed text is
produced by the transcription pipeline separately, so nothing is lost — held-
back words appear when the phrase finalizes. Reset the buffer between phrases /
whenever the underlying segment finalizes.

Pure/stdlib so it can be imported and unit-tested without the monolith.
"""

from typing import List


class LocalAgreementBuffer:
    """Word-level LocalAgreement-2 stabilizer for one evolving hypothesis."""

    def __init__(self) -> None:
        self._committed: List[str] = []  # words safe to display (never revised)
        self._prev: List[str] = []       # previous full hypothesis (words)

    def reset(self) -> None:
        """Start a fresh hypothesis (call when the current segment finalizes)."""
        self._committed = []
        self._prev = []

    def stabilize(self, text: str) -> str:
        """Feed the latest raw hypothesis; return the confirmed prefix to show.

        A word is confirmed once it appears at the same position in two
        consecutive hypotheses. Already-shown words are never revised: if the
        new hypothesis diverges from the committed prefix (a mid-segment
        correction), the committed prefix is kept and nothing new is confirmed
        until agreement resumes.
        """
        words = (text or "").split()
        n = len(self._committed)

        # New hypothesis must still begin with what we've already shown; if not
        # (a deep in-segment correction), don't rewrite shown text.
        if words[:n] != self._committed:
            self._prev = words
            return " ".join(self._committed)

        new_tail = words[n:]
        prev_tail = self._prev[n:]
        # Confirm the longest common prefix of the new and previous tails.
        i = 0
        while i < len(new_tail) and i < len(prev_tail) and new_tail[i] == prev_tail[i]:
            i += 1
        if i:
            self._committed.extend(new_tail[:i])
        self._prev = words
        return " ".join(self._committed)

    @property
    def committed_text(self) -> str:
        """The currently-confirmed text (what was last returned by stabilize)."""
        return " ".join(self._committed)

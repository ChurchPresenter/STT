"""What becomes of one finished sentence: keep it, keep it with a credit cut out, or deny it.

The decision used to be written out five times in ``speech_to_text.py`` (a sentence and a
remainder on each of the segment path and the phrase-timeout path, plus the stop flush), each
copy a few lines of "is it a hallucination, is it music, which reason wins". Fixing how Whisper's
stock subtitle credit is handled meant five identical edits and left a sixth to be missed.
This is the one place that rule lives, and it takes everything it needs as arguments so the
replay harness (:mod:`stt.asr_replay`) can ask the same question production asks.

Precedence is the one the callers always had: a hallucination outranks CJK, which outranks
music, and only the winning reason is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from stt.text_utils import is_whisper_hallucination, strip_hallucinations

REASON_HALLUCINATION = "hallucination"
REASON_CJK = "cjk"


@dataclass(frozen=True)
class Disposition:
    """The outcome for one sentence."""

    text: str                     # what to store (the credit already cut out when stripped)
    denied: bool
    reason: Optional[str]         # None when kept; else "hallucination", "cjk" or the music reason
    stripped: bool = False        # a credit was cut out and the speech around it kept
    hallucination: bool = False   # the input contained a known hallucination (kept or not)
    music: bool = False           # denied for music, and for nothing that outranks it


def decide(
    text: str,
    *,
    phrases: Sequence[str],
    cjk_deny: bool = False,
    music_label: Optional[str] = None,
    transcribe_music: bool = False,
    music_reason: str = "music",
) -> Disposition:
    """Decide ``text``'s fate.

    ``music_label`` is the audio type the segment was finalised with; it only denies when
    ``transcribe_music`` is off. A hallucination glued to real speech is cut out and the
    speech kept, so the sentence is denied for it only when nothing real remains.
    """
    hallucination = is_whisper_hallucination(text, phrases)
    stripped = False
    if hallucination:
        kept = strip_hallucinations(text, phrases)
        if kept and not is_whisper_hallucination(kept, phrases):
            text, stripped = kept, True
    still_hallucination = hallucination and not stripped
    music = (not transcribe_music) and music_label == "Music"
    denied = still_hallucination or cjk_deny or music
    if not denied:
        reason: Optional[str] = None
    elif still_hallucination:
        reason = REASON_HALLUCINATION
    elif cjk_deny:
        reason = REASON_CJK
    else:
        reason = music_reason
    return Disposition(
        text=text,
        denied=denied,
        reason=reason,
        stripped=stripped,
        hallucination=hallucination,
        music=music and not (still_hallucination or cjk_deny),
    )

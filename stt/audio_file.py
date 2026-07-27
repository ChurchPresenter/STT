"""Audio-file helpers for the transcription pipeline.

Extracted from the monolith so the logic is importable and unit-tested without
the monolith's import-time side effects. Stdlib-only: duration is read from a
WAV header via the ``wave`` module, so no ML/audio deps are required. Callers
that need non-WAV durations layer their own fallback (e.g. librosa) on top; this
helper deliberately returns ``None`` rather than raising on anything it can't
read, so a missing/garbage/non-WAV path degrades gracefully.
"""

import contextlib
import wave
from typing import Optional


def wav_duration_seconds(path: str) -> Optional[float]:
    """Total playback length of a WAV file in seconds, or ``None``.

    Reads only the header (frame count / sample rate) — no decode — so it is
    cheap even for long files. Returns ``None`` for a missing file, a non-WAV
    file, a zero sample rate, or any read error.
    """
    try:
        with contextlib.closing(wave.open(path, "rb")) as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        if rate <= 0:
            return None
        return frames / float(rate)
    except (wave.Error, OSError, EOFError, ValueError):
        return None

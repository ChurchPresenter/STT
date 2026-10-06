"""The monolith cannot be imported, so the wiring of the filter counters is checked on its source."""

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "speech_to_text.py").read_text(encoding="utf-8")


def _calls():
    return re.findall(r"decide_segment\((?!text)[^\n]*\)", SRC)


def test_every_music_aware_call_passes_the_audio_tag():
    # A call that knows the audio type but not the tag could never raise the speech-tagged alert.
    music_aware = [c for c in _calls() if "music_label=" in c]
    assert len(music_aware) == 4
    assert all("audio_tag=" in c for c in music_aware)


def test_health_exposes_filter_stats_while_running():
    assert '"filters": ts.get("filter_stats") if running else None' in SRC


def test_a_new_session_zeroes_the_filter_counts():
    assert "_disposition_stats.reset()" in SRC


def test_filter_log_tag_is_allowlisted_for_diagnostics():
    from stt.diagnostics import LOG_TAGS
    assert "FILTER" in LOG_TAGS and '[FILTER]' in SRC

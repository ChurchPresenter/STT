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


def test_corrections_page_gets_the_filter_stats_not_health():
    assert '@app.route("/api/corrections/filter-stats"' in SRC
    assert 'ts.get("filter_stats") if ts.get("running") else None' in SRC
    assert '"filters": ts.get("filter_stats")' not in SRC.split('def get_corrections_filter_stats')[0]


def test_corrections_template_polls_the_filter_stats():
    html = (Path(__file__).resolve().parent.parent / "templates" / "corrections.html").read_text(encoding="utf-8")
    assert "/api/corrections/filter-stats" in html and 'id="filter-alert"' in html
    health = (Path(__file__).resolve().parent.parent / "templates" / "health.html").read_text(encoding="utf-8")
    assert "c-filters" not in health


def test_a_new_session_zeroes_the_filter_counts():
    assert "_disposition_stats.reset()" in SRC


def test_filter_log_tag_is_allowlisted_for_diagnostics():
    from stt.diagnostics import LOG_TAGS
    assert "FILTER" in LOG_TAGS and '[FILTER]' in SRC


def test_music_override_is_governed_by_the_gate():
    assert "_music_gate.allow(time.time())" in SRC
    assert '_std_cfg.get("music_override_backoff", True)' in SRC


def test_every_decode_feeds_the_gate_and_a_new_session_reconfigures_it():
    assert '_music_gate.record(_credit_only.denied and _credit_only.reason == "hallucination", time.time())' in SRC
    assert "decide_segment(current_text, count=False)" in SRC
    assert "_music_gate.configure(" in SRC


def test_music_gate_log_tag_is_allowlisted():
    from stt.diagnostics import LOG_TAGS
    assert "MUSIC-GATE" in LOG_TAGS and "[MUSIC-GATE]" in SRC

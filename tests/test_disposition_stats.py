"""Counting what the sentence filters did (stt/disposition_stats.py)."""

from stt.disposition_stats import DEGRADED_AT, ERROR_AT, WINDOW_SECONDS, DispositionStats
from stt.segment_disposition import Disposition

KEPT = Disposition(text="x", denied=False, reason=None)
STRIPPED = Disposition(text="x", denied=False, reason=None, stripped=True, hallucination=True)
HALLUC = Disposition(text="x", denied=True, reason="hallucination", hallucination=True)
CJK = Disposition(text="x", denied=True, reason="cjk")
MUSIC = Disposition(text="x", denied=True, reason="music:0.5", music=True)


class TestCounts:
    def test_each_outcome_is_counted_in_its_bucket(self):
        s = DispositionStats()
        for d in (KEPT, KEPT, STRIPPED, HALLUC, CJK, MUSIC):
            s.record(d, "Speaking", 0.0)
        snap = s.snapshot(0.0)
        assert (snap["kept"], snap["denied_hallucination"], snap["denied_cjk"], snap["denied_music"]) == (3, 1, 1, 1)
        assert snap["stripped_credits"] == 1

    def test_music_reason_threshold_does_not_split_the_bucket(self):
        s = DispositionStats()
        s.record(Disposition("x", True, "music:0.7", music=True), "Music", 0.0)
        s.record(Disposition("x", True, "music:0.5", music=True), "Music", 0.0)
        assert s.snapshot(0.0)["denied_music"] == 2


class TestSuspect:
    def test_sung_music_is_not_suspect(self):
        s = DispositionStats()
        for t in range(20):
            s.record(MUSIC, "Music", float(t))
        snap = s.snapshot(20.0)
        assert snap["suspect_total"] == 0 and snap["status"] == "healthy"

    def test_speech_tagged_music_denials_raise_the_alert_in_steps(self):
        s = DispositionStats()
        for t in range(DEGRADED_AT):
            s.record(MUSIC, "Speech", float(t))
        assert s.snapshot(10.0)["status"] == "degraded"
        for t in range(DEGRADED_AT, ERROR_AT):
            s.record(MUSIC, "Speech", float(t))
        assert s.snapshot(10.0)["status"] == "error"

    def test_alert_clears_when_the_window_passes_but_the_total_stays(self):
        s = DispositionStats()
        for t in range(ERROR_AT):
            s.record(MUSIC, "Speech", float(t))
        later = ERROR_AT + WINDOW_SECONDS + 1
        snap = s.snapshot(later)
        assert snap["status"] == "healthy" and snap["suspect_recent"] == 0 and snap["suspect_total"] == ERROR_AT

    def test_kept_speech_never_counts(self):
        s = DispositionStats()
        for t in range(50):
            s.record(KEPT, "Speech", float(t))
        assert s.snapshot(50.0)["suspect_total"] == 0

"""Calibration analysis (stt/calibration.py)."""

import pytest

from stt.calibration import _percentile, analyze_calibration_data


def _empty():
    return {"noise_samples": [], "speech_samples": [], "silence_durations": [], "vad_probabilities": []}


class TestPercentile:
    def test_matches_numpy_linear_interpolation(self):
        # numpy.percentile(range(1..11), 75) == 8.5 with the default 'linear' method.
        assert _percentile(list(range(1, 12)), 75) == 8.5

    def test_endpoints(self):
        xs = [10, 20, 30, 40]
        assert _percentile(xs, 0) == 10
        assert _percentile(xs, 100) == 40

    def test_unsorted_input(self):
        assert _percentile([40, 10, 30, 20], 50) == 25.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _percentile([], 50)


class TestAnalyze:
    def test_noise_and_speech_gives_high_confidence_threshold(self):
        data = _empty()
        data["noise_samples"] = [{"energy": 100} for _ in range(12)]
        data["speech_samples"] = [{"energy": 2000} for _ in range(12)]
        r = analyze_calibration_data(data)
        assert "energy_threshold" in r["suggestions"]
        # threshold sits between noise and speech, clamped to [100, 10000]
        assert 100 <= r["suggestions"]["energy_threshold"] <= 10000
        assert r["analysis"]["threshold_confidence"] == "high"
        assert r["confidence"] == "high"  # >=10 of each

    def test_noise_only_warns_and_marks_low(self):
        data = _empty()
        data["noise_samples"] = [{"energy": 200} for _ in range(6)]
        r = analyze_calibration_data(data)
        assert r["analysis"]["threshold_confidence"] == "low"
        assert any("No speech detected" in w for w in r.get("warnings", []))
        assert r["suggestions"]["energy_threshold"] >= 300  # conservative floor
        assert r["confidence"] == "medium"  # 6 noise samples

    def test_silence_durations_suggest_phrase_timeout(self):
        data = _empty()
        data["silence_durations"] = [0.8, 1.0, 1.2, 6.0]  # 6.0 filtered out (>=5)
        r = analyze_calibration_data(data)
        # median of [0.8,1.0,1.2] = 1.0 -> *1.3 = 1.3, clamped to [1,5]
        assert r["suggestions"]["phrase_timeout"] == pytest.approx(1.3)
        assert r["analysis"]["pause_pattern"]["total_pauses"] == 4

    @pytest.mark.parametrize("avg,expected", [(0.9, 0.6), (0.7, 0.5), (0.4, 0.4)])
    def test_vad_threshold_branches(self, avg, expected):
        data = _empty()
        data["vad_probabilities"] = [avg, avg]
        r = analyze_calibration_data(data)
        assert r["suggestions"]["vad_threshold"] == expected
        assert r["suggestions"]["vad_enabled"] is True

    def test_empty_data_is_low_confidence_no_suggestions(self):
        r = analyze_calibration_data(_empty())
        assert r["confidence"] == "low"
        assert r["suggestions"] == {}

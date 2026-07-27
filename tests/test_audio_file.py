"""WAV duration reading (stt/audio_file.py)."""

import wave

import pytest

from stt.audio_file import wav_duration_seconds


def _write_wav(path, seconds, rate=16000, channels=1, sampwidth=2):
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(rate)
        wav.writeframes(b"\x00" * frames * channels * sampwidth)


class TestWavDurationSeconds:
    def test_reads_duration_of_a_generated_wav(self, tmp_path):
        p = tmp_path / "tone.wav"
        _write_wav(p, seconds=2.5)
        assert wav_duration_seconds(str(p)) == pytest.approx(2.5)

    def test_fractional_and_odd_rate(self, tmp_path):
        p = tmp_path / "odd.wav"
        _write_wav(p, seconds=1, rate=44100)
        # 44100 frames / 44100 Hz == 1.0s regardless of the odd rate
        assert wav_duration_seconds(str(p)) == pytest.approx(1.0)

    def test_missing_file_returns_none(self, tmp_path):
        assert wav_duration_seconds(str(tmp_path / "nope.wav")) is None

    def test_non_wav_file_returns_none(self, tmp_path):
        p = tmp_path / "garbage.wav"
        p.write_bytes(b"this is not a RIFF/WAVE header")
        assert wav_duration_seconds(str(p)) is None

    def test_empty_wav_is_zero_not_none(self, tmp_path):
        p = tmp_path / "silent.wav"
        _write_wav(p, seconds=0)
        assert wav_duration_seconds(str(p)) == pytest.approx(0.0)

    def test_directory_path_returns_none(self, tmp_path):
        assert wav_duration_seconds(str(tmp_path)) is None

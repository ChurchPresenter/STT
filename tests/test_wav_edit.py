"""Trimming session recordings (stt/wav_edit.py).

The cases that matter are the ones peculiar to these files: a header that lies
about how long the recording is, and a trim that must never write over the
recording it was given.
"""

import os
import struct

import pytest

from stt.wav_edit import (
    WavError,
    clamp_range,
    find_speech_bounds,
    peaks,
    read_info,
    trim,
    trimmed_name,
)

RATE = 16000


def write_wav(path, frames, *, rate=RATE, channels=1, width=2,
              declared_frames=None, samples=None, extra_chunk=False):
    """A PCM WAV. ``declared_frames`` fakes the stale header these captures have."""
    data = (samples if samples is not None
            else struct.pack("<%dh" % (frames * channels), *([0] * frames * channels)))
    declared = len(data) if declared_frames is None else declared_frames * channels * width
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        fh.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                       rate * channels * width, channels * width, width * 8))
        if extra_chunk:                      # real files carry these before data
            fh.write(b"LIST" + struct.pack("<I", 4) + b"INFO")
        fh.write(b"data" + struct.pack("<I", declared))
        fh.write(data)
    return str(path)


def tone(frames, amplitude=8000, channels=1):
    """Frames of a loud constant value — "speech" for the bounds tests."""
    return struct.pack("<%dh" % (frames * channels), *([amplitude] * frames * channels))


def quiet(frames, channels=1):
    return struct.pack("<%dh" % (frames * channels), *([0] * frames * channels))


class TestReadInfo:
    def test_reads_the_format(self, tmp_path):
        info = read_info(write_wav(tmp_path / "a.wav", RATE))
        assert (info.channels, info.sample_width, info.sample_rate) == (1, 2, RATE)
        assert info.frames == RATE
        assert info.seconds == pytest.approx(1.0)

    def test_a_stale_header_is_corrected_from_the_file(self, tmp_path):
        # The real fault: a capture written as a stream never patches its size
        # fields, so a long recording can declare one second of audio.
        p = write_wav(tmp_path / "b.wav", RATE * 60, declared_frames=RATE)
        info = read_info(p)
        assert info.header_frames == RATE          # what it claimed
        assert info.frames == RATE * 60            # what is actually there
        assert info.header_is_stale
        assert info.seconds == pytest.approx(60.0)

    def test_an_honest_header_is_trusted(self, tmp_path):
        info = read_info(write_wav(tmp_path / "c.wav", RATE * 3))
        assert not info.header_is_stale
        assert info.frames == RATE * 3

    def test_chunks_before_data_are_skipped(self, tmp_path):
        # A fixed 44-byte offset would read the LIST chunk as audio.
        info = read_info(write_wav(tmp_path / "d.wav", RATE, extra_chunk=True))
        assert info.frames == RATE
        assert info.data_offset > 44

    def test_not_a_wav(self, tmp_path):
        p = tmp_path / "e.wav"
        p.write_bytes(b"this is not a wav file at all")
        with pytest.raises(WavError):
            read_info(str(p))

    def test_stereo_and_width_are_honoured(self, tmp_path):
        p = write_wav(tmp_path / "f.wav", RATE, channels=2)
        info = read_info(p)
        assert info.channels == 2 and info.frame_size == 4 and info.frames == RATE


class TestClampRange:
    def setup_method(self):
        self.info = None

    def test_range_maps_to_frames(self, tmp_path):
        info = read_info(write_wav(tmp_path / "a.wav", RATE * 10))
        assert clamp_range(info, 2.0, 5.0) == (2 * RATE, 3 * RATE)

    def test_end_defaults_to_the_end_of_file(self, tmp_path):
        info = read_info(write_wav(tmp_path / "a.wav", RATE * 10))
        assert clamp_range(info, 4.0) == (4 * RATE, 6 * RATE)

    def test_past_the_end_is_clamped(self, tmp_path):
        info = read_info(write_wav(tmp_path / "a.wav", RATE * 10))
        assert clamp_range(info, 8.0, 999.0) == (8 * RATE, 2 * RATE)

    def test_negative_start_is_clamped(self, tmp_path):
        info = read_info(write_wav(tmp_path / "a.wav", RATE * 10))
        assert clamp_range(info, -5.0, 1.0) == (0, RATE)

    def test_an_empty_range_is_refused(self, tmp_path):
        # Better to fail than to write a zero-length recording.
        info = read_info(write_wav(tmp_path / "a.wav", RATE * 10))
        with pytest.raises(WavError):
            clamp_range(info, 5.0, 5.0)
        with pytest.raises(WavError):
            clamp_range(info, 6.0, 2.0)


class TestFindSpeechBounds:
    def test_finds_speech_after_leading_silence(self, tmp_path):
        p = write_wav(tmp_path / "a.wav", 0,
                      samples=quiet(RATE * 2) + tone(RATE) + quiet(RATE * 2))
        first, last = find_speech_bounds(p)
        assert first == pytest.approx(2.0, abs=0.1)
        assert last == pytest.approx(3.0, abs=0.1)

    def test_a_wholly_silent_file_returns_the_whole_span(self, tmp_path):
        # Never hand back an empty range: trimming by it would destroy the file.
        p = write_wav(tmp_path / "b.wav", RATE * 3)
        assert find_speech_bounds(p) == (0.0, pytest.approx(3.0))

    def test_speech_from_the_first_sample(self, tmp_path):
        p = write_wav(tmp_path / "c.wav", 0, samples=tone(RATE) + quiet(RATE))
        first, _last = find_speech_bounds(p)
        assert first == pytest.approx(0.0, abs=0.05)

    def test_threshold_is_respected(self, tmp_path):
        # A quiet hiss counts as silence at the default and as speech below it.
        p = write_wav(tmp_path / "d.wav", 0, samples=quiet(RATE) + tone(RATE, 100))
        assert find_speech_bounds(p, threshold=0.01)[0] == pytest.approx(0.0)
        assert find_speech_bounds(p, threshold=0.001)[0] == pytest.approx(1.0, abs=0.1)


class TestTrim:
    def test_writes_the_selected_range(self, tmp_path):
        src = write_wav(tmp_path / "a.wav", RATE * 10)
        out = trim(src, str(tmp_path / "out.wav"), 2.0, 5.0)
        assert out.frames == 3 * RATE
        assert out.seconds == pytest.approx(3.0)
        assert not out.header_is_stale       # the new header tells the truth

    def test_the_audio_is_the_same_samples(self, tmp_path):
        body = quiet(RATE) + tone(RATE) + quiet(RATE)
        src = write_wav(tmp_path / "a.wav", 0, samples=body)
        dest = str(tmp_path / "out.wav")
        trim(src, dest, 1.0, 2.0)
        info = read_info(dest)
        with open(dest, "rb") as fh:
            fh.seek(info.data_offset)
            assert fh.read() == tone(RATE)

    def test_trimming_a_stale_header_file_uses_the_real_length(self, tmp_path):
        src = write_wav(tmp_path / "a.wav", RATE * 30, declared_frames=RATE)
        out = trim(src, str(tmp_path / "out.wav"), 10.0)
        assert out.seconds == pytest.approx(20.0)

    def test_refuses_to_overwrite_an_existing_file(self, tmp_path):
        src = write_wav(tmp_path / "a.wav", RATE)
        dest = tmp_path / "out.wav"
        dest.write_bytes(b"precious")
        with pytest.raises(WavError):
            trim(src, str(dest), 0.0)
        assert dest.read_bytes() == b"precious"

    def test_refuses_to_edit_in_place(self, tmp_path):
        src = write_wav(tmp_path / "a.wav", RATE)
        before = open(src, "rb").read()
        with pytest.raises(WavError):
            trim(src, src, 0.0, 0.5)
        assert open(src, "rb").read() == before

    def test_an_empty_selection_writes_nothing(self, tmp_path):
        src = write_wav(tmp_path / "a.wav", RATE * 5)
        dest = str(tmp_path / "out.wav")
        with pytest.raises(WavError):
            trim(src, dest, 3.0, 3.0)
        assert not os.path.exists(dest)

    def test_stereo_round_trips(self, tmp_path):
        src = write_wav(tmp_path / "a.wav", RATE * 4, channels=2)
        out = trim(src, str(tmp_path / "out.wav"), 1.0, 3.0)
        assert out.channels == 2
        assert out.seconds == pytest.approx(2.0)


class TestTrimmedName:
    def test_sits_beside_the_original(self, tmp_path):
        src = str(tmp_path / "2026-08-05_182905.wav")
        assert trimmed_name(src).endswith("2026-08-05_182905_trimmed.wav")

    def test_does_not_collide(self, tmp_path):
        src = tmp_path / "a.wav"
        src.write_bytes(b"")
        (tmp_path / "a_trimmed.wav").write_bytes(b"")
        assert trimmed_name(str(src)).endswith("a_trimmed2.wav")


class TestPeaks:
    """The envelope a waveform is drawn from."""

    def test_silence_reads_flat_and_speech_reads_loud(self, tmp_path):
        p = write_wav(tmp_path / "a.wav", 0,
                      samples=quiet(RATE) + tone(RATE, 16000) + quiet(RATE))
        values = peaks(p, buckets=30)
        assert len(values) == 30
        assert max(values[:10]) == 0.0          # first second silent
        assert min(values[10:20]) > 0.4         # second second loud
        assert max(values[20:]) == 0.0          # third silent

    def test_values_are_scaled_to_full_scale(self, tmp_path):
        p = write_wav(tmp_path / "b.wav", 0, samples=tone(RATE, 32767))
        assert max(peaks(p, buckets=10)) == pytest.approx(1.0, abs=0.001)

    def test_a_range_reads_only_that_range(self, tmp_path):
        # Zooming into the silent tail must not show the speech before it.
        p = write_wav(tmp_path / "c.wav", 0, samples=tone(RATE, 16000) + quiet(RATE))
        assert max(peaks(p, buckets=10, start_seconds=1.0)) == 0.0
        assert max(peaks(p, buckets=10, end_seconds=1.0)) > 0.4

    def test_bucket_count_is_honoured(self, tmp_path):
        p = write_wav(tmp_path / "d.wav", RATE * 4)
        assert len(peaks(p, buckets=200)) == 200

    def test_more_buckets_than_frames_is_capped(self, tmp_path):
        p = write_wav(tmp_path / "e.wav", 50)
        assert len(peaks(p, buckets=5000)) == 50

    def test_an_empty_range_raises(self, tmp_path):
        p = write_wav(tmp_path / "f.wav", RATE)
        with pytest.raises(WavError):
            peaks(p, start_seconds=0.5, end_seconds=0.5)

"""The full-quality listening branch of the transcription capture (stt/audio_capture.py).

The meter's full-quality stream used to open the device a second time, which an ALSA
plughw: device refuses ("Device or resource busy"). It is now one more output of the
transcription ffmpeg, written to an inherited pipe. What matters: the 16 kHz Whisper
output and the .ts backup are untouched, and the reader can never stall ffmpeg.
"""

import os
import sys
import threading

import pytest

from stt import audio_capture
from stt.audio_capture import FFmpegAudioCapture
from stt.hq_audio import chunk_bytes


def capture(tmp_path, ts_enabled=True, **kw):
    return FFmpegAudioCapture(device_name="plughw:1,0", backup_dir=str(tmp_path),
                              filename_format="FIXED", ts_enabled=ts_enabled, **kw)


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(audio_capture, "_IS_WINDOWS", False)


def outputs(cmd):
    """{map label: the arguments of that output}"""
    out, i = {}, 0
    while "-map" in cmd[i:]:
        i = cmd.index("-map", i)
        nxt = cmd.index("-map", i + 1) if "-map" in cmd[i + 1:] else len(cmd)
        out[cmd[i + 1]] = cmd[i + 2:nxt]
        i = nxt
    return out


class TestCommand:
    def test_adds_a_third_branch_on_the_extra_pipe(self, tmp_path, linux):
        src = capture(tmp_path, hq_format=(48000, 2), on_hq_chunk=lambda c: None)
        cmd = src._get_ffmpeg_command(hq_fd=7)
        assert cmd[cmd.index("-filter_complex") + 1] == "[0:a]asplit=3[a1][a2][a3]"
        outs = outputs(cmd)
        assert outs["[a1]"][-1] == "pipe:1"
        assert outs["[a2]"][-1] == src.backup_file
        hq = outs["[a3]"]
        assert hq[-1] == "pipe:7"
        assert hq[hq.index("-ar") + 1] == "48000" and hq[hq.index("-ac") + 1] == "2"
        assert hq[hq.index("-f") + 1] == "s16le"

    def test_whisper_output_keeps_16k_mono(self, tmp_path, linux):
        src = capture(tmp_path, hq_format=(48000, 2), on_hq_chunk=lambda c: None)
        cmd = src._get_ffmpeg_command(hq_fd=7)
        head = cmd[:cmd.index("-filter_complex")]
        assert head[-4:] == ["-ar", "16000", "-ac", "1"]  # before the first output: applies to pipe:1 only

    def test_without_backup_splits_in_two(self, tmp_path, linux):
        src = capture(tmp_path, ts_enabled=False, hq_format=(48000, 2), on_hq_chunk=lambda c: None)
        cmd = src._get_ffmpeg_command(hq_fd=5)
        assert cmd[cmd.index("-filter_complex") + 1] == "[0:a]asplit=2[a1][a2]"
        assert outputs(cmd)["[a2]"][-1] == "pipe:5"

    @pytest.mark.parametrize("ts_enabled", [True, False])
    def test_without_the_pipe_the_command_is_unchanged(self, tmp_path, linux, monkeypatch, ts_enabled):
        monkeypatch.setattr(audio_capture, "datetime", _FixedNow)
        plain = capture(tmp_path, ts_enabled=ts_enabled)._get_ffmpeg_command()
        with_hq = capture(tmp_path, ts_enabled=ts_enabled, hq_format=(48000, 2),
                          on_hq_chunk=lambda c: None)._get_ffmpeg_command()
        assert with_hq == plain

    def test_windows_keeps_the_second_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audio_capture, "_IS_WINDOWS", True)
        src = capture(tmp_path, hq_format=(48000, 2), on_hq_chunk=lambda c: None)
        assert src.hq_format is None
        assert src._open_hq_pipe() == (None, None)


class _FixedNow:
    @staticmethod
    def now():
        import datetime as dt
        return dt.datetime(2026, 1, 1, 12, 0, 0)


class TestReader:
    def run_reader(self, src, payload):
        r, w = os.pipe()
        t = threading.Thread(target=src._read_hq, args=(r,))
        t.start()
        os.write(w, payload)
        os.close(w)
        t.join(timeout=2)
        assert not t.is_alive()

    def test_delivers_whole_100ms_chunks(self, tmp_path, linux):
        got = []
        src = capture(tmp_path, hq_format=(48000, 2), on_hq_chunk=got.append)
        size = chunk_bytes(48000, 2)
        payload = bytes(range(256)) * (size * 5 // 2 // 256) + b"\x00" * (size * 5 // 2 % 256)
        self.run_reader(src, payload)
        assert [len(c) for c in got] == [size, size]
        assert b"".join(got) == payload[:2 * size]

    def test_a_failing_handler_does_not_stop_the_drain(self, tmp_path, linux):
        calls = []

        def boom(chunk):
            calls.append(chunk)
            raise RuntimeError("listener gone")
        src = capture(tmp_path, hq_format=(48000, 2), on_hq_chunk=boom)
        self.run_reader(src, bytes(chunk_bytes(48000, 2) * 3))
        assert len(calls) == 3

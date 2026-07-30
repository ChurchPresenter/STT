"""The ffmpeg command built for each platform (stt/audio_capture.py).

This is the one place where all three supported platforms diverge in pure
logic — ALSA, avfoundation and dshow name devices differently, and each has a
backup-recording variant. It is also the code least likely to be exercised by
whoever changes it: a developer on one platform never runs the other two
branches, so a typo in the dshow arguments would ship and only surface as
"transcription does not start" on a Windows install.

sys.platform is patched rather than skipping per-runner, so every branch is
checked everywhere. No ffmpeg is launched: the method only assembles a list.
"""

import os
import sys

import pytest

from stt.audio_capture import FFmpegAudioCapture


@pytest.fixture
def source(tmp_path):
    """A capture source whose backup directory is disposable."""
    return FFmpegAudioCapture(sample_rate=16000, backup_dir=str(tmp_path / "backup"),
                             filename_format="%Y-%m-%d_%H%M%S", ts_enabled=False)


def build(src, platform, monkeypatch):
    monkeypatch.setattr(sys, "platform", platform)
    return src._get_ffmpeg_command()


class TestPlatformInputArguments:
    """Each platform's capture backend and device syntax."""

    @pytest.mark.parametrize("platform,fmt", [
        ("linux", "alsa"), ("linux2", "alsa"),
        ("darwin", "avfoundation"),
        ("win32", "dshow"),
    ])
    def test_capture_backend_per_platform(self, source, monkeypatch, platform, fmt):
        cmd = build(source, platform, monkeypatch)
        assert cmd[cmd.index("-f") + 1] == fmt

    def test_linux_defaults_to_the_alsa_default_device(self, source, monkeypatch):
        cmd = build(source, "linux", monkeypatch)
        assert cmd[cmd.index("-i") + 1] == "default"

    def test_macos_defaults_to_the_first_avfoundation_device(self, source, monkeypatch):
        cmd = build(source, "darwin", monkeypatch)
        assert cmd[cmd.index("-i") + 1] == ":0"

    def test_windows_defaults_to_a_named_dshow_microphone(self, source, monkeypatch):
        cmd = build(source, "win32", monkeypatch)
        assert cmd[cmd.index("-i") + 1] == "audio=Microphone"

    def test_named_device_is_passed_through_on_linux(self, tmp_path, monkeypatch):
        src = FFmpegAudioCapture(device_name="plughw:1,0", backup_dir=str(tmp_path),
                                ts_enabled=False)
        assert build(src, "linux", monkeypatch)[build(src, "linux", monkeypatch).index("-i") + 1] == "plughw:1,0"

    def test_named_device_is_prefixed_for_avfoundation(self, tmp_path, monkeypatch):
        src = FFmpegAudioCapture(device_name="Scarlett", backup_dir=str(tmp_path),
                                ts_enabled=False)
        cmd = build(src, "darwin", monkeypatch)
        assert cmd[cmd.index("-i") + 1] == ":Scarlett", "avfoundation indexes with a leading colon"

    def test_named_device_is_prefixed_for_dshow(self, tmp_path, monkeypatch):
        src = FFmpegAudioCapture(device_name="Line In (USB)", backup_dir=str(tmp_path),
                                ts_enabled=False)
        cmd = build(src, "win32", monkeypatch)
        assert cmd[cmd.index("-i") + 1] == "audio=Line In (USB)", (
            "dshow needs the audio= prefix; without it ffmpeg looks for a video device")

    def test_an_unknown_platform_is_refused_rather_than_guessed(self, source, monkeypatch):
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            build(source, "sunos5", monkeypatch)


class TestOutputIsAlwaysPipeablePcm:
    """Whatever the platform, the pipe carries what the reader expects."""

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_stdout_is_signed_16_bit_mono_at_the_requested_rate(self, source, monkeypatch, platform):
        cmd = build(source, platform, monkeypatch)
        assert "pipe:1" in cmd
        assert cmd[cmd.index("-ar") + 1] == "16000"
        assert cmd[cmd.index("-ac") + 1] == "1"
        assert "s16le" in cmd

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_sample_rate_follows_the_constructor(self, tmp_path, monkeypatch, platform):
        src = FFmpegAudioCapture(sample_rate=48000, backup_dir=str(tmp_path), ts_enabled=False)
        assert build(src, platform, monkeypatch)[build(src, platform, monkeypatch).index("-ar") + 1] == "48000"

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_overwrites_without_prompting(self, source, monkeypatch, platform):
        # Without -y ffmpeg blocks on a confirmation prompt no one can answer.
        assert "-y" in build(source, platform, monkeypatch)


class TestBackupRecording:
    """ts_enabled splits the stream: PCM to the pipe, MPEG-TS to disk."""

    @pytest.fixture
    def recording(self, tmp_path):
        return FFmpegAudioCapture(backup_dir=str(tmp_path / "backup"),
                                 filename_format="%Y-%m-%d_%H%M%S", ts_enabled=True)

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_splits_into_a_pipe_and_a_file(self, recording, monkeypatch, platform):
        cmd = build(recording, platform, monkeypatch)
        assert "asplit=2[a1][a2]" in " ".join(cmd)
        assert "pipe:1" in cmd
        assert cmd[-1] == recording.backup_file

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_backup_is_mpegts_with_a_real_codec(self, recording, monkeypatch, platform):
        # MPEG-TS with mp2 is what survives a power cut mid-service; raw PCM in
        # a .ts would not be recoverable.
        cmd = build(recording, platform, monkeypatch)
        assert "mpegts" in cmd and "mp2" in cmd

    def test_creates_the_backup_directory(self, recording, monkeypatch):
        build(recording, "linux", monkeypatch)
        assert os.path.isdir(os.path.dirname(recording.backup_file))

    def test_prefix_is_appended_after_the_timestamp(self, tmp_path, monkeypatch):
        src = FFmpegAudioCapture(backup_dir=str(tmp_path), filename_format="FIXED",
                                filename_prefix="service", ts_enabled=True)
        build(src, "linux", monkeypatch)
        assert os.path.basename(src.backup_file) == "FIXED_service.ts"

    def test_without_a_prefix_the_timestamp_stands_alone(self, tmp_path, monkeypatch):
        src = FFmpegAudioCapture(backup_dir=str(tmp_path), filename_format="FIXED",
                                ts_enabled=True)
        build(src, "linux", monkeypatch)
        assert os.path.basename(src.backup_file) == "FIXED.ts"

    def test_no_backup_file_when_recording_is_off(self, source, monkeypatch):
        cmd = build(source, "linux", monkeypatch)
        assert "mpegts" not in cmd
        assert not any(str(c).endswith(".ts") for c in cmd)


class TestFilePlayback:
    """A path instead of a device: how a session is replayed for testing.

    This branch is taken before the platform check, so a wav plays back
    identically everywhere — which is what makes it usable as a test input.
    """

    @pytest.fixture
    def wav(self, tmp_path):
        path = tmp_path / "service.wav"
        path.write_bytes(b"RIFF....WAVE")
        return str(path)

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_a_file_path_is_read_directly_on_every_platform(self, tmp_path, wav, monkeypatch, platform):
        src = FFmpegAudioCapture(device_name=wav, backup_dir=str(tmp_path), ts_enabled=False)
        cmd = build(src, platform, monkeypatch)
        assert cmd[cmd.index("-i") + 1] == wav
        assert "alsa" not in cmd and "dshow" not in cmd and "avfoundation" not in cmd

    def test_playback_is_rate_limited_to_real_time(self, tmp_path, wav, monkeypatch):
        # Without -re ffmpeg decodes the whole file instantly and the pipeline
        # sees a service's worth of audio in one burst.
        src = FFmpegAudioCapture(device_name=wav, backup_dir=str(tmp_path), ts_enabled=False)
        assert "-re" in build(src, "linux", monkeypatch)

    def test_playback_still_records_a_backup_when_enabled(self, tmp_path, wav, monkeypatch):
        src = FFmpegAudioCapture(device_name=wav, backup_dir=str(tmp_path / "b"),
                                filename_format="FIXED", ts_enabled=True)
        cmd = build(src, "linux", monkeypatch)
        assert cmd[-1] == src.backup_file
        assert "mpegts" in cmd

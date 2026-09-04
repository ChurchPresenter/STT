"""Testable slices of stt/audio_capture.py: device parsing/resolution and
ffmpeg command construction.

A live microphone is out of scope, but the capture loop is not entirely: the
open-failure path is exercised against a real ffmpeg at the bottom of this file
(skipped when ffmpeg is absent). It has to be. The bug it pins was that on Unix
none of the loop's three open-failure branches ever ran — select() reports a
closed pipe as readable, so a failed device reaches the EOF branch, which
reported nothing — and no amount of testing the decision in isolation could
show that the signal never arrived."""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import pytest

from stt.audio_capture import (
    FFmpegAudioCapture,
    create_compatible_audio_source,
    summarise_ffmpeg_error,
    parse_asound_cards,
    resolve_audio_device_by_name,
)

CARDS = """ 0 [NVidia         ]: HDA-Intel - HDA NVidia
                      HDA NVidia at 0xfcffc000 irq 22
 1 [USB            ]: USB-Audio - Blue Yeti USB Microphone
                      Blue Microphones Blue Yeti at usb-0000:00:14.0-2
"""


class TestParseAsoundCards:
    def test_parses_cards_with_plughw_names(self):
        devices = parse_asound_cards(CARDS)
        assert [d["name"] for d in devices] == ["plughw:0,0", "plughw:1,0"]
        # The type/description split happens at the FIRST hyphen, so a hyphenated
        # driver name ("HDA-Intel") leaks its tail into the display name.
        assert devices[0]["display_name"] == "Intel - HDA NVidia"
        assert devices[1]["card_id"] == "USB"

    def test_first_device_default_when_none_deprioritized(self):
        devices = parse_asound_cards(CARDS)
        assert [d["is_default"] for d in devices] == [True, False]

    def test_deprioritized_first_card_yields_second_default(self):
        # HDMI/GPU audio should not win the default slot over a real mic
        devices = parse_asound_cards(CARDS, deprioritize_markers=["nvidia"])
        assert [d["is_default"] for d in devices] == [False, True]

    def test_all_deprioritized_falls_back_to_first(self):
        devices = parse_asound_cards(CARDS, deprioritize_markers=["nvidia", "usb"])
        assert [d["is_default"] for d in devices] == [True, False]

    def test_internal_flag_stripped_from_output(self):
        for d in parse_asound_cards(CARDS, deprioritize_markers=["nvidia"]):
            assert "is_deprioritized" not in d

    def test_empty_or_garbage_content(self):
        assert parse_asound_cards("") == []
        assert parse_asound_cards("no cards here\njust noise") == []


DEVICES = [
    {"name": "plughw:0,0", "card_id": "NVidia", "display_name": "HDA NVidia"},
    {"name": "plughw:1,0", "card_id": "USB", "display_name": "Blue Yeti USB Microphone"},
]


class TestResolveDeviceByName:
    def test_matches_display_name_substring_case_insensitive(self):
        dev = resolve_audio_device_by_name("blue yeti", DEVICES)
        assert dev["name"] == "plughw:1,0"

    def test_matches_card_id(self):
        assert resolve_audio_device_by_name("NVidia", DEVICES)["name"] == "plughw:0,0"

    def test_card_id_contained_in_saved_name(self):
        # e.g. saved "USB Audio Device" matches card_id "USB"
        assert resolve_audio_device_by_name("USB Audio Device", DEVICES)["name"] == "plughw:1,0"

    def test_no_match_or_empty(self):
        assert resolve_audio_device_by_name("nonexistent mic", DEVICES) is None
        assert resolve_audio_device_by_name("", DEVICES) is None
        assert resolve_audio_device_by_name("   ", DEVICES) is None
        assert resolve_audio_device_by_name(None, DEVICES) is None

    def test_a_device_with_no_card_id_does_not_match_everything(self):
        # STT#12: Windows (and macOS) enumeration never sets card_id, so it's
        # '' on every device. The old third arm was `card_id in needle`, which
        # for an empty card_id is `'' in needle` — true of every non-empty
        # string, so devices[0] matched unconditionally regardless of the
        # saved name. A saved name that matches nothing must still miss.
        no_card_id_devices = [
            {"name": "dshow:0", "card_id": "", "display_name": "Conexant HD Audio"},
            {"name": "dshow:1", "card_id": "", "display_name": "Line In (Realtek)"},
        ]
        assert resolve_audio_device_by_name("UR22mkII", no_card_id_devices) is None
        assert resolve_audio_device_by_name("Line In (Realtek)", no_card_id_devices)["name"] == "dshow:1"


class TestInit:
    def test_chunk_size_and_sample_width(self):
        cap = FFmpegAudioCapture(sample_rate=16000, chunk_duration=0.5, ts_enabled=False)
        assert cap.chunk_size == 8000
        assert cap.SAMPLE_WIDTH == 2
        assert cap.SAMPLE_RATE == 16000

    def test_ts_disabled_has_no_backup_dir(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert cap.backup_dir is None

    def test_explicit_backup_dir_honored(self, tmp_path):
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), ts_enabled=True)
        assert cap.backup_dir == str(tmp_path)

    def test_filename_defaults(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert cap.filename_format == "%Y-%m-%d_%H%M%S"
        assert cap.filename_prefix == ""

    def test_flush_buffer_sets_event(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert not cap._flush_event.is_set()
        cap.flush_buffer()
        assert cap._flush_event.is_set()

    def test_playback_finished_event_starts_unset(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert isinstance(cap.playback_finished, threading.Event)
        assert not cap.playback_finished.is_set()

    def test_open_signal_starts_unset_and_unfailed(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert isinstance(cap._open_event, threading.Event)
        assert not cap._open_event.is_set()
        assert cap._open_error is None

    def test_open_timeout_defaults_and_is_overridable(self):
        assert FFmpegAudioCapture(ts_enabled=False).open_timeout == 5.0
        assert FFmpegAudioCapture(ts_enabled=False, open_timeout=1.5).open_timeout == 1.5

    def test_stall_tracking_starts_clear(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert cap.last_data_at is None
        assert cap.is_stalled is False

    def test_signal_eof_sets_event_only_for_a_file_source(self, tmp_path):
        # A real file path -> EOF means the file ended -> event set.
        f = tmp_path / "clip.wav"
        f.write_bytes(b"RIFF")
        cap_file = FFmpegAudioCapture(device_name=str(f), ts_enabled=False)
        cap_file._signal_eof_if_file()
        assert cap_file.playback_finished.is_set()

        # A mic device name is not a file -> EOF must NOT auto-stop the session.
        cap_mic = FFmpegAudioCapture(device_name="plughw:1,0", ts_enabled=False)
        cap_mic._signal_eof_if_file()
        assert not cap_mic.playback_finished.is_set()


class TestGetFfmpegCommand:
    def test_backup_file_named_from_format_and_prefix(self, tmp_path):
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), filename_prefix="sunday", ts_enabled=True)
        cap.device_name = None
        cmd = cap._get_ffmpeg_command()
        assert cap.backup_file.startswith(str(tmp_path) + os.sep)
        assert cap.backup_file.endswith("_sunday.ts")
        assert cmd[-1] == cap.backup_file  # backup file is the mpegts output

    def test_backup_dir_created(self, tmp_path):
        backup = tmp_path / "2026" / "07"
        cap = FFmpegAudioCapture(backup_dir=str(backup), ts_enabled=True)
        cap._get_ffmpeg_command()
        assert backup.is_dir()

    def test_ts_split_counter_increments(self, tmp_path):
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), ts_enabled=True)
        cap._get_ffmpeg_command()
        cap._get_ffmpeg_command()
        assert cap._ts_file_count == 2

    def test_file_playback_mode(self, tmp_path):
        wav = tmp_path / "input.wav"
        wav.write_bytes(b"\0")
        cap = FFmpegAudioCapture(device_name=str(wav), ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert cmd[:3] == ["ffmpeg", "-y", "-re"]  # -re: real-time pacing for file input
        assert str(wav) in cmd
        assert "s16le" in cmd and "pipe:1" in cmd

    def test_mic_mode_no_ts_is_pcm_only(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        cap = FFmpegAudioCapture(device_name="plughw:1,0", ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert ["-f", "alsa", "-i", "plughw:1,0"] == cmd[2:6]
        assert "mpegts" not in cmd
        assert "-filter_complex" not in cmd

    def test_mic_mode_with_ts_splits_to_backup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), ts_enabled=True)
        cmd = cap._get_ffmpeg_command()
        assert "alsa" in cmd
        assert "-filter_complex" in cmd
        assert "mpegts" in cmd and cmd[-1] == cap.backup_file

    def test_darwin_device_gets_colon_prefix(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        cap = FFmpegAudioCapture(device_name="1", ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert ["-f", "avfoundation", "-i", ":1"] == cmd[2:6]

    def test_windows_device_gets_audio_prefix(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cap = FFmpegAudioCapture(device_name="Blue Yeti", ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert ["-f", "dshow", "-i", "audio=Blue Yeti"] == cmd[2:6]


class TestCreateCompatibleAudioSource:
    def test_source_has_queue_and_interface_attrs(self):
        src = create_compatible_audio_source(sample_rate=8000, ts_enabled=False)
        assert src.SAMPLE_RATE == 8000
        assert src.SAMPLE_WIDTH == 2
        assert src.data_queue is not None
        assert src.data_queue.empty()


class _FakeProcess:
    """A subprocess.Popen stand-in that records how it was ended."""

    _next_pid = 4000

    def __init__(self):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


class TestWaitForOpenOrRaise:
    """The actual fix for STT#12's device-open-failure bug.

    start() used to return the instant the capture thread was spawned; Popen,
    _get_ffmpeg_command() and every open error lived entirely inside that
    background thread, so the caller's fallback loop always saw success. The
    thread now signals `_open_event`/`_open_error` and start() waits briefly
    on them via `_wait_for_open_or_raise()`. Exercised directly (no real
    subprocess/thread) so the decision is covered without the flakiness of
    threaded timing -- the capture loop's own thread wiring is out of scope
    here, same as the rest of this file.
    """

    def test_raises_when_the_thread_reported_a_failure(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        cap.running = True
        cap._open_error = RuntimeError("device busy")
        cap._open_event.set()
        with pytest.raises(RuntimeError, match="Failed to open audio device"):
            cap._wait_for_open_or_raise()
        assert cap.running is False, "a failed open must not leave running=True"

    def test_the_original_error_is_chained(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        cap.running = True
        original = FileNotFoundError("ffmpeg")
        cap._open_error = original
        cap._open_event.set()
        with pytest.raises(RuntimeError) as excinfo:
            cap._wait_for_open_or_raise()
        assert excinfo.value.__cause__ is original

    def test_does_not_raise_when_the_thread_reported_success(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        cap.running = True
        cap._open_event.set()  # success: no error attached
        cap._wait_for_open_or_raise()  # must not raise
        assert cap.running is True

    def test_a_device_still_pending_after_the_timeout_is_not_treated_as_failed(self):
        # Only a DEMONSTRATED failure raises; a device that's merely slow to
        # produce its first chunk must not be penalized.
        cap = FFmpegAudioCapture(ts_enabled=False, open_timeout=0.05)
        cap.running = True
        cap._wait_for_open_or_raise()  # event never set -> times out -> no raise
        assert cap.running is True


class TestStopIsInterlockedWithRespawn:
    """A stalled stream is respawned; a respawn during teardown left an orphan.

    The capture loop restarts ffmpeg after ~10s without PCM — very reachable on a box
    where a multi-GB model download is starving the stream. stop() terminates the
    process it can see and then clears the handle, so a Popen landing in between wrote
    into a field about to be nulled: nothing held the new ffmpeg, and it kept appending
    to its .ts capture file long after transcription reported "stopped". Both sides
    wait up to 2s on a terminate, so the window is seconds wide.
    """

    def capture(self):
        cap = FFmpegAudioCapture(device_name="test-device")
        cap.running = True
        cap.process = _FakeProcess()
        return cap

    def test_stop_terminates_the_running_process_and_clears_it(self):
        cap = self.capture()
        proc = cap.process
        cap.stop()
        assert proc.terminated is True
        assert cap.process is None
        assert cap.running is False

    def test_stop_kills_a_process_that_appears_during_teardown(self):
        # Simulates the race directly: the capture thread wins the assignment after
        # stop() has already terminated the one it knew about.
        cap = self.capture()
        cap.stop()
        assert cap.process is None

        stray = _FakeProcess()
        cap.process = stray  # what the racing respawn used to do
        cap.stop()
        assert stray.killed or stray.terminated, "the orphan must not be left running"
        assert cap.process is None

    def test_the_lock_exists_and_is_held_while_swapping(self):
        # The guard is the lock plus the running flag; a respawn that ignores either
        # reintroduces the orphan.
        cap = self.capture()
        assert hasattr(cap, "_process_lock")
        acquired = cap._process_lock.acquire(blocking=False)
        assert acquired, "the lock must not be held outside a swap"
        cap._process_lock.release()

    def test_running_is_cleared_before_the_process_is_touched(self):
        # _restart_ffmpeg checks self.running, so it has to be false by the time
        # stop() starts terminating anything.
        cap = self.capture()
        order = []

        class _Recorder(_FakeProcess):
            def terminate(inner):
                order.append(("terminate", cap.running))
                super().terminate()

        cap.process = _Recorder()
        cap.stop()
        assert order == [("terminate", False)]


class TestResolvesExactBeforeSubstring:
    """Two mics sharing a prefix — the case Windows produces routinely.

    Substring matching alone cannot separate them: the saved "#2" name contains
    the shorter device's name, and the shorter saved name is contained in "#2".
    Whichever the operator picked, the answer depended on enumeration order and
    was wrong half the time.
    """

    PAIR = ["Microphone (USB Audio Device)", "Microphone (USB Audio Device) #2"]

    @staticmethod
    def _devices(names):
        return [{"name": n, "index": i, "display_name": n, "card_id": n, "alt_name": None}
                for i, n in enumerate(names)]

    @pytest.mark.parametrize("order", [PAIR, list(reversed(PAIR))])
    @pytest.mark.parametrize("saved", PAIR)
    def test_a_prefix_sharing_pair_resolves_to_the_saved_one(self, order, saved):
        got = resolve_audio_device_by_name(saved, self._devices(order))
        assert got["name"] == saved

    def test_an_alt_name_separates_two_identically_named_devices(self):
        """When even the display names are equal, dshow's alternative name is
        the only thing left to tell them apart."""
        devices = [
            {"name": "Microphone", "display_name": "Microphone", "card_id": "Microphone",
             "alt_name": r"@device_cm_{A}\wave_{1}", "index": 0},
            {"name": "Microphone", "display_name": "Microphone", "card_id": "Microphone",
             "alt_name": r"@device_cm_{A}\wave_{2}", "index": 1},
        ]
        assert resolve_audio_device_by_name(r"@device_cm_{A}\wave_{2}", devices)["index"] == 1

    def test_substring_matching_still_works_for_an_alsa_card_id(self):
        """The fallback earns its place: an ALSA card_id genuinely is a fragment
        of the fuller description a name is saved from, and vice versa."""
        devices = [
            {"name": "plughw:1,0", "card_id": "UR22mkII",
             "display_name": "Steinberg UR22mkII at usb-0000:00:14.0-1"},
            {"name": "plughw:0,0", "card_id": "PCH", "display_name": "HDA Intel PCH"},
        ]
        assert resolve_audio_device_by_name("UR22mkII", devices)["name"] == "plughw:1,0"
        assert resolve_audio_device_by_name(
            "Steinberg UR22mkII at usb-0000:00:14.0-1", devices)["name"] == "plughw:1,0"
        assert resolve_audio_device_by_name("Nonexistent", devices) is None

    def test_an_exact_match_later_in_the_list_beats_an_earlier_substring_one(self):
        devices = self._devices(["Microphone (USB Audio Device) extended", "Microphone (USB Audio Device)"])
        got = resolve_audio_device_by_name("Microphone (USB Audio Device)", devices)
        assert got["name"] == "Microphone (USB Audio Device)"


class TestFfmpegErrorSummary:
    """What an operator is told when a device will not open.

    ffmpeg states the reason plainly; it used to reach [DEBUG-TS-STDERR] in the
    log and nowhere else, so the caller got a generic "no data".
    """

    REAL = [
        "ffmpeg version 8.1 Copyright (c) 2000-2026 the FFmpeg developers",
        "  built with Apple clang version 17.0.0",
        "  configuration: --prefix=/opt/homebrew --enable-shared",
        "  libavutil      60.  8.100 / 60.  8.100",
        "[in#0 @ 0x8b6c10000] Error opening input: Input/output error",
        "Error opening input file ::99.",
        "Error opening input files: Input/output error",
    ]

    def test_the_banner_is_dropped(self):
        summary = summarise_ffmpeg_error(self.REAL)
        assert "ffmpeg version" not in summary
        assert "clang" not in summary
        assert "libavutil" not in summary

    def test_the_reason_survives(self):
        assert "Input/output error" in summarise_ffmpeg_error(self.REAL)

    def test_the_heap_address_tag_is_stripped(self):
        """It changes every run and means nothing to a reader."""
        assert "0x8b6c10000" not in summarise_ffmpeg_error(self.REAL)
        assert "Error opening input: Input/output error" in summarise_ffmpeg_error(self.REAL)

    def test_progress_lines_are_not_the_error(self):
        """A session's worth of `size=` lines must not push the reason out."""
        lines = ["Device or resource busy"] + [f"size=     {n}kB time=00:00:0{n%10}" for n in range(50)]
        assert summarise_ffmpeg_error(lines) == "Device or resource busy"

    def test_a_fault_reported_three_times_reads_as_one(self):
        repeated = ["Device or resource busy"] * 3
        assert summarise_ffmpeg_error(repeated) == "Device or resource busy"

    def test_nothing_meaningful_yields_an_empty_summary(self):
        assert summarise_ffmpeg_error([]) == ""
        assert summarise_ffmpeg_error(["", "   ", "ffmpeg version 8.1"]) == ""

    def test_the_summary_is_bounded(self):
        assert len(summarise_ffmpeg_error(["x" * 5000])) <= 300


class TestSignalOpenFailure:
    """Every path that ends the capture loop without data reports the same way.

    Previously each branch invented its own: the Windows one had a message with
    the exit code, the Unix EOF branch — the one that actually fires, because
    select() reports a closed pipe as readable — set nothing at all, and the
    caller fell through to a generic backstop.
    """

    class _Exited:
        """An ffmpeg that has already gone; poll() answering is the whole point."""

        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def _cap(self, returncode=1, stderr_lines=()):
        cap = FFmpegAudioCapture(ts_enabled=False, device_name="mic")
        cap.process = self._Exited(returncode)
        cap._stderr_tail.extend(stderr_lines)
        return cap

    def test_the_exit_code_and_reason_reach_the_error(self):
        cap = self._cap(returncode=251, stderr_lines=["Error opening input: Input/output error"])
        cap._signal_open_failure(251)
        assert cap._open_event.is_set()
        assert "exit code 251" in str(cap._open_error)
        assert "Input/output error" in str(cap._open_error)

    def test_no_stderr_still_gives_an_actionable_sentence(self):
        cap = self._cap(returncode=1)
        cap._signal_open_failure(1)
        assert "may be in use by another application" in str(cap._open_error)

    def test_the_exit_code_is_derived_when_the_caller_has_none(self):
        cap = self._cap(returncode=99)
        cap._signal_open_failure(None)
        assert "exit code 99" in str(cap._open_error)

    def test_it_is_a_no_op_once_the_device_has_proven_itself(self):
        """After the first chunk of real audio this is a mid-session ffmpeg
        death, which the restart path owns — not an open failure."""
        cap = self._cap()
        cap._open_event.set()  # first data already arrived
        cap._signal_open_failure(1)
        assert cap._open_error is None

    def test_the_first_cause_is_kept(self):
        cap = self._cap(returncode=251, stderr_lines=["Error opening input: Input/output error"])
        cap._signal_open_failure(251)
        first = cap._open_error
        cap._signal_open_failure(2)
        assert cap._open_error is first

    def test_a_thread_that_died_with_no_process_uses_the_fallback(self):
        cap = FFmpegAudioCapture(ts_enabled=False, device_name="mic")
        cap.process = None
        cap._signal_open_failure(None, fallback="capture thread exited before producing data")
        assert "capture thread exited before producing data" in str(cap._open_error)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestOpenFailureReachesTheCaller:
    """End to end against a real ffmpeg, because the unit tests above could not
    have caught what was wrong here.

    The capture loop had three places that reported an open failure and a
    backstop. On Unix none of the three fired: select() reports a closed pipe as
    *readable*, so a device that fails to open reaches the EOF branch — which
    reported nothing — long before any select timeout. The failure still
    surfaced, via the backstop, but stripped of the exit code and of ffmpeg's own
    account of what went wrong. Only running a real ffmpeg shows that.
    """

    def _capture(self, device, tmp_path):
        cap = FFmpegAudioCapture(device_name=device, backup_dir=str(tmp_path),
                                 ts_enabled=False, open_timeout=10.0)
        cap.data_queue = queue.Queue()
        return cap

    def test_a_device_that_cannot_be_opened_raises_out_of_start(self, tmp_path):
        cap = self._capture("/nonexistent/device.wav", tmp_path)
        try:
            with pytest.raises(RuntimeError, match="Failed to open audio device"):
                cap.start()
            assert cap.running is False, "a failed open must not leave running=True"
        finally:
            cap.stop()

    def test_the_failure_carries_the_exit_code_and_ffmpegs_reason(self, tmp_path):
        cap = self._capture("/nonexistent/device.wav", tmp_path)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                cap.start()
        finally:
            cap.stop()
        message = str(excinfo.value)
        assert "exit code" in message, message
        assert "Error opening input" in message, message
        assert "ffmpeg version" not in message, "the build banner is not a diagnosis"

    def test_a_source_that_opens_does_not_raise(self, tmp_path):
        """The other half: a working device must not be penalised, and must not
        wait out the open timeout either."""
        wav = tmp_path / "tone.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-ar", "16000", "-ac", "1", str(wav)],
            capture_output=True, check=True)
        cap = self._capture(str(wav), tmp_path)
        started = time.monotonic()
        try:
            cap.start()
            assert time.monotonic() - started < cap.open_timeout
            assert cap.running is True
        finally:
            cap.stop()

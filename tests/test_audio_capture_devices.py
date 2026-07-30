"""Audio device enumeration per platform (stt/audio_capture.py).

This fills the microphone dropdown an operator picks from before a service. It
parses ffmpeg's *stderr* on macOS and Windows — an output format nobody
controls — and reads /proc/asound/cards on Linux. A parsing slip shows an empty
list or the wrong device, and the operator finds out when the room is full.

Every branch is exercised on every platform by patching sys.platform and
stubbing subprocess.run with captured real-world output; no ffmpeg runs.
"""

import subprocess
import sys

import pytest

from stt.audio_capture import FFmpegAudioCapture, parse_asound_cards

list_devices = FFmpegAudioCapture.list_devices

# Verbatim shape of `ffmpeg -f avfoundation -list_devices true -i ""` stderr.
MACOS_STDERR = """[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] Built-in Microphone
[AVFoundation indev @ 0x7f8] [1] Scarlett 2i2 USB
"""

# Verbatim shape of `ffmpeg -list_devices true -f dshow -i dummy` stderr.
WINDOWS_STDERR = '''[dshow @ 000001] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001]  "Integrated Webcam" (video)
[dshow @ 000001]     Alternative name "@device_pnp_\\\\?\\usb#vid_0c45"
[dshow @ 000001] DirectShow audio devices
[dshow @ 000001]  "Microphone (Realtek(R) Audio)" (audio)
[dshow @ 000001]  "Line In (Scarlett 2i2 USB)" (audio)
'''


class _Result:
    def __init__(self, stdout="", stderr=""):
        self.stdout, self.stderr, self.returncode = stdout, stderr, 0


@pytest.fixture
def ffmpeg(monkeypatch):
    """Stubs subprocess.run and records the commands it was given."""
    state = {"cmds": [], "result": _Result()}

    def fake_run(cmd, **kw):
        state["cmds"].append(list(cmd))
        if isinstance(state["result"], Exception):
            raise state["result"]
        return state["result"]

    monkeypatch.setattr(subprocess, "run", fake_run)
    return state


class TestMacOS:
    @pytest.fixture(autouse=True)
    def on_macos(self, monkeypatch, ffmpeg):
        monkeypatch.setattr(sys, "platform", "darwin")
        ffmpeg["result"] = _Result(stderr=MACOS_STDERR)

    def test_lists_only_the_audio_devices(self, ffmpeg):
        names = [d["display_name"] for d in list_devices()]
        assert names == ["Built-in Microphone", "Scarlett 2i2 USB"], (
            "the video section must not leak into the microphone dropdown")

    def test_the_name_is_the_avfoundation_index(self, ffmpeg):
        # The command builder turns this into ':N', so it has to be the index.
        assert [d["name"] for d in list_devices()] == ["0", "1"]

    def test_the_index_is_ffmpeg_s_not_the_list_position(self, ffmpeg):
        """The two coincide in the common case, which hides a real difference.

        avfoundation numbers devices itself, and the number is what "-i :N"
        selects. Reporting the list position instead would open the wrong
        microphone whenever the numbering is not 0,1,2… .
        """
        ffmpeg["result"] = _Result(stderr=(
            "[AVFoundation indev @ 0x1] AVFoundation audio devices:\n"
            "[AVFoundation indev @ 0x1] [2] Scarlett 2i2 USB\n"
            "[AVFoundation indev @ 0x1] [5] Loopback\n"))
        devices = list_devices()
        assert [d["name"] for d in devices] == ["2", "5"]
        assert [d["index"] for d in devices] == [2, 5]
        assert not devices[0]["is_default"], "default is index 0, which is absent here"

    def test_video_listed_after_audio_does_not_leak_in(self, ffmpeg):
        """Section order is not guaranteed across ffmpeg builds."""
        ffmpeg["result"] = _Result(stderr=(
            "[AVFoundation indev @ 0x1] AVFoundation audio devices:\n"
            "[AVFoundation indev @ 0x1] [0] Built-in Microphone\n"
            "[AVFoundation indev @ 0x1] AVFoundation video devices:\n"
            "[AVFoundation indev @ 0x1] [0] FaceTime HD Camera\n"))
        assert [d["display_name"] for d in list_devices()] == ["Built-in Microphone"]

    def test_the_first_device_is_the_default(self, ffmpeg):
        devices = list_devices()
        assert devices[0]["is_default"] is True
        assert devices[1]["is_default"] is False

    def test_a_name_containing_the_separator_survives(self, ffmpeg):
        ffmpeg["result"] = _Result(stderr=(
            "[AVFoundation indev @ 0x1] AVFoundation audio devices:\n"
            "[AVFoundation indev @ 0x1] [0] Weird] Mic\n"))
        assert list_devices()[0]["display_name"] == "Weird] Mic"

    def test_it_asks_avfoundation_for_the_list(self, ffmpeg):
        list_devices()
        assert "avfoundation" in ffmpeg["cmds"][0]
        assert "-list_devices" in ffmpeg["cmds"][0]


class TestWindows:
    @pytest.fixture(autouse=True)
    def on_windows(self, monkeypatch, ffmpeg):
        monkeypatch.setattr(sys, "platform", "win32")
        ffmpeg["result"] = _Result(stderr=WINDOWS_STDERR)

    def test_extracts_the_quoted_device_names(self, ffmpeg):
        assert [d["name"] for d in list_devices()] == [
            "Microphone (Realtek(R) Audio)", "Line In (Scarlett 2i2 USB)"]

    def test_devices_are_indexed_in_order(self, ffmpeg):
        assert [d["index"] for d in list_devices()] == [0, 1]

    def test_it_asks_dshow_for_the_list(self, ffmpeg):
        list_devices()
        assert "dshow" in ffmpeg["cmds"][0]

    def test_no_audio_devices_yields_an_empty_list(self, ffmpeg):
        ffmpeg["result"] = _Result(stderr='[dshow @ 1] DirectShow video devices\n'
                                          '[dshow @ 1]  "Integrated Webcam" (video)\n')
        assert list_devices() == []

    def test_the_parser_keys_on_the_audio_tag_not_the_section_header(self, ffmpeg):
        """Worth stating: a quoted line without "(audio)" is skipped.

        Modern ffmpeg tags every device line, so this holds in practice — but it
        means a build that only prints the "DirectShow audio devices" header
        would yield an empty dropdown rather than the devices under it.
        """
        ffmpeg["result"] = _Result(stderr='[dshow @ 1] DirectShow audio devices\n'
                                          '[dshow @ 1]  "Some Mic"\n')
        assert list_devices() == []


class TestLinux:
    @pytest.fixture(autouse=True)
    def on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")

    def test_reads_the_alsa_card_list_when_present(self, monkeypatch, ffmpeg, tmp_path):
        cards = tmp_path / "cards"
        cards.write_text(" 0 [PCH  ]: HDA-Intel - HDA Intel PCH\n"
                         "                      HDA Intel PCH at 0xf7f10000 irq 33\n",
                         encoding="utf-8")
        monkeypatch.setattr("stt.audio_capture.os.path.exists",
                            lambda p: str(p) == "/proc/asound/cards")
        real_open = open
        monkeypatch.setattr("builtins.open",
                            lambda p, *a, **k: (real_open(str(cards), *a, **k)
                                                if str(p) == "/proc/asound/cards"
                                                else real_open(p, *a, **k)))
        devices = list_devices()
        assert devices, "a machine with a sound card must not fall back to guesses"
        assert ffmpeg["cmds"] == [], "no need to shell out when /proc has the answer"

    def test_falls_back_to_arecord_when_proc_is_absent(self, monkeypatch, ffmpeg):
        monkeypatch.setattr("stt.audio_capture.os.path.exists", lambda p: False)
        ffmpeg["result"] = _Result(stdout="default\nplughw:CARD=PCH,DEV=0\n")
        devices = list_devices()
        assert ffmpeg["cmds"] and ffmpeg["cmds"][0][0] == "arecord"
        assert [d["name"] for d in devices] == ["default", "plughw:CARD=PCH,DEV=0"]

    def test_without_arecord_it_offers_the_standard_alsa_names(self, monkeypatch, ffmpeg):
        monkeypatch.setattr("stt.audio_capture.os.path.exists", lambda p: False)
        ffmpeg["result"] = FileNotFoundError("arecord")
        names = [d["name"] for d in list_devices()]
        assert names == ["default", "plughw:0,0"], (
            "an empty dropdown would leave the operator with nothing to pick")


class TestFailuresAreNotFatal:
    """The dropdown degrades to empty rather than breaking the settings page."""

    @pytest.mark.parametrize("platform", ["darwin", "win32"])
    def test_ffmpeg_missing_yields_an_empty_list(self, monkeypatch, ffmpeg, platform):
        monkeypatch.setattr(sys, "platform", platform)
        ffmpeg["result"] = FileNotFoundError("ffmpeg")
        assert list_devices() == []

    @pytest.mark.parametrize("platform", ["darwin", "win32"])
    def test_a_timeout_yields_an_empty_list(self, monkeypatch, ffmpeg, platform):
        monkeypatch.setattr(sys, "platform", platform)
        ffmpeg["result"] = subprocess.TimeoutExpired("ffmpeg", 5)
        assert list_devices() == []


class TestAsoundParsing:
    """parse_asound_cards is what the Linux path relies on."""

    CARDS = (" 0 [PCH            ]: HDA-Intel - HDA Intel PCH\n"
             "                      HDA Intel PCH at 0xf7f10000 irq 33\n"
             " 1 [USB            ]: USB-Audio - Scarlett 2i2 USB\n"
             "                      Focusrite Scarlett 2i2 USB at usb-0000:00:14.0-2\n")

    def test_every_card_is_listed(self):
        assert len(parse_asound_cards(self.CARDS)) == 2

    def test_the_first_card_is_the_default(self):
        devices = parse_asound_cards(self.CARDS)
        assert devices[0]["is_default"] is True
        assert devices[1]["is_default"] is False

    def test_a_deprioritised_card_is_never_the_default(self):
        """HDMI and similar outputs are never the microphone the operator wants."""
        devices = parse_asound_cards(self.CARDS, ["hda intel"])
        assert devices[0]["is_default"] is False
        assert any(d["is_default"] for d in devices), "something must still be default"

    def test_empty_input_yields_nothing(self):
        assert parse_asound_cards("") == []

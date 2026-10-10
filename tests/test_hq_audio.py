"""The high-quality listening capture (stt/hq_audio.py).

No ffmpeg is launched: `HqCapture` takes a `popen`, and the fake below hands
back canned stdout/stderr and an exit state the test controls. `step()` is
driven with explicit times, so back-off is checked without sleeping.
"""

import io
import threading

import pytest

from stt import hq_audio
from stt.hq_audio import (
    Chunker,
    HqCapture,
    HqTarget,
    backoff_delay,
    chunk_bytes,
    hq_command,
    pump,
    resolve_target,
    stream_info,
)

STEREO = HqTarget(device="2", sample_rate=48000, channels=2)


class FakeProc:
    def __init__(self, stdout=b"", stderr=b""):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def die(self, code=1):
        self.returncode = code

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakePopen:
    def __init__(self, **proc_kwargs):
        self.calls = []
        self.procs = []
        self.proc_kwargs = proc_kwargs
        self.fail_with = None

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self.fail_with:
            raise self.fail_with
        proc = FakeProc(**self.proc_kwargs)
        self.procs.append(proc)
        return proc


def make(target=STEREO, **proc_kwargs):
    state = {"target": target}
    popen = FakePopen(**proc_kwargs)
    logs = []
    chunks = []
    cap = HqCapture(lambda: state["target"], chunks.append, popen=popen, log=logs.append, platform="darwin")
    return cap, state, popen, logs, chunks


class TestResolveTarget:
    def test_wanted_while_running_with_a_listener_and_a_device(self):
        assert resolve_target({}, True, "2", 1) == HqTarget("2", 48000, 2)

    @pytest.mark.parametrize("settings,running,device,listeners", [
        ({"enabled": False}, True, "2", 1),
        ({}, False, "2", 1),
        ({}, True, None, 1),
        ({}, True, "", 1),
        ({}, True, "2", 0),
    ])
    def test_nothing_otherwise(self, settings, running, device, listeners):
        assert resolve_target(settings, running, device, listeners) is None

    def test_configured_format(self):
        assert resolve_target({"sample_rate": 44100, "channels": 1}, True, "2", 1) == HqTarget("2", 44100, 1)

    @pytest.mark.parametrize("bad", [0, -1, "48000", True, 10 ** 7, None])
    def test_bad_values_fall_back(self, bad):
        assert resolve_target({"sample_rate": bad, "channels": bad}, True, "2", 1) == HqTarget("2", 48000, 2)

    def test_non_dict_settings(self):
        assert resolve_target(None, True, "2", 1) == HqTarget("2", 48000, 2)


class TestCommand:
    def test_outputs_interleaved_pcm_at_the_target_format(self):
        cmd = hq_command(STEREO, platform="darwin")
        assert cmd[0] == "ffmpeg"
        assert ["-f", "avfoundation", "-i", ":2"] == cmd[cmd.index("-f"):cmd.index("-f") + 4]
        assert cmd[cmd.index("-ar") + 1] == "48000"
        assert cmd[cmd.index("-ac") + 1] == "2"
        assert cmd[-5:] == ["-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
        assert "-nostdin" in cmd  # never reads the server's stdin (the shutdown channel)

    @pytest.mark.parametrize("platform,name,fmt,dev", [
        ("linux", "plughw:1,0", "alsa", "plughw:1,0"),
        ("win32", "Blue Yeti", "dshow", "audio=Blue Yeti"),
    ])
    def test_opens_the_device_the_same_way_as_transcription(self, platform, name, fmt, dev):
        cmd = hq_command(HqTarget(name, 48000, 2), platform=platform)
        assert cmd[cmd.index("-f") + 1] == fmt
        assert cmd[cmd.index("-i") + 1] == dev

    def test_no_backup_output(self):
        assert "mpegts" not in hq_command(STEREO, platform="linux")

    def test_stream_info_describes_the_chunks(self):
        info = stream_info(STEREO)
        assert info == {"sample_rate": 48000, "channels": 2, "bit_depth": 16,
                        "format": "s16le", "chunk_seconds": hq_audio.CHUNK_SECONDS}


class TestChunking:
    def test_chunk_is_100ms_of_whole_frames(self):
        assert chunk_bytes(48000, 2) == 4800 * 2 * 2
        assert chunk_bytes(44100, 1) == 4410 * 2
        assert chunk_bytes(48000, 2) % (2 * 2) == 0

    def test_partial_reads_become_exact_chunks(self):
        c = Chunker(4)
        assert c.feed(b"ab") == []
        assert c.feed(b"cdefghij") == [b"abcd", b"efgh"]
        assert c.feed(b"kl") == [b"ijkl"]

    def test_rejects_empty_chunk_size(self):
        with pytest.raises(ValueError):
            Chunker(0)

    def test_pump_reads_to_eof_and_drops_the_trailing_partial(self):
        out = []
        pump(io.BytesIO(b"x" * 10), Chunker(4), out.append, read_size=3)
        assert out == [b"xxxx", b"xxxx"]


class TestBackoff:
    def test_doubles_then_caps(self):
        assert [backoff_delay(n) for n in range(0, 5)] == [0.0, 1.0, 2.0, 4.0, 8.0]
        assert backoff_delay(50) == hq_audio.MAX_BACKOFF_SECONDS


class TestSupervision:
    def test_idle_without_a_target(self):
        cap, _, popen, _, _ = make(target=None)
        assert cap.step(0) == "idle"
        assert popen.calls == []

    def test_starts_and_streams_chunks(self):
        size = chunk_bytes(48000, 2)
        cap, _, popen, logs, chunks = make(stdout=b"\1" * (size * 2 + 7))
        assert cap.step(0) == "started"
        cap.reader.join(timeout=5)
        assert chunks == [b"\1" * size] * 2
        assert len(popen.calls) == 1
        assert any("48000 Hz, 2 ch" in m for m in logs)

    def test_keeps_running_while_wanted(self):
        cap, _, popen, _, _ = make()
        cap.step(0)
        assert cap.step(1) == "running"
        assert len(popen.calls) == 1

    def test_stops_when_no_longer_wanted(self):
        cap, state, popen, _, _ = make()
        cap.step(0)
        proc = popen.procs[0]
        state["target"] = None
        assert cap.step(1) == "stopped"
        assert proc.terminated
        assert cap.process is None

    def test_restarts_on_a_target_change(self):
        cap, state, popen, _, _ = make()
        cap.step(0)
        state["target"] = HqTarget("3", 48000, 2)
        assert cap.step(1) == "started"
        assert popen.procs[0].terminated
        assert popen.calls[1][popen.calls[1].index("-i") + 1] == ":3"

    def test_a_death_backs_off_before_restarting_and_logs_once(self):
        cap, _, popen, logs, _ = make(stderr=b"[avfoundation @ 0x1] Device or resource busy\n")
        cap.step(0)
        popen.procs[0].die(1)
        assert cap.step(10) == "failed"
        assert cap.step(10.5) == "waiting"  # 1 s back-off
        assert cap.step(11) == "started"
        popen.procs[1].die(1)
        assert cap.step(12) == "failed"
        assert cap.step(13) == "waiting"  # 2 s back-off now
        assert cap.step(14) == "started"
        failures = [m for m in logs if "exited" in m]
        assert len(failures) == 1  # a streak is reported once, not every retry
        assert "Device or resource busy" in failures[0]

    def test_recovery_is_logged_after_staying_up(self):
        cap, _, popen, logs, _ = make()
        cap.step(0)
        popen.procs[0].die(1)
        cap.step(1)
        cap.step(2)  # restarted
        cap.step(3)
        assert cap.failures == 1  # not yet proven stable
        cap.step(2 + hq_audio.STABLE_SECONDS)
        assert cap.failures == 0
        assert any("recovered" in m for m in logs)

    def test_popen_failure_is_retried_with_backoff(self):
        cap, _, popen, logs, _ = make()
        popen.fail_with = FileNotFoundError("ffmpeg")
        assert cap.step(0) == "failed"
        assert cap.step(0.5) == "waiting"
        popen.fail_with = None
        assert cap.step(1) == "started"
        assert sum("could not start" in m for m in logs) == 1

    def test_stopping_resets_the_failure_streak(self):
        cap, state, popen, _, _ = make()
        cap.step(0)
        popen.procs[0].die(1)
        cap.step(1)
        state["target"] = None
        cap.step(2)
        assert cap.failures == 0
        state["target"] = STEREO
        assert cap.step(2) == "started"  # no leftover back-off

    def test_run_stops_the_process_on_exit(self):
        cap, _, popen, _, _ = make()
        stop = threading.Event()
        started = threading.Event()
        real_step = cap.step

        def step(now):
            result = real_step(now)
            started.set()
            return result

        cap.step = step
        t = threading.Thread(target=cap.run, args=(stop, 0.01))
        t.start()
        assert started.wait(timeout=5)
        stop.set()
        t.join(timeout=5)
        assert not t.is_alive()
        assert popen.procs[0].terminated

    def test_run_survives_a_target_callback_error(self):
        logs = []
        calls = []
        stop = threading.Event()

        def target():
            calls.append(1)
            if len(calls) >= 2:
                stop.set()
            raise RuntimeError("boom")

        cap = HqCapture(target, lambda _: None, popen=FakePopen(), log=logs.append)
        cap.run(stop, 0)
        assert len(calls) >= 2
        assert any("supervisor error: boom" in m for m in logs)

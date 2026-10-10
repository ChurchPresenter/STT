"""High-quality live listening capture, beside the 16 kHz transcription feed.

The transcription capture (stt/audio_capture.py) produces 16 kHz mono int16,
which is what Whisper wants and what the existing ``audio_chunk`` room carries.
SongListener hardcodes that format, so it must never change. Meters and
listeners need more: a loudness meter reading 16 kHz mono sees no stereo
correlation, under-reads true peak and loses a stereo source's level by about
3 dB in the downmix.

So this opens the same device a second time, in its own ffmpeg, at full rate
(48 kHz stereo by default). It runs only while someone is listening and
transcription is running. It is a separate process on purpose: if it fails to
open the device (an exclusive ALSA ``hw:`` device, a busy dshow input), the
capture that matters is untouched. A third output on the transcription ffmpeg
would make one blocked HQ reader stall the transcript.

Pure parts (`hq_command`, `Chunker`, `backoff_delay`, `HqTarget`) are
separate from the process supervision so the supervision can be driven
step-by-step in tests with a fake ``Popen``.
"""

from __future__ import annotations

import collections
import subprocess
import threading
import time
from typing import IO, Any, Callable, List, NamedTuple, Optional

from stt.audio_capture import _CREATE_NO_WINDOW, ffmpeg_input_args, summarise_ffmpeg_error

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2
CHUNK_SECONDS = 0.1  # 100 ms keeps a meter responsive without flooding Socket.IO
BYTES_PER_SAMPLE = 2  # s16le
MAX_BACKOFF_SECONDS = 30.0
STABLE_SECONDS = 10.0  # alive this long after a restart = recovered


class HqTarget(NamedTuple):
    """What the HQ capture should be doing; ``None`` instead means "nothing"."""

    device: Optional[str]
    sample_rate: int
    channels: int


def resolve_target(settings: Any, running: bool, device: Optional[str], listeners: int) -> Optional[HqTarget]:
    """The capture wanted right now, or None.

    ``settings`` is the ``audio.listen_stream`` config block (any shape; bad
    values fall back to the defaults). Nothing runs without a listener, while
    transcription is stopped, or before the worker has said which device it
    opened: guessing a device here could open a different microphone from the
    one being transcribed.
    """
    settings = settings if isinstance(settings, dict) else {}
    if not settings.get("enabled", True) or not running or listeners <= 0 or not device:
        return None

    def positive(key: str, default: int, limit: int) -> int:
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= limit:
            return default
        return value

    return HqTarget(str(device), positive("sample_rate", DEFAULT_SAMPLE_RATE, 192000),
                    positive("channels", DEFAULT_CHANNELS, 8))


def hq_command(target: HqTarget, platform: Optional[str] = None) -> List[str]:
    """The ffmpeg command for an HQ capture: interleaved s16le PCM on stdout."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        *ffmpeg_input_args(target.device, platform),
        "-ar", str(target.sample_rate), "-ac", str(target.channels),
        "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
    ]


def chunk_bytes(sample_rate: int, channels: int, seconds: float = CHUNK_SECONDS) -> int:
    """Bytes in one chunk: whole frames only, so a chunk never splits a sample."""
    frames = max(1, int(sample_rate * seconds))
    return frames * channels * BYTES_PER_SAMPLE


def stream_info(target: HqTarget) -> dict:
    """The ``audio_stream_hq_info`` payload a listener reads before the first chunk."""
    return {
        "sample_rate": target.sample_rate,
        "channels": target.channels,
        "bit_depth": BYTES_PER_SAMPLE * 8,
        "format": "s16le",
        "chunk_seconds": CHUNK_SECONDS,
    }


def backoff_delay(failures: int) -> float:
    """Seconds to wait before restart number ``failures`` (1-based): 1, 2, 4 … 30."""
    if failures <= 0:
        return 0.0
    return min(MAX_BACKOFF_SECONDS, 2.0 ** (failures - 1))


class Chunker:
    """Turns arbitrary pipe reads into fixed-size chunks."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("chunk size must be positive")
        self.size = size
        self._pending = bytearray()

    def feed(self, data: bytes) -> List[bytes]:
        self._pending += data
        out = []
        while len(self._pending) >= self.size:
            out.append(bytes(self._pending[:self.size]))
            del self._pending[:self.size]
        return out


def pump(stream: IO[bytes], chunker: Chunker, on_chunk: Callable[[bytes], None],
         read_size: int = 65536) -> None:
    """Read ``stream`` to EOF, handing every whole chunk to ``on_chunk``.

    A trailing partial chunk is dropped: it is less than 100 ms at the end of
    a capture that is being torn down anyway.
    """
    while True:
        data = stream.read1(read_size) if hasattr(stream, "read1") else stream.read(read_size)
        if not data:
            return
        for chunk in chunker.feed(data):
            on_chunk(chunk)


class HqCapture:
    """Keeps one HQ ffmpeg running exactly while `get_target()` asks for one.

    `step()` is the whole decision: start, stop, restart on a target change, or
    restart with back-off after ffmpeg dies. `run()` just calls it on a timer,
    which is why no stop path elsewhere in the server needs to remember to stop
    this: the next step sees transcription stopped and tears it down.
    """

    def __init__(
        self,
        get_target: Callable[[], Optional[HqTarget]],
        on_chunk: Callable[[bytes], None],
        popen: Callable[..., Any] = subprocess.Popen,
        log: Callable[[str], None] = print,
        platform: Optional[str] = None,
    ) -> None:
        self._get_target = get_target
        self._on_chunk = on_chunk
        self._popen = popen
        self._log = log
        self._platform = platform
        self._lock = threading.Lock()
        self.process: Any = None
        self.target: Optional[HqTarget] = None
        self.reader: Optional[threading.Thread] = None
        self.failures = 0
        self._retry_at = 0.0
        self._stable_at = 0.0
        self._stderr_tail: collections.deque = collections.deque(maxlen=20)
        self._stderr_reader: Optional[threading.Thread] = None

    def step(self, now: float) -> str:
        """Bring the process in line with the target; returns what it did."""
        want = self._get_target()
        with self._lock:
            proc = self.process
            alive = proc is not None and proc.poll() is None
            if want is None:
                if proc is not None:
                    self._stop_locked()
                    self.failures = 0
                    return "stopped"
                self.failures = 0
                return "idle"
            if alive and want == self.target:
                if self.failures and now >= self._stable_at:
                    # Stayed up long enough after a restart: recovered.
                    self._log("[AUDIO-HQ] capture recovered")
                    self.failures = 0
                return "running"
            if proc is not None and not alive and want == self.target:
                # Died on its own while still wanted.
                self.failures += 1
                self._retry_at = now + backoff_delay(self.failures)
                detail = self._stderr_summary()
                if self.failures == 1:
                    self._log(f"[AUDIO-HQ] capture exited (code {proc.returncode}){': ' + detail if detail else ''}; retrying")
                self._clear_locked()
                return "failed"
            if proc is not None:
                self._stop_locked()  # target changed (device, rate or channels)
            if now < self._retry_at:
                return "waiting"
            return self._start_locked(want, now)

    def _start_locked(self, want: HqTarget, now: float) -> str:
        cmd = hq_command(want, self._platform)
        try:
            proc = self._popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW)
        except OSError as e:
            self.failures += 1
            self._retry_at = now + backoff_delay(self.failures)
            if self.failures == 1:
                self._log(f"[AUDIO-HQ] could not start ffmpeg: {e}; retrying")
            return "failed"
        self.process = proc
        self.target = want
        chunker = Chunker(chunk_bytes(want.sample_rate, want.channels))
        self.reader = threading.Thread(target=self._read, args=(proc, chunker),
                                       name="audio-hq-reader", daemon=True)
        self.reader.start()
        # Drained continuously so ffmpeg can never block on a full stderr pipe;
        # the tail is what a failure quotes.
        self._stderr_tail = collections.deque(maxlen=20)
        self._stderr_reader = threading.Thread(target=self._drain_stderr, args=(proc, self._stderr_tail),
                                               name="audio-hq-stderr", daemon=True)
        self._stderr_reader.start()
        self._stable_at = now + STABLE_SECONDS
        if self.failures == 0:
            self._log(f"[AUDIO-HQ] capturing {want.sample_rate} Hz, {want.channels} ch")
        return "started"

    def _read(self, proc: Any, chunker: Chunker) -> None:
        try:
            pump(proc.stdout, chunker, self._on_chunk)
        except (OSError, ValueError):
            pass  # pipe closed by stop()

    @staticmethod
    def _drain_stderr(proc: Any, tail: collections.deque) -> None:
        try:
            for line in proc.stderr:
                tail.append(line.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError, TypeError):
            pass  # pipe closed by stop()

    def _stderr_summary(self) -> str:
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=0.5)  # ffmpeg has exited; let the last lines land
        return summarise_ffmpeg_error(list(self._stderr_tail))

    def _stop_locked(self) -> None:
        proc = self.process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._clear_locked()

    def _clear_locked(self) -> None:
        proc = self.process
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        self.process = None
        self.target = None
        self.reader = None
        self._stderr_reader = None

    def stop(self) -> None:
        with self._lock:
            if self.process is not None:
                self._stop_locked()

    def run(self, stop_event: threading.Event, interval: float = 1.0) -> None:
        """Supervise until ``stop_event`` is set; for a daemon thread."""
        while not stop_event.is_set():
            try:
                self.step(time.monotonic())
            except Exception as e:  # never let supervision die silently
                self._log(f"[AUDIO-HQ] supervisor error: {e}")
            stop_event.wait(interval)
        self.stop()

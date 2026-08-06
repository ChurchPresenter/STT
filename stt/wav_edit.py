"""Trimming session recordings: cut a range, or cut the silence off the front.

A capture starts when the operator starts it and stops when they remember to,
so a two-hour file routinely holds half an hour of nothing before anyone speaks
— one real service measured here was silent for its first 31 minutes 53 seconds,
a third of the file. Trimming that off is worth doing before a recording is
handed to anything else.

These are 16 kHz mono PCM WAVs, so a trim is a byte range: no decoding, no
re-encoding, no ffmpeg, and the output is sample-exact rather than
re-compressed. Silence is found the same way — by reading the samples.

Two things about these files in particular:

* **The header lies about length.** They are written as a stream and the size
  fields are never patched when recording stops, so a 476 MB file can declare
  one second of audio. What is on disk is the truth, so the data size is taken
  from the file length when the header disagrees.
* **Nothing here writes over its input.** A trim produces a new file. The
  operation is exact but it is not reversible, and a recording is not something
  to be clever with.

Stdlib only, so it is testable without audio tooling installed.
"""

from __future__ import annotations

import os
import struct
import threading
from collections import OrderedDict
from typing import Any, List, NamedTuple, Optional, Tuple

#: Read this much at a time when scanning for silence.
_CHUNK_FRAMES = 1 << 16

#: Only every Nth sample is examined when looking for silence. Speech does not
#: hide between adjacent samples at 16 kHz, and reading every one of 115 million
#: of them in Python costs minutes for an answer that does not change.
_STRIDE = 16

#: Drawing a waveform reads at most this many frames per bucket, taken as
#: :data:`_PROBES_PER_BUCKET` evenly spaced probes across it. A full-file view of
#: a two-hour recording puts eight seconds of audio behind every horizontal pixel
#: — reading all of it means reading the whole 230 MB file to draw 900 numbers,
#: which measured at 3.4 s per open on the deployed box. The envelope was never
#: exact anyway (:data:`_STRIDE` already examines one sample in sixteen); probing
#: keeps that character and cuts the reading by roughly four. Measured against
#: full reads of an hour of speech-shaped audio, the largest bucket differed by
#: 0.01 of full scale. Zoomed-in views have small buckets and are read whole, so
#: looking closer at anything still shows every sample.
_PROBE_FRAMES = 1024
_PROBES_PER_BUCKET = 16

#: How precisely a silence boundary is reported. Fine enough to be a trim point,
#: coarse enough that a two-hour file is still a handful of seconds to scan.
_WINDOW_SECONDS = 0.05


class WavInfo(NamedTuple):
    """What the header says, corrected by what is actually on disk."""

    channels: int
    sample_width: int          # bytes per sample
    sample_rate: int
    frames: int                # total frames, from the file size when the header is stale
    data_offset: int           # byte offset of the first sample
    header_frames: int         # what the header claimed, kept so callers can see the lie

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def frame_size(self) -> int:
        return self.channels * self.sample_width

    @property
    def header_is_stale(self) -> bool:
        """Whether the declared length is materially short of the file."""
        return self.header_frames < self.frames * 0.9


class WavError(ValueError):
    """The file is not a PCM WAV this module can work with."""


def read_info(path: str) -> WavInfo:
    """Parse the header of a PCM WAV, or raise :class:`WavError`.

    Walks the RIFF chunks rather than assuming the canonical 44-byte layout:
    real files carry ``LIST``/``fact`` chunks before the data, and a fixed offset
    would read those as audio.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        riff = fh.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise WavError("not a RIFF/WAVE file")

        channels = sample_width = sample_rate = 0
        while True:
            head = fh.read(8)
            if len(head) < 8:
                raise WavError("no data chunk")
            chunk_id, chunk_size = struct.unpack("<4sI", head)
            if chunk_id == b"fmt ":
                fmt = fh.read(chunk_size)
                if len(fmt) < 16:
                    raise WavError("truncated fmt chunk")
                audio_format, channels, sample_rate, _br, _ba, bits = struct.unpack(
                    "<HHIIHH", fmt[:16])
                if audio_format != 1:
                    raise WavError(f"not uncompressed PCM (format {audio_format})")
                sample_width = bits // 8
            elif chunk_id == b"data":
                data_offset = fh.tell()
                if not channels or not sample_width or not sample_rate:
                    raise WavError("data chunk before fmt")
                frame_size = channels * sample_width
                on_disk = max(size - data_offset, 0)
                # The declared size is trusted only when it accounts for most of
                # the file; a stream-written capture never patches it.
                declared = min(chunk_size, on_disk)
                usable = declared if declared >= 0.9 * on_disk else on_disk
                return WavInfo(channels, sample_width, sample_rate,
                               usable // frame_size, data_offset,
                               declared // frame_size)
            else:
                fh.seek(chunk_size + (chunk_size & 1), os.SEEK_CUR)


def clamp_range(info: WavInfo, start_seconds: float,
                end_seconds: Optional[float] = None) -> Tuple[int, int]:
    """``(start_frame, frame_count)`` for a requested range, clamped to the file.

    Raises :class:`WavError` when the range selects nothing, rather than quietly
    writing a zero-length recording.
    """
    start_frame = max(0, min(info.frames, round(start_seconds * info.sample_rate)))
    if end_seconds is None:
        end_frame = info.frames
    else:
        end_frame = max(0, min(info.frames, round(end_seconds * info.sample_rate)))
    if end_frame <= start_frame:
        raise WavError("the selected range is empty")
    return start_frame, end_frame - start_frame


def _peak(block: bytes, sample_width: int) -> int:
    """Largest absolute sample in a block of 16-bit PCM, sampled by stride."""
    if sample_width != 2:
        return 1  # only 16-bit is analysed; treat anything else as "not silent"
    count = len(block) // 2
    if not count:
        return 0
    samples = struct.unpack("<%dh" % count, block[:count * 2])
    loudest = 0
    for i in range(0, count, _STRIDE):
        value = samples[i]
        value = -value if value < 0 else value
        if value > loudest:
            loudest = value
    return loudest


def find_speech_bounds(path: str, threshold: float = 0.01,
                       info: Optional[WavInfo] = None) -> Tuple[float, float]:
    """``(first, last)`` seconds where the recording is above ``threshold``.

    ``threshold`` is a fraction of full scale — 0.01 is about -40 dBFS, which
    clears the noise floor of a desk microphone without clipping a quiet first
    word. Returns ``(0.0, duration)`` when nothing anywhere crosses it, so a
    caller trimming by this can never produce an empty file from a quiet one.
    """
    info = info or read_info(path)
    if info.sample_width != 2 or not info.frames:
        return 0.0, info.seconds if info else 0.0

    limit = int(threshold * 32767)
    first: Optional[int] = None
    last = 0
    frame_size = info.frame_size
    # Resolution of the answer. Reading is done in large blocks for speed, but
    # the verdict is per window — a whole block would put the boundary anywhere
    # within four seconds at 16 kHz, which is not a trim point.
    window = max(1, int(info.sample_rate * _WINDOW_SECONDS))

    with open(path, "rb") as fh:
        fh.seek(info.data_offset)
        frame = 0
        remaining = info.frames
        while remaining > 0:
            want = min(_CHUNK_FRAMES, remaining)
            block = fh.read(want * frame_size)
            if not block:
                break
            got = len(block) // frame_size
            for offset in range(0, got, window):
                span = min(window, got - offset)
                piece = block[offset * frame_size:(offset + span) * frame_size]
                if _peak(piece, info.sample_width) > limit:
                    if first is None:
                        first = frame + offset
                    last = frame + offset + span
            frame += got
            remaining -= got

    if first is None:
        return 0.0, info.seconds
    return first / info.sample_rate, min(last, info.frames) / info.sample_rate


def trim(path: str, dest: str, start_seconds: float,
         end_seconds: Optional[float] = None) -> WavInfo:
    """Write the selected range of ``path`` to ``dest``; return the new file's info.

    A straight copy of the byte range with a correct header in front, so the
    output is sample-identical to that stretch of the input. Refuses to write
    over an existing file, and refuses ``dest == path``: a recording is not
    something to edit in place.
    """
    if os.path.exists(dest):
        raise WavError(f"{os.path.basename(dest)} already exists")
    if os.path.realpath(dest) == os.path.realpath(path):
        raise WavError("refusing to overwrite the recording being trimmed")

    info = read_info(path)
    start_frame, count = clamp_range(info, start_seconds, end_seconds)
    frame_size = info.frame_size
    data_bytes = count * frame_size
    byte_rate = info.sample_rate * frame_size

    with open(path, "rb") as src, open(dest, "wb") as out:
        out.write(b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE")
        out.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, info.channels,
                                        info.sample_rate, byte_rate, frame_size,
                                        info.sample_width * 8))
        out.write(b"data" + struct.pack("<I", data_bytes))
        src.seek(info.data_offset + start_frame * frame_size)
        remaining = data_bytes
        while remaining > 0:
            block = src.read(min(1 << 20, remaining))
            if not block:
                break
            out.write(block)
            remaining -= len(block)
    return read_info(dest)


def trimmed_name(path: str, suffix: str = "trimmed") -> str:
    """A sibling filename for the output, without colliding with what is there."""
    base, ext = os.path.splitext(path)
    candidate = f"{base}_{suffix}{ext}"
    n = 2
    while os.path.exists(candidate):
        candidate = f"{base}_{suffix}{n}{ext}"
        n += 1
    return candidate


def peaks(path: str, buckets: int = 800, start_seconds: float = 0.0,
          end_seconds: Optional[float] = None,
          info: Optional[WavInfo] = None) -> List[float]:
    """Loudness envelope of a range, as ``buckets`` values between 0 and 1.

    One value per horizontal pixel of a waveform: the largest absolute sample in
    that slice of time, scaled to full scale. Peak rather than average, because
    a waveform drawn from averages of a 16 kHz signal is a flat smear — the
    point of looking at one is to see where sound starts and stops.

    Reading is confined to the requested range, which is what makes zooming
    cheap: every zoom reads only the part being looked at. A bucket wider than
    the probe budget is sampled rather than read whole — see
    :data:`_PROBE_FRAMES` — so the first, whole-file draw does not have to read
    the whole file.
    """
    info = info or read_info(path)
    if info.sample_width != 2 or not info.frames or buckets < 1:
        return []

    start_frame, count = clamp_range(info, start_seconds, end_seconds)
    buckets = min(buckets, count) or 1
    per_bucket = count / buckets
    frame_size = info.frame_size
    probe_budget = _PROBE_FRAMES * _PROBES_PER_BUCKET

    out: List[float] = []
    with open(path, "rb") as fh:
        for index in range(buckets):
            begin = start_frame + int(index * per_bucket)
            finish = start_frame + int((index + 1) * per_bucket)
            span = max(1, finish - begin)
            if span <= probe_budget:
                fh.seek(info.data_offset + begin * frame_size)
                block = fh.read(span * frame_size)
                loudest = _peak(block, info.sample_width) if block else 0
            else:
                step = span // _PROBES_PER_BUCKET
                loudest = 0
                for probe in range(_PROBES_PER_BUCKET):
                    fh.seek(info.data_offset + (begin + probe * step) * frame_size)
                    block = fh.read(_PROBE_FRAMES * frame_size)
                    if not block:
                        break
                    value = _peak(block, info.sample_width)
                    if value > loudest:
                        loudest = value
            out.append(min(1.0, loudest / 32767.0))
    return out


class PeaksCache:
    """Recently drawn waveform envelopes, kept so a redraw is free.

    The trim editor asks for the same envelope over and over: opening a file,
    closing it and opening it again, zooming out to where it started, two
    operators looking at the same recording. Each of those is a fresh read of
    the file for an answer that has not changed.

    Identity includes size and modification time, so a recording still being
    written to never serves a stale envelope — it simply misses and is read
    again. Entries are small (a few hundred floats), and the least recently used
    is dropped past ``max_entries``.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Tuple[Any, ...], List[float]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def envelope(self, path: str, buckets: int = 800, start_seconds: float = 0.0,
                 end_seconds: Optional[float] = None,
                 info: Optional[WavInfo] = None) -> List[float]:
        """:func:`peaks`, served from the cache when the same range was asked for."""
        try:
            stat = os.stat(path)
            key: Tuple[Any, ...] = (os.path.realpath(path), stat.st_size, stat.st_mtime_ns,
                                    int(buckets), round(float(start_seconds), 3),
                                    None if end_seconds is None else round(float(end_seconds), 3))
        except OSError:
            # Un-stat-able: let peaks() raise whatever it raises, uncached.
            return peaks(path, buckets, start_seconds, end_seconds, info=info)

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return list(cached)
            self.misses += 1

        computed = peaks(path, buckets, start_seconds, end_seconds, info=info)
        with self._lock:
            self._entries[key] = computed
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return list(computed)

    def clear(self) -> None:
        """Forget everything (mainly for tests)."""
        with self._lock:
            self._entries.clear()


#: Default silence gate, in dBFS. -40 dB is 1% of full scale: below the noise
#: floor of a desk microphone in a hall, above the quietest speech that matters.
DEFAULT_SILENCE_DB = -40.0


def db_to_amplitude(db: float) -> float:
    """dBFS to a fraction of full scale — -40 dB is 0.01, 0 dB is 1.0.

    The gate is set in dB because that is the unit the level actually behaves
    in: halving the amplitude is -6 dB wherever you are on the scale, whereas
    the same step expressed as a fraction means something different at 0.5 than
    at 0.01. An operator nudging a threshold is thinking in dB whether or not
    the box says so.

    Clamped to (0, 1]: 0 dB is full scale and nothing can exceed it, and a gate
    at exactly zero would call the whole recording speech.
    """
    if db >= 0:
        return 1.0
    return max(1e-6, min(1.0, 10.0 ** (db / 20.0)))


def amplitude_to_db(amplitude: float) -> float:
    """A fraction of full scale back to dBFS; silence reports as -inf."""
    if amplitude <= 0:
        return float("-inf")
    import math
    return 20.0 * math.log10(min(1.0, amplitude))

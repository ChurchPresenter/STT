"""A binary broadcast reaching websocket and polling listeners at once.

``engineio.packet.Packet.encode`` caches its first result and returns it to every
later caller, whatever ``b64`` they pass. That is harmless for text, which
encodes the same either way, and wrong for binary: a websocket writer asks for
raw bytes (``b64=False``), a long-polling response asks for a ``'b'``-prefixed
base64 string (``b64=True``), and whichever comes second gets the first one's
answer.

python-socketio makes that reachable. ``Manager.emit`` builds *one* engineio
packet for a broadcast and queues the same object to every participant, so
``socketio.emit("audio_chunk_hq", <bytes>, room=...)`` hands one packet to
every listener in the room. With one browser on a websocket and one stuck on
polling, which is what a Cloudflare tunnel tends to produce, the poller's
``Payload.encode`` concatenated cached bytes onto a str. The result was
``TypeError`` (STT-2R) and a failed poll. Going the other way, a websocket
listener would have been sent the base64 text frame as audio.

Still present in python-engineio 4.14.0. :func:`install` wraps ``encode`` so the
cache is kept per encoding for binary packets. Text packets are left on the
original path, cache included, because they are nearly all of the traffic.

Stdlib-only: the class is passed in, so tests can exercise the wrapper without
engineio installed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

#: Set on the class once wrapped, so a second install is a no-op.
_MARKER = "_stt_encode_per_transport"


def install(packet_cls: Any) -> bool:
    """Make ``packet_cls.encode`` cache binary packets per ``b64`` flag.

    Returns True if the class was patched now, False if it already was.
    """
    if getattr(packet_cls, _MARKER, False):
        return False
    original: Callable[..., Any] = packet_cls.encode

    def encode(self: Any, b64: bool = False) -> Any:
        if not getattr(self, "binary", False):
            return original(self, b64=b64)
        cache: Dict[bool, Any] = self.__dict__.setdefault("_stt_encoded", {})
        key = bool(b64)
        if key not in cache:
            # Clear the upstream cache so the original encodes for *this* flag
            # rather than returning what the other transport asked for.
            self.encode_cache = None
            cache[key] = original(self, b64=b64)
        return cache[key]

    encode.__doc__ = original.__doc__
    packet_cls.encode = encode
    setattr(packet_cls, _MARKER, True)
    return True

import base64

import pytest

from stt import engineio_compat


class _FakePacket:
    """The shape of engineio.packet.Packet.encode: cache the first answer, whatever b64 was."""

    def __init__(self, data):
        self.data = data
        self.binary = isinstance(data, bytes)
        self.encode_cache = None
        self.calls = 0

    def encode(self, b64=False):
        if self.encode_cache:
            return self.encode_cache
        self.calls += 1
        if self.binary:
            out = "b" + base64.b64encode(self.data).decode("utf-8") if b64 else self.data
        else:
            out = "4" + self.data
        self.encode_cache = out
        return out


def _patched_class():
    cls = type("Packet", (_FakePacket,), {})
    engineio_compat.install(cls)
    return cls


def test_unpatched_fake_reproduces_the_bug():
    pkt = _FakePacket(b"\x00\x01")
    assert pkt.encode(b64=False) == b"\x00\x01"
    assert pkt.encode(b64=True) == b"\x00\x01"  # the wrong answer the poller got


@pytest.mark.parametrize("first", [False, True])
def test_binary_packet_answers_each_transport_in_either_order(first):
    pkt = _patched_class()(b"\x00\x01")
    pkt.encode(b64=first)
    assert pkt.encode(b64=False) == b"\x00\x01"
    assert pkt.encode(b64=True) == "bAAE="


def test_binary_encodings_are_still_cached():
    pkt = _patched_class()(b"abc")
    for _ in range(3):
        pkt.encode(b64=True)
        pkt.encode(b64=False)
    assert pkt.calls == 2


def test_text_packets_keep_the_original_path():
    pkt = _patched_class()("hello")
    assert pkt.encode(b64=True) == "4hello"
    assert pkt.encode(b64=False) == "4hello"
    assert pkt.calls == 1
    assert "_stt_encoded" not in pkt.__dict__


def test_install_is_idempotent():
    cls = type("Packet", (_FakePacket,), {})
    assert engineio_compat.install(cls) is True
    wrapped = cls.encode
    assert engineio_compat.install(cls) is False
    assert cls.encode is wrapped


def test_real_engineio_broadcast_packet_serves_poller_after_websocket():
    packet = pytest.importorskip("engineio.packet")
    payload = pytest.importorskip("engineio.payload")
    engineio_compat.install(packet.Packet)
    pkt = packet.Packet(packet.MESSAGE, b"\x00\x01")
    assert pkt.encode() == b"\x00\x01"  # websocket listener first
    assert payload.Payload(packets=[pkt]).encode() == "bAAE="  # then the poller
    assert pkt.encode() == b"\x00\x01"

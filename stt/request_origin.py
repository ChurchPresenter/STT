"""Where a request actually came from, and what that means for trust.

The Cloudflare tunnel forwards to ``http://127.0.0.1:<port>``, so every request
through it reaches Flask with ``remote_addr == "127.0.0.1"`` — indistinguishable,
at the transport layer, from someone sitting at the machine. ``check_ip_whitelist``
treats localhost as fully trusted, so without this module the public URL grants
the same access as the console.

Classification therefore comes from the headers Cloudflare's edge adds.

**Which headers, and why only these.** Measured against a live quick tunnel:

- ``CF-Connecting-IP`` — set by the edge to the real client IP. A client that
  tries to send its own is **rejected by Cloudflare with a 403 before the request
  reaches us**, so this value cannot be forged or suppressed by a visitor.
- ``CF-Ray`` — a client-supplied value is overwritten by the edge. Useful as a
  corroborating marker and as the id that ties a row to Cloudflare's own logs.
- ``X-Forwarded-For`` — **never read.** The edge *appends* to it, so a client
  sending ``X-Forwarded-For: 127.0.0.1`` arrives as ``127.0.0.1,<real ip>``.
  Trusting its first entry is exactly how a tunnel visitor would hand itself the
  localhost free pass. Same for ``X-Real-IP``, which the edge does not set at all.

The failure direction matters: a LAN client can *add* CF headers and be treated
as tunnelled, which only removes its own privileges. Nothing a caller can send
gains trust.

Stdlib-only, Flask-free — headers come in as a plain mapping.
"""

from __future__ import annotations

import ipaddress
from typing import Mapping, NamedTuple, Optional

#: Same machine, over loopback, not via any proxy — the console.
ORIGIN_LOCAL = "local"
#: Another device on a local network.
ORIGIN_LAN = "lan"
#: Arrived through the Cloudflare tunnel, i.e. from the public internet.
ORIGIN_TUNNEL = "tunnel"

_CF_CLIENT_IP_HEADER = "CF-Connecting-IP"
_CF_RAY_HEADER = "CF-Ray"

_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1", "localhost", ""})


class RequestOrigin(NamedTuple):
    """How a request reached us.

    ``client_ip`` is the caller worth logging and binding a session to: the real
    visitor for a tunnelled request, ``remote_addr`` otherwise.
    """

    kind: str
    client_ip: str
    ray_id: Optional[str]

    @property
    def is_tunnelled(self) -> bool:
        return self.kind == ORIGIN_TUNNEL


def classify(remote_addr: Optional[str], headers: Optional[Mapping[str, str]] = None) -> RequestOrigin:
    """Classify a request from its transport address and headers.

    Presence of ``CF-Connecting-IP`` is what makes a request tunnelled — it is
    the header Cloudflare refuses to accept from clients. ``CF-Ray`` alone is not
    enough to *trust* as an IP source, but it still marks the request as
    tunnelled so it cannot fall back to the localhost shortcut.
    """
    remote_addr = (remote_addr or "").strip()
    headers = headers or {}

    cf_ip = _header(headers, _CF_CLIENT_IP_HEADER)
    ray_id = _header(headers, _CF_RAY_HEADER)

    if cf_ip:
        # The edge sets this and rejects client-supplied copies, so it is the
        # one caller-visible value we can believe.
        return RequestOrigin(ORIGIN_TUNNEL, cf_ip, ray_id)

    if ray_id:
        # Cloudflare in front of us but no usable client IP: still untrusted.
        # Keep remote_addr for the record rather than inventing one.
        return RequestOrigin(ORIGIN_TUNNEL, remote_addr, ray_id)

    if _is_loopback(remote_addr):
        return RequestOrigin(ORIGIN_LOCAL, remote_addr, None)

    return RequestOrigin(ORIGIN_LAN, remote_addr, None)


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup that tolerates a plain dict.

    Flask's header object is already case-insensitive; a plain mapping (as tests
    and the socket path pass) is not, so fall back to a scan.
    """
    try:
        value = headers.get(name)
    except Exception:  # pragma: no cover - defensive against odd mappings
        return None
    if value is None:
        lowered = name.lower()
        for key, candidate in headers.items():
            if str(key).lower() == lowered:
                value = candidate
                break
    value = (value or "").strip()
    return value or None


def _is_loopback(address: str) -> bool:
    if address in _LOOPBACK_LITERALS:
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def is_trusted_transport(origin: RequestOrigin) -> bool:
    """Whether the caller's address may be used to grant access on its own.

    False for tunnelled requests: their ``remote_addr`` is our own loopback and
    their ``client_ip`` is an arbitrary internet address, so neither the
    localhost shortcut nor an empty-whitelist "allow all" may apply.
    """
    return not origin.is_tunnelled


# ─── SocketIO surface ───────────────────────────────────────────────────────
#
# The websocket has no authentication of its own, so gating HTTP alone would be
# theatre: these events change stored transcripts and live output. Listed as data
# so the split is greppable and testable rather than spread across handlers.
#
# Everything NOT listed here stays open to tunnelled connections, because the
# caption display needs it: request_all_entries, request_all_translation_entries,
# join/leave_audio_stream, join/leave_tts_audio, connect, disconnect.

GATED_SOCKET_EVENTS = frozenset({
    "submit_correction",
    "mark_reviewed",
    "submit_translation_correction",
    "select_translation_alternative",
    "toggle_delay",
    "set_delay_seconds",
    "approve_staged",
    "discard_staged",
    "set_segment_denied",
    "set_segment_marked",
})


def socket_event_requires_auth(event: str, origin_kind: str) -> bool:
    """Whether this event must be refused unless the connection authenticated.

    Only tunnelled connections are gated — a LAN display or the operator's own
    browser behaves exactly as before, so this cannot break a live service.
    """
    return origin_kind == ORIGIN_TUNNEL and event in GATED_SOCKET_EVENTS

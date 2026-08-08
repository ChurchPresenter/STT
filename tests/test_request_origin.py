"""Request origin classification and the socket gate (stt/request_origin.py)."""

from stt.request_origin import (
    GATED_SOCKET_EVENTS,
    ORIGIN_LAN,
    ORIGIN_LOCAL,
    ORIGIN_TUNNEL,
    classify,
    is_trusted_transport,
    socket_event_requires_auth,
)

# What a real tunnelled request carries, measured against a live quick tunnel.
# remote_addr is our own loopback because cloudflared forwards to 127.0.0.1.
TUNNEL_HEADERS = {
    "Cf-Connecting-Ip": "71.75.202.167",
    "Cf-Ray": "a27ad15e4858ed8b-ATL",
    "Cf-Ipcountry": "US",
    "Cf-Visitor": '{"scheme":"https"}',
    "X-Forwarded-For": "71.75.202.167",
    "X-Forwarded-Proto": "https",
}


class TestClassify:
    def test_tunnelled_request_is_recognised(self):
        origin = classify("127.0.0.1", TUNNEL_HEADERS)
        assert origin.kind == ORIGIN_TUNNEL
        assert origin.is_tunnelled is True

    def test_reports_the_real_visitor_not_our_loopback(self):
        assert classify("127.0.0.1", TUNNEL_HEADERS).client_ip == "71.75.202.167"

    def test_captures_the_ray_id_for_correlation(self):
        assert classify("127.0.0.1", TUNNEL_HEADERS).ray_id == "a27ad15e4858ed8b-ATL"

    def test_header_lookup_is_case_insensitive(self):
        # Flask's header object folds case; a plain dict does not, and the socket
        # path passes a plain dict.
        origin = classify("127.0.0.1", {"CF-CONNECTING-IP": "203.0.113.9"})
        assert origin.kind == ORIGIN_TUNNEL
        assert origin.client_ip == "203.0.113.9"

    def test_console_request_is_local(self):
        origin = classify("127.0.0.1", {})
        assert origin.kind == ORIGIN_LOCAL
        assert origin.client_ip == "127.0.0.1"
        assert origin.ray_id is None

    def test_ipv6_loopback_is_local(self):
        assert classify("::1", {}).kind == ORIGIN_LOCAL

    def test_other_loopback_addresses_are_local(self):
        assert classify("127.0.0.2", {}).kind == ORIGIN_LOCAL

    def test_lan_client_is_lan(self):
        origin = classify("192.168.2.62", {})
        assert origin.kind == ORIGIN_LAN
        assert origin.client_ip == "192.168.2.62"

    def test_missing_remote_addr_is_not_treated_as_lan(self):
        assert classify(None, {}).kind == ORIGIN_LOCAL

    def test_unparseable_address_is_lan_not_local(self):
        # Fail towards less trust, never towards more.
        assert classify("not-an-ip", {}).kind == ORIGIN_LAN


class TestSpoofResistance:
    def test_forwarded_for_is_never_used_as_the_client_ip(self):
        # Cloudflare APPENDS to X-Forwarded-For, so a visitor sending
        # "X-Forwarded-For: 127.0.0.1" arrives as "127.0.0.1,<real ip>".
        # Reading its first entry is exactly how a tunnel visitor would hand
        # itself the localhost free pass.
        headers = dict(TUNNEL_HEADERS, **{"X-Forwarded-For": "127.0.0.1,71.75.202.167"})
        origin = classify("127.0.0.1", headers)
        assert origin.client_ip == "71.75.202.167"
        assert origin.kind == ORIGIN_TUNNEL

    def test_forwarded_for_alone_grants_nothing(self):
        # No CF headers: a direct LAN caller claiming to be forwarded is still
        # judged on its actual address.
        origin = classify("192.168.2.99", {"X-Forwarded-For": "127.0.0.1"})
        assert origin.kind == ORIGIN_LAN
        assert origin.client_ip == "192.168.2.99"

    def test_x_real_ip_is_ignored(self):
        origin = classify("192.168.2.99", {"X-Real-IP": "127.0.0.1"})
        assert origin.kind == ORIGIN_LAN
        assert origin.client_ip == "192.168.2.99"

    def test_ray_without_a_client_ip_is_still_untrusted(self):
        # Cloudflare in front of us but no usable IP: must not fall back to the
        # localhost shortcut just because the address looks local.
        origin = classify("127.0.0.1", {"Cf-Ray": "abc-ATL"})
        assert origin.kind == ORIGIN_TUNNEL
        assert is_trusted_transport(origin) is False

    def test_lan_client_forging_cf_headers_only_loses_privileges(self):
        # The safe failure direction: pretending to be tunnelled is allowed
        # because it removes trust rather than granting it.
        origin = classify("192.168.2.99", {"Cf-Connecting-Ip": "203.0.113.5"})
        assert origin.kind == ORIGIN_TUNNEL
        assert is_trusted_transport(origin) is False

    def test_blank_header_values_do_not_count_as_present(self):
        origin = classify("127.0.0.1", {"Cf-Connecting-Ip": "   ", "Cf-Ray": ""})
        assert origin.kind == ORIGIN_LOCAL


class TestTrustedTransport:
    def test_local_and_lan_may_be_judged_on_address(self):
        assert is_trusted_transport(classify("127.0.0.1", {})) is True
        assert is_trusted_transport(classify("192.168.2.62", {})) is True

    def test_tunnelled_may_not(self):
        assert is_trusted_transport(classify("127.0.0.1", TUNNEL_HEADERS)) is False


class TestSocketGate:
    def test_mutating_events_are_gated_over_the_tunnel(self):
        for event in ("submit_correction", "set_delay_seconds", "approve_staged"):
            assert socket_event_requires_auth(event, ORIGIN_TUNNEL) is True, event

    def test_display_events_stay_open_over_the_tunnel(self):
        # The caption display needs these, and / is deliberately passwordless.
        for event in ("request_all_entries", "request_all_translation_entries",
                      "join_audio_stream", "join_tts_audio", "connect", "disconnect"):
            assert socket_event_requires_auth(event, ORIGIN_TUNNEL) is False, event

    def test_lan_and_local_connections_are_never_gated(self):
        # This is the one that would break a live service if it regressed.
        for kind in (ORIGIN_LAN, ORIGIN_LOCAL):
            for event in GATED_SOCKET_EVENTS:
                assert socket_event_requires_auth(event, kind) is False, (kind, event)

    def test_unknown_event_is_not_gated(self):
        # A new display-side event shouldn't start failing silently; gating is
        # opt-in via the explicit set.
        assert socket_event_requires_auth("some_new_event", ORIGIN_TUNNEL) is False

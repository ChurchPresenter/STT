"""How Machine B decides a request is its paired Machine A.

The rules live in stt/pair_tokens; these cases cover the wiring in the monolith
around them — that a token is read off the Authorization header, that a machine
which has moved address is followed rather than refused, that unpairing revokes,
and that the claim endpoint hands a token to a paired machine exactly once.

The routes are extracted from the monolith and run against a stub namespace (see
tests/conftest.py) — the module cannot be imported, and CI installs no Flask.
"""

import threading
import time

import pytest

from conftest import extract_definitions
from stt import pair_tokens as pair_tokens_mod

A_IP = "192.168.2.62"
STRANGER_IP = "192.168.2.90"

_NAMES = [
    "_is_trusted_translation_client",
    "_pair_tokens",
    "_save_pair_tokens",
    "_paired_client_ok",
    "_rebind_trusted_client",
    "_pair_token_grace_ips",
    "translation_pair_token",
]


class _Request:
    def __init__(self, remote_addr=A_IP, token=None):
        self.remote_addr = remote_addr
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}


def build(*, trusted=(A_IP,), tokens=None, remote_addr=A_IP, token=None, ports=None,
          grace=None):
    """A Machine B with the given pairing state, ready to answer one request."""
    config = {"live_translation": {
        "trusted_clients": list(trusted),
        "trusted_client_tokens": dict(tokens or {}),
    }}
    saves = []
    ns = extract_definitions(
        "speech_to_text.py", _NAMES,
        extra_globals={
            "config": config,
            "request": _Request(remote_addr, token),
            "jsonify": lambda obj: obj,
            "save_config": lambda cfg: saves.append(dict(cfg)),
            "_pair_tokens_mod": pair_tokens_mod,
            "_trusted_translation_clients": set(trusted),
            "_translation_client_ports": dict(ports or {}),
            "_forget_client_port": lambda ip: None,
            "_pair_token_grace": dict(grace or {}),
            "_pair_token_grace_lock": threading.Lock(),
            "PAIR_TOKEN_GRACE_SECONDS": 120,
            "_remember_client_port": lambda ip, port, persist=True: None,
            "app": type("A", (), {"route": staticmethod(lambda *a, **kw: (lambda f: f))})(),
        },
    )
    ns["_config"] = config
    ns["_saves"] = saves
    return ns


def issue_to(ip):
    """A token store holding one token for ``ip``, plus that token."""
    return pair_tokens_mod.issue({}, ip)


class TestPairedClientOk:
    def test_a_trusted_address_with_no_token_is_accepted(self):
        # A Machine A on an older build keeps working until it updates.
        assert build()["_paired_client_ok"]() is True

    def test_an_unknown_address_is_refused(self):
        assert build(remote_addr=STRANGER_IP)["_paired_client_ok"]() is False

    def test_a_token_is_accepted(self):
        tokens, token = issue_to(A_IP)
        assert build(tokens=tokens, token=token)["_paired_client_ok"]() is True

    def test_the_address_alone_stops_working_once_a_token_exists(self):
        # The DHCP hole: whoever inherits the address gets nothing, because the
        # paired machine now proves itself with something the address cannot.
        tokens, _token = issue_to(A_IP)
        assert build(tokens=tokens)["_paired_client_ok"]() is False

    def test_a_revoked_token_is_refused_from_the_paired_address(self):
        tokens, _token = issue_to(A_IP)
        stale = pair_tokens_mod.mint_token()
        assert build(tokens=tokens, token=stale)["_paired_client_ok"]() is False

    def test_a_machine_that_moved_address_is_followed(self):
        tokens, token = issue_to(A_IP)
        ns = build(tokens=tokens, remote_addr="192.168.2.77", token=token,
                   ports={A_IP: 8080})
        assert ns["_paired_client_ok"]() is True
        trusted = ns["_config"]["live_translation"]["trusted_clients"]
        assert trusted == ["192.168.2.77"]
        # And it is still authorized on the next request from the new address.
        assert ns["_pair_tokens"]().popitem()[1] == "192.168.2.77"

    def test_following_a_machine_persists_the_move(self):
        tokens, token = issue_to(A_IP)
        ns = build(tokens=tokens, remote_addr="192.168.2.77", token=token)
        ns["_paired_client_ok"]()
        assert ns["_saves"], "the move must survive a restart"


class TestClaimEndpoint:
    def test_a_paired_machine_is_given_a_token(self):
        ns = build()
        body = ns["translation_pair_token"]()
        assert body["success"] is True
        assert body["token"]
        # Stored as a fingerprint, never as the token itself.
        stored = ns["_pair_tokens"]()
        assert body["token"] not in stored
        assert stored[pair_tokens_mod.fingerprint(body["token"])] == A_IP

    def test_an_unpaired_machine_is_refused(self):
        ns = build(remote_addr=STRANGER_IP)
        _body, status = ns["translation_pair_token"]()
        assert status == 403
        assert ns["_pair_tokens"]() == {}

    def test_a_second_claim_without_the_current_token_is_refused(self):
        # Otherwise anything holding the address could rotate the real machine's
        # token out from under it.
        tokens, _token = issue_to(A_IP)
        ns = build(tokens=tokens)
        _body, status = ns["translation_pair_token"]()
        assert status == 403

    def test_the_current_token_can_rotate_itself(self):
        tokens, token = issue_to(A_IP)
        ns = build(tokens=tokens, token=token)
        body = ns["translation_pair_token"]()
        assert body["success"] is True
        assert body["token"] != token
        # The old one no longer works.
        assert not pair_tokens_mod.authorize(
            ns["_pair_tokens"](), [A_IP], A_IP, token).authorized

    def test_a_machine_that_lost_the_token_it_was_just_given_can_ask_again(self):
        # B issued one but A never stored it (a failed write, a kill between the
        # response and the save). Without this window A can neither talk nor ask,
        # and only someone at B's keyboard could rescue it — mid-service.
        tokens, _token = issue_to(A_IP)
        ns = build(tokens=tokens, grace={A_IP: time.time() + 60})
        body = ns["translation_pair_token"]()
        assert body["success"] is True

    def test_the_re_claim_window_closes(self):
        tokens, _token = issue_to(A_IP)
        ns = build(tokens=tokens, grace={A_IP: time.time() - 1})
        _body, status = ns["translation_pair_token"]()
        assert status == 403

    def test_the_window_belongs_to_the_address_it_was_issued_to(self):
        tokens, _token = issue_to(A_IP)
        ns = build(trusted=(A_IP, STRANGER_IP), tokens=tokens, remote_addr=STRANGER_IP,
                   grace={A_IP: time.time() + 60})
        # The stranger holds no token of its own, so it claims on its own merit —
        # what it must not do is inherit A's window and rotate A's token.
        body = ns["translation_pair_token"]()
        assert body["success"] is True
        assert set(ns["_pair_tokens"]().values()) == {A_IP, STRANGER_IP}

    def test_a_stranger_holding_no_token_cannot_claim_for_another_machine(self):
        tokens, _token = issue_to(A_IP)
        ns = build(tokens=tokens, remote_addr=STRANGER_IP)
        _body, status = ns["translation_pair_token"]()
        assert status == 403


class TestRevocation:
    def test_forgetting_an_address_drops_its_token(self):
        # What unpair/unpair-me do, in the shape the routes use.
        tokens, token = issue_to(A_IP)
        remaining = pair_tokens_mod.forget(tokens, A_IP)
        ns = build(tokens=remaining, token=token)
        assert ns["_paired_client_ok"]() is False


@pytest.mark.parametrize("header", ["", "Bearer", "Basic abc", "Bearer    "])
def test_a_malformed_authorization_header_falls_back_to_the_address(header):
    # Nothing presented is not the same as something wrong: an old client sends
    # no header at all, and must keep working.
    ns = build()
    ns["request"].headers = {"Authorization": header}
    assert ns["_paired_client_ok"]() is True


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _Session:
    """Stands in for the pooled requests.Session Machine A calls B through."""

    def __init__(self, response=None, raises=None):
        self.response = response or _Response(200, {"success": True, "token": "SECRET"})
        self.raises = raises
        self.calls = []

    def post(self, url, headers=None, timeout=None, **kw):
        self.calls.append({"url": url, "headers": headers or {}})
        if self.raises:
            raise self.raises
        return self.response


def build_machine_a(*, token=None, session=None):
    """Machine A's side: what it holds and what it sends."""
    config = {"live_translation": {"remote": {"endpoint": "http://b:8080"}}}
    if token:
        config["live_translation"]["remote"]["token"] = token
    session = session or _Session()
    saves = []
    ns = extract_definitions(
        "speech_to_text.py",
        ["_remote_pair_token", "_remote_auth_headers", "_claim_remote_pair_token",
         "_forget_remote_pair_token"],
        extra_globals={
            "config": config,
            "save_config": lambda cfg: saves.append(1),
            "_get_remote_http_session": lambda: session,
        },
    )
    ns["_config"] = config
    ns["_session"] = session
    ns["_saves"] = saves
    return ns


class TestMachineASide:
    def test_no_header_until_a_token_is_held(self):
        assert build_machine_a()["_remote_auth_headers"]() == {}

    def test_the_token_is_sent_as_a_bearer_header(self):
        ns = build_machine_a(token="SECRET")
        assert ns["_remote_auth_headers"]() == {"Authorization": "Bearer SECRET"}

    def test_claiming_stores_the_token(self):
        ns = build_machine_a()
        assert ns["_claim_remote_pair_token"]("http://b:8080") is True
        assert ns["_config"]["live_translation"]["remote"]["token"] == "SECRET"
        assert ns["_saves"], "the token must survive a restart"

    def test_claiming_is_skipped_when_one_is_already_held(self):
        ns = build_machine_a(token="OLD")
        assert ns["_claim_remote_pair_token"]("http://b:8080") is False
        assert ns["_session"].calls == []
        assert ns["_config"]["live_translation"]["remote"]["token"] == "OLD"

    def test_a_rotation_presents_the_current_token(self):
        ns = build_machine_a(token="OLD")
        assert ns["_claim_remote_pair_token"]("http://b:8080", force=True) is True
        assert ns["_session"].calls[0]["headers"] == {"Authorization": "Bearer OLD"}
        assert ns["_config"]["live_translation"]["remote"]["token"] == "SECRET"

    def test_an_older_remote_that_has_no_such_endpoint_changes_nothing(self):
        # B answers 404: the pairing keeps working on the address path.
        ns = build_machine_a(session=_Session(_Response(404)))
        assert ns["_claim_remote_pair_token"]("http://b:8080") is False
        assert "token" not in ns["_config"]["live_translation"]["remote"]

    def test_an_unreachable_remote_changes_nothing(self):
        ns = build_machine_a(session=_Session(raises=OSError("connection refused")))
        assert ns["_claim_remote_pair_token"]("http://b:8080") is False
        assert "token" not in ns["_config"]["live_translation"]["remote"]

    def test_a_junk_response_changes_nothing(self):
        ns = build_machine_a(session=_Session(_Response(200, {"success": True})))
        assert ns["_claim_remote_pair_token"]("http://b:8080") is False
        assert "token" not in ns["_config"]["live_translation"]["remote"]

    def test_forgetting_clears_the_token(self):
        ns = build_machine_a(token="SECRET")
        ns["_forget_remote_pair_token"]()
        assert "token" not in ns["_config"]["live_translation"]["remote"]
        assert ns["_remote_auth_headers"]() == {}

"""Machine A's single door to its paired Machine B.

Two things are being pinned down. That the bearer token is attached, because B
refuses a paired address that holds a token unless the token is presented — and
two call sites had quietly stopped attaching it, which read as "the peer is
unreachable" while the peer was answering its neighbours fine.

And that a refusal heals. If B forgets A's token — config restored from a
backup, hand-edited, the machine re-imaged — then A presents a token B has never
seen, which authorize() refuses outright and deliberately never downgrades to
the address path. A held a token, so it never asked for a new one; B would have
issued one for the asking. The pairing stayed broken until an operator fixed it
by hand at B's keyboard, mid-service.
"""

import threading

import pytest

from conftest import extract_definitions

ENDPOINT = "http://192.168.2.52:8080"

_NAMES = [
    "_remote_pair_token",
    "_remote_auth_headers",
    "_store_remote_pair_token",
    "_forget_remote_pair_token",
    "_peer_reclaim_allowed",
    "_peer_request",
]


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _Session:
    """Records what was sent, and answers from a script of status codes."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.sent = []

    def request(self, method, url, **kwargs):
        self.sent.append({"method": method, "url": url, **kwargs})
        return _Response(self.statuses.pop(0) if self.statuses else 200)


def build(*, token="live-token", statuses=(200,), claim_result=True, reclaim_last=0.0):
    """Machine A holding ``token``, talking to a B that answers ``statuses``."""
    config = {"live_translation": {"remote": {"token": token} if token else {}}}
    session = _Session(statuses)
    claims = []

    def _claim(endpoint, force=False):
        claims.append({"endpoint": endpoint, "force": force})
        if not claim_result:
            return False
        config["live_translation"]["remote"]["token"] = "fresh-token"
        return True

    ns = extract_definitions(
        "speech_to_text.py", _NAMES,
        extra_globals={
            "config": config,
            "save_config": lambda cfg: None,
            "_get_remote_http_session": lambda: session,
            "_claim_remote_pair_token": _claim,
            "_PEER_RECLAIM_MIN_INTERVAL": 60.0,
            "_peer_reclaim_last": reclaim_last,
            "_peer_reclaim_lock": threading.Lock(),
        },
    )
    ns["_session"] = session
    ns["_claims"] = claims
    ns["_config"] = config
    return ns


def _auth(sent):
    return sent["headers"].get("Authorization")


class TestTheTokenIsAttached:
    def test_every_call_carries_the_bearer_token(self):
        ns = build()
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert _auth(ns["_session"].sent[0]) == "Bearer live-token"

    def test_the_url_is_the_endpoint_plus_the_path(self):
        ns = build()
        ns["_peer_request"]("GET", ENDPOINT + "/", "/api/health")
        assert ns["_session"].sent[0]["url"] == ENDPOINT + "/api/health"

    def test_an_unpaired_machine_sends_no_header(self):
        # An older B still recognises us by address; sending "Bearer None" would
        # be worse than sending nothing.
        ns = build(token=None)
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert "Authorization" not in ns["_session"].sent[0]["headers"]

    def test_caller_headers_are_kept_alongside_the_token(self):
        ns = build()
        ns["_peer_request"]("POST", ENDPOINT, "/api/translate", headers={"X-Trace": "abc"})
        sent = ns["_session"].sent[0]
        assert sent["headers"]["X-Trace"] == "abc"
        assert _auth(sent) == "Bearer live-token"

    def test_the_token_wins_over_a_caller_supplied_authorization(self):
        ns = build()
        ns["_peer_request"]("GET", ENDPOINT, "/api/health",
                            headers={"Authorization": "Bearer stale"})
        assert _auth(ns["_session"].sent[0]) == "Bearer live-token"

    def test_keyword_arguments_reach_the_session(self):
        ns = build()
        ns["_peer_request"]("POST", ENDPOINT, "/api/translate", json={"text": "hi"}, timeout=15)
        sent = ns["_session"].sent[0]
        assert sent["json"] == {"text": "hi"}
        assert sent["timeout"] == 15


class TestHealingAForgottenPairing:
    def test_a_success_never_triggers_a_claim(self):
        ns = build(statuses=(200,))
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert ns["_claims"] == []
        assert len(ns["_session"].sent) == 1

    def test_a_refusal_reclaims_a_token_and_retries(self):
        ns = build(statuses=(403, 200))
        response = ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert response.status_code == 200
        assert ns["_claims"] == [{"endpoint": ENDPOINT, "force": True}]
        assert len(ns["_session"].sent) == 2

    def test_the_retry_presents_the_new_token(self):
        """The whole point — the second attempt must not repeat the rejected one."""
        ns = build(statuses=(403, 200))
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert _auth(ns["_session"].sent[0]) == "Bearer live-token"
        assert _auth(ns["_session"].sent[1]) == "Bearer fresh-token"

    def test_it_retries_only_once(self):
        # A B that refuses everything must not become an infinite loop.
        ns = build(statuses=(403, 403))
        response = ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert response.status_code == 403
        assert len(ns["_session"].sent) == 2

    def test_a_failed_claim_returns_the_refusal(self):
        ns = build(statuses=(403,), claim_result=False)
        response = ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert response.status_code == 403
        assert len(ns["_session"].sent) == 1

    def test_a_failed_claim_gives_the_old_token_back(self):
        # The refusal may have had nothing to do with pairing, and a token we
        # cannot replace is still better than none.
        ns = build(statuses=(403,), claim_result=False)
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert ns["_config"]["live_translation"]["remote"]["token"] == "live-token"

    def test_the_claim_is_made_without_the_rejected_token(self):
        """B refuses a claim that presents an unknown token, so it has to be
        dropped before asking — otherwise the ask fails the same way."""
        seen = {}
        ns = build(statuses=(403, 200))
        original_claim = ns["_claim_remote_pair_token"]

        def _spy(endpoint, force=False):
            seen["token_at_claim_time"] = ns["_remote_pair_token"]()
            return original_claim(endpoint, force=force)

        ns["_claim_remote_pair_token"] = _spy
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert seen["token_at_claim_time"] is None

    @pytest.mark.parametrize("status", [200, 401, 404, 500, 502])
    def test_only_a_403_triggers_the_heal(self, status):
        ns = build(statuses=(status,))
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert ns["_claims"] == []


class TestTheHandshakeOptsOut:
    def test_self_heal_off_does_not_claim(self):
        # Pairing runs before a token exists; a refusal there means "not paired
        # yet", and claiming would recurse into the thing being set up.
        ns = build(statuses=(403,))
        response = ns["_peer_request"]("POST", ENDPOINT, "/api/translate/pair/request",
                                       self_heal=False)
        assert response.status_code == 403
        assert ns["_claims"] == []
        assert len(ns["_session"].sent) == 1


class TestRateLimiting:
    def test_a_second_refusal_inside_the_window_does_not_claim_again(self):
        # A peer that refuses for a reason re-pairing cannot fix must not turn
        # every call into a claim.
        ns = build(statuses=(403, 200, 403))
        ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert len(ns["_claims"]) == 1

        response = ns["_peer_request"]("GET", ENDPOINT, "/api/health")
        assert response.status_code == 403
        assert len(ns["_claims"]) == 1

    def test_the_allowance_is_shared_across_paths(self):
        ns = build(statuses=(403, 200, 403))
        ns["_peer_request"]("POST", ENDPOINT, "/api/translate/preload")
        ns["_peer_request"]("POST", ENDPOINT, "/api/translate/sync-dictionary")
        assert len(ns["_claims"]) == 1

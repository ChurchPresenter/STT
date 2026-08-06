"""Unit tests for stt.pair_tokens — what makes a request a paired Machine A."""

from stt.pair_tokens import (
    authorize,
    bind,
    can_claim,
    fingerprint,
    forget,
    issue,
    mint_token,
    parse_bearer,
    rebind,
)

A_IP = "192.168.2.62"
OTHER_IP = "192.168.2.90"


def paired(ip=A_IP):
    """A store holding one paired machine, plus the token it was given."""
    tokens, token = issue({}, ip)
    return tokens, token


def test_a_token_is_unguessable_and_unique():
    first, second = mint_token(), mint_token()
    assert first != second
    assert len(first) > 30


def test_the_stored_fingerprint_is_not_the_token():
    token = mint_token()
    fp = fingerprint(token)
    assert token not in fp
    assert fp == fingerprint(token)          # stable
    assert fp != fingerprint(mint_token())   # per token


def test_parse_bearer_accepts_the_header_and_rejects_the_rest():
    assert parse_bearer("Bearer abc123") == "abc123"
    assert parse_bearer("bearer abc123") == "abc123"   # case of the scheme
    assert parse_bearer("Basic abc123") is None
    assert parse_bearer("abc123") is None
    assert parse_bearer("Bearer ") is None
    assert parse_bearer("") is None
    assert parse_bearer(None) is None


class TestAuthorize:
    def test_a_known_token_is_accepted(self):
        tokens, token = paired()
        assert authorize(tokens, [A_IP], A_IP, token).authorized

    def test_a_known_token_works_from_a_new_address(self):
        # The whole point: the machine moved, the trust moved with it.
        tokens, token = paired()
        decision = authorize(tokens, [A_IP], "192.168.2.77", token)
        assert decision.authorized
        assert decision.rebind_from == A_IP

    def test_an_unknown_token_is_refused_even_from_the_paired_address(self):
        # A revoked secret must not be quietly downgraded to the address check.
        tokens, _token = paired()
        assert not authorize(tokens, [A_IP], A_IP, mint_token()).authorized

    def test_an_address_with_a_token_may_not_skip_presenting_it(self):
        # This is what closes the DHCP hole: once upgraded, the address alone
        # stops being a way in, so whoever inherits it gets nothing.
        tokens, _token = paired()
        assert not authorize(tokens, [A_IP], A_IP, None).authorized

    def test_a_trusted_address_without_a_token_still_works(self):
        # A machine on an older build keeps working until it updates.
        assert authorize({}, [A_IP], A_IP, None).authorized

    def test_an_untrusted_address_is_refused(self):
        assert not authorize({}, [A_IP], OTHER_IP, None).authorized

    def test_a_missing_address_is_refused(self):
        assert not authorize({}, [A_IP], None, None).authorized

    def test_one_machines_token_does_not_let_another_address_skip_the_check(self):
        # B is paired with two machines; only one has upgraded.
        tokens, token = paired(A_IP)
        assert authorize(tokens, [A_IP, OTHER_IP], OTHER_IP, None).authorized
        assert authorize(tokens, [A_IP, OTHER_IP], OTHER_IP, token).authorized


class TestClaim:
    def test_a_paired_address_with_no_token_may_claim_one(self):
        assert can_claim({}, [A_IP], A_IP)

    def test_an_unpaired_address_may_not(self):
        assert not can_claim({}, [A_IP], OTHER_IP)

    def test_an_address_that_already_holds_one_may_not_claim_again(self):
        tokens, _token = paired()
        assert not can_claim(tokens, [A_IP], A_IP)

    def test_a_forgotten_machine_can_claim_again(self):
        # Unpaired then re-paired: its old token is gone, so it recovers by
        # claiming a new one rather than being stuck presenting a dead secret.
        tokens, _token = paired()
        tokens = forget(tokens, A_IP)
        assert can_claim(tokens, [A_IP], A_IP)


class TestStore:
    def test_binding_replaces_the_addresss_previous_token(self):
        tokens, first = paired()
        tokens = bind(tokens, A_IP, mint_token())
        assert len(tokens) == 1
        assert not authorize(tokens, [A_IP], A_IP, first).authorized

    def test_rebinding_moves_every_entry_for_an_address(self):
        tokens, token = paired()
        moved = rebind(tokens, A_IP, "10.0.0.5")
        assert authorize(moved, [], "10.0.0.5", token).authorized
        assert not authorize(moved, [], "10.0.0.5", None).authorized

    def test_forgetting_revokes_the_token(self):
        tokens, token = paired()
        tokens = forget(tokens, A_IP)
        assert tokens == {}
        assert not authorize(tokens, [], A_IP, token).authorized

    def test_forgetting_leaves_other_machines_alone(self):
        tokens, token = paired(A_IP)
        tokens = bind(tokens, OTHER_IP, mint_token())
        tokens = forget(tokens, OTHER_IP)
        assert authorize(tokens, [A_IP], A_IP, token).authorized


class TestReclaimWindow:
    """A token that was issued but never stored must not lock the pairing out."""

    def test_the_address_may_claim_again_inside_the_window(self):
        tokens, _token = paired()
        assert can_claim(tokens, [A_IP], A_IP, grace_ips=[A_IP])

    def test_and_may_not_outside_it(self):
        tokens, _token = paired()
        assert not can_claim(tokens, [A_IP], A_IP, grace_ips=[])

    def test_the_window_does_not_extend_to_other_addresses(self):
        tokens, _token = paired(A_IP)
        tokens = bind(tokens, OTHER_IP, mint_token())
        assert not can_claim(tokens, [A_IP, OTHER_IP], OTHER_IP, grace_ips=[A_IP])

    def test_the_window_never_admits_an_unpaired_address(self):
        assert not can_claim({}, [A_IP], OTHER_IP, grace_ips=[OTHER_IP])

    def test_the_window_does_not_weaken_the_request_check(self):
        # It only governs claiming. An un-authenticated request from an address
        # that has a token is still refused, window or not.
        tokens, _token = paired()
        assert not authorize(tokens, [A_IP], A_IP, None).authorized

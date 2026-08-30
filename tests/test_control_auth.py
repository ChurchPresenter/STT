"""Letting a stream-deck button reach a route that changes something.

The trade being made: a GET that changes state can be fired by any page a browser on the
network loads, and the operator's own browser is already authorised by a cookie or by
being on the whitelist. So the GET form is honoured only when the token is in the URL —
which a drive-by page cannot know, and a button configured once by the person who owns the
machine has forever.
"""

from stt.control_auth import KEY_PARAM, allowed, refuse_reason

TOKEN = "s3cret-token"


class TestGetNeedsTheToken:
    def test_the_right_token_is_allowed(self):
        assert refuse_reason({KEY_PARAM: TOKEN}, token=TOKEN) is None
        assert allowed({KEY_PARAM: TOKEN}, token=TOKEN) is True

    def test_no_token_on_the_request_is_refused(self):
        # The cookie-and-whitelist case: exactly what a drive-by page would ride on.
        reason = refuse_reason({}, token=TOKEN)
        assert reason and "?key=" in reason

    def test_the_wrong_token_is_refused(self):
        assert refuse_reason({KEY_PARAM: "nope"}, token=TOKEN)

    def test_a_partly_right_token_is_refused(self):
        assert refuse_reason({KEY_PARAM: TOKEN[:-1]}, token=TOKEN)

    def test_an_empty_key_parameter_is_not_a_token(self):
        assert refuse_reason({KEY_PARAM: ""}, token=TOKEN)


class TestNoTokenConfigured:
    def test_control_links_are_off_rather_than_open(self):
        # "No token means everyone" would turn a machine that never opted in into an open
        # control surface.
        reason = refuse_reason({KEY_PARAM: "anything"}, token="")
        assert reason and "no access token" in reason

    def test_it_says_how_to_turn_them_on(self):
        assert "Server Settings" in (refuse_reason({}, token="") or "")


class TestPostIsUnchanged:
    """A POST arrives through the routes that already exist, with their own gate."""

    def test_a_post_needs_nothing_extra(self):
        assert refuse_reason({}, token=TOKEN, method="POST") is None

    def test_a_post_is_unaffected_by_a_missing_token(self):
        assert refuse_reason({}, token="", method="POST") is None

    def test_the_method_is_read_case_insensitively(self):
        assert refuse_reason({}, token=TOKEN, method="post") is None

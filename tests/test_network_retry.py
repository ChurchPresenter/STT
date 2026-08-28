"""Transient-network classification and retry (stt/downloads.py).

The shapes here are the ones a model download actually dies on: a TLS handshake
that ends early, a name that will not resolve, a certificate a proxy replaced.
"""

import socket
import ssl

import pytest

from stt import downloads


def _ssl_eof():
    """The failure from STT-17/18: httpx wraps httpcore wraps an SSL EOF."""
    inner = ssl.SSLError(
        "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)"
    )
    try:
        try:
            raise inner
        except ssl.SSLError as exc:
            raise ConnectionError(str(exc)) from exc
    except ConnectionError as exc:
        return exc


class FakeTransportError(Exception):
    """Stands in for httpx.ConnectError, which these tests cannot import."""


class TestClassification:
    def test_ssl_eof_chain_is_transient(self):
        assert downloads.is_transient_network_error(_ssl_eof()) is True

    def test_unimportable_library_class_matched_by_name(self):
        exc = FakeTransportError("something went wrong")
        exc.__class__ = type("ConnectError", (Exception,), {})
        assert downloads.is_transient_network_error(exc) is True

    def test_message_hint_matches_when_class_is_generic(self):
        assert downloads.is_transient_network_error(
            Exception("Connection reset by peer")
        ) is True

    def test_dns_failure_is_transient(self):
        assert downloads.is_transient_network_error(
            socket.gaierror("[Errno -2] Name or service not known")
        ) is True

    def test_certificate_rejection_is_not_transient(self):
        # Stable condition: retrying it fails identically and delays the message.
        exc = ssl.SSLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        assert downloads.is_transient_network_error(exc) is False

    def test_programming_error_is_not_transient(self):
        assert downloads.is_transient_network_error(KeyError("siblings")) is False

    def test_cause_chain_walk_is_bounded_on_a_cycle(self):
        a, b = Exception("a"), Exception("b")
        a.__cause__ = b
        b.__cause__ = a
        assert downloads.is_transient_network_error(a) is False


class TestMessages:
    def test_non_network_error_gets_no_message(self):
        # The guard that keeps this from swallowing real bugs.
        assert downloads.network_error_message(KeyError("siblings")) is None

    def test_ssl_eof_message_names_the_host_and_not_the_c_file(self):
        msg = downloads.network_error_message(_ssl_eof())
        assert "huggingface.co" in msg
        assert "_ssl.c" not in msg

    def test_certificate_message_names_the_likely_cause(self):
        exc = ssl.SSLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        msg = downloads.network_error_message(exc)
        assert "antivirus" in msg or "proxy" in msg

    def test_dns_message_differs_from_the_generic_one(self):
        dns = downloads.network_error_message(socket.gaierror("getaddrinfo failed"))
        generic = downloads.network_error_message(_ssl_eof())
        assert dns != generic
        assert "DNS" in dns

    def test_timeout_message_differs_from_the_generic_one(self):
        timeout = downloads.network_error_message(TimeoutError("connection timed out"))
        assert "timed out" in timeout

    def test_host_is_configurable(self):
        msg = downloads.network_error_message(_ssl_eof(), host="example.test")
        assert "example.test" in msg


class TestRetry:
    def _sleeper(self):
        slept = []
        return slept, slept.append

    def test_returns_first_result_without_sleeping(self):
        slept, sleep = self._sleeper()
        assert downloads.call_with_retry(lambda: "files", sleep=sleep, log=lambda m: None) == "files"
        assert slept == []

    def test_retries_transient_failure_then_succeeds(self):
        slept, sleep = self._sleeper()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise _ssl_eof()
            return ["config.json"]

        result = downloads.call_with_retry(flaky, sleep=sleep, log=lambda m: None)
        assert result == ["config.json"]
        assert len(calls) == 3
        assert slept == [2.0, 4.0]  # linear backoff, no sleep after the success

    def test_gives_up_after_max_attempts_and_reraises_the_original(self):
        slept, sleep = self._sleeper()
        original = _ssl_eof()

        def always_fails():
            raise original

        with pytest.raises(ConnectionError) as caught:
            downloads.call_with_retry(always_fails, max_attempts=3, sleep=sleep, log=lambda m: None)
        assert caught.value is original
        assert len(slept) == 2  # no sleep after the final attempt

    def test_non_transient_failure_is_not_retried(self):
        slept, sleep = self._sleeper()
        calls = []

        def bug():
            calls.append(1)
            raise KeyError("siblings")

        with pytest.raises(KeyError):
            downloads.call_with_retry(bug, sleep=sleep, log=lambda m: None)
        assert len(calls) == 1
        assert slept == []

    def test_retry_is_logged_with_the_description(self):
        messages = []
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise _ssl_eof()
            return "ok"

        downloads.call_with_retry(flaky, description="File list for facebook/nllb",
                                  sleep=lambda _: None, log=messages.append)
        assert any("File list for facebook/nllb" in m for m in messages)

    def test_max_attempts_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            downloads.call_with_retry(lambda: None, max_attempts=0)

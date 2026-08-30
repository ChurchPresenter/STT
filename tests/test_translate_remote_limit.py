"""/api/translate must bound the text a paired machine can send.

The route accepted arbitrary text behind only an emptiness check, and that text then
became a TextTranslationCache key — held whole, for up to server_cache_size entries.
A caption is short: over 70,153 captions from real services the longest was ~1,100
characters, and that one was a Whisper repetition artifact rather than speech. So
anything large here is a malformed or runaway payload and is rejected before it
reaches either the model or the cache.

The route is extracted from the monolith and run against a stub namespace (see
tests/conftest.py) — the module cannot be imported, and CI installs no Flask.
"""

import pytest

from conftest import extract_definitions
from stt.coercion import coerce_float, coerce_int
from stt.llm_priority import KIND_CAPTION, PeerActivity


class _Request:
    def __init__(self, payload, remote_addr="192.168.2.62"):
        self._payload = payload
        self.remote_addr = remote_addr

    def get_json(self):
        return self._payload


def call_translate(payload, *, remote=None, translated="TRANSLATED", activity=None):
    """Run the route; returns (body, status). Status is 200 when the route omits it."""
    calls = {"translated": 0, "cached": 0}
    activity = PeerActivity() if activity is None else activity

    def _translate_live_text(text, *a, **kw):
        calls["translated"] += 1
        return translated

    class _Cache:
        def get(self, *a, **kw):
            return None

        def set(self, *a, **kw):
            calls["cached"] += 1

    ns = extract_definitions(
        "speech_to_text.py", ["translate_remote"],
        extra_globals={
            "config": {"live_translation": {"remote": remote if remote is not None else {}}},
            "request": _Request(payload),
            "jsonify": lambda obj: obj,
            # The real helpers, not stubs: the clamp semantics are part of what the
            # cap relies on for an unusable configured value.
            "coerce_int": coerce_int,
            "coerce_float": coerce_float,
            "_paired_client_ok": lambda ip=None: True,
            "_register_translation_client": lambda ip: None,
            "_peer_activity": activity,
            "_ACT_CAPTION": KIND_CAPTION,
            "time": __import__("time"),
            "get_server_text_cache": _Cache,
            "translate_live_text": _translate_live_text,
            "_should_cache_translation": lambda a, b: True,
            "app": type("A", (), {"route": staticmethod(lambda *a, **kw: (lambda f: f))})(),
        },
    )
    result = ns["translate_remote"]()
    body, status = result if isinstance(result, tuple) else (result, 200)
    return body, status, calls


class TestCaptionActivityIsRecorded:
    """The route is where the machine holding the model learns captions are in flight.

    Nothing else on this machine knows: it is not transcribing, so its own translation
    loop never runs and never reports a backlog. Without this record the sermon summariser
    reads the machine as idle and takes the generation lock out from under a live service.
    """

    def test_a_translated_caption_is_recorded(self):
        activity = PeerActivity()
        call_translate({"text": "Слово"}, activity=activity)
        assert activity.last_seen(KIND_CAPTION) > 0

    def test_an_oversized_payload_is_not(self):
        # It never reaches the model, so it must not hold a summary back either.
        activity = PeerActivity()
        call_translate({"text": "я" * 9000}, activity=activity)
        assert activity.last_seen(KIND_CAPTION) == 0

    def test_an_empty_caption_is_not(self):
        activity = PeerActivity()
        call_translate({"text": "   "}, activity=activity)
        assert activity.last_seen(KIND_CAPTION) == 0


class TestLengthCap:
    def test_oversized_text_is_rejected_with_413(self):
        body, status, _ = call_translate({"text": "я" * 9000})
        assert status == 413
        assert body["success"] is False
        assert "too long" in body["error"].lower()

    def test_the_model_is_never_asked(self):
        # Rejecting after translating would defeat the point: the cost this guards
        # against is the model call and the cache key, not the response size.
        _, _, calls = call_translate({"text": "я" * 9000})
        assert calls["translated"] == 0
        assert calls["cached"] == 0

    def test_a_real_sized_caption_passes(self):
        # p99.9 over the archive is 238 characters; the longest ever seen ~1,100.
        body, status, calls = call_translate({"text": "Мир вам. " * 120})
        assert status == 200
        assert body["success"] is True
        assert calls["translated"] == 1

    def test_the_limit_is_configurable(self):
        body, status, _ = call_translate({"text": "x" * 500}, remote={"max_text_chars": 200})
        assert status == 413
        assert "200-char" in body["error"]

    def test_boundary_is_inclusive(self):
        _, status, _ = call_translate({"text": "x" * 200}, remote={"max_text_chars": 200})
        assert status == 200
        _, status, _ = call_translate({"text": "x" * 201}, remote={"max_text_chars": 200})
        assert status == 413

    def test_a_missing_setting_uses_the_shipped_default(self):
        # 8000 — about 7x the largest caption ever recorded here.
        _, status, _ = call_translate({"text": "x" * 8000}, remote={})
        assert status == 200
        _, status, _ = call_translate({"text": "x" * 8001}, remote={})
        assert status == 413

    @pytest.mark.parametrize("bad", [None, "lots", -1])
    def test_an_unusable_setting_falls_back_to_the_default(self, bad):
        # Never 0: a bad value must not reject every caption the pair sends.
        _, status, _ = call_translate({"text": "x" * 100}, remote={"max_text_chars": bad})
        assert status == 200

    def test_empty_text_still_short_circuits_before_the_cap(self):
        body, status, calls = call_translate({"text": "   "})
        assert status == 200
        assert body["translated_text"] == ""
        assert calls["translated"] == 0

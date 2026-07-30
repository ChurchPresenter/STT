"""Reading request params from JSON, form, or query (stt/http_params.py)."""

import pytest

from stt.http_params import merge_request_params


class TestPrecedence:
    def test_json_beats_form_and_query(self):
        merged = merge_request_params({"translation": "ru"}, {"translation": "es"},
                                      {"translation": "fr"})
        assert merged["translation"] == "ru"

    def test_form_beats_query(self):
        merged = merge_request_params(None, {"translation": "es"}, {"translation": "fr"})
        assert merged["translation"] == "es"

    def test_sources_are_unioned_not_replaced(self):
        # A surface may split its intent across a URL and a body; both arrive.
        merged = merge_request_params({"translation": "ru"}, None, {"transcription": "en"})
        assert merged == {"translation": "ru", "transcription": "en"}

    def test_query_alone_is_enough(self):
        assert merge_request_params(None, None, {"translation": "ru"}) == {"translation": "ru"}

    def test_form_alone_is_enough(self):
        assert merge_request_params(None, {"translation": "ru"}) == {"translation": "ru"}


class TestUnsentValues:
    """A field left empty must read as "not sent", never as "set to nothing"."""

    def test_blank_json_value_does_not_blank_a_lower_source(self):
        merged = merge_request_params({"translation": ""}, None, {"translation": "ru"})
        assert merged["translation"] == "ru"

    def test_whitespace_only_is_treated_as_blank(self):
        merged = merge_request_params({"translation": "   "}, None, {"translation": "ru"})
        assert merged["translation"] == "ru"

    def test_none_value_is_dropped(self):
        assert merge_request_params({"translation": None, "transcription": "en"}) == {
            "transcription": "en"}

    def test_blank_everywhere_yields_no_key(self):
        assert merge_request_params({"translation": ""}, {"translation": ""},
                                    {"translation": ""}) == {}

    def test_false_and_zero_are_real_values(self):
        merged = merge_request_params({"enabled": False, "count": 0})
        assert merged == {"enabled": False, "count": 0}


class TestWhitespaceStripping:
    def test_language_code_with_trailing_space_is_usable(self):
        # The Companion button that prompted this carried a trailing space.
        assert merge_request_params({"translation": "ru "})["translation"] == "ru"

    def test_leading_and_inner_whitespace(self):
        assert merge_request_params({"translation": "  ru"})["translation"] == "ru"
        assert merge_request_params({"text": "a b"})["text"] == "a b"

    def test_non_strings_pass_through_untouched(self):
        merged = merge_request_params({"count": 3, "flag": True, "items": [1, 2]})
        assert merged == {"count": 3, "flag": True, "items": [1, 2]}


class TestMalformedInput:
    """Bad input must degrade to "nothing sent", never raise into a 500."""

    @pytest.mark.parametrize("body", [None, {}, [], "a string", 42, True])
    def test_non_mapping_json_body_is_ignored(self, body):
        assert merge_request_params(body) == {}

    def test_non_mapping_json_still_lets_other_sources_through(self):
        assert merge_request_params(["not", "a", "dict"], None,
                                    {"translation": "ru"}) == {"translation": "ru"}

    @pytest.mark.parametrize("form", [None, "nope", 7])
    def test_non_mapping_form_is_ignored(self, form):
        assert merge_request_params({"translation": "ru"}, form) == {"translation": "ru"}

    def test_all_sources_absent(self):
        assert merge_request_params(None, None, None) == {}

    def test_non_string_keys_are_stringified(self):
        assert merge_request_params({1: "a"}) == {"1": "a"}


class TestRealPayloads:
    def test_the_companion_button_body(self):
        # Exactly what button 10/1/1 sends, trailing space and all.
        body = {"transcription": "ru", "translation": "en"}
        assert merge_request_params(body, {}, {}) == body

    def test_the_same_intent_as_a_query_string(self):
        # /api/language?transcription=ru&translation=en - what a plain GET-style
        # surface would send, and previously a 400.
        assert merge_request_params(None, None,
                                   {"transcription": "ru", "translation": "en"}) == {
            "transcription": "ru", "translation": "en"}

    def test_wrong_content_type_falls_back_to_form(self):
        # Flask hands us json_body=None when the Content-Type isn't JSON; the
        # body still arrives parsed as form data.
        assert merge_request_params(None, {"transcription": "ru", "translation": "en"}) == {
            "transcription": "ru", "translation": "en"}

"""LLM translation request building and response validation (stt/llm_translate.py).

The rejection fixtures are verbatim outputs observed while measuring candidate models
against real captions — not invented cases.
"""

import pytest

from stt.llm_translate import (
    build_chat_messages,
    build_chat_payload,
    extract_chat_text,
    looks_like_reasoning_model,
    validate_translation,
)

SRC = "Да будет мир Твой, Господи, с нами всегда."
GOOD = "May Your peace, Lord, remain with us always."


class TestBuildChatMessages:
    def test_system_then_user(self):
        m = build_chat_messages("Мир вам.", "SYS")
        assert [x["role"] for x in m] == ["system", "user"]
        assert m[0]["content"] == "SYS"
        assert m[1]["content"] == "Мир вам."

    def test_draft_switches_to_post_editing(self):
        m = build_chat_messages("Мир вам.", "SYS", draft="Peace be with you.")
        assert "Russian: Мир вам." in m[1]["content"]
        assert "Draft translation: Peace be with you." in m[1]["content"]


class TestBuildChatPayload:
    def test_defaults_are_deterministic_and_pinned(self):
        p = build_chat_payload("m", "Мир вам.", "SYS")
        assert p["temperature"] == 0, "captions must be reproducible for review"
        assert p["keep_alive"] == -1, "an unpinned model measured p90 4.89s vs p50 0.29s"
        assert p["stream"] is False
        assert p["max_tokens"] == 120

    def test_keep_alive_can_be_omitted_for_providers_that_reject_it(self):
        assert "keep_alive" not in build_chat_payload("m", "x", "SYS", keep_alive=None)

    def test_extra_merges(self):
        p = build_chat_payload("m", "x", "SYS", extra={"top_p": 0.9})
        assert p["top_p"] == 0.9


class TestExtractChatText:
    def test_openai_shape(self):
        assert extract_chat_text({"choices": [{"message": {"content": "hi"}}]}) == "hi"

    def test_ollama_shape(self):
        assert extract_chat_text({"message": {"content": "hi"}}) == "hi"

    def test_plain_completion_shapes(self):
        assert extract_chat_text({"response": "hi"}) == "hi"
        assert extract_chat_text({"choices": [{"text": "hi"}]}) == "hi"

    @pytest.mark.parametrize("resp", [None, {}, "str", 7, {"choices": []},
                                      {"choices": [{}]}, {"message": {}}])
    def test_unusable_shapes_give_none(self, resp):
        assert extract_chat_text(resp) is None


class TestValidateAccepts:
    def test_a_plain_translation(self):
        assert validate_translation(GOOD, SRC, "en") == GOOD

    def test_strips_quote_wrapper(self):
        assert validate_translation(f'"{GOOD}"', SRC, "en") == GOOD
        assert validate_translation(f"«{GOOD}»", SRC, "en") == GOOD

    def test_strips_label_prefixes(self):
        for prefix in ("Translation: ", "English: ", "Output: ", "Corrected: "):
            assert validate_translation(prefix + GOOD, SRC, "en") == GOOD

    def test_strips_nested_label_and_quotes(self):
        assert validate_translation(f'Translation: "{GOOD}"', SRC, "en") == GOOD

    def test_strips_closed_think_block(self):
        raw = f"<think>the user wants a translation</think>{GOOD}"
        assert validate_translation(raw, SRC, "en") == GOOD

    def test_short_source_may_expand_freely(self):
        # "Щит веры." -> "The shield of faith." is a legitimate expansion.
        assert validate_translation("The shield of faith.", "Щит веры.", "en")

    def test_same_script_target_skips_the_script_check(self):
        # en->de cannot be checked this way; Cyrillic screen must not misfire.
        assert validate_translation("Friede sei mit euch.", "Peace be with you.", "de")


class TestValidateRejects:
    """Every fixture here is output a real model actually produced."""

    def test_reasoning_preamble(self):
        raw = ("Okay, let's tackle this translation request. The user wants me to "
               "translate a Russian sentence into English.")
        assert validate_translation(raw, SRC, "en") is None

    def test_reasoning_preamble_second_form(self):
        raw = "We are given a live caption in Russian for a Baptist church service."
        assert validate_translation(raw, SRC, "en") is None

    def test_source_language_leaking_through(self):
        # llama3.2:1b returned this mixed-language output.
        raw = "Мы preparing for the meeting, speaking with one another"
        assert validate_translation(raw, SRC, "en") is None

    def test_wholly_untranslated(self):
        raw = "Здесь приводится совсем другое сравнение."
        assert validate_translation(raw, SRC, "en") is None

    def test_refusal(self):
        raw = ("I can't assist with providing translations that may be mistranslated "
               "or have incorrect meaning.")
        assert validate_translation(raw, SRC, "en") is None

    def test_echoed_prompt_keeps_only_the_answer(self):
        # The 1b echoed our framing back; only the part after the draft label is real.
        raw = f"Russian: {SRC}\nDraft translation: {GOOD}"
        assert validate_translation(raw, SRC, "en") == GOOD

    def test_echoed_prompt_with_no_answer_is_rejected(self):
        raw = f"Russian: {SRC}"
        assert validate_translation(raw, SRC, "en") is None

    @pytest.mark.parametrize("raw", [None, "", "   ", '""', "Translation:"])
    def test_empty_and_label_only(self, raw):
        assert validate_translation(raw, SRC, "en") is None

    def test_commentary_padding(self):
        raw = GOOD + " " + ("This phrase is a common liturgical blessing which "
                            "originates in Hebrew and carries connotations of "
                            "wholeness and wellbeing, often used at the close of "
                            "a service to dismiss the congregation. " * 3)
        assert validate_translation(raw, SRC, "en") is None

    def test_short_source_still_has_a_bounded_budget(self):
        """The regression that a full-service run exposed.

        Asked to translate the reference "1 Фессалоникийцам 5 глава.", the model answered
        with the reference *and then recited the passage*. An earlier version exempted
        sources under three words from the expansion check, which let this through.
        """
        recitation = ("1 Corinthians 11:1-24\n\n1 Now, brothers and sisters, about the "
                      "Lord's Supper, which I received from the Lord, I also passed on "
                      "to you, that the Lord Jesus on the night he was betrayed took "
                      "bread and gave thanks.")
        assert validate_translation(recitation, "1 Фессалоникийцам 5 глава.", "en") is None

    def test_multi_paragraph_output_is_a_document_not_a_caption(self):
        assert validate_translation("A line.\n\nAnother paragraph.", "Одна строка.",
                                    "en") is None

    def test_short_source_legitimate_expansion_still_passes(self):
        # 2 content words in, 4 out — inside the absolute floor, must survive.
        assert validate_translation("The shield of faith.", "Щит веры.",
                                    "en") == "The shield of faith."
        assert validate_translation("Let us pray together now.", "Помолимся вместе.",
                                    "en") is not None


class TestLooksLikeReasoningModel:
    def test_separate_thinking_field(self):
        assert looks_like_reasoning_model({"message": {"thinking": "hmm", "content": "hi"}})
        assert looks_like_reasoning_model({"thinking": "hmm", "response": "hi"})

    def test_inline_reasoning_in_content(self):
        # The case actually observed: reasoning in `content`, not separable.
        resp = {"message": {"content": "Okay, let's tackle this translation request."}}
        assert looks_like_reasoning_model(resp)

    def test_think_tag(self):
        assert looks_like_reasoning_model({"message": {"content": "<think>hm</think>hi"}})

    def test_a_plain_answer_is_not_flagged(self):
        assert not looks_like_reasoning_model({"message": {"content": GOOD}})

    def test_unusable_shapes(self):
        assert not looks_like_reasoning_model({})
        assert not looks_like_reasoning_model("nope")

"""LLM translation request building and response validation (stt/llm_translate.py).

The rejection fixtures are verbatim outputs observed while measuring candidate models
against real captions — not invented cases.
"""

import os

import pytest

from stt.llm_translate import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    build_chat_messages,
    build_chat_payload,
    build_system_prompt,
    extract_chat_text,
    language_name,
    local_model_path,
    looks_like_reasoning_model,
    resolve_gpu_layers,
    scan_gguf_models,
    validate_translation,
)

NAMES = {"en": "English", "es": "Spanish", "de": "German", "ru": "Russian"}

SRC = "Да будет мир Твой, Господи, с нами всегда."
GOOD = "May Your peace, Lord, remain with us always."


class TestBuildChatMessages:
    def test_system_then_user(self):
        m = build_chat_messages("Мир вам.", "SYS")
        assert [x["role"] for x in m] == ["system", "user"]
        assert m[0]["content"] == "SYS"
        assert m[1]["content"] == "Мир вам."

    def test_draft_switches_to_post_editing(self):
        m = build_chat_messages("Мир вам.", "SYS", draft="Peace be with you.",
                                source_name="Russian")
        assert "Russian: Мир вам." in m[1]["content"]
        assert "Draft translation: Peace be with you." in m[1]["content"]

    def test_source_label_defaults_to_a_neutral_word(self):
        # The source is frequently "auto", so the label must not assert a language.
        m = build_chat_messages("Мир вам.", "SYS", draft="Peace be with you.")
        assert "Source: Мир вам." in m[1]["content"]
        assert "Russian" not in m[1]["content"]


class TestLanguageName:
    def test_known_code(self):
        assert language_name("es", NAMES) == "Spanish"

    def test_case_insensitive(self):
        assert language_name("ES", NAMES) == "Spanish"

    def test_unknown_code_returns_the_code(self):
        # Not a guess and not empty: the operator can see what was actually sent.
        assert language_name("zz", NAMES) == "zz"

    @pytest.mark.parametrize("code", [None, "", "   "])
    def test_missing_code_is_empty(self, code):
        assert language_name(code, NAMES) == ""

    def test_without_a_catalog(self):
        assert language_name("es") == "es"


class TestBuildSystemPrompt:
    """The regression: a Spanish session that silently produced English."""

    def test_template_is_filled_with_the_target_language(self):
        p = build_system_prompt(DEFAULT_SYSTEM_PROMPT_TEMPLATE, "es", NAMES)
        assert "Spanish Bibles" in p
        assert "{language}" not in p
        assert "English" not in p, "the shipped prompt must not assert English"

    def test_english_target_reads_naturally(self):
        p = build_system_prompt(DEFAULT_SYSTEM_PROMPT_TEMPLATE, "en", NAMES)
        assert "English Bibles" in p
        assert "Translate into English." in p

    def test_target_is_stated_explicitly(self):
        assert "Translate into German." in build_system_prompt("", "de", NAMES)

    def test_custom_prompt_still_gets_the_target_appended(self):
        """A custom prompt is written for the language configured at the time.

        Without the appended directive, switching the target later would leave the
        model still being told to produce the old language — silently, because the
        validator's wrong-script screen cannot tell English from Spanish.
        """
        custom = "You translate for a Baptist service. Render вечеря as communion."
        p = build_system_prompt(custom, "es", NAMES)
        assert p.startswith(custom)
        assert "Translate into Spanish." in p

    def test_custom_prompt_may_use_the_placeholder_too(self):
        p = build_system_prompt("Answer in {language} only.", "de", NAMES)
        assert "Answer in German only." in p

    def test_blank_base_falls_back_to_the_shipped_template(self):
        p = build_system_prompt("   ", "en", NAMES)
        assert "live captions for a church service" in p

    def test_unknown_target_still_names_something(self):
        p = build_system_prompt(DEFAULT_SYSTEM_PROMPT_TEMPLATE, "zz", NAMES)
        assert "Translate into zz." in p

    def test_missing_target_leaves_the_prompt_usable(self):
        p = build_system_prompt(DEFAULT_SYSTEM_PROMPT_TEMPLATE, "", NAMES)
        assert "{language}" not in p
        assert "Translate into" not in p


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


class TestLocalModelPath:
    """The in-process provider stores a GGUF where every other model lives."""

    def test_repo_slash_becomes_double_dash(self):
        p = local_model_path("/m", "bartowski/Qwen2.5-7B-Instruct-GGUF", "q4.gguf")
        # os.path.join, so the separator is the platform's own.
        assert p == os.path.join("/m", "bartowski--Qwen2.5-7B-Instruct-GGUF", "q4.gguf")
        assert "bartowski--Qwen2.5-7B-Instruct-GGUF" in p, "the repo slash must not survive"

    def test_repo_without_a_slash(self):
        assert local_model_path("/m", "somerepo", "q4.gguf") == os.path.join(
            "/m", "somerepo", "q4.gguf")


class TestScanGgufModels:
    """The inverse of local_model_path, driving the settings-page picker."""

    def _make(self, root, repo_dir, *files):
        d = root / repo_dir
        d.mkdir(parents=True)
        for name in files:
            (d / name).write_bytes(b"x" * 10)
        return d

    def test_finds_a_repo_and_restores_the_slash(self, tmp_path):
        self._make(tmp_path, "ggml-org--gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf")
        found = scan_gguf_models(str(tmp_path))
        assert [m["repo"] for m in found] == ["ggml-org/gemma-3-4b-it-GGUF"]
        assert found[0]["files"][0]["name"] == "gemma-3-4b-it-Q4_K_M.gguf"
        assert found[0]["files"][0]["size_bytes"] == 10

    def test_round_trips_with_local_model_path(self, tmp_path):
        repo = "ggml-org/gemma-3-4b-it-GGUF"
        path = local_model_path(str(tmp_path), repo, "q4.gguf")
        os.makedirs(os.path.dirname(path))
        open(path, "wb").write(b"x")
        assert scan_gguf_models(str(tmp_path))[0]["repo"] == repo

    def test_lists_every_quantisation(self, tmp_path):
        self._make(tmp_path, "some--repo", "m-Q4_K_M.gguf", "m-Q8_0.gguf", "README.md")
        files = [f["name"] for f in scan_gguf_models(str(tmp_path))[0]["files"]]
        assert files == ["m-Q4_K_M.gguf", "m-Q8_0.gguf"], "non-gguf files must not appear"

    def test_directory_without_a_gguf_is_omitted(self, tmp_path):
        # An NMT model directory shares this tree; it is not a choice here.
        self._make(tmp_path, "google--madlad400-3b-mt", "model.safetensors")
        assert scan_gguf_models(str(tmp_path)) == []

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert scan_gguf_models(str(tmp_path / "nope")) == []

    def test_loose_files_at_the_root_are_ignored(self, tmp_path):
        (tmp_path / "stray.gguf").write_bytes(b"x")
        assert scan_gguf_models(str(tmp_path)) == []

    def test_repos_are_sorted(self, tmp_path):
        self._make(tmp_path, "zzz--b", "b.gguf")
        self._make(tmp_path, "aaa--a", "a.gguf")
        assert [m["repo"] for m in scan_gguf_models(str(tmp_path))] == ["aaa/a", "zzz/b"]


class TestResolveGpuLayers:
    def test_auto_uses_the_gpu_when_there_is_one(self):
        assert resolve_gpu_layers("auto", has_gpu=True) == -1

    def test_auto_falls_back_to_cpu(self):
        # CPU is viable, not an error: captions are short (p50 19 output tokens).
        assert resolve_gpu_layers("auto", has_gpu=False) == 0

    def test_explicit_int_is_honoured(self):
        assert resolve_gpu_layers(20, has_gpu=True) == 20
        assert resolve_gpu_layers(0, has_gpu=True) == 0

    def test_numeric_string_is_honoured(self):
        assert resolve_gpu_layers("20", has_gpu=False) == 20

    @pytest.mark.parametrize("value", [None, "", "nonsense", [], {}])
    def test_unusable_values_fall_back_to_auto(self, value):
        assert resolve_gpu_layers(value, has_gpu=True) == -1
        assert resolve_gpu_layers(value, has_gpu=False) == 0

    def test_bool_is_not_mistaken_for_an_int_count(self):
        # bool is an int subclass; True would otherwise mean "offload 1 layer".
        assert resolve_gpu_layers(True, has_gpu=True) in (-1, 1)


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

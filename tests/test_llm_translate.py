"""LLM translation request building and response validation (stt/llm_translate.py).

The rejection fixtures are verbatim outputs observed while measuring candidate models
against real captions — not invented cases.
"""

import json
import os

import pytest

from stt.llm_translate import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    build_chat_messages,
    build_chat_payload,
    build_system_prompt,
    estimate_tokens,
    extract_chat_text,
    fit_context_prefix,
    input_fits,
    input_token_budget,
    language_name,
    local_model_path,
    looks_like_reasoning_model,
    resolve_gpu_layers,
    scan_gguf_models,
    uses_local_llm,
    numbers_survived,
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


class TestScriptureReferences:
    """The two ways an LLM fails on a scripture reference.

    Asked for a *reference* it answers with the passage it remembers; asked for a verse
    *range* it counts the range out. Both are fluent, correctly-scripted output of plausible
    length, so every other check here waves them through. What moves in both cases is the
    figures — a recited passage drops the chapter and verse, an enumeration invents numbers
    nobody said.

    The cases below are constructed to that shape. The observed originals are congregation
    speech and are not kept in this repository; TestRealServiceCaptions runs against them
    where they exist.
    """

    RECITED = [
        # A reference, answered with a passage: the chapter and verse vanish.
        ("Послание к Римлянам, 8 глава, 28 стих.",
         "And we know that all things work together for good to them that love God."),
        # The passage is quoted correctly but the reference in front of it is dropped.
        ("Мы читаем в 5 главе, 14 стихе, что вы свет мира.",
         "You are the light of the world."),
    ]

    @pytest.mark.parametrize("source,output", RECITED)
    def test_a_recited_passage_is_refused(self, source, output):
        assert validate_translation(output, source, "en") is None

    def test_a_counted_out_verse_range_is_refused(self):
        # A range answered with its own contents. This passed the older checks because a
        # five-word source buys a fifteen-word budget and the list is thirteen tokens, and
        # because it uses single newlines rather than the blank lines the document check
        # looks for.
        assert validate_translation("2\n3\n4\n5\n6\n7\n8\n9",
                                    "с 2 по 9 стихи.", "en") is None

    def test_a_reference_translated_as_a_reference_passes(self):
        # Most references are handled correctly and must keep passing.
        for source, output in [
            ("Евангелие от Иоанна, 3 глава, 14 стих.", "The Gospel of John, chapter 3, verse 14."),
            ("В Евреям, 9 глава, 22 стих.", "Hebrews, chapter 9, verse 22."),
            ("Мы прочитаем об этом в 1 главе, 9 стихе.",
             "We will read about this in chapter 1, verse 9."),
        ]:
            assert validate_translation(output, source, "en") == output

    def test_a_number_spelled_out_is_not_a_dropped_number(self):
        assert validate_translation("chapter three, verse sixteen is where we are",
                                    "3 глава, 16 стих", "en") is None
        assert validate_translation("the third chapter", "3 глава", "en") == "the third chapter"

    def test_one_invented_number_is_tolerated(self):
        # A translation may write in figures what the source wrote in words. That is not
        # the enumeration failure, and rejecting it would be noisier than the fault.
        assert validate_translation("We sang 2 hymns before the sermon.",
                                    "Мы спели два гимна перед проповедью.", "en") is not None

    def test_prose_with_no_numbers_is_unaffected(self):
        assert validate_translation("This is an ordinary sentence with no numbers.",
                                    "Это обычное предложение без чисел.", "en") is not None


class TestRealServiceCaptions:
    """The validator against whole services, where the fixtures for them exist locally.

    Every caption a service produced is a far better test than any case anyone invents —
    it is the only way to know a new rule is not quietly rejecting good translations at
    scale. But those captions are verbatim congregation speech, so the file holding them is
    gitignored and generated on the machine that recorded the session. Absent, these skip:
    CI checks the constructed cases above, and this pair of assertions is the extra
    confidence available to whoever has the recordings.
    """

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "real_captions.json")

    @pytest.fixture
    def captions(self):
        if not os.path.exists(self.PATH):
            pytest.skip("no local caption fixtures; see .gitignore")
        with open(self.PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_every_known_bad_caption_is_refused(self, captions):
        survived = [(src, out) for src, out in captions["must_reject"]
                    if validate_translation(out, src, "en") is not None]
        assert not survived, "a known-bad caption passed validation"

    def test_the_good_captions_are_not_rejected(self, captions):
        # The rule has to be precise, not merely strict: rejecting sound translations
        # costs the LLM's quality on every one of them.
        rejected = [(src, out) for src, out in captions["must_accept"]
                    if validate_translation(out, src, "en") is None]
        assert not rejected, f"{len(rejected)} good captions would now fall back to NMT"


class TestNumbersSurvived:
    def test_a_missing_figure_fails(self):
        assert not numbers_survived("в 4 главе, с 12 стиха", "in the epistle", "en")

    def test_all_figures_present_passes(self):
        assert numbers_survived("4 глава, 12 стих", "chapter 4, verse 12", "en")

    def test_a_crowd_of_invented_figures_fails(self):
        assert not numbers_survived("с 2 по 9 стихи", "1 2 3 4 5 6 7 8 9 10 11 12 13", "en")

    def test_no_figures_either_side_is_fine(self):
        assert numbers_survived("Помолимся.", "Let us pray.", "en")

    def test_an_unknown_target_language_compares_digits_only(self):
        # The safe direction: without a numeral table, a spelled-out form reads as missing
        # rather than being waved through.
        assert not numbers_survived("3 глава", "kolmas luku", "fi")
        assert numbers_survived("3 глава", "luku 3", "fi")


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

    def test_transformers_model_shipping_ggufs_is_omitted(self, tmp_path):
        """MADLAD's own repo ships GGUFs beside its safetensors.

        Downloading it as a translation model therefore offered
        "madlad400-3b-mt" in the LLM picker — an NMT model with no chat
        template, which cannot answer a chat-completion request at all.
        """
        self._make(tmp_path, "google--madlad400-3b-mt",
                   "model.safetensors", "model-q4k.gguf", "config.json")
        assert scan_gguf_models(str(tmp_path)) == []

    @pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin",
                                          "flax_model.msgpack", "tf_model.h5"])
    def test_any_transformers_weight_format_disqualifies_the_directory(self, tmp_path, weights):
        self._make(tmp_path, "some--repo", weights, "m-Q4_K_M.gguf")
        assert scan_gguf_models(str(tmp_path)) == []

    def test_a_real_gguf_release_still_lists(self, tmp_path):
        # Metadata alongside the quantisations is normal and must not disqualify it.
        self._make(tmp_path, "bartowski--Qwen2.5-7B-Instruct-GGUF",
                   "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "README.md", "config.json",
                   ".gitattributes")
        found = scan_gguf_models(str(tmp_path))
        assert [m["repo"] for m in found] == ["bartowski/Qwen2.5-7B-Instruct-GGUF"]

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


class TestUsesLocalLlm:
    """Decides whether this machine holds GGUF weights it is responsible for."""

    def test_local_provider_under_the_llm_method(self):
        assert uses_local_llm({"translation_method": "llm", "llm": {"provider": "local"}})

    def test_provider_is_case_and_space_insensitive(self):
        assert uses_local_llm({"translation_method": "llm", "llm": {"provider": " LOCAL "}})

    def test_endpoint_provider_is_another_machines_model(self):
        assert not uses_local_llm({"translation_method": "llm", "llm": {"provider": "endpoint"}})

    def test_absent_provider_defaults_to_endpoint(self):
        # Not "local": guessing local would unload weights this box never loaded.
        assert not uses_local_llm({"translation_method": "llm", "llm": {}})
        assert not uses_local_llm({"translation_method": "llm"})

    def test_nmt_methods_never_use_the_llm(self):
        # A configured local LLM is irrelevant while an NMT method is selected.
        local = {"provider": "local"}
        for method in ("nllb", "madlad", "whisper_translate", "whisper_forced_lang"):
            assert not uses_local_llm({"translation_method": method, "llm": local})

    def test_absent_method_defaults_to_nllb(self):
        assert not uses_local_llm({"llm": {"provider": "local"}})

    def test_unusable_shapes(self):
        assert not uses_local_llm({})
        assert not uses_local_llm(None)


# A word-per-token counter: deterministic, and independent of any real vocabulary, so
# these tests assert the budgeting arithmetic rather than a tokenizer's behaviour.
def _words(text):
    return len(text.split())


class TestEstimateTokens:
    """The fallback used when no real tokenizer is available (the endpoint provider)."""

    def test_cyrillic_costs_more_than_latin_per_character(self):
        # A BPE vocabulary trained mostly on English splits Cyrillic harder. Under-
        # estimating it is what would let a prompt past the budget and into the
        # exception this whole path exists to avoid.
        cyr = "абвгдежзийклмнопрстуфхцчшщэюя" * 4
        lat = "abcdefghijklmnopqrstuvwxyzabc" * 4
        assert len(cyr) == len(lat)
        assert estimate_tokens(cyr) > estimate_tokens(lat)

    def test_never_zero_for_non_empty_input(self):
        # A zero would read as "fits" against any budget, including an exhausted one.
        assert estimate_tokens("a") >= 1
        assert estimate_tokens(".") >= 1

    def test_empty_is_free(self):
        assert estimate_tokens("") == 0

    def test_grows_with_length(self):
        assert estimate_tokens(SRC * 4) > estimate_tokens(SRC)


class TestInputTokenBudget:
    def test_reserves_reply_prompt_and_margin(self):
        # 1000 ctx - 100 reply - 3 prompt words - 64 margin
        assert input_token_budget(1000, 100, "one two three", counter=_words, margin=64) == 833

    def test_floors_at_zero_when_the_window_is_tiny(self):
        # Never negative: callers compare against it, and a negative budget would
        # invert the comparison rather than reject.
        assert input_token_budget(512, 1024, "a system prompt", counter=_words) == 0

    def test_uses_the_estimate_when_no_counter_is_given(self):
        assert input_token_budget(2048, 160, "") == 2048 - 160 - 64

    def test_a_raising_counter_degrades_to_the_estimate(self):
        # A tokenizer call must never break a caption.
        def boom(_):
            raise RuntimeError("tokenizer gone")

        assert input_token_budget(2048, 160, "hello", counter=boom) > 0


class TestInputFits:
    def test_a_real_caption_fits_the_shipped_defaults(self):
        budget = input_token_budget(2048, 160, DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        assert input_fits(SRC, budget)

    def test_an_exhausted_budget_admits_nothing(self):
        assert not input_fits("a", 0)
        assert not input_fits("a", -5)

    def test_boundary_is_inclusive(self):
        assert input_fits("one two three", 3, counter=_words)
        assert not input_fits("one two three four", 3, counter=_words)


class TestFitContextPrefix:
    def test_everything_kept_when_it_all_fits(self):
        ctx = ["one", "two", "three"]
        assert fit_context_prefix(ctx, "target", 100, counter=_words) == ctx

    def test_sheds_oldest_first(self):
        # budget 3 words: "two three target" fits, "one two three target" does not.
        assert fit_context_prefix(["one", "two", "three"], "target", 3, counter=_words) == ["two", "three"]

    def test_returns_empty_when_not_even_one_entry_fits(self):
        # The target alone may still fit — that is the caller's separate check, because
        # the target is the caption and can never be dropped.
        assert fit_context_prefix(["one"], "target", 1, counter=_words) == []

    def test_never_drops_the_target(self):
        # Whatever comes back is prefix only; the caption is not this function's to cut.
        kept = fit_context_prefix(["a", "b"], "the caption", 0, counter=_words)
        assert kept == []

    def test_blank_context_entries_are_dropped(self):
        assert fit_context_prefix(["", "two", ""], "target", 100, counter=_words) == ["two"]

    def test_empty_context_is_returned_unchanged(self):
        assert fit_context_prefix([], "target", 100, counter=_words) == []


class TestArchiveHeadroom:
    """Pins the measured headroom, so a default change cannot silently spend it.

    Figures are from 87 real sessions (70,153 captions, 96.3% Cyrillic): p99.9 is 238
    characters, and the worst context-stacked input ever recorded was ~1,800 characters
    at the maximum context_window of 5.
    """

    # Built to the measured size rather than by repeat count, so the figure that matters
    # is stated once and a change to the filler sentence cannot quietly shrink it.
    WORST_STACKED = ("Да будет мир Твой, Господи, с нами всегда, " * 45)[:1800]

    def test_the_worst_real_input_fits_the_shipped_defaults(self):
        assert len(self.WORST_STACKED) > 1750
        budget = input_token_budget(2048, 160, DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        assert input_fits(self.WORST_STACKED, budget)

    def test_the_same_input_does_not_fit_the_smallest_configurable_window(self):
        # n_ctx bottoms out at 512 in the UI; that is the one setting that turns this
        # guard from inert into load-bearing.
        budget = input_token_budget(512, 160, DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        assert not input_fits(self.WORST_STACKED, budget)

    def test_a_typical_caption_still_fits_the_smallest_window(self):
        # Degrading must cost context, not captions.
        budget = input_token_budget(512, 160, DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        assert input_fits(SRC, budget)


class TestExpansionCeiling:
    """The ratio alone leaves a long source too much room to hide a recitation in.

    A context window prepends prior captions before translating, so the source the
    validator measures against is several captions long. Under the ratio alone a
    30-word combined source bought a 90-word budget — enough for a correct caption
    plus a recited passage behind it. Figures below come from 36,264 aligned
    source/translation pairs from real services.
    """

    def test_short_sources_are_unaffected(self):
        # The ratio must keep governing here: p99 expansion at 1-3 words is 5.00, and
        # a legitimate "Щит веры." -> "The shield of faith." must still pass.
        assert validate_translation("The shield of faith.", "Щит веры.", "en")

    def test_a_six_word_source_keeps_its_full_ratio_budget(self):
        # 6 words -> ratio allows 18, ceiling allows 30; the ratio still binds.
        src = " ".join(["слово"] * 6)
        assert validate_translation(" ".join(["word"] * 18), src, "en")
        assert validate_translation(" ".join(["word"] * 19), src, "en") is None

    def test_a_long_source_is_capped_below_the_ratio(self):
        # 30 words: the ratio would allow 90, the ceiling allows 54. Real translations
        # of a 26+ word source never exceeded 1.73x, and never grew by more than 22.
        src = " ".join(["слово"] * 30)
        assert validate_translation(" ".join(["word"] * 54), src, "en")
        assert validate_translation(" ".join(["word"] * 55), src, "en") is None

    def test_the_context_recitation_shape_is_now_rejected(self):
        # A correct caption for a context-stacked source, with a passage riding along
        # behind it. Under the ratio alone this fit inside the 90-word budget.
        src = " ".join(["слово"] * 30)
        caption = " ".join(["word"] * 34)          # a plausible 1.13x translation
        recitation = " ".join(["verse"] * 50)      # and then the passage
        assert validate_translation(caption + " " + recitation, src, "en") is None
        assert validate_translation(caption, src, "en") == caption

    def test_real_expansion_at_length_still_passes(self):
        # The largest ratio ever recorded for a 26+ word source was 1.73x.
        src = " ".join(["слово"] * 30)
        assert validate_translation(" ".join(["word"] * 51), src, "en")

    def test_the_ceiling_is_configurable(self):
        src = " ".join(["слово"] * 30)
        # slack 10 -> allowed 40 (inclusive); slack 40 -> the ratio's 90 binds again.
        assert validate_translation(" ".join(["word"] * 40), src, "en", max_slack_words=10)
        assert validate_translation(" ".join(["word"] * 41), src, "en", max_slack_words=10) is None
        assert validate_translation(" ".join(["word"] * 41), src, "en", max_slack_words=40)

    def test_the_floor_still_wins_for_a_tiny_source(self):
        # min_word_budget must not be clipped by the ceiling on a 1-word source.
        assert validate_translation(" ".join(["word"] * 8), "Аминь.", "en")

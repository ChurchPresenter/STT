"""NLLB static catalog (stt/nllb_catalog.py)."""

from stt.nllb_catalog import (
    MADLAD_LANG_CODES,
    NLLB_LANG_CODES,
    TRANSLATION_LANGUAGES,
    build_madlad_input,
    get_default_madlad_models,
    get_default_nllb_models,
    get_nllb_model_description,
    is_madlad_model,
    LLM_LANG_CODES,
    languages_for_method,
    madlad_anti_repetition_defaults,
    madlad_target_code,
    resolve_translation_model_id,
    supported_target,
)

MADLAD_DEFAULT = "google/madlad400-3b-mt"


class TestResolveTranslationModelId:
    """The engine and the model id must agree, or madlad silently runs NLLB weights."""

    def test_madlad_engine_with_stale_nllb_id_falls_back_to_default(self):
        cfg = {"translation_method": "madlad",
               "translation_model": "facebook/nllb-200-distilled-600M"}
        assert resolve_translation_model_id(cfg, MADLAD_DEFAULT) == MADLAD_DEFAULT

    def test_madlad_engine_with_madlad_id_is_untouched(self):
        cfg = {"translation_method": "madlad", "translation_model": "google/madlad400-7b-mt"}
        assert resolve_translation_model_id(cfg, MADLAD_DEFAULT) == "google/madlad400-7b-mt"

    def test_madlad_engine_with_no_model_gets_the_default(self):
        assert resolve_translation_model_id({"translation_method": "madlad"},
                                            MADLAD_DEFAULT) == MADLAD_DEFAULT

    def test_nllb_engine_keeps_its_model_even_if_odd(self):
        cfg = {"translation_method": "nllb", "translation_model": "something/custom"}
        assert resolve_translation_model_id(cfg, MADLAD_DEFAULT) == "something/custom"

    def test_engine_defaults_to_nllb(self):
        cfg = {"translation_model": "facebook/nllb-200-distilled-600M"}
        assert resolve_translation_model_id(cfg, MADLAD_DEFAULT) == \
            "facebook/nllb-200-distilled-600M"

    def test_empty_and_none_config_yield_empty_string(self):
        assert resolve_translation_model_id(None, MADLAD_DEFAULT) == ""
        assert resolve_translation_model_id({}, MADLAD_DEFAULT) == ""


class TestLanguageTables:
    def test_auto_and_unknown_fall_back_to_english(self):
        assert NLLB_LANG_CODES["auto"] == "eng_Latn"
        assert NLLB_LANG_CODES.get("zz", "eng_Latn") == "eng_Latn"

    def test_known_codes_map_to_flores(self):
        assert NLLB_LANG_CODES["es"] == "spa_Latn"
        assert NLLB_LANG_CODES["ru"] == "rus_Cyrl"
        assert NLLB_LANG_CODES["zh"] == "zho_Hans"

    def test_every_nllb_code_has_a_name(self):
        # TRANSLATION_LANGUAGES is the union of both engines' codes, so every
        # NLLB code (except 'auto', which the UI adds itself) must have a name,
        # or the picker would show a code where a language label belongs.
        for code in NLLB_LANG_CODES:
            if code != "auto":
                assert code in TRANSLATION_LANGUAGES, code

    def test_flores_codes_are_well_formed(self):
        # NLLB codes are '<lang>_<Script>' (e.g. eng_Latn).
        for code in NLLB_LANG_CODES.values():
            lang, _, script = code.partition("_")
            assert lang and script and script[0].isupper(), code

    def test_full_flores_200_set_is_present(self):
        # We expose the whole FLORES-200 set (202 languages + 'auto').
        assert len(NLLB_LANG_CODES) >= 200
        # A couple of long-tail languages that only exist post-expansion.
        assert "zho-Hant" in NLLB_LANG_CODES
        assert NLLB_LANG_CODES["zho-Hant"] == "zho_Hant"


class TestModelDescription:
    def test_known_model(self):
        assert "600M" in get_nllb_model_description("facebook/nllb-200-distilled-600M")

    def test_unknown_model_generic_fallback(self):
        assert get_nllb_model_description("facebook/whatever") == \
            "NLLB translation model - 200+ languages supported"


class TestDefaultModels:
    def test_shape_and_ordering(self):
        models = get_default_nllb_models()
        assert len(models) == 4
        required = {"model_id", "name", "size", "size_order", "downloads", "likes", "description"}
        for m in models:
            assert required <= set(m)
        # size_order is ascending 1..N so the UI lists smallest first.
        orders = [m["size_order"] for m in models]
        assert orders == sorted(orders)

    def test_returns_a_fresh_list_each_call(self):
        a = get_default_nllb_models()
        a[0]["name"] = "mutated"
        assert get_default_nllb_models()[0]["name"] == "nllb-200-distilled-600M"


class TestMadladCodes:
    def test_auto_and_unknown_fall_back_to_english(self):
        assert MADLAD_LANG_CODES["auto"] == "en"
        assert madlad_target_code("zz") == "en"

    def test_common_codes_are_identity(self):
        assert madlad_target_code("es") == "es"
        assert madlad_target_code("ru") == "ru"
        assert madlad_target_code("zh") == "zh"

    def test_hebrew_uses_google_iw_convention(self):
        assert madlad_target_code("he") == "iw"

    def test_every_madlad_code_has_a_name(self):
        # Every MADLAD code (except 'auto') must have a UI name.
        for code in MADLAD_LANG_CODES:
            if code != "auto":
                assert code in TRANSLATION_LANGUAGES, code

    def test_full_madlad_400_set_is_present(self):
        # MADLAD-400 advertises 400+ languages; we expose the full set.
        assert len(MADLAD_LANG_CODES) >= 400
        # Hawaiian is MADLAD-only (not in FLORES-200).
        assert "haw" in MADLAD_LANG_CODES
        assert "haw" not in NLLB_LANG_CODES

    def test_build_input_prefixes_target_tag(self):
        assert build_madlad_input("hello", "es") == "<2es> hello"
        assert build_madlad_input("hello", "he") == "<2iw> hello"
        # Unknown target still yields a valid (English) tag, never a crash.
        assert build_madlad_input("hi", "zz") == "<2en> hi"


class TestEngineDetectionAndValidation:
    def test_is_madlad_model(self):
        assert is_madlad_model("google/madlad400-3b-mt") is True
        assert is_madlad_model("google/MADLAD400-7b-mt") is True
        assert is_madlad_model("facebook/nllb-200-distilled-600M") is False
        assert is_madlad_model("") is False

    def test_supported_target_per_engine(self):
        assert supported_target("es", "nllb") is True
        assert supported_target("es", "madlad") is True
        assert supported_target("es", "llm") is True
        assert supported_target("zz", "nllb") is False
        assert supported_target("zz", "madlad") is False
        # MADLAD-only language is valid for MADLAD, rejected for NLLB.
        assert supported_target("haw", "madlad") is True
        assert supported_target("haw", "nllb") is False
        # The LLM list is deliberately short: a low-resource language the NMT
        # engines cover is not offered, because a small quantised instruction
        # model translates it badly and a bad caption is worse than none.
        assert supported_target("ace-Latn", "nllb") is True
        assert supported_target("ace-Latn", "llm") is False


class TestLanguagesForMethod:
    def test_returns_named_supported_codes_without_auto(self):
        for method in ("nllb", "madlad", "llm"):
            langs = languages_for_method(method)
            assert "auto" not in langs
            table = {"madlad": MADLAD_LANG_CODES, "llm": LLM_LANG_CODES}.get(
                method, NLLB_LANG_CODES)
            # Exactly the engine's non-auto codes, each with a display name.
            assert set(langs) == {c for c in table if c != "auto"}
            for code, name in langs.items():
                assert name and name == TRANSLATION_LANGUAGES[code]

    def test_madlad_offers_more_than_nllb(self):
        assert len(languages_for_method("madlad")) > len(languages_for_method("nllb"))

    def test_unknown_method_is_treated_as_nllb(self):
        assert languages_for_method("anything-else") == languages_for_method("nllb")

    def test_llm_gets_its_own_short_list(self):
        """Not NLLB's and not MADLAD's — a judgement about output quality.

        An LLM has no target-token table, so its supported set is not a property
        of the model file. Offering the NMT lists would promise hundreds of
        low-resource languages a small quantised model handles badly.
        """
        llm = languages_for_method("llm")
        assert 10 <= len(llm) <= 60, "the point of this list is that it is short"
        assert len(llm) < len(languages_for_method("nllb"))
        assert llm != languages_for_method("nllb")
        assert llm != languages_for_method("madlad")

    def test_llm_covers_the_languages_this_is_deployed_into(self):
        llm = languages_for_method("llm")
        for code in ("en", "es", "de", "fr", "ru", "uk", "pl"):
            assert code in llm, f"{code} must be offered"

    def test_every_llm_code_carries_a_display_name(self):
        # The name is what reaches the model's prompt, so a missing one would
        # send it a bare code it was never trained to interpret.
        for code, name in LLM_LANG_CODES.items():
            assert name and not name.islower(), f"{code} needs a language name"

    def test_english_is_present_in_every_engine(self):
        # switchLanguageSet() falls back to "en" when a target is unsupported.
        for method in ("nllb", "madlad", "llm"):
            assert "en" in languages_for_method(method)


class TestMadladAntiRepetition:
    def test_neutral_defaults_are_nudged(self):
        assert madlad_anti_repetition_defaults(1.0, 0) == (1.1, 4)

    def test_operator_tuned_values_pass_through(self):
        assert madlad_anti_repetition_defaults(1.3, 3) == (1.3, 3)

    def test_only_the_neutral_one_is_nudged(self):
        assert madlad_anti_repetition_defaults(1.2, 0) == (1.2, 4)
        assert madlad_anti_repetition_defaults(1.0, 5) == (1.1, 5)


class TestDefaultMadladModels:
    def test_shape_and_ordering(self):
        models = get_default_madlad_models()
        assert len(models) >= 1
        required = {"model_id", "name", "size", "size_order", "downloads", "likes", "description"}
        for m in models:
            assert required <= set(m)
            assert is_madlad_model(m["model_id"])
        orders = [m["size_order"] for m in models]
        assert orders == sorted(orders)

    def test_returns_a_fresh_list_each_call(self):
        a = get_default_madlad_models()
        a[0]["name"] = "mutated"
        assert get_default_madlad_models()[0]["name"] == "madlad400-3b-mt"

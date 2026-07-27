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
    madlad_target_code,
    supported_target,
)


class TestLanguageTables:
    def test_auto_and_unknown_fall_back_to_english(self):
        assert NLLB_LANG_CODES["auto"] == "eng_Latn"
        assert NLLB_LANG_CODES.get("zz", "eng_Latn") == "eng_Latn"

    def test_known_codes_map_to_flores(self):
        assert NLLB_LANG_CODES["es"] == "spa_Latn"
        assert NLLB_LANG_CODES["ru"] == "rus_Cyrl"
        assert NLLB_LANG_CODES["zh"] == "zho_Hans"

    def test_every_ui_name_has_a_flores_code(self):
        # Every human-readable language (except 'auto', which is code-only)
        # must resolve to an NLLB code, or the UI would offer a dead option.
        for code in TRANSLATION_LANGUAGES:
            assert code in NLLB_LANG_CODES, code

    def test_flores_codes_are_well_formed(self):
        # NLLB codes are '<lang>_<Script>' (e.g. eng_Latn).
        for code in NLLB_LANG_CODES.values():
            lang, _, script = code.partition("_")
            assert lang and script and script[0].isupper(), code


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

    def test_every_ui_name_has_a_madlad_code(self):
        # Both engines must cover every UI-offered language.
        for code in TRANSLATION_LANGUAGES:
            assert code in MADLAD_LANG_CODES, code

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
        assert supported_target("zz", "nllb") is False
        assert supported_target("zz", "madlad") is False


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

"""CTranslate2 translation-backend helpers (stt/ct2_translate.py)."""

import math

from stt.ct2_translate import (
    ct2_model_dir,
    decode_ct2_tokens,
    madlad_source_tokens,
    nllb_ct2_target_prefix,
    nllb_source_tokens,
    resolve_compute_type,
    score_to_confidence,
    strip_target_prefix,
)


class FakeTokenizer:
    """Minimal stand-in: tokens are just the whitespace-split words, prefixed
    by the current src_lang so we can assert it was applied. ids == tokens."""

    def __init__(self):
        self.src_lang = None

    def encode(self, text):
        return list(text.split())

    def convert_ids_to_tokens(self, ids):
        head = [self.src_lang] if self.src_lang else []
        return head + list(ids)

    def convert_tokens_to_ids(self, tokens):
        return list(tokens)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class TestComputeType:
    def test_explicit_passthrough(self):
        assert resolve_compute_type("int8", "cpu") == "int8"
        assert resolve_compute_type("float16", "cuda") == "float16"

    def test_auto_picks_by_device(self):
        assert resolve_compute_type("auto", "cpu") == "int8"
        assert resolve_compute_type("auto", "mps") == "int8"   # CT2 has no Metal -> CPU int8
        assert resolve_compute_type("auto", "cuda") == "int8_float16"


class TestModelDir:
    def test_cache_dir_keyed_by_type(self):
        assert ct2_model_dir("/m/google--madlad400-3b-mt", "int8") == \
            "/m/google--madlad400-3b-mt-ct2-int8"
        assert ct2_model_dir("/m/google--madlad400-3b-mt", "int8_float16").endswith("-ct2-int8_float16")

    def test_trailing_sep_stripped(self):
        assert ct2_model_dir("/m/x/", "int8") == "/m/x-ct2-int8"


class TestNllbTokens:
    def test_target_prefix_and_strip(self):
        assert nllb_ct2_target_prefix("spa_Latn") == [["spa_Latn"]]
        assert strip_target_prefix(["spa_Latn", "hola", "mundo"], "spa_Latn") == ["hola", "mundo"]
        # No leading prefix -> unchanged
        assert strip_target_prefix(["hola"], "spa_Latn") == ["hola"]
        assert strip_target_prefix([], "spa_Latn") == []

    def test_source_tokens_set_src_lang(self):
        tok = FakeTokenizer()
        out = nllb_source_tokens(tok, "hello world", "eng_Latn")
        assert tok.src_lang == "eng_Latn"
        assert out == ["eng_Latn", "hello", "world"]


class TestMadladTokens:
    def test_source_tokens_prefix_target_tag(self):
        tok = FakeTokenizer()
        out = madlad_source_tokens(tok, "hello world", "es")
        # build_madlad_input prepends "<2es>"; no src_lang set for MADLAD
        assert out == ["<2es>", "hello", "world"]
        assert tok.src_lang is None

    def test_hebrew_uses_iw(self):
        tok = FakeTokenizer()
        assert madlad_source_tokens(tok, "hi", "he")[0] == "<2iw>"


class TestDecodeAndConfidence:
    def test_decode_roundtrip(self):
        assert decode_ct2_tokens(FakeTokenizer(), ["hola", "mundo"]) == "hola mundo"

    def test_score_to_confidence(self):
        assert score_to_confidence(None) is None
        assert score_to_confidence(0.0) == 1.0
        assert score_to_confidence(-0.5) == math.exp(-0.5)

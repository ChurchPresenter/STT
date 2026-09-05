"""CTranslate2 translation-backend helpers (stt/ct2_translate.py)."""

import pathlib
import math
import os

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

    def test_auto_drops_fp16_on_a_pre_volta_card(self):
        # Pascal and older have no fast fp16 path. CT2 accepts int8_float16 and
        # reports back something else, so it looks like it worked while naming a
        # type the card does not list — and it builds a second cache directory of
        # the same weights beside the int8 one.
        assert resolve_compute_type("auto", "cuda", 6.1) == "int8"
        assert resolve_compute_type("auto", "cuda", 5.2) == "int8"

    def test_volta_and_newer_keep_fp16(self):
        assert resolve_compute_type("auto", "cuda", 7.0) == "int8_float16"
        assert resolve_compute_type("auto", "cuda", 8.9) == "int8_float16"

    def test_an_unknown_capability_keeps_the_long_standing_answer(self):
        # The caller cannot always determine it, and int8_float16 is right on every
        # card new enough to ship with CUDA 12.
        assert resolve_compute_type("auto", "cuda", None) == "int8_float16"

    def test_an_explicit_choice_still_wins_on_any_card(self):
        # The operator pinning a type is not overridden by a capability probe.
        assert resolve_compute_type("int8_float16", "cuda", 6.1) == "int8_float16"


class TestModelDir:
    def test_cache_dir_keyed_by_type(self):
        assert ct2_model_dir("/m/google--madlad400-3b-mt", "int8") == \
            "/m/google--madlad400-3b-mt-ct2-int8"
        assert ct2_model_dir("/m/google--madlad400-3b-mt", "int8_float16").endswith("-ct2-int8_float16")

    def test_trailing_sep_stripped(self):
        # Either separator: Windows accepts "/" as well as os.sep, and a models
        # directory entered with a trailing slash otherwise produced
        # "C:/models/-ct2-int8".
        assert ct2_model_dir("/m/x/", "int8") == "/m/x-ct2-int8"
        assert ct2_model_dir("/m/x" + os.sep, "int8") == "/m/x-ct2-int8"
        assert ct2_model_dir("/m/x", "int8") == "/m/x-ct2-int8"


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


class TestManifestCheckSkipsTheConversionPath:
    """A converted model whose HF weights were reclaimed must still load.

    Converting leaves the weights in a "-ct2-" sibling and the original directory
    is often stripped afterwards to save several gigabytes — a workflow the Model
    Manager endorses, counting a conversion as downloaded. But the manifest still
    lists pytorch_model.bin, so a check that runs before the CT2 branch refuses a
    setup that works, for a file the CT2 path never reads.
    """

    def test_the_manifest_does_report_the_reclaimed_weights(self, tmp_path):
        """Load-bearing: without the guard, this list is what refuses the load."""
        from stt import model_files

        model_dir = tmp_path / "google--madlad400-3b-mt"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"x" * 1400)
        (model_dir / "tokenizer.json").write_bytes(b"x" * 1400)
        model_files.write_manifest(str(model_dir), "google/madlad400-3b-mt", {
            "pytorch_model.bin": model_files.FileExpectation(size=11_800_000_000),
            "config.json": model_files.FileExpectation(size=1400),
            "tokenizer.json": model_files.FileExpectation(size=1400),
        })

        assert "pytorch_model.bin" in model_files.manifest_mismatches(str(model_dir))

    def test_the_check_is_guarded_by_use_ct2(self):
        """The monolith cannot be imported, so assert the shape that matters."""
        import re

        source = pathlib.Path("speech_to_text.py").read_text(encoding="utf-8")
        guarded = re.search(
            r"if not use_ct2:\s*\n\s*_incomplete = _model_files\.manifest_mismatches",
            source)
        assert guarded, (
            "manifest_mismatches must run only off the CT2 path — a converted "
            "model supplies only its tokenizer from this directory"
        )

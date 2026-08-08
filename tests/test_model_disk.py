"""On-disk model-weight detection (stt/model_disk.py)."""

import os

import pytest

from stt.model_disk import (
    dir_has_weights,
    dir_is_writable,
    has_weight_file,
    is_weight_file,
    ct2_variant_names,
    model_presence,
)


class TestIsWeightFile:
    @pytest.mark.parametrize("name", [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",                                  # CTranslate2 / faster-whisper
        "pytorch_model-00001-of-00003.bin",           # sharded pytorch
        "model-00001-of-00002.safetensors",           # sharded safetensors
        "base.pt",                                     # whisper checkpoint
        "large-v3.pt",
        "gemma-3-4b-it-Q4_K_M.gguf",                   # llama.cpp / LLM translation
        "model-q2k.gguf",
    ])
    def test_recognizes_weight_files(self, name):
        assert is_weight_file(name)

    @pytest.mark.parametrize("name", [
        "config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "vocab.json",
        "training_args.bin",       # a .bin that is NOT a weight file
        "optimizer.bin",
        "README.md",
        "model.safetensors.index.json",
        "",
    ])
    def test_rejects_non_weight_files(self, name):
        assert not is_weight_file(name)


class TestGgufIsADownloadedModel:
    """A GGUF directory holds nothing but the .gguf file.

    Without recognising that extension a downloaded LLM was invisible to the
    Model Manager — not listed, and so not deletable — despite typically being
    the largest single file on disk.
    """

    def test_gguf_only_directory_counts_as_downloaded(self, tmp_path):
        d = tmp_path / "ggml-org--gemma-3-4b-it-GGUF"
        d.mkdir()
        (d / "gemma-3-4b-it-Q4_K_M.gguf").write_bytes(b"x")
        assert dir_has_weights(str(d))

    def test_still_false_once_the_gguf_is_deleted(self, tmp_path):
        d = tmp_path / "ggml-org--gemma-3-4b-it-GGUF"
        d.mkdir()
        (d / "README.md").write_text("leftover")
        assert not dir_has_weights(str(d))


class TestHasWeightFile:
    def test_true_when_any_weight_present(self):
        assert has_weight_file(["config.json", "tokenizer.json", "model.safetensors"])

    def test_false_for_config_only_leftover(self):
        # The partial-delete case: weights gone, metadata remains.
        assert not has_weight_file(["config.json", "tokenizer.json", "training_args.bin"])

    def test_false_for_empty(self):
        assert not has_weight_file([])


class TestDirHasWeights:
    def test_true_for_downloaded_model(self, tmp_path):
        d = tmp_path / "facebook--nllb-200-distilled-600M"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00")
        assert dir_has_weights(str(d))

    def test_false_after_weights_deleted(self, tmp_path):
        # Reproduces the reported bug: the directory survives a delete of its
        # weights but must no longer count as downloaded.
        d = tmp_path / "facebook--nllb-200-distilled-600M"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "tokenizer.json").write_text("{}")
        assert not dir_has_weights(str(d))

    def test_false_for_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert not dir_has_weights(str(d))

    def test_false_for_missing_path(self, tmp_path):
        assert not dir_has_weights(str(tmp_path / "does-not-exist"))

    def test_false_when_path_is_a_file(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"\x00")
        # A file path is not a model directory; os.listdir raises -> False.
        assert not dir_has_weights(str(f))


class TestDirIsWritable:
    def test_true_for_writable_dir(self, tmp_path):
        assert dir_is_writable(str(tmp_path))

    def test_false_for_missing_dir(self, tmp_path):
        assert not dir_is_writable(str(tmp_path / "nope"))

    @pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else True,
                        reason="root bypasses permission bits; POSIX-only test")
    def test_false_for_readonly_dir(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)  # r-x: no write
        try:
            assert not dir_is_writable(str(ro))
        finally:
            os.chmod(ro, 0o700)  # restore so tmp cleanup can remove it



class TestCt2VariantNames:
    def test_finds_the_conversion_beside_the_model(self):
        entries = ["google--madlad400-3b-mt", "google--madlad400-3b-mt-ct2-int8_float16"]
        assert ct2_variant_names(entries, "google--madlad400-3b-mt") == ["int8_float16"]

    def test_multiple_compute_types_are_all_reported(self):
        entries = ["m", "m-ct2-int8", "m-ct2-int8_float16", "m-ct2-float16"]
        assert ct2_variant_names(entries, "m") == ["float16", "int8", "int8_float16"]

    def test_no_conversion_is_empty(self):
        assert ct2_variant_names(["m", "other-ct2-int8"], "m") == []

    def test_bare_marker_without_a_compute_type_is_not_a_conversion(self):
        assert ct2_variant_names(["m-ct2-"], "m") == []

    def test_a_longer_model_name_is_not_mistaken_for_a_conversion(self):
        # "madlad400-10b-mt" must never be reported as a conversion of
        # "madlad400-3b-mt", nor the reverse.
        entries = ["google--madlad400-10b-mt", "google--madlad400-10b-mt-ct2-int8"]
        assert ct2_variant_names(entries, "google--madlad400-3b-mt") == []

    def test_empty_listing(self):
        assert ct2_variant_names([], "m") == []


class TestModelPresence:
    def test_hf_weights_only(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "model.safetensors").write_text("w")
        p = model_presence(str(tmp_path), "m")
        assert p.has_weights is True
        assert p.ct2_variants == []
        assert p.downloaded is True

    def test_conversion_only_counts_as_downloaded(self, tmp_path):
        # The shape on .62: the HF directory keeps its tokenizer/config but the
        # weights were reclaimed after conversion. The model still runs.
        hf = tmp_path / "google--madlad400-3b-mt"
        hf.mkdir()
        (hf / "config.json").write_text("{}")
        (hf / "spiece.model").write_text("tok")
        ct2 = tmp_path / "google--madlad400-3b-mt-ct2-int8_float16"
        ct2.mkdir()
        (ct2 / "model.bin").write_text("w")

        p = model_presence(str(tmp_path), "google--madlad400-3b-mt")
        assert p.has_weights is False
        assert p.ct2_variants == ["int8_float16"]
        assert p.downloaded is True

    def test_both_present(self, tmp_path):
        hf = tmp_path / "m"
        hf.mkdir()
        (hf / "model.safetensors").write_text("w")
        (tmp_path / "m-ct2-int8").mkdir()
        p = model_presence(str(tmp_path), "m")
        assert p.has_weights is True
        assert p.ct2_variants == ["int8"]

    def test_neither_present_is_not_downloaded(self, tmp_path):
        # The regression that matters most: never tell the operator a missing
        # model is present.
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text("{}")
        p = model_presence(str(tmp_path), "m")
        assert p.downloaded is False

    def test_missing_models_dir_reports_nothing(self, tmp_path):
        p = model_presence(str(tmp_path / "nope"), "m")
        assert p.downloaded is False
        assert p.ct2_variants == []

"""On-disk model-weight detection (stt/model_disk.py)."""

import os

import pytest

from stt.model_disk import (
    dir_has_weights,
    dir_is_writable,
    has_weight_file,
    is_weight_file,
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


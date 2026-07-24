"""On-disk model-weight detection (stt/model_disk.py)."""

import os

import pytest

from stt.model_disk import (
    dir_has_weights,
    dir_is_writable,
    has_weight_file,
    is_weight_file,
    migrate_model_dirs,
    resolve_writable_models_dir,
    stranded_model_dirs,
)


def _make_model_dir(base, name, weight="model.safetensors"):
    d = base / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / weight).write_bytes(b"\x00\x01")
    return d


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


class TestResolveWritableModelsDir:
    def test_uses_preferred_when_writable(self, tmp_path):
        preferred = tmp_path / "repo-models"
        fallback = tmp_path / "home-models"
        chosen, used_fallback = resolve_writable_models_dir(str(preferred), str(fallback))
        assert chosen == str(preferred)
        assert used_fallback is False
        assert preferred.is_dir()

    def test_creates_preferred_if_absent(self, tmp_path):
        preferred = tmp_path / "does-not-exist-yet"
        fallback = tmp_path / "home-models"
        chosen, used_fallback = resolve_writable_models_dir(str(preferred), str(fallback))
        assert chosen == str(preferred)
        assert used_fallback is False

    @pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else True,
                        reason="root bypasses permission bits; POSIX-only test")
    def test_falls_back_when_preferred_unwritable(self, tmp_path):
        # Reproduces the reported box: preferred models/ exists but is owned by
        # another user (simulated via a read-only dir), so a normal-user server
        # must transparently land downloads in the writable home fallback.
        preferred = tmp_path / "root-owned-models"
        preferred.mkdir()
        os.chmod(preferred, 0o500)
        fallback = tmp_path / "home-models"
        try:
            chosen, used_fallback = resolve_writable_models_dir(str(preferred), str(fallback))
            assert chosen == str(fallback)
            assert used_fallback is True
            assert fallback.is_dir()
        finally:
            os.chmod(preferred, 0o700)


class TestStrandedModelDirs:
    def test_lists_only_weighted_dirs_missing_from_fallback(self, tmp_path):
        preferred = tmp_path / "repo-models"
        fallback = tmp_path / "home-models"
        fallback.mkdir()
        _make_model_dir(preferred, "facebook--nllb-200-distilled-600M")
        _make_model_dir(preferred, "faster-whisper-small")
        # A config-only leftover (no weights) is not a real model -> not stranded.
        (preferred / "half-deleted").mkdir()
        (preferred / "half-deleted" / "config.json").write_text("{}")
        # Already mirrored in the fallback -> not stranded.
        _make_model_dir(preferred, "whisper-base", weight="base.pt")
        _make_model_dir(fallback, "whisper-base", weight="base.pt")

        assert stranded_model_dirs(str(preferred), str(fallback)) == [
            "facebook--nllb-200-distilled-600M",
            "faster-whisper-small",
        ]

    def test_empty_when_preferred_missing(self, tmp_path):
        assert stranded_model_dirs(str(tmp_path / "nope"), str(tmp_path / "fb")) == []


class TestMigrateModelDirs:
    def test_copies_and_leaves_source(self, tmp_path):
        preferred = tmp_path / "repo-models"
        fallback = tmp_path / "home-models"
        fallback.mkdir()
        _make_model_dir(preferred, "facebook--nllb-200-distilled-600M")

        done = migrate_model_dirs(
            ["facebook--nllb-200-distilled-600M"], str(preferred), str(fallback))

        assert done == ["facebook--nllb-200-distilled-600M"]
        # Copied into the writable location...
        assert dir_has_weights(str(fallback / "facebook--nllb-200-distilled-600M"))
        # ...and the original is left in place (never moved/deleted).
        assert dir_has_weights(str(preferred / "facebook--nllb-200-distilled-600M"))

    def test_skips_when_target_already_has_weights(self, tmp_path):
        preferred = tmp_path / "repo-models"
        fallback = tmp_path / "home-models"
        _make_model_dir(preferred, "m", weight="model.bin")
        _make_model_dir(fallback, "m", weight="model.bin")
        # Sentinel proves copytree did not run over the existing target.
        (fallback / "m" / "sentinel.txt").write_text("keep")

        done = migrate_model_dirs(["m"], str(preferred), str(fallback))

        assert done == ["m"]
        assert (fallback / "m" / "sentinel.txt").exists()

    def test_reports_failures_without_raising(self, tmp_path):
        preferred = tmp_path / "repo-models"
        fallback = tmp_path / "home-models"
        preferred.mkdir()
        fallback.mkdir()
        logged = []
        # 'ghost' has no source directory -> copytree raises -> logged, omitted.
        done = migrate_model_dirs(["ghost"], str(preferred), str(fallback), log=logged.append)
        assert done == []
        assert logged and "ghost" in logged[0]

"""Completeness checks for the model families that had none.

faster-whisper was the only family where "downloaded" meant "loadable". The rest
reported presence, so an interrupted transfer produced a model the UI called
ready and the loader met much later — a crash from a C++ reader, a confusing
failure at speech time, or a silent fall back to a worse detector.

These functions live in the monolith, so they are extracted from its AST (see
conftest) and run against a stub namespace.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import extract_definitions


# --- PANNs checkpoint ------------------------------------------------------


@pytest.fixture
def panns(tmp_path):
    ns = extract_definitions(
        "speech_to_text.py",
        ["panns_checkpoint_ok"],
        extra_globals={"panns_checkpoint_path": lambda cfg=None: str(tmp_path / "cnn14.pth"),
                       "_PANNS_CKPT_MIN_BYTES": 250_000_000},
    )
    return ns["panns_checkpoint_ok"], tmp_path / "cnn14.pth"


def test_a_missing_checkpoint_is_not_ok(panns):
    is_ok, _ = panns
    assert is_ok() is False


def test_a_truncated_checkpoint_is_not_ok(panns):
    """A dropped transfer used to leave this, and every check called it present."""
    is_ok, path = panns
    path.write_bytes(b"x" * 5_000_000)  # 5 MB of a ~312 MB model
    assert is_ok() is False


def test_a_zero_byte_checkpoint_is_not_ok(panns):
    is_ok, path = panns
    path.write_bytes(b"")
    assert is_ok() is False


def test_a_full_size_checkpoint_is_ok(panns):
    is_ok, path = panns
    path.write_bytes(b"\0" * 260_000_000)
    assert is_ok() is True


def test_an_unreadable_path_is_not_ok_rather_than_raising(panns):
    is_ok, path = panns
    path.mkdir()  # a directory where a file belongs
    assert is_ok() is False


# --- Piper voices ----------------------------------------------------------


@pytest.fixture
def piper(tmp_path):
    ns = extract_definitions(
        "speech_to_text.py",
        ["_is_piper_model_downloaded"],
        extra_globals={"_get_piper_model_dir": lambda mid: str(tmp_path / mid)},
    )
    return ns["_is_piper_model_downloaded"]


def _voice(tmp_path, name, *, onnx=True, config=True, onnx_bytes=b"weights",
           config_bytes=b"{}"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if onnx:
        (d / f"{name}.onnx").write_bytes(onnx_bytes)
    if config:
        (d / f"{name}.onnx.json").write_bytes(config_bytes)
    return d


def test_a_complete_voice_is_downloaded(piper, tmp_path):
    _voice(tmp_path, "en_US-lessac-medium")
    assert piper("en_US-lessac-medium") is True


def test_a_voice_missing_its_config_is_not_downloaded(piper, tmp_path):
    """PiperVoice.load raises without the .onnx.json; only the .onnx was required."""
    _voice(tmp_path, "en_US-lessac-medium", config=False)
    assert piper("en_US-lessac-medium") is False


def test_a_zero_byte_config_does_not_count_as_present(piper, tmp_path):
    _voice(tmp_path, "en_US-lessac-medium", config_bytes=b"")
    assert piper("en_US-lessac-medium") is False


def test_a_zero_byte_onnx_does_not_count_as_present(piper, tmp_path):
    _voice(tmp_path, "en_US-lessac-medium", onnx_bytes=b"")
    assert piper("en_US-lessac-medium") is False


def test_a_missing_directory_is_not_downloaded(piper):
    assert piper("no-such-voice") is False


# --- Supertonic ------------------------------------------------------------


@pytest.fixture
def supertonic(tmp_path):
    ns = extract_definitions(
        "speech_to_text.py",
        ["_is_supertonic_downloaded", "_supertonic_onnx_files"],
        extra_globals={"_get_supertonic_dir": lambda: str(tmp_path / "supertonic")},
    )
    ns["_is_supertonic_downloaded"].__globals__.update(
        _supertonic_onnx_files=ns["_supertonic_onnx_files"])
    return ns["_is_supertonic_downloaded"], tmp_path / "supertonic"


def test_an_absent_supertonic_is_not_downloaded(supertonic):
    is_downloaded, _ = supertonic
    assert is_downloaded() is False


def test_a_zero_byte_module_is_refused(supertonic):
    """Its download runs inside the third-party package, so nothing resumes it."""
    is_downloaded, root = supertonic
    onnx = root / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "encoder.onnx").write_bytes(b"real")
    (onnx / "decoder.onnx").write_bytes(b"")
    assert is_downloaded() is False


def test_a_directory_with_no_modules_at_all_is_refused(supertonic):
    is_downloaded, root = supertonic
    (root / "onnx").mkdir(parents=True)
    assert is_downloaded() is False


def test_modules_are_found_at_any_depth(tmp_path):
    """Whether the set is *complete* is the package's call, not ours.

    All this side owns is finding the files to size-check, wherever the package
    chooses to lay them out — so that is what is asserted here.
    """
    ns = extract_definitions("speech_to_text.py", ["_supertonic_onnx_files"])
    nested = tmp_path / "onnx" / "v3"
    nested.mkdir(parents=True)
    (nested / "encoder.onnx").write_bytes(b"real")
    (tmp_path / "top.onnx").write_bytes(b"real")

    found = ns["_supertonic_onnx_files"](str(tmp_path))
    assert sorted(os.path.basename(f) for f in found) == ["encoder.onnx", "top.onnx"]


# --- the CT2 conversion is built under a temporary name --------------------


def test_a_conversion_is_staged_and_moved_into_place():
    """An interrupted convert used to leave half a model that then got loaded."""
    import pathlib

    source = pathlib.Path("speech_to_text.py").read_text(encoding="utf-8")
    assert 'staging = f"{ct2_dir}.converting"' in source
    assert "os.replace(staging, ct2_dir)" in source
    assert '_model_files.dir_status(ct2_dir, "ct2").complete' in source, (
        "the guard must judge the directory, not merely that the path exists"
    )

"""download_hf_repo_files: "already exists" now has to mean "and is whole".

The loop lives in the monolith, so it is extracted and run against a stub
namespace with the huggingface_hub calls faked — see conftest.extract_definitions.
What is under test is the decision the loop makes per file, not HTTP: whether it
skips, repairs, or fetches, and whether it leaves behind a manifest another
component can check a year later.
"""

import os
import threading

import pytest

from conftest import extract_definitions
from stt import model_files as _model_files


class FakeSibling:
    def __init__(self, rfilename, size, sha256=None):
        self.rfilename = rfilename
        self.size = size
        self.lfs = type("Lfs", (), {"sha256": sha256})() if sha256 else None


class FakeInfo:
    def __init__(self, siblings):
        self.siblings = siblings


#: The real shape of a Systran faster-whisper repo, sizes shrunk.
REPO_FILES = {
    "model.bin": (500, "a" * 64),
    "config.json": (40, None),
    "tokenizer.json": (100, None),
    "vocabulary.txt": (60, None),
}


def build(fetched, cancelled=None):
    """Namespace for the loop; `fetched` collects (filename, expected_size).

    The fake transfer writes exactly the number of bytes it was told to expect,
    so a file that was asked for lands verified — which is what lets the skip /
    repair decisions be read off `fetched` alone.
    """

    def fake_download(url, dest_path, cancel_check=None, log=None,
                      expected_size=None, expected_sha256=None, **kw):
        fetched.append((os.path.basename(dest_path), expected_size))
        with open(dest_path, "wb") as handle:
            handle.write(b"x" * (expected_size or 8))
        return "ok"

    return extract_definitions(
        "speech_to_text.py", ["hf_file_expectations", "download_hf_repo_files"],
        {
            "_model_files": _model_files,
            "download_url_to_file": fake_download,
            "call_with_retry": lambda fn, **kw: fn(),
            "select_repo_files": lambda f, include: [n for n in f if n in include],
            "cancelled_downloads": cancelled if cancelled is not None else set(),
            "active_downloads": {},
            "active_downloads_lock": threading.Lock(),
            "save_download_progress": lambda: None,
        })


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("huggingface_hub")
    mod.list_repo_files = lambda repo_id: list(REPO_FILES)
    mod.hf_hub_url = lambda repo_id, filename: f"https://hub.test/{repo_id}/{filename}"

    class HfApi:
        def model_info(self, repo_id, files_metadata=False):
            if getattr(mod, "_broken", False):
                raise RuntimeError("hub unreachable")
            return FakeInfo([FakeSibling(n, *REPO_FILES[n]) for n in REPO_FILES])

    mod.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return mod


def run(tmp_path, fetched, cancelled=None):
    ns = build(fetched, cancelled=cancelled)
    return ns["download_hf_repo_files"](
        "Systran/faster-whisper-small", str(tmp_path), "key", log=lambda _m: None)


class TestFreshDownload:
    def test_every_file_is_fetched_with_its_expected_size(self, tmp_path):
        fetched = []
        assert run(tmp_path, fetched) == "ok"
        assert dict(fetched) == {"model.bin": 500, "config.json": 40,
                                 "tokenizer.json": 100, "vocabulary.txt": 60}

    def test_a_manifest_is_written(self, tmp_path):
        run(tmp_path, [])
        manifest = _model_files.read_manifest(str(tmp_path))
        assert manifest["model.bin"].size == 500
        assert manifest["model.bin"].sha256 == "a" * 64

    def test_the_result_reads_as_a_complete_model(self, tmp_path):
        run(tmp_path, [])
        assert _model_files.faster_whisper_status(str(tmp_path)).state == "complete"


class TestRepair:
    def test_a_truncated_file_is_re_fetched(self, tmp_path):
        """The bug: `getsize(dest) > 0` called this "already exists", so a
        re-download could never repair it and the folder had to be deleted by
        hand."""
        (tmp_path / "model.bin").write_bytes(b"x" * 12)
        (tmp_path / "config.json").write_bytes(b"x" * 40)
        (tmp_path / "tokenizer.json").write_bytes(b"x" * 100)
        (tmp_path / "vocabulary.txt").write_bytes(b"x" * 60)
        fetched = []
        run(tmp_path, fetched)
        assert [name for name, _ in fetched] == ["model.bin"]

    def test_whole_files_are_still_skipped(self, tmp_path):
        for name, (size, _sha) in REPO_FILES.items():
            (tmp_path / name).write_bytes(b"x" * size)
        fetched = []
        run(tmp_path, fetched)
        assert fetched == []

    def test_a_repaired_directory_ends_up_complete(self, tmp_path):
        (tmp_path / "model.bin").write_bytes(b"x" * 12)
        run(tmp_path, [])
        assert _model_files.faster_whisper_status(str(tmp_path)).state == "complete"


class TestDegradedHub:
    def test_a_hub_that_will_not_answer_does_not_block_the_download(self, tmp_path, fake_hub):
        """Sizes are an improvement, not a prerequisite: without them this has to
        fall back to the old presence rule rather than refusing to download."""
        fake_hub._broken = True
        fetched = []
        assert run(tmp_path, fetched) == "ok"
        assert sorted(name for name, _ in fetched) == [
            "config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
        assert all(size is None for _n, size in fetched)


class TestCancellation:
    def test_a_cancelled_download_writes_no_manifest(self, tmp_path):
        """The manifest is the claim "this finished". A cancel did not."""
        assert run(tmp_path, [], cancelled={"key"}) == "cancelled"
        assert _model_files.read_manifest(str(tmp_path)) == {}

"""model_files: what "downloaded" has to mean before a loader sees the bytes.

The sizes here are toy; the shapes are the real ones. A faster-whisper directory
holds model.bin / config.json / tokenizer.json and a vocabulary file whose
extension varies by repo — vocabulary.txt on the small models, vocabulary.json on
the large ones — which is why that one is required as an either/or.
"""

import json
import os

import pytest

from stt import model_files as mf


def write(path, data=b"x" * 32):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


@pytest.fixture
def model_dir(tmp_path):
    """A complete faster-whisper directory with a manifest describing it."""
    d = tmp_path / "faster-whisper-small"
    write(str(d / "model.bin"), b"w" * 500)
    write(str(d / "config.json"), b"c" * 50)
    write(str(d / "tokenizer.json"), b"t" * 100)
    write(str(d / "vocabulary.txt"), b"v" * 10)
    mf.write_manifest(str(d), "Systran/faster-whisper-small", {
        "model.bin": mf.FileExpectation(500, mf.file_sha256(str(d / "model.bin"))),
        "config.json": mf.FileExpectation(50),
        "tokenizer.json": mf.FileExpectation(100),
        "vocabulary.txt": mf.FileExpectation(10),
    })
    return str(d)


class TestPartPath:
    def test_part_is_a_sibling_of_the_destination(self):
        """os.replace is only atomic within one filesystem, so it must not
        wander off into a temp directory."""
        dest = os.path.join("models", "faster-whisper-small", "model.bin")
        assert os.path.dirname(mf.part_path(dest)) == os.path.dirname(dest)
        assert mf.part_path(dest).endswith(".part")


class TestVerifyFile:
    def test_missing_file_fails(self, tmp_path):
        assert not mf.verify_file(str(tmp_path / "nope"), 10)

    def test_empty_file_always_fails(self, tmp_path):
        """Zero bytes is wreckage even when we know nothing about the size."""
        p = write(str(tmp_path / "empty"), b"")
        assert not mf.verify_file(p)
        assert not mf.verify_file(p, None)

    def test_unknown_size_accepts_any_non_empty_file(self, tmp_path):
        assert mf.verify_file(write(str(tmp_path / "f")))

    def test_truncated_file_fails_on_size(self, tmp_path):
        p = write(str(tmp_path / "model.bin"), b"x" * 40)
        assert not mf.verify_file(p, 500)
        assert mf.verify_file(p, 40)

    def test_sha_is_checked_when_given(self, tmp_path):
        p = write(str(tmp_path / "model.bin"), b"x" * 40)
        good = mf.file_sha256(p)
        assert mf.verify_file(p, 40, good)
        assert mf.verify_file(p, 40, good.upper())  # hub casing must not matter
        assert not mf.verify_file(p, 40, "0" * 64)

    def test_a_directory_is_not_a_file(self, tmp_path):
        """getsize() succeeds on a directory; the size check must not pass it."""
        d = tmp_path / "d"
        d.mkdir()
        assert not mf.verify_file(str(d), 500)


class TestFilesNeedingDownload:
    def test_a_truncated_file_is_re_downloaded(self, tmp_path):
        """The bug this replaces: `getsize(dest) > 0` called a truncated file
        'already exists', so a re-download could never repair one."""
        write(str(tmp_path / "model.bin"), b"x" * 40)
        need = mf.files_needing_download(str(tmp_path), {"model.bin": mf.FileExpectation(500)})
        assert need == ["model.bin"]

    def test_a_complete_file_is_skipped(self, tmp_path):
        write(str(tmp_path / "model.bin"), b"x" * 40)
        assert mf.files_needing_download(str(tmp_path), {"model.bin": mf.FileExpectation(40)}) == []

    def test_absent_files_are_listed(self, tmp_path):
        need = mf.files_needing_download(str(tmp_path), {
            "a.json": mf.FileExpectation(5), "b.json": mf.FileExpectation(5),
        })
        assert sorted(need) == ["a.json", "b.json"]


class TestManifest:
    def test_round_trip(self, model_dir):
        expected = mf.read_manifest(model_dir)
        assert expected["model.bin"].size == 500
        assert expected["model.bin"].sha256
        assert expected["config.json"].sha256 is None

    def test_manifest_is_dot_prefixed(self, model_dir):
        """The model-manager listings skip dotfiles, so the sidecar must never
        show up as a model of its own."""
        assert os.path.basename(mf.manifest_path(model_dir)).startswith(".")

    def test_no_manifest_reads_empty(self, tmp_path):
        """Every model downloaded before the sidecar existed has none, and must
        keep working — so absent can never mean 'not downloaded'."""
        assert mf.read_manifest(str(tmp_path)) == {}

    def test_corrupt_manifest_reads_empty(self, tmp_path):
        write(str(tmp_path / mf.MANIFEST_NAME), b"{not json")
        assert mf.read_manifest(str(tmp_path)) == {}

    def test_manifest_with_junk_entries_is_survivable(self, tmp_path):
        with open(str(tmp_path / mf.MANIFEST_NAME), "w") as handle:
            json.dump({"files": {"a": "notadict", "b": {"size": "big"}, "c": {"size": 7}}}, handle)
        got = mf.read_manifest(str(tmp_path))
        assert "a" not in got
        assert got["b"].size is None
        assert got["c"].size == 7

    def test_write_manifest_does_not_raise_on_an_unwritable_dir(self, tmp_path):
        """A model that downloaded fine must not be reported as failed because
        its sidecar could not be written."""
        mf.write_manifest(str(tmp_path / "does-not-exist"), "repo", {})

    def test_mismatches_finds_a_truncated_file(self, model_dir):
        write(os.path.join(model_dir, "model.bin"), b"w" * 12)
        assert mf.manifest_mismatches(model_dir) == ["model.bin"]

    def test_no_manifest_reports_no_mismatches(self, tmp_path):
        write(str(tmp_path / "model.bin"))
        assert mf.manifest_mismatches(str(tmp_path)) == []


class TestFasterWhisperStatus:
    def test_complete(self, model_dir):
        status = mf.faster_whisper_status(model_dir)
        assert status.state == "complete"
        assert status.complete
        assert status.missing == []

    def test_absent_directory(self, tmp_path):
        assert mf.faster_whisper_status(str(tmp_path / "nothing")).state == "absent"

    def test_missing_tokenizer_is_incomplete(self, model_dir):
        """The #8 hang: with tokenizer.json gone, WhisperModel goes to the
        network with no timeout we can set, so this has to be refused up front."""
        os.remove(os.path.join(model_dir, "tokenizer.json"))
        status = mf.faster_whisper_status(model_dir)
        assert status.state == "incomplete"
        assert status.missing == ["tokenizer.json"]

    def test_truncated_weights_are_incomplete_even_though_the_file_is_there(self, model_dir):
        """Every filename check passes here; only the manifest catches it."""
        write(os.path.join(model_dir, "model.bin"), b"w" * 12)
        status = mf.faster_whisper_status(model_dir)
        assert status.state == "incomplete"
        assert status.missing == ["model.bin"]

    def test_a_missing_vocabulary_is_incomplete(self, model_dir):
        """Measured, not assumed: with no vocabulary file at all, ctranslate2
        raises "Cannot load the vocabulary from the model directory"."""
        os.remove(os.path.join(model_dir, "vocabulary.txt"))
        status = mf.faster_whisper_status(model_dir)
        assert status.state == "incomplete"
        assert status.missing == ["vocabulary.txt or vocabulary.json"]

    def test_either_vocabulary_filename_satisfies_it(self, tmp_path):
        """Systran ships vocabulary.txt on the small models and vocabulary.json
        on the large ones, so a fixed name would call half the catalogue broken."""
        for vocab in ("vocabulary.txt", "vocabulary.json"):
            d = tmp_path / f"fw-{vocab}"
            for name in ("model.bin", "config.json", "tokenizer.json", vocab):
                write(str(d / name))
            assert mf.faster_whisper_status(str(d)).state == "complete", vocab

    def test_a_pre_manifest_install_still_reads_complete(self, tmp_path):
        """Directories downloaded before the sidecar existed have no manifest
        and must keep loading."""
        d = tmp_path / "faster-whisper-tiny"
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
            write(str(d / name))
        assert mf.faster_whisper_status(str(d)).state == "complete"

    def test_a_weightless_leftover_is_incomplete_not_absent(self, tmp_path):
        """A partially deleted folder still occupies disk and still gets picked
        up by the loader, so it must stay visible rather than being hidden."""
        d = tmp_path / "faster-whisper-medium"
        write(str(d / "config.json"))
        status = mf.faster_whisper_status(str(d))
        assert status.state == "incomplete"
        assert "model.bin" in status.missing


class TestMissingRequired:
    def test_alternatives_are_reported_as_one_entry(self, tmp_path):
        write(str(tmp_path / "a.json"))
        assert mf.missing_required(str(tmp_path), [("a.json", "b.json")]) == []
        assert mf.missing_required(str(tmp_path), [("x.json", "y.json")]) == ["x.json or y.json"]


class TestDescribeMissing:
    def test_empty(self):
        assert mf.describe_missing([]) == ""

    def test_one(self):
        assert mf.describe_missing(["tokenizer.json"]) == "tokenizer.json"

    def test_two(self):
        assert mf.describe_missing(["model.bin", "tokenizer.json"]) == "model.bin and tokenizer.json"

    def test_three(self):
        assert mf.describe_missing(["a", "b", "c"]) == "a, b and c"


class TestBytesOnDisk:
    """What Remove would throw away.

    Transfers resume now, so an incomplete directory is not junk — it is however
    much of a multi-gigabyte download survived a bad connection. Nothing in the
    UI distinguished 1.4 GB of progress from an empty stub, which made the choice
    between Repair and Remove a guess.
    """

    def test_a_staged_part_counts_towards_what_would_be_lost(self, tmp_path):
        d = tmp_path / "faster-whisper-large-v3"
        d.mkdir()
        (d / "config.json").write_bytes(b"x" * 100)
        (d / "model.bin.part").write_bytes(b"x" * 5000)

        assert mf.bytes_on_disk(str(d)) == 5100

    def test_an_absent_directory_is_zero_rather_than_an_error(self, tmp_path):
        assert mf.bytes_on_disk(str(tmp_path / "nope")) == 0

    def test_an_empty_directory_is_zero(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert mf.bytes_on_disk(str(d)) == 0

    def test_subdirectories_do_not_break_the_count(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "sub").mkdir()
        (d / "model.bin").write_bytes(b"x" * 42)
        assert mf.bytes_on_disk(str(d)) == 42


class TestDescribeBytes:
    def test_it_scales_to_a_unit_a_person_can_read(self):
        assert mf.describe_bytes(0) == "0 B"
        assert mf.describe_bytes(999) == "999 B"
        assert mf.describe_bytes(1536) == "1.5 KB"
        assert mf.describe_bytes(3 * 1024 ** 3) == "3.0 GB"

    def test_a_multi_terabyte_figure_still_renders(self):
        assert mf.describe_bytes(5 * 1024 ** 4).endswith("GB")

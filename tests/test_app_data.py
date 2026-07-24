"""One-time APP_DIR relocation (stt/app_data.py)."""

import os

import pytest

from stt.app_data import (
    DEFAULT_MIGRATION_ITEMS,
    migrate_app_data,
    removable_originals,
    tree_stats,
)


def _seed_old(old):
    """A realistic old repo-dir layout: config + a session DB + a model."""
    (old / "config").mkdir(parents=True)
    (old / "config" / "config.json").write_text('{"a": 1}')
    (old / "_AUTOMATIC_BACKUP" / "2026" / "07").mkdir(parents=True)
    (old / "_AUTOMATIC_BACKUP" / "2026" / "07" / "s.db").write_bytes(b"sqlite")
    (old / "models" / "facebook--nllb").mkdir(parents=True)
    (old / "models" / "facebook--nllb" / "model.safetensors").write_bytes(b"\x00")
    (old / "download_progress.json").write_text("{}")


class TestMigrateAppData:
    def test_copies_dirs_and_files_and_leaves_originals(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        _seed_old(old)
        new.mkdir()

        results = migrate_app_data(str(old), str(new))
        status = dict(results)

        # Copied into the new root...
        assert (new / "config" / "config.json").read_text() == '{"a": 1}'
        assert (new / "_AUTOMATIC_BACKUP" / "2026" / "07" / "s.db").exists()
        assert (new / "models" / "facebook--nllb" / "model.safetensors").exists()
        assert (new / "download_progress.json").exists()
        # ...and the originals are still there (copy, never move).
        assert (old / "config" / "config.json").exists()
        assert (old / "models" / "facebook--nllb" / "model.safetensors").exists()

        assert status["config"] == "copied"
        assert status["models"] == "copied"
        assert status["_AUTOMATIC_BACKUP"] == "copied"
        assert status["download_progress.json"] == "copied"
        # Items with no source are skipped, not errored.
        assert status["logs"] == "skipped"
        assert status["server.log"] == "skipped"

    def test_does_not_overwrite_existing_target(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        _seed_old(old)
        (new / "config").mkdir(parents=True)
        (new / "config" / "config.json").write_text('{"existing": true}')

        status = dict(migrate_app_data(str(old), str(new)))

        assert status["config"] == "skipped"
        # Existing target config is untouched.
        assert (new / "config" / "config.json").read_text() == '{"existing": true}'

    def test_second_run_is_a_noop(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        _seed_old(old)
        new.mkdir()

        migrate_app_data(str(old), str(new))
        second = dict(migrate_app_data(str(old), str(new)))

        # Everything that existed is now present in the target -> all skipped.
        assert second["config"] == "skipped"
        assert second["models"] == "skipped"
        assert second["download_progress.json"] == "skipped"

    def test_empty_old_root_skips_all(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        old.mkdir()
        new.mkdir()
        results = migrate_app_data(str(old), str(new))
        assert {s for _, s in results} == {"skipped"}
        assert [n for n, _ in results] == list(DEFAULT_MIGRATION_ITEMS)

    def test_custom_item_list(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        (old / "only").mkdir(parents=True)
        (old / "only" / "f.txt").write_text("x")
        new.mkdir()
        status = dict(migrate_app_data(str(old), str(new), items=["only"]))
        assert status == {"only": "copied"}
        assert (new / "only" / "f.txt").exists()

    @pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else True,
                        reason="root bypasses permission bits; POSIX-only test")
    def test_unreadable_source_is_reported_not_raised(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        secret = old / "logs"
        secret.mkdir(parents=True)
        (secret / "server.log").write_text("hi")
        os.chmod(secret, 0o000)  # unreadable -> copytree raises inside
        new.mkdir()
        logged = []
        try:
            status = dict(migrate_app_data(str(old), str(new), items=["logs"], log=logged.append))
        finally:
            os.chmod(secret, 0o700)
        assert status["logs"].startswith("error:")
        assert logged and "logs" in logged[0]


class TestTreeStats:
    def test_counts_files_and_bytes(self, tmp_path):
        d = tmp_path / "m"
        (d / "sub").mkdir(parents=True)
        (d / "a.bin").write_bytes(b"1234")
        (d / "sub" / "b.bin").write_bytes(b"56")
        assert tree_stats(str(d)) == (2, 6)

    def test_single_file(self, tmp_path):
        f = tmp_path / "x.json"
        f.write_bytes(b"abc")
        assert tree_stats(str(f)) == (1, 3)

    def test_missing_path(self, tmp_path):
        assert tree_stats(str(tmp_path / "nope")) == (0, 0)


class TestRemovableOriginals:
    def test_returns_fully_mirrored_items(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        _seed_old(old)
        migrate_app_data(str(old), str(new))  # full clean copy

        removable = removable_originals(str(old), str(new))
        assert set(removable) == {"config", "models", "_AUTOMATIC_BACKUP",
                                  "download_progress.json"}

    def test_excludes_partial_copy(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        _seed_old(old)
        migrate_app_data(str(old), str(new))
        # Simulate a target that lost a shard after migration (fewer bytes/files):
        (new / "models" / "facebook--nllb" / "model.safetensors").unlink()

        removable = removable_originals(str(old), str(new))
        assert "models" not in removable          # not safe — target is short
        assert "config" in removable              # still fully mirrored

    def test_excludes_missing_target(self, tmp_path):
        old, new = tmp_path / "repo", tmp_path / "home.stt"
        _seed_old(old)
        new.mkdir()  # nothing migrated
        assert removable_originals(str(old), str(new)) == []

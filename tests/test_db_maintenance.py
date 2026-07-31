"""Retiring WAL sidecars without losing data (stt/db_maintenance.py).

The temptation here is os.remove, and it is wrong: a WAL can hold committed
rows the main database file does not have yet, so deleting it discards a
service's last minutes silently — invisible until someone reads the transcript.
The first test below is the one that matters; a blind-delete implementation
passes every other test in this file and fails that one.
"""

import os
import sqlite3

import pytest

from stt.db_maintenance import (
    SIDECAR_SUFFIXES,
    checkpoint_and_release,
    resolve_sidecars,
    sweep_orphaned_sidecars,
)


def make_session_db(path, rows=3, leave_wal=True):
    """A WAL-mode session database, optionally still carrying its sidecars.

    Holding the connection open is what leaves the -wal behind: it is exactly
    the state a terminated worker leaves on disk.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany("INSERT INTO transcriptions (text) VALUES (?)",
                     [(f"caption {i}",) for i in range(rows)])
    conn.commit()
    if leave_wal:
        # Deliberately not closed: the sidecars stay, and the rows live in the
        # -wal rather than in the .db.
        return conn
    # A genuinely clean database, built without using the code under test: a
    # TRUNCATE checkpoint alone would leave both files on disk (measured), so
    # the mode is switched to DELETE, which is what removes them.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.commit()
    conn.close()
    for suffix in ("-wal", "-shm"):
        if os.path.exists(str(path) + suffix):
            os.remove(str(path) + suffix)
    return None


def read_rows(path):
    """Row count as a delivered copy would see it — the .db alone.

    Returns 0 when the table is not there at all, which is what an unrecovered
    WAL looks like: the schema itself is still only in the sidecar.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0]
    except sqlite3.DatabaseError:
        return 0
    finally:
        conn.close()


def killed_worker_state(tmp_path, rows=5):
    """A .db + -wal pair whose rows are in the WAL and NOT in the main file.

    Copying the files out from under a live connection reproduces what a killed
    process leaves behind, which is the state that matters here. Closing the
    connection instead would not: on macOS close() checkpoints, so the data
    would already be in the main file and deleting the WAL would lose nothing —
    an earlier version of this test made exactly that mistake and passed
    against a naive os.remove implementation.
    """
    source = tmp_path / "live.db"
    conn = make_session_db(source, rows=rows)
    victim = tmp_path / "killed" / "session.db"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_bytes(source.read_bytes())
    (victim.parent / "session.db-wal").write_bytes((tmp_path / "live.db-wal").read_bytes())
    conn.close()
    return victim


class TestDataIsNeverLost:
    """The reason this module is not a call to os.remove."""

    def test_rows_living_only_in_the_wal_survive(self, tmp_path):
        db = killed_worker_state(tmp_path, rows=5)
        # The premise, asserted rather than assumed: the main file is empty and
        # every row is in the sidecar.
        assert os.path.exists(str(db) + "-wal")
        assert read_rows(db) == 0, "premise: the main file does not have the rows yet"

        assert checkpoint_and_release(str(db))
        assert read_rows(db) == 5, (
            "the rows were in the WAL — deleting it would have discarded them")
        assert resolve_sidecars(str(db)) == []

    def test_a_delivered_copy_is_complete_afterwards(self, tmp_path):
        """What the file mover sends is the .db; it must stand alone."""
        db = tmp_path / "session.db"
        conn = make_session_db(db, rows=4)
        conn.close()
        checkpoint_and_release(str(db))

        delivered = tmp_path / "delivered.db"
        delivered.write_bytes(db.read_bytes())  # no sidecars, as delivery does
        assert read_rows(delivered) == 4


class TestCheckpointAndRelease:
    def test_sidecars_are_gone_afterwards(self, tmp_path):
        db = tmp_path / "session.db"
        conn = make_session_db(db)
        conn.close()
        assert checkpoint_and_release(str(db))
        assert resolve_sidecars(str(db)) == []

    def test_an_already_tidy_database_is_fine(self, tmp_path):
        db = tmp_path / "session.db"
        make_session_db(db, leave_wal=False)
        assert checkpoint_and_release(str(db)) is True
        assert read_rows(db) == 3

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert checkpoint_and_release(str(tmp_path / "nope.db")) is False

    def test_a_directory_is_refused(self, tmp_path):
        assert checkpoint_and_release(str(tmp_path)) is False

    def test_a_file_that_is_not_a_database_is_refused(self, tmp_path):
        junk = tmp_path / "notes.db"
        junk.write_text("this is not sqlite", encoding="utf-8")
        assert checkpoint_and_release(str(junk)) is False

    def test_an_empty_path_is_refused(self):
        assert checkpoint_and_release("") is False


class TestResolveSidecars:
    def test_lists_only_what_exists(self, tmp_path):
        db = tmp_path / "s.db"
        db.write_bytes(b"")
        (tmp_path / "s.db-wal").write_bytes(b"")
        assert resolve_sidecars(str(db)) == [str(db) + "-wal"]

    def test_all_three_suffixes_are_recognised(self, tmp_path):
        db = tmp_path / "s.db"
        db.write_bytes(b"")
        for suffix in SIDECAR_SUFFIXES:
            (tmp_path / f"s.db{suffix}").write_bytes(b"")
        assert len(resolve_sidecars(str(db))) == 3

    def test_nothing_for_a_clean_database(self, tmp_path):
        db = tmp_path / "s.db"
        db.write_bytes(b"")
        assert resolve_sidecars(str(db)) == []

    def test_empty_path(self):
        assert resolve_sidecars("") == []


class TestSweep:
    def _dirty(self, directory, name, rows=2):
        directory.mkdir(parents=True, exist_ok=True)
        db = directory / name
        conn = make_session_db(db, rows=rows)
        conn.close()
        return db

    def test_cleans_every_database_it_finds(self, tmp_path):
        a = self._dirty(tmp_path / "2026" / "07", "a.db")
        b = self._dirty(tmp_path / "2026" / "06", "b.db")
        result = sweep_orphaned_sidecars([str(tmp_path)])
        assert result["cleaned"] == 2
        assert resolve_sidecars(str(a)) == [] and resolve_sidecars(str(b)) == []

    def test_databases_already_tidy_are_not_opened(self, tmp_path):
        make_session_db(tmp_path / "clean.db", leave_wal=False)
        result = sweep_orphaned_sidecars([str(tmp_path)])
        assert result["scanned"] == 0, "no sidecars means nothing to do"
        assert result["cleaned"] == 0

    def test_the_live_session_is_left_alone(self, tmp_path):
        """Its sidecars are load-bearing; checkpointing under the writer is wrong."""
        live = self._dirty(tmp_path, "live.db")
        result = sweep_orphaned_sidecars([str(tmp_path)], skip_paths=[str(live)])
        assert result["skipped_active"] == 1
        assert result["cleaned"] == 0
        assert resolve_sidecars(str(live)), "the sidecars must still be there"

    def test_skip_matches_through_a_symlink(self, tmp_path):
        live = self._dirty(tmp_path, "live.db")
        link = tmp_path / "alias.db"
        try:
            os.symlink(str(live), str(link))
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        result = sweep_orphaned_sidecars([str(tmp_path)], skip_paths=[str(link)])
        assert result["skipped_active"] == 1

    def test_a_recently_stopped_session_can_be_left_to_settle(self, tmp_path):
        db = self._dirty(tmp_path, "just_stopped.db")
        os.utime(str(db), (1000.0, 1000.0))
        result = sweep_orphaned_sidecars([str(tmp_path)], min_age_s=60, now=1030.0)
        assert result["skipped_recent"] == 1
        assert resolve_sidecars(str(db))

    def test_an_older_session_is_swept_under_the_same_rule(self, tmp_path):
        db = self._dirty(tmp_path, "old.db")
        os.utime(str(db), (1000.0, 1000.0))
        result = sweep_orphaned_sidecars([str(tmp_path)], min_age_s=60, now=9999.0)
        assert result["cleaned"] == 1
        assert resolve_sidecars(str(db)) == []

    def test_a_swept_database_keeps_rows_that_were_only_in_its_wal(self, tmp_path):
        """The same guarantee, through the sweep the server actually runs."""
        db = killed_worker_state(tmp_path, rows=7)
        assert read_rows(db) == 0
        result = sweep_orphaned_sidecars([str(db.parent)])
        assert result["cleaned"] == 1
        assert read_rows(db) == 7

    def test_a_corrupt_database_is_counted_not_raised(self, tmp_path):
        junk = tmp_path / "corrupt.db"
        junk.write_text("not sqlite", encoding="utf-8")
        (tmp_path / "corrupt.db-wal").write_bytes(b"junk")
        result = sweep_orphaned_sidecars([str(tmp_path)])
        assert result["failed"] == 1
        assert "corrupt.db" in result["errors"]
        assert (tmp_path / "corrupt.db-wal").exists(), (
            "a file we could not open is not ours to delete")

    def test_non_database_files_are_ignored(self, tmp_path):
        (tmp_path / "session.wav").write_bytes(b"RIFF")
        (tmp_path / "session.srt").write_text("1\n", encoding="utf-8")
        assert sweep_orphaned_sidecars([str(tmp_path)])["scanned"] == 0

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        result = sweep_orphaned_sidecars([str(tmp_path / "gone")])
        assert result["cleaned"] == 0 and result["errors"] == []

    def test_no_directories_at_all(self):
        assert sweep_orphaned_sidecars([])["scanned"] == 0

    def test_the_same_directory_twice_is_not_double_counted(self, tmp_path):
        self._dirty(tmp_path, "a.db")
        result = sweep_orphaned_sidecars([str(tmp_path), str(tmp_path)])
        assert result["cleaned"] == 1

    def test_the_summary_reports_what_happened(self, tmp_path):
        self._dirty(tmp_path, "a.db")
        result = sweep_orphaned_sidecars([str(tmp_path)])
        assert set(result) == {"scanned", "cleaned", "skipped_active",
                               "skipped_recent", "failed", "errors"}

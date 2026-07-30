"""Destination reachability checks (stt/file_mover.py).

This is what the settings page calls before an operator commits a delivery
target, so a wrong answer either blocks a valid NAS share or accepts one that
silently drops every session afterwards. Both outcomes are discovered late — at
the end of a service, when the recording is meant to arrive.

The function is named ``test_destination_accessible`` in the module, which
pytest would otherwise collect as a test (and error on its required argument),
so it is imported under an alias.
"""

import os
import sys

import pytest

from stt.file_mover import is_smb_path
from stt.file_mover import test_destination_accessible as check_destination

# The smbclient helpers are imported by file_mover only when smbprotocol is
# installed. CI installs the dev dependencies only, so every stub below uses
# raising=False and the suite behaves the same with or without the library.


class TestLocalPaths:
    def test_a_writable_directory_is_accessible(self, tmp_path):
        assert check_destination(str(tmp_path)) is True

    def test_a_missing_directory_is_created(self, tmp_path):
        target = tmp_path / "2026" / "07"
        assert check_destination(str(target)) is True
        assert target.is_dir(), "the delivery tree is created on demand"

    def test_the_probe_file_is_cleaned_up(self, tmp_path):
        check_destination(str(tmp_path))
        assert os.listdir(str(tmp_path)) == [], (
            "a probe left behind would be delivered as if it were a session")

    def test_a_path_that_cannot_be_created_is_refused(self, tmp_path):
        # A file where a directory should be: makedirs raises, and the answer
        # must be False rather than an exception reaching the settings page.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        assert check_destination(str(blocker / "sub")) is False

    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="Windows ignores the POSIX write bit on directories (ACLs govern), "
               "and root ignores it everywhere")
    def test_a_read_only_directory_is_refused(self, tmp_path):
        target = tmp_path / "readonly"
        target.mkdir()
        target.chmod(0o500)
        try:
            assert check_destination(str(target)) is False
        finally:
            target.chmod(0o700)  # so tmp_path cleanup can remove it


class TestSmbPaths:
    """The SMB branch, up to the point where a real server would be needed."""

    def test_unc_and_posix_forms_are_both_recognised(self):
        assert is_smb_path("//nas/share/recordings")
        assert is_smb_path(r"\\nas\share\recordings")
        assert not is_smb_path("/mnt/nas/recordings")

    def test_smb_without_credentials_is_refused(self, monkeypatch):
        # Anonymous SMB is not attempted: it would appear to work against a
        # guest-writable share and then fail for the real one.
        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", True)
        assert check_destination("//nas/share") is False

    def test_smb_without_the_library_is_refused(self, monkeypatch):
        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", False)
        assert check_destination("//nas/share", username="u", password="p") is False

    @pytest.mark.parametrize("path", ["//nas", r"\\nas", "//"])
    def test_a_path_without_a_share_is_refused(self, path, monkeypatch):
        # "//server" alone names no share, so nothing could be written to it.
        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", True)
        assert check_destination(path, username="u", password="p") is False

    def test_a_reachable_share_probes_and_cleans_up(self, monkeypatch):
        """The success path, with the SMB calls stubbed."""
        opened, removed, sessions = [], [], []

        class _Handle:
            def __enter__(self_):
                return self_

            def __exit__(self_, *exc):
                return False

            def write(self_, data):
                opened.append(data)

        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", True)
        monkeypatch.setattr("stt.file_mover.register_session",
                            lambda server, **kw: sessions.append((server, kw.get("username"))),
                            raising=False)
        monkeypatch.setattr("stt.file_mover.open_file",
                            lambda path, mode: _Handle(), raising=False)
        monkeypatch.setattr("stt.file_mover.smb_remove",
                            lambda path: removed.append(path), raising=False)

        assert check_destination("//nas/share/rec", username="u", password="p") is True
        assert sessions == [("nas", "u")], "the session is registered for the server, not the share"
        assert opened == ["test"]
        assert removed and removed[0].endswith(".file_mover_test"), "the probe is removed"

    def test_a_failing_write_is_refused_not_raised(self, monkeypatch):
        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", True)
        monkeypatch.setattr("stt.file_mover.register_session",
                            lambda server, **kw: None, raising=False)

        def boom(path, mode):
            raise OSError("STATUS_ACCESS_DENIED")

        monkeypatch.setattr("stt.file_mover.open_file", boom, raising=False)
        assert check_destination("//nas/share", username="u", password="p") is False

    def test_a_failing_session_is_refused_not_raised(self, monkeypatch):
        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", True)

        def boom(server, **kw):
            raise ValueError("bad credentials")

        monkeypatch.setattr("stt.file_mover.register_session", boom, raising=False)
        assert check_destination("//nas/share", username="u", password="p") is False

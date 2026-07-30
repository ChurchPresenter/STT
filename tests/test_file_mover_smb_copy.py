"""Direct SMB delivery of a finished session (stt/file_mover.py).

This is how a recording reaches the NAS after a service. It runs unattended,
and its two failure modes are quiet: a wrong path parse writes to the wrong
share, and a missing directory step silently drops the file. The directory
walk is the fiddly part — an SMB path has a server and a share before any real
directory, and mkdir on either of those is an error rather than a no-op.

No server is contacted: the smbclient helpers are recorded. They exist only
when smbprotocol is installed, which CI does not have, so every stub uses
raising=False.
"""

import pytest

from stt.file_mover import copy_file_to_smb_direct


@pytest.fixture
def smb(monkeypatch, tmp_path):
    """Records what the copy would have done, and lets tests shape the share."""
    state = {"existing": set(), "mkdirs": [], "sessions": [], "written": [], "opened": []}

    class _Dst:
        def __enter__(self_):
            return self_

        def __exit__(self_, *exc):
            return False

        def write(self_, data):
            state["written"].append(data)

    def fake_open_file(path, mode="rb"):
        state["opened"].append((path, mode))
        return _Dst()

    monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", True)
    monkeypatch.setattr("stt.file_mover.register_session",
                        lambda server, **kw: state["sessions"].append((server, kw)),
                        raising=False)
    monkeypatch.setattr("stt.file_mover.smb_exists",
                        lambda path: path in state["existing"], raising=False)
    monkeypatch.setattr("stt.file_mover.mkdir",
                        lambda path: state["mkdirs"].append(path), raising=False)
    monkeypatch.setattr("stt.file_mover.open_file", fake_open_file, raising=False)
    monkeypatch.setattr("stt.file_mover.shutil.copyfileobj",
                        lambda src, dst: dst.write(src.read()))

    source = tmp_path / "2026-05-21_183919.db"
    source.write_bytes(b"session-bytes")
    state["source"] = str(source)
    return state


class TestGuards:
    def test_without_the_library_it_reports_rather_than_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("stt.file_mover.SMB_AVAILABLE", False)
        ok, err = copy_file_to_smb_direct(str(tmp_path / "x"), "//nas/share/x", "u", "p")
        assert ok is False
        assert "smbprotocol" in err

    @pytest.mark.parametrize("dest", ["//nas", r"\\nas", "//"])
    def test_a_path_naming_no_share_is_refused(self, smb, dest):
        ok, err = copy_file_to_smb_direct(smb["source"], dest, "u", "p")
        assert ok is False
        assert "Invalid SMB path" in err
        assert smb["sessions"] == [], "no session is opened for an unusable path"


class TestPathHandling:
    def test_backslash_form_is_normalised(self, smb):
        ok, _ = copy_file_to_smb_direct(smb["source"], r"\\nas\share\rec\s.db", "u", "p")
        assert ok is True
        assert smb["opened"][0][0] == "//nas/share/rec/s.db", (
            "the UNC form an operator pastes from Windows must reach smbclient as POSIX")

    def test_the_session_is_registered_for_the_server_only(self, smb):
        copy_file_to_smb_direct(smb["source"], "//nas/share/rec/s.db", "u", "p")
        assert smb["sessions"][0][0] == "nas"
        assert smb["sessions"][0][1]["username"] == "u"
        assert smb["sessions"][0][1]["auth_protocol"] == "ntlm"


class TestDirectoryWalk:
    def test_missing_directories_are_created_in_order(self, smb):
        copy_file_to_smb_direct(smb["source"], "//nas/share/2026/07/s.db", "u", "p")
        assert smb["mkdirs"] == ["//nas/share", "//nas/share/2026", "//nas/share/2026/07"]

    def test_the_server_alone_is_never_mkdir_ed(self, smb):
        """"//nas" is not a directory; creating it is an error, not a no-op."""
        copy_file_to_smb_direct(smb["source"], "//nas/share/s.db", "u", "p")
        assert "//nas" not in smb["mkdirs"]

    def test_existing_directories_are_left_alone(self, smb):
        smb["existing"].update({"//nas/share", "//nas/share/2026"})
        copy_file_to_smb_direct(smb["source"], "//nas/share/2026/07/s.db", "u", "p")
        assert smb["mkdirs"] == ["//nas/share/2026/07"]

    def test_nothing_is_created_when_the_whole_tree_exists(self, smb):
        smb["existing"].add("//nas/share/2026/07")
        copy_file_to_smb_direct(smb["source"], "//nas/share/2026/07/s.db", "u", "p")
        assert smb["mkdirs"] == []

    def test_a_failed_mkdir_does_not_abort_the_delivery(self, smb, monkeypatch):
        """A share where mkdir is denied but the directory already exists.

        Racing another machine's delivery is normal, so a mkdir failure is
        logged and the copy is still attempted — the file is what matters.
        """
        def denied(path):
            raise PermissionError("STATUS_ACCESS_DENIED")

        monkeypatch.setattr("stt.file_mover.mkdir", denied, raising=False)
        ok, err = copy_file_to_smb_direct(smb["source"], "//nas/share/2026/s.db", "u", "p")
        assert ok is True and err is None


class TestTheCopyItself:
    def test_the_file_contents_are_written_to_the_share(self, smb):
        ok, err = copy_file_to_smb_direct(smb["source"], "//nas/share/s.db", "u", "p")
        assert (ok, err) == (True, None)
        assert smb["written"] == [b"session-bytes"]

    def test_the_destination_is_opened_binary(self, smb):
        copy_file_to_smb_direct(smb["source"], "//nas/share/s.db", "u", "p")
        assert smb["opened"][0][1] == "wb", "text mode would corrupt a sqlite file"

    def test_a_missing_source_file_is_reported_not_raised(self, smb):
        ok, err = copy_file_to_smb_direct("/nope/missing.db", "//nas/share/s.db", "u", "p")
        assert ok is False and err

    def test_a_refused_write_is_reported_not_raised(self, smb, monkeypatch):
        def denied(path, mode="rb"):
            raise OSError("STATUS_DISK_FULL")

        monkeypatch.setattr("stt.file_mover.open_file", denied, raising=False)
        ok, err = copy_file_to_smb_direct(smb["source"], "//nas/share/s.db", "u", "p")
        assert ok is False
        assert "DISK_FULL" in err

    def test_bad_credentials_are_reported_not_raised(self, smb, monkeypatch):
        def denied(server, **kw):
            raise ValueError("STATUS_LOGON_FAILURE")

        monkeypatch.setattr("stt.file_mover.register_session", denied, raising=False)
        ok, err = copy_file_to_smb_direct(smb["source"], "//nas/share/s.db", "u", "p")
        assert ok is False
        assert "LOGON_FAILURE" in err

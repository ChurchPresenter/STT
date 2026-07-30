"""Mounting an SMB share (stt/file_mover.py).

Two things here are worth pinning rather than trusting to review. On Linux the
password must reach mount.cifs through a temp credentials file, never through
argv — anything on the command line is world-readable in /proc/<pid>/cmdline
for the duration of the mount. And on Windows "already in use" is success, not
failure: a share mounted by a previous run must not fail the delivery.

Nothing is mounted: subprocess.run and the /proc/mounts read are stubbed. The
module-level `open` stub works because a module global shadows the builtin for
code in that module.
"""

import os
import subprocess
import sys

import pytest

from stt.file_mover import mount_smb_share


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def mount(monkeypatch):
    """Records the mount command instead of running it."""
    state = {"cmds": [], "result": _Result(0), "proc_mounts": "", "creds_written": [],
             "unlinked": []}

    def fake_run(cmd, **kw):
        state["cmds"].append(list(cmd))
        # Capture the credentials file while it still exists — the function
        # deletes it in a finally block.
        for part in cmd:
            if isinstance(part, str) and part.startswith("credentials="):
                path = part.split("=", 1)[1]
                try:
                    with open(path, encoding="utf-8") as f:
                        state["creds_written"].append((path, f.read(),
                                                       os.stat(path).st_mode & 0o777))
                except OSError:
                    pass
        return state["result"]

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("stt.file_mover.os.makedirs", lambda *a, **k: None)

    real_unlink = os.unlink

    def fake_unlink(path):
        state["unlinked"].append(path)
        real_unlink(path)

    monkeypatch.setattr("stt.file_mover.os.unlink", fake_unlink)
    return state


def _stub_proc_mounts(monkeypatch, state):
    """Make the /proc/mounts read work on a machine that has no /proc."""
    import io

    real_open = open

    def fake_open(path, *a, **kw):
        if str(path) == "/proc/mounts":
            return io.StringIO(state["proc_mounts"])
        return real_open(path, *a, **kw)

    monkeypatch.setattr("stt.file_mover.open", fake_open, raising=False)


class TestWindows:
    @pytest.fixture(autouse=True)
    def on_windows(self, monkeypatch):
        monkeypatch.setattr("stt.file_mover.platform.system", lambda: "Windows")

    def test_uses_net_use_with_a_backslash_path(self, mount):
        assert mount_smb_share("//nas/share", "u", "p") is True
        cmd = mount["cmds"][0]
        assert cmd[:2] == ["net", "use"]
        assert cmd[2] == r"\\nas\share", "net use does not accept forward slashes"

    def test_a_domain_qualifies_the_user(self, mount):
        mount_smb_share("//nas/share", "u", "p", domain="CORP")
        assert r"/user:CORP\u" in mount["cmds"][0]

    def test_without_a_domain_the_user_is_bare(self, mount):
        mount_smb_share("//nas/share", "u", "p")
        assert "/user:u" in mount["cmds"][0]

    def test_an_already_mounted_share_is_success(self, mount):
        """A previous run's mount must not fail this delivery."""
        mount["result"] = _Result(2, stdout="The local device name is already in use.")
        assert mount_smb_share("//nas/share", "u", "p") is True

    def test_a_real_failure_is_reported(self, mount):
        mount["result"] = _Result(2, stderr="System error 53 has occurred.")
        assert mount_smb_share("//nas/share", "u", "p") is False


class TestLinux:
    @pytest.fixture(autouse=True)
    def on_linux(self, monkeypatch, mount):
        monkeypatch.setattr("stt.file_mover.platform.system", lambda: "Linux")
        _stub_proc_mounts(monkeypatch, mount)

    def test_mounts_cifs_at_a_server_share_mount_point(self, mount):
        assert mount_smb_share("//nas/recordings", "u", "p") is True
        cmd = mount["cmds"][0]
        assert cmd[:5] == ["sudo", "mount", "-t", "cifs", "//nas/recordings"]
        assert cmd[5] == "/mnt/nas_recordings"

    def test_the_password_never_reaches_the_command_line(self, mount):
        """argv is world-readable in /proc/<pid>/cmdline while mount runs."""
        mount_smb_share("//nas/share", "operator", "hunter2", domain="CORP")
        flat = " ".join(mount["cmds"][0])
        assert "hunter2" not in flat
        assert "operator" not in flat
        assert "credentials=" in flat, "credentials go through a file instead"

    def test_the_credentials_file_holds_the_secrets_and_is_private(self, mount):
        mount_smb_share("//nas/share", "operator", "hunter2", domain="CORP")
        assert mount["creds_written"], "the credentials file was never written"
        _path, body, mode = mount["creds_written"][0]
        assert "username=operator" in body
        assert "password=hunter2" in body
        assert "domain=CORP" in body
        if sys.platform != "win32":
            # POSIX permission bits only mean something on POSIX; on Windows
            # st_mode always reports 0o666 because NTFS uses ACLs. This branch
            # is the Linux one anyway — /proc/<pid>/cmdline is what it protects.
            assert mode == 0o600, (
                f"credentials file must not be readable by others (was {mode:o})")

    def test_the_credentials_file_is_deleted_afterwards(self, mount):
        mount_smb_share("//nas/share", "u", "p")
        path = mount["creds_written"][0][0]
        assert path in mount["unlinked"]
        assert not os.path.exists(path)

    def test_it_is_deleted_even_when_the_mount_fails(self, mount, monkeypatch):
        def boom(cmd, **kw):
            for part in cmd:
                if isinstance(part, str) and part.startswith("credentials="):
                    mount["creds_written"].append((part.split("=", 1)[1], "", 0o600))
            raise OSError("mount blew up")

        monkeypatch.setattr(subprocess, "run", boom)
        assert mount_smb_share("//nas/share", "u", "p") is False
        path = mount["creds_written"][0][0]
        assert not os.path.exists(path), "a leaked credentials file would outlive the failure"

    def test_no_credentials_file_when_there_are_no_credentials(self, mount):
        mount_smb_share("//nas/share", "", "")
        assert not any("credentials=" in str(p) for p in mount["cmds"][0])

    def test_an_already_mounted_point_short_circuits(self, mount):
        mount["proc_mounts"] = "//nas/share /mnt/nas_share cifs rw 0 0\n"
        assert mount_smb_share("//nas/share", "u", "p") is True
        assert mount["cmds"] == [], "mounting twice would fail or stack mounts"

    def test_a_path_naming_no_share_is_refused(self, mount):
        assert mount_smb_share("//nas", "u", "p") is False
        assert mount["cmds"] == []

    def test_a_failed_mount_is_reported(self, mount):
        mount["result"] = _Result(32, stderr="mount error(13): Permission denied")
        assert mount_smb_share("//nas/share", "u", "p") is False


class TestOtherPlatforms:
    def test_macos_is_refused_rather_than_guessed(self, mount, monkeypatch):
        # No mount.cifs on macOS; the direct-SMB path is used there instead.
        monkeypatch.setattr("stt.file_mover.platform.system", lambda: "Darwin")
        assert mount_smb_share("//nas/share", "u", "p") is False
        assert mount["cmds"] == []

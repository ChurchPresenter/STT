"""_git_usable: macOS xcode-select git shim must not count as a real git.

On a Mac without the Command Line Tools, /usr/bin/git is a shim that pops
Apple's GUI installer dialog and exits 1 — provisioning must treat it as
"no git" and use the archive fallback instead.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt.watchdog as watchdog  # noqa: E402


def _patch(monkeypatch, *, platform, which, xcode_rc=None, probe_rc=0):
    monkeypatch.setattr(watchdog.sys, "platform", platform)
    # IS_WINDOWS is decided at import time, so the simulated platform has to
    # drive it too — otherwise these cases take the Windows probe branch when
    # the suite runs on the Windows CI runner.
    monkeypatch.setattr(watchdog, "IS_WINDOWS", platform == "win32")
    monkeypatch.setattr(watchdog, "_which", lambda name: which)
    # realpath is the host's, not the simulated platform's: on Windows it turns
    # "/usr/bin/git" into "C:\\usr\\bin\\git" and the macOS shim check never
    # fires. The branch under test is the logic, not the host's path rules.
    monkeypatch.setattr(watchdog.os.path, "realpath", lambda path: path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        rc = probe_rc if "--version" in cmd else xcode_rc
        return subprocess.CompletedProcess(cmd, rc)

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    return calls


def test_no_git_on_path(monkeypatch):
    _patch(monkeypatch, platform="darwin", which=None)
    assert watchdog._git_usable() is False


def test_darwin_shim_without_clt_is_unusable(monkeypatch):
    calls = _patch(monkeypatch, platform="darwin", which="/usr/bin/git", xcode_rc=2)
    assert watchdog._git_usable() is False
    assert calls == [["xcode-select", "-p"]]


def test_darwin_usr_bin_git_with_clt_is_usable(monkeypatch):
    _patch(monkeypatch, platform="darwin", which="/usr/bin/git", xcode_rc=0)
    assert watchdog._git_usable() is True


def test_darwin_homebrew_git_skips_clt_check(monkeypatch):
    calls = _patch(monkeypatch, platform="darwin", which="/opt/homebrew/bin/git")
    assert watchdog._git_usable() is True
    assert calls == []


def test_darwin_xcode_select_missing_is_unusable(monkeypatch):
    monkeypatch.setattr(watchdog.sys, "platform", "darwin")
    monkeypatch.setattr(watchdog, "IS_WINDOWS", False)
    monkeypatch.setattr(watchdog, "_which", lambda name: "/usr/bin/git")
    # As in _patch: the host's realpath would rewrite the POSIX path.
    monkeypatch.setattr(watchdog.os.path, "realpath", lambda path: path)

    def raise_oserror(cmd, **kwargs):
        raise FileNotFoundError("xcode-select")

    monkeypatch.setattr(watchdog.subprocess, "run", raise_oserror)
    assert watchdog._git_usable() is False


def test_linux_git_is_usable_without_clt_check(monkeypatch):
    calls = _patch(monkeypatch, platform="linux", which="/usr/bin/git")
    assert watchdog._git_usable() is True
    assert calls == []


class TestWindowsProbesGit:
    r"""Windows has its own shim problem: an App Execution Alias in
    %LOCALAPPDATA%\Microsoft\WindowsApps answers a PATH lookup with a 0-byte
    reparse point, and spawning it raises OSError [WinError 1920] when no Store
    app backs it. Reported from the field as a first-run setup crash. Being on
    PATH is therefore not evidence — git has to actually run, so that a broken
    one routes provisioning to MinGit and the source archive instead.
    """

    ALIAS = r"C:\Users\User\AppData\Local\Microsoft\WindowsApps\git.exe"

    def test_a_git_that_runs_is_usable(self, monkeypatch):
        calls = _patch(monkeypatch, platform="win32", which=r"C:\Program Files\Git\cmd\git.exe")
        assert watchdog._git_usable() is True
        assert calls and calls[0][1:] == ["--version"], "the probe runs the resolved git"

    def test_a_git_that_exits_nonzero_is_unusable(self, monkeypatch):
        _patch(monkeypatch, platform="win32", which=self.ALIAS, probe_rc=1)
        assert watchdog._git_usable() is False

    def test_an_unspawnable_alias_stub_is_unusable(self, monkeypatch):
        """[WinError 1920] — the exact field failure."""
        _patch(monkeypatch, platform="win32", which=self.ALIAS)

        def raise_oserror(cmd, **kwargs):
            raise OSError(22, "The file cannot be accessed by the system")

        monkeypatch.setattr(watchdog.subprocess, "run", raise_oserror)
        assert watchdog._git_usable() is False

    def test_a_hanging_probe_is_unusable(self, monkeypatch):
        """A shim that prompts instead of answering must not stall setup."""
        _patch(monkeypatch, platform="win32", which=self.ALIAS)

        def raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 30)

        monkeypatch.setattr(watchdog.subprocess, "run", raise_timeout)
        assert watchdog._git_usable() is False

    def test_no_git_on_path_never_probes(self, monkeypatch):
        calls = _patch(monkeypatch, platform="win32", which=None)
        assert watchdog._git_usable() is False
        assert calls == []


class TestHaveGitCheckout:
    """The updater must gate on _git_usable(), not shutil.which: MinGit lives
    on the augmented PATH only, and the macOS shim answers shutil.which but
    cannot run a fetch/reset."""

    def _source(self, monkeypatch, tmp_path, *, dot_git):
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(tmp_path))
        if dot_git:
            (tmp_path / ".git").mkdir()

    def test_usable_git_and_checkout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(watchdog, "_git_usable", lambda: True)
        self._source(monkeypatch, tmp_path, dot_git=True)
        assert watchdog._have_git_checkout() is True

    def test_checkout_without_usable_git(self, monkeypatch, tmp_path):
        # macOS shim over a checkout git once made: must fall back to archives
        # rather than run a reset that the shim fails.
        monkeypatch.setattr(watchdog, "_git_usable", lambda: False)
        self._source(monkeypatch, tmp_path, dot_git=True)
        assert watchdog._have_git_checkout() is False

    def test_usable_git_without_checkout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(watchdog, "_git_usable", lambda: True)
        self._source(monkeypatch, tmp_path, dot_git=False)
        assert watchdog._have_git_checkout() is False

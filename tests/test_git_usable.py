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


def _patch(monkeypatch, *, platform, which, xcode_rc=None):
    monkeypatch.setattr(watchdog.sys, "platform", platform)
    monkeypatch.setattr(watchdog, "_which", lambda name: which)
    # realpath is the host's, not the simulated platform's: on Windows it turns
    # "/usr/bin/git" into "C:\\usr\\bin\\git" and the macOS shim check never
    # fires. The branch under test is the logic, not the host's path rules.
    monkeypatch.setattr(watchdog.os.path, "realpath", lambda path: path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, xcode_rc)

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

"""Provisioner steps that decide what gets installed (stt/watchdog.py).

The watchdog provisions a fresh machine: uv, a managed Python, git, ffmpeg. It
runs once, on someone else's computer, usually with nobody watching — so a wrong
branch here is discovered as "the installer hung" or "it says uv is installed
and it isn't", with no operator able to diagnose it.

Nothing is downloaded or executed: _run, _which and the download helpers are
stubbed, and the assertions are about which branch was taken and what it was
asked to run.
"""

import pytest

from stt import watchdog


def make_provisioner(monkeypatch, *, which=None, run_codes=None, downloads=None):
    """A Provisioner with every side effect replaced by a recorder."""
    p = watchdog.Provisioner.__new__(watchdog.Provisioner)
    p._uv = None
    p.logs = []
    p.log = lambda msg: p.logs.append(str(msg))
    p.runs = []
    p.downloaded = []

    codes = dict(run_codes or {})

    def fake_run(cmd, desc=None, check=True, **kw):
        p.runs.append(list(cmd))
        code = codes.get(tuple(cmd[:1]), 0)
        if code and check:
            raise watchdog.ProvisionError(f"{desc} failed")
        return code

    p._run = fake_run
    p._download_file = lambda url, dest: p.downloaded.append(("file", url, dest))
    p._download_text = lambda url: (p.downloaded.append(("text", url, None)) or
                                    (downloads or {}).get(url, "#!/bin/sh\ntrue\n"))

    seq = list(which if isinstance(which, list) else [which])

    def fake_which(name):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(watchdog, "_which", fake_which)
    return p


class TestUvAlreadyPresent:
    def test_an_existing_uv_is_used_and_nothing_is_installed(self, monkeypatch):
        p = make_provisioner(monkeypatch, which="/usr/local/bin/uv")
        p._step_uv()
        assert p._uv == "/usr/local/bin/uv"
        assert p.runs == [], "an installed uv must never be reinstalled"
        assert p.downloaded == [], "and nothing downloaded"

    def test_it_says_which_uv_it_found(self, monkeypatch):
        p = make_provisioner(monkeypatch, which="/opt/uv")
        p._step_uv()
        assert any("/opt/uv" in line for line in p.logs), (
            "the path matters when several uv installs exist")


class TestUvInstallOnPosix:
    @pytest.fixture(autouse=True)
    def posix(self, monkeypatch):
        monkeypatch.setattr(watchdog, "IS_WINDOWS", False)

    def test_the_install_script_is_fetched_and_run(self, monkeypatch):
        # absent at the first probe, present after installing
        p = make_provisioner(monkeypatch, which=[None, "/root/.local/bin/uv"])
        p._step_uv()
        assert ("text", "https://astral.sh/uv/install.sh", None) in p.downloaded
        assert any(cmd[0] == "sh" for cmd in p.runs), "the downloaded script is executed"
        assert p._uv == "/root/.local/bin/uv"

    def test_an_install_that_leaves_no_uv_on_path_is_an_error(self, monkeypatch):
        """Silence here would fail later, during the first real dependency install."""
        p = make_provisioner(monkeypatch, which=[None, None])
        with pytest.raises(watchdog.ProvisionError, match="did not put 'uv' on PATH"):
            p._step_uv()

    def test_windows_powershell_is_not_used_on_posix(self, monkeypatch):
        p = make_provisioner(monkeypatch, which=[None, "/usr/bin/uv"])
        p._step_uv()
        assert not any("powershell" in c[0] for c in p.runs)


class TestUvInstallOnWindows:
    @pytest.fixture(autouse=True)
    def windows(self, monkeypatch):
        monkeypatch.setattr(watchdog, "IS_WINDOWS", True)

    def test_the_powershell_installer_runs_first(self, monkeypatch):
        p = make_provisioner(monkeypatch, which=[None, "C:/uv.exe", "C:/uv.exe"])
        p._step_uv()
        assert p.runs and p.runs[0][0] == "powershell"
        assert p.downloaded == [], "the zip fallback is only for a failed installer"

    def test_tls_1_2_is_forced_in_the_installer_command(self, monkeypatch):
        """Stock PowerShell may still default to TLS 1.0, which astral.sh rejects."""
        p = make_provisioner(monkeypatch, which=[None, "C:/uv.exe", "C:/uv.exe"])
        p._step_uv()
        assert any("3072" in part for part in p.runs[0]), (
            "3072 is the SecurityProtocol bit for TLS 1.2")

    def test_a_blocked_installer_falls_back_to_the_github_zip(self, monkeypatch):
        # PowerShell "succeeds" but uv is still not on PATH: proxy or AV ate it.
        p = make_provisioner(monkeypatch, which=[None, None, "C:/uv.exe", "C:/uv.exe"])
        monkeypatch.setattr(watchdog.zipfile, "ZipFile", _FakeZip)
        monkeypatch.setattr(watchdog.os, "remove", lambda path: None)
        p._step_uv()
        kinds = [d[0] for d in p.downloaded]
        assert "file" in kinds, "the zip is fetched over the certifi-backed downloader"
        assert any(".zip" in d[1] for d in p.downloaded)

    @pytest.mark.parametrize("machine,expected", [
        ("arm64", "aarch64"), ("AARCH64", "aarch64"),
        ("AMD64", "x86_64"), ("x86_64", "x86_64"),
    ])
    def test_the_zip_matches_the_machine_architecture(self, monkeypatch, machine, expected):
        p = make_provisioner(monkeypatch, which=[None, None, "C:/uv.exe", "C:/uv.exe"])
        monkeypatch.setattr(watchdog.platform, "machine", lambda: machine)
        monkeypatch.setattr(watchdog.zipfile, "ZipFile", _FakeZip)
        monkeypatch.setattr(watchdog.os, "remove", lambda path: None)
        p._step_uv()
        url = next(d[1] for d in p.downloaded if d[0] == "file")
        assert f"uv-{expected}-pc-windows-msvc.zip" in url


class _FakeZip:
    """Stands in for zipfile.ZipFile so nothing is extracted to disk."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extractall(self, dest):
        self.dest = dest


class TestSpawnFailures:
    """A command that cannot be spawned at all.

    Reported from a Windows install: CreateProcess on the system powershell.exe
    returned [WinError 5] Access is denied (AppLocker/AV policy), and the
    PermissionError escaped _run — so setup died on step 2/8 even though
    _step_uv has a pure-Python fallback right below the PowerShell call.
    """

    def provisioner(self, monkeypatch, error):
        p = watchdog.Provisioner.__new__(watchdog.Provisioner)
        p._uv = None
        p.logs = []
        p.log = lambda msg: p.logs.append(str(msg))

        def boom(*a, **kw):
            raise error

        monkeypatch.setattr(watchdog.subprocess, "Popen", boom)
        monkeypatch.setattr(watchdog, "_which", lambda name: None)
        return p

    @pytest.mark.parametrize("error", [
        PermissionError(13, "Access is denied"),
        FileNotFoundError(2, "No such file"),
        OSError(8, "Exec format error"),
    ])
    def test_an_unspawnable_command_is_a_failure_not_a_crash(self, monkeypatch, error):
        p = self.provisioner(monkeypatch, error)
        assert p._run(["powershell", "-Command", "x"], check=False) == 1

    def test_it_is_reported_as_a_provision_error_when_checked(self, monkeypatch):
        p = self.provisioner(monkeypatch, PermissionError(13, "Access is denied"))
        with pytest.raises(watchdog.ProvisionError, match="could not be started"):
            p._run(["powershell", "-Command", "x"])

    def test_a_missing_command_still_says_not_found(self, monkeypatch):
        p = self.provisioner(monkeypatch, FileNotFoundError(2, "No such file"))
        with pytest.raises(watchdog.ProvisionError, match="not found"):
            p._run(["git", "status"])

    def test_a_blocked_powershell_still_reaches_the_zip_fallback(self, monkeypatch):
        """The whole point: policy blocking PowerShell must not end the install."""
        p = self.provisioner(monkeypatch, PermissionError(13, "Access is denied"))
        p.downloaded = []
        p._download_file = lambda url, dest: p.downloaded.append(url)
        monkeypatch.setattr(watchdog, "IS_WINDOWS", True)
        monkeypatch.setattr(watchdog.zipfile, "ZipFile", _FakeZip)
        monkeypatch.setattr(watchdog.os, "remove", lambda path: None)

        found = [None, None, "C:/uv.exe", "C:/uv.exe"]
        monkeypatch.setattr(watchdog, "_which",
                            lambda name: found.pop(0) if len(found) > 1 else found[0])

        p._step_uv()
        assert any(".zip" in url for url in p.downloaded)
        assert p._uv == "C:/uv.exe"


def test_provision_error_is_an_exception():
    """Callers catch it by type to distinguish setup failure from a crash."""
    assert issubclass(watchdog.ProvisionError, Exception)
    assert isinstance(watchdog.ProvisionError("x"), Exception)


def test_the_module_exposes_what_these_tests_patch():
    """A rename would otherwise make the stubs silently patch nothing."""
    for name in ("_which", "IS_WINDOWS", "ProvisionError", "Provisioner"):
        assert hasattr(watchdog, name), name
    assert isinstance(watchdog.UV_PYTHON_VERSION, str)


class TestPinLatestRelease:
    """Which release tag a fresh install checks out.

    A release workflow that tags and pushes the branch separately can leave a tag
    on a commit that reached no branch — and that is the *newest* tag, so it wins
    the version comparison and a fresh install silently checks out code that was
    never released. Observed once, from a branch push rejected as non-fast-forward
    while the tag in the same command went through.
    """

    def pin(self, monkeypatch, tags, on_branch=(), shallow=False):
        p = make_provisioner(monkeypatch)
        seen = {}

        def fake_subprocess_run(cmd, **kw):
            joined = " ".join(cmd)
            if "tag" in cmd and "--list" in cmd:
                return type("R", (), {"returncode": 0, "stdout": "\n".join(tags)})()
            if "--is-shallow-repository" in cmd:
                return type("R", (), {"returncode": 0, "stdout": "true" if shallow else "false"})()
            if "merge-base" in cmd:
                seen.setdefault("checked", []).append(cmd[-2])
                return type("R", (), {"returncode": 0 if cmd[-2] in on_branch else 1, "stdout": ""})()
            return type("R", (), {"returncode": 0, "stdout": ""})()

        monkeypatch.setattr(watchdog.subprocess, "run", fake_subprocess_run)
        monkeypatch.setattr(watchdog, "_which", lambda n: "/usr/bin/" + n)
        p._pin_latest_release()
        resets = [c[-1] for c in p.runs if "reset" in c]
        return p, resets, seen

    def test_the_newest_tag_on_the_branch_wins(self, monkeypatch):
        _p, resets, _ = self.pin(monkeypatch, ["26.2.140", "26.2.150"],
                                on_branch=("26.2.140", "26.2.150"))
        assert resets == ["26.2.150"]

    def test_a_tag_on_no_branch_is_skipped_for_the_next_one(self, monkeypatch):
        # The exact failure: the newest tag is the orphan, so it must not win.
        p, resets, _ = self.pin(monkeypatch, ["26.2.140", "26.2.158"],
                                on_branch=("26.2.140",))
        assert resets == ["26.2.140"]
        assert any("skipping 26.2.158" in m for m in p.logs)

    def test_no_usable_tag_stays_on_the_branch(self, monkeypatch):
        p, resets, _ = self.pin(monkeypatch, ["26.2.158"], on_branch=())
        assert resets == []
        assert any("staying on default branch" in m for m in p.logs)

    def test_no_tags_at_all_stays_on_the_branch(self, monkeypatch):
        p, resets, _ = self.pin(monkeypatch, [], on_branch=())
        assert resets == []
        assert any("no release tags" in m for m in p.logs)

    def test_a_shallow_clone_trusts_the_tag(self, monkeypatch):
        # Fails open: a shallow clone has no history to walk, and refusing every
        # tag there would leave every install on the default branch.
        _p, resets, seen = self.pin(monkeypatch, ["26.2.158"], on_branch=(), shallow=True)
        assert resets == ["26.2.158"]
        assert "checked" not in seen, "no reachability check is possible on a shallow clone"

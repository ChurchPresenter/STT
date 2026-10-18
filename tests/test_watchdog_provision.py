"""Provisioner steps that decide what gets installed (stt/watchdog.py).

The watchdog provisions a fresh machine: uv, a managed Python, git, ffmpeg. It
runs once, on someone else's computer, usually with nobody watching — so a wrong
branch here is discovered as "the installer hung" or "it says uv is installed
and it isn't", with no operator able to diagnose it.

Nothing is downloaded or executed: _run, _which and the download helpers are
stubbed, and the assertions are about which branch was taken and what it was
asked to run.
"""

import io
import os

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


class TestBrokenWingetAlias:
    r"""winget resolves on PATH and cannot be spawned.

    Reported twice from one Windows install: %LOCALAPPDATA%\Microsoft\WindowsApps
    holds a 0-byte App Execution Alias that answers the PATH lookup, and
    CreateProcess on it raises OSError [WinError 1920] when no Store app backs
    it. The alias cannot be told apart from a working one by inspection (both
    are 0-byte reparse points), so the spawn is the probe — and each step has to
    survive it and take its portable fallback.
    """

    ALIAS_ERROR = OSError(22, "The file cannot be accessed by the system")

    def provisioner(self, monkeypatch, *, present):
        """`present` decides what _which answers; winget is always resolvable."""
        p = watchdog.Provisioner.__new__(watchdog.Provisioner)
        p.logs = []
        p.log = lambda msg: p.logs.append(str(msg))

        def boom(*a, **kw):
            raise self.ALIAS_ERROR

        monkeypatch.setattr(watchdog.subprocess, "Popen", boom)
        monkeypatch.setattr(watchdog, "IS_WINDOWS", True)
        monkeypatch.setattr(watchdog, "_which", lambda name: present.get(name))
        return p

    def test_git_falls_back_to_mingit(self, monkeypatch):
        alias = r"C:\Users\User\AppData\Local\Microsoft\WindowsApps\winget.EXE"
        p = self.provisioner(monkeypatch, present={"winget": alias})
        monkeypatch.setattr(watchdog, "_git_usable", lambda: False)
        p.mingit = False
        monkeypatch.setattr(p, "_install_mingit", lambda: setattr(p, "mingit", True))

        p._step_git()

        assert p.mingit is True, "the portable git is the whole point of the fallback"
        assert any("could not be started" in m for m in p.logs)

    def test_ffmpeg_falls_back_to_the_static_build(self, monkeypatch):
        alias = r"C:\Users\User\AppData\Local\Microsoft\WindowsApps\winget.EXE"
        p = self.provisioner(monkeypatch, present={"winget": alias})
        p.static = False
        monkeypatch.setattr(p, "_install_static_ffmpeg", lambda: setattr(p, "static", True))

        p._step_ffmpeg()

        assert p.static is True
        assert any("could not be started" in m for m in p.logs)


def test_provision_error_is_an_exception():
    """Callers catch it by type to distinguish setup failure from a crash."""
    assert issubclass(watchdog.ProvisionError, Exception)
    assert isinstance(watchdog.ProvisionError("x"), Exception)


def test_the_module_exposes_what_these_tests_patch():
    """A rename would otherwise make the stubs silently patch nothing."""
    for name in ("_which", "IS_WINDOWS", "ProvisionError", "Provisioner"):
        assert hasattr(watchdog, name), name
    assert isinstance(watchdog.UV_PYTHON_VERSION, str)


class TestNewerReleaseHint:
    """Whether a permanent SETUP dep failure should point at a newer installer.

    SETUP runs on the bootstrapper baked into the executable itself, before any
    source checkout exists — none of the self-repair machinery elsewhere in
    this module (auto-update, _fresh_provisioner) can reach it. A newer release
    is the only fix a stuck machine can actually apply, so this is the one
    failure path responsible for saying so.
    """

    def test_a_transient_failure_gets_no_hint(self, monkeypatch):
        monkeypatch.setattr(watchdog, "read_bundle_version", lambda: "26.3.16")
        monkeypatch.setattr(watchdog.Provisioner, "_latest_release_tag",
                            lambda self: "26.9.1")
        assert watchdog._newer_release_hint("Connection timed out") is None

    def test_a_permanent_failure_with_no_newer_release_gets_no_hint(self, monkeypatch):
        monkeypatch.setattr(watchdog, "read_bundle_version", lambda: "26.9.1")
        monkeypatch.setattr(watchdog.Provisioner, "_latest_release_tag",
                            lambda self: "26.9.1")
        msg = "Build failures usually indicate a problem with the package"
        assert watchdog._newer_release_hint(msg) is None

    def test_a_permanent_failure_with_a_newer_release_points_at_it(self, monkeypatch):
        monkeypatch.setattr(watchdog, "read_bundle_version", lambda: "26.3.16")
        monkeypatch.setattr(watchdog.Provisioner, "_latest_release_tag",
                            lambda self: "v26.9.1")
        msg = "Build failures usually indicate a problem with the package"
        hint = watchdog._newer_release_hint(msg)
        assert hint is not None
        assert "26.9.1" in hint
        assert "releases/latest" in hint

    def test_offline_or_no_releases_gets_no_hint(self, monkeypatch):
        monkeypatch.setattr(watchdog, "read_bundle_version", lambda: "26.3.16")
        monkeypatch.setattr(watchdog.Provisioner, "_latest_release_tag",
                            lambda self: None)
        msg = "Build failures usually indicate a problem with the package"
        assert watchdog._newer_release_hint(msg) is None


class TestSourceFallsBackToTheArchive:
    """git passed the usability probe and still could not clone.

    A proxy that blocks the git protocol, a filtered github.com, a half-installed
    git: the archive path already exists for the git-less case and needs only
    HTTPS, so setup must use it instead of ending on step 6/8.
    """

    def source(self, monkeypatch, tmp_path, *, clone_code):
        p = make_provisioner(monkeypatch, which="/usr/bin/git",
                             run_codes={("git",): clone_code})
        src = tmp_path / "app"
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(src))
        monkeypatch.setattr(watchdog, "STT_SCRIPT", str(src / "speech_to_text.py"))
        monkeypatch.setattr(watchdog, "_git_usable", lambda: True)
        monkeypatch.setattr(p, "_pin_latest_release", lambda: None)

        # A clone that "succeeds" leaves a checkout behind; _step_source wipes
        # non-git leftovers before cloning, so the file has to appear here.
        recorded_run = p._run

        def run_with_clone_effect(cmd, **kw):
            code = recorded_run(cmd, **kw)
            if "clone" in cmd:
                src.mkdir(parents=True, exist_ok=True)
                (src / "speech_to_text.py").write_text("# app\n", encoding="utf-8")
            return code

        p._run = run_with_clone_effect

        def fake_zipball():
            p.zipball = True
            src.mkdir(parents=True, exist_ok=True)
            (src / "speech_to_text.py").write_text("# app\n", encoding="utf-8")

        p.zipball = False
        monkeypatch.setattr(p, "_fetch_source_zipball", fake_zipball)
        return p

    def test_a_failed_clone_fetches_the_archive(self, monkeypatch, tmp_path):
        p = self.source(monkeypatch, tmp_path, clone_code=128)
        p._step_source()
        assert p.zipball is True
        assert any("falling back to the source archive" in m for m in p.logs)

    def test_a_successful_clone_does_not(self, monkeypatch, tmp_path):
        p = self.source(monkeypatch, tmp_path, clone_code=0)
        p._step_source()
        assert p.zipball is False, "the archive is a fallback, not a second download"

    def test_an_archive_that_produces_nothing_still_fails(self, monkeypatch, tmp_path):
        """The fallback must not turn a broken install into a silent success."""
        p = self.source(monkeypatch, tmp_path, clone_code=128)
        monkeypatch.setattr(p, "_fetch_source_zipball", lambda: None)
        with pytest.raises(watchdog.ProvisionError, match=r"speech_to_text\.py"):
            p._step_source()


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

    def test_a_git_that_cannot_be_spawned_stays_on_the_branch(self, monkeypatch):
        """This call bypasses _run, so an OSError here used to crash setup."""
        p = make_provisioner(monkeypatch, which="/usr/bin/git")

        def raise_oserror(cmd, **kw):
            raise OSError(22, "The file cannot be accessed by the system")

        monkeypatch.setattr(watchdog.subprocess, "run", raise_oserror)
        p._pin_latest_release()
        assert [c for c in p.runs if "reset" in c] == []
        assert any("staying on default branch" in m for m in p.logs)

    def test_a_shallow_clone_trusts_the_tag(self, monkeypatch):
        # Fails open: a shallow clone has no history to walk, and refusing every
        # tag there would leave every install on the default branch.
        _p, resets, seen = self.pin(monkeypatch, ["26.2.158"], on_branch=(), shallow=True)
        assert resets == ["26.2.158"]
        assert "checked" not in seen, "no reachability check is possible on a shallow clone"


class TestUnsupportedPlatform:
    """A Mac that is not Apple Silicon cannot be provisioned at all.

    PyTorch published its last macOS x86_64 wheels with 2.2.2, so the pinned
    torch never resolves there and uv fails with "no wheels with a matching
    platform tag" — the field report that produced this gate had retried that
    same resolution every ten minutes, thirteen times and counting.
    """

    def _mac(self, monkeypatch, *, hardware_arm64, process="arm64", macos="15.1"):
        monkeypatch.setattr(watchdog.sys, "platform", "darwin")
        monkeypatch.setattr(watchdog.platform, "machine", lambda: process)
        monkeypatch.setattr(watchdog.platform, "mac_ver", lambda: (macos, ("", "", ""), process))
        monkeypatch.setattr(watchdog, "_mac_hardware_is_arm64",
                            lambda: hardware_arm64)

    def test_intel_mac_is_refused(self, monkeypatch):
        self._mac(monkeypatch, hardware_arm64=False, process="x86_64")
        reason = watchdog._unsupported_platform_reason()
        assert reason and "Intel" in reason
        assert "Apple Silicon" in reason

    def test_apple_silicon_is_supported(self, monkeypatch):
        self._mac(monkeypatch, hardware_arm64=True, process="arm64")
        assert watchdog._unsupported_platform_reason() is None

    def test_an_intel_build_under_rosetta_gets_its_own_message(self, monkeypatch):
        # The hardware is fine; the wrong build was installed. Telling this
        # operator to buy an Apple Silicon Mac would be wrong — they have one.
        self._mac(monkeypatch, hardware_arm64=True, process="x86_64")
        reason = watchdog._unsupported_platform_reason()
        assert reason and "Rosetta" in reason
        assert "arm64" in reason
        # A different diagnosis from the Intel-hardware one, not the same text.
        self._mac(monkeypatch, hardware_arm64=False, process="x86_64")
        assert reason != watchdog._unsupported_platform_reason()

    @pytest.mark.parametrize("macos", ["10.15.7", "11.0", "11.7.10"])
    def test_an_apple_silicon_mac_on_an_old_macos_is_refused(self, monkeypatch, macos):
        """The failure that produced this gate: scipy and scikit-learn have
        never published an arm64 wheel tagged below macosx_12_0 for the Python
        we install, so uv found nothing to install and nothing to fall back to.
        It built llvmlite from source instead and failed for two hours, saying
        "build error" rather than "your macOS is too old"."""
        self._mac(monkeypatch, hardware_arm64=True, macos=macos)
        reason = watchdog._unsupported_platform_reason()
        assert reason and "macOS" in reason
        # The advice must be the one that works: updating is free, and every
        # Apple Silicon Mac can take it. Not "buy a new computer".
        assert "Software Update" in reason
        assert "Intel" not in reason

    @pytest.mark.parametrize("macos", ["12.0", "13.6.1", "15.1", "26.0"])
    def test_a_supported_macos_passes(self, monkeypatch, macos):
        self._mac(monkeypatch, hardware_arm64=True, macos=macos)
        assert watchdog._unsupported_platform_reason() is None

    def test_an_unreadable_macos_version_is_not_a_refusal(self, monkeypatch):
        """A failed lookup must not lock anyone out — let setup run and report
        what it actually hits."""
        self._mac(monkeypatch, hardware_arm64=True, macos="")
        assert watchdog._unsupported_platform_reason() is None
        monkeypatch.setattr(watchdog.platform, "mac_ver",
                            lambda: (_ for _ in ()).throw(OSError("no")))
        assert watchdog._macos_major() is None

    def test_the_old_macos_message_is_distinct_from_the_intel_one(self, monkeypatch):
        self._mac(monkeypatch, hardware_arm64=True, macos="11.0")
        old_os = watchdog._unsupported_platform_reason()
        self._mac(monkeypatch, hardware_arm64=False, process="x86_64", macos="11.0")
        assert old_os != watchdog._unsupported_platform_reason()

    @pytest.mark.parametrize("plat,machine", [
        ("linux", "x86_64"), ("linux", "aarch64"), ("win32", "AMD64"),
    ])
    def test_other_platforms_are_never_gated(self, monkeypatch, plat, machine):
        monkeypatch.setattr(watchdog.sys, "platform", plat)
        monkeypatch.setattr(watchdog.platform, "machine", lambda: machine)
        assert watchdog._unsupported_platform_reason() is None

    def test_sysctl_failure_falls_back_to_the_process_arch(self, monkeypatch):
        def boom(*a, **kw):
            raise OSError("sysctl: not found")

        monkeypatch.setattr(watchdog.subprocess, "run", boom)
        monkeypatch.setattr(watchdog.platform, "machine", lambda: "x86_64")
        assert watchdog._mac_hardware_is_arm64() is False
        monkeypatch.setattr(watchdog.platform, "machine", lambda: "arm64")
        assert watchdog._mac_hardware_is_arm64() is True

    def test_sysctl_reports_the_hardware(self, monkeypatch):
        class Result:
            def __init__(self, out):
                self.returncode, self.stdout = 0, out

        monkeypatch.setattr(watchdog.subprocess, "run", lambda *a, **kw: Result("1\n"))
        monkeypatch.setattr(watchdog.platform, "machine", lambda: "x86_64")
        assert watchdog._mac_hardware_is_arm64() is True

    def test_run_aborts_before_any_step(self, monkeypatch):
        p = make_provisioner(monkeypatch, which=None)
        self._mac(monkeypatch, hardware_arm64=False, process="x86_64")
        with pytest.raises(watchdog.UnsupportedPlatformError):
            p.run()
        assert p.runs == []       # nothing was executed
        assert p.downloaded == []  # nothing was downloaded
        assert p.logs == []        # not even a "[1/8]" line

    def test_the_updater_path_is_gated_too(self, monkeypatch):
        p = make_provisioner(monkeypatch, which=None)
        self._mac(monkeypatch, hardware_arm64=False, process="x86_64")
        with pytest.raises(watchdog.UnsupportedPlatformError):
            p.install_deps_only()
        assert p.runs == []

    def test_it_is_a_provision_error(self, monkeypatch):
        # Existing handlers catch ProvisionError; this must not slip past them
        # as an unhandled crash.
        assert issubclass(watchdog.UnsupportedPlatformError, watchdog.ProvisionError)


class TestPermanentDependencyFailures:
    """Which uv failures a retry can fix, and which it cannot.

    Both shapes below came off real installs. A resolver dead end ("no solution
    found") and a source build that will not compile are equally hopeless on
    the next attempt; a download that was cut off is not.
    """

    @pytest.mark.parametrize("message", [
        # The llvmlite sdist build, from a macOS 11 Apple Silicon Mac whose
        # platform tag no longer matched any published wheel.
        "command failed (1): uv pip install -r requirements.txt — last output: "
        "returned non-zero exit status 1. | hint: Build failures usually indicate "
        "a problem with the package or the build environment",
        "command failed (1): uv pip install — last output: "
        "× No solution found when resolving dependencies",
        "command failed (1): uv pip install — last output: "
        "no wheels with a matching platform tag",
        "command failed (2): uv pip install — last output: "
        "your requirements are unsatisfiable",
        "command failed (1): uv pip install — last output: Failed to build `llvmlite`",
    ])
    def test_a_hopeless_install_is_recognised(self, message):
        assert watchdog._is_permanent_dep_failure(message) is True

    @pytest.mark.parametrize("message", [
        "command failed (1): uv pip install — last output: "
        "error sending request for url (https://pypi.org/simple/torch/)",
        "timed out after 7200s: uv pip install -r requirements.txt",
        "command failed (1): uv pip install — last output: connection reset by peer",
        "",
        None,
    ])
    def test_a_transient_failure_is_not(self, message):
        assert watchdog._is_permanent_dep_failure(message) is False

    def test_resolution_failures_remain_a_narrower_case(self):
        """_is_resolution_failure means "nothing was installed, the venv is
        untouched", which the updater relies on to skip a repair reinstall. A
        build failure is permanent but *did* reach the venv, so the two
        classifiers must not be collapsed into one."""
        build = "returned non-zero exit status 1. | hint: Build failures usually indicate"
        assert watchdog._is_permanent_dep_failure(build) is True
        assert watchdog._is_resolution_failure(build) is False


class TestProvisionFingerprint:
    """One incident must be one Sentry issue.

    The message ends in the last five lines of uv output, which name a
    per-run temporary build directory — so the raw message put every attempt on
    every machine in a group of its own.
    """

    def test_the_output_tail_does_not_reach_the_fingerprint(self):
        base = "command failed (1): uv pip install -r requirements.txt"
        one = watchdog._provision_fingerprint(
            watchdog.ProvisionError(base + " — last output: /tmp/.tmpG19Y73/bin/python"))
        two = watchdog._provision_fingerprint(
            watchdog.ProvisionError(base + " — last output: /tmp/.tmp0cvvGI/bin/python"))
        assert one == two

    def test_home_directories_do_not_reach_the_fingerprint(self):
        """Otherwise two machines failing identically are two issues, and the
        operator's username is the thing telling them apart."""
        one = watchdog._provision_fingerprint(
            watchdog.ProvisionError("command failed (1): /Users/xmedia/.local/bin/uv pip install"))
        two = watchdog._provision_fingerprint(
            watchdog.ProvisionError("command failed (1): /Users/pastudios/.local/bin/uv pip install"))
        assert one == two
        assert not any("xmedia" in part for part in one)

    def test_different_failures_stay_apart(self):
        build = watchdog._provision_fingerprint(watchdog.ProvisionError(
            "command failed (1): uv pip install — last output: "
            "hint: Build failures usually indicate a problem"))
        blip = watchdog._provision_fingerprint(watchdog.ProvisionError(
            "command failed (1): uv pip install — last output: connection reset"))
        assert build != blip

    def test_the_exception_type_is_part_of_the_key(self):
        message = "cannot run here"
        assert watchdog._provision_fingerprint(watchdog.ProvisionError(message)) != \
            watchdog._provision_fingerprint(watchdog.UnsupportedPlatformError(message))


class TestStaticFfmpegMirrors:
    """gyan.dev answered a Windows first run with 503 and setup failed, though
    BtbN's GitHub build of the same thing was up."""

    def provisioner(self, monkeypatch, tmp_path, *, failing=()):
        p = make_provisioner(monkeypatch)
        monkeypatch.setattr(watchdog, "IS_WINDOWS", True)
        monkeypatch.setattr(watchdog.sys, "platform", "win32")
        monkeypatch.setattr(watchdog, "_FFMPEG_BIN_DIR", str(tmp_path / "bin"))
        p.installed = False

        def download(url, dest):
            p.downloaded.append(url)
            if url in failing:
                raise watchdog.urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)

        p._download_file = download
        p._extract_ffmpeg = lambda archive, dest: setattr(p, "installed", True)
        monkeypatch.setattr(watchdog, "_which", lambda name: "ffmpeg.exe" if p.installed else None)
        return p

    def test_a_mirror_answering_503_falls_through_to_the_next(self, monkeypatch, tmp_path):
        first, second, *_ = watchdog._STATIC_FFMPEG_URLS["win32"]
        p = self.provisioner(monkeypatch, tmp_path, failing={first})

        p._install_static_ffmpeg()

        assert p.downloaded == [first, second]
        assert p.installed
        assert any("503" in m for m in p.logs)

    def test_the_first_mirror_is_enough_when_it_works(self, monkeypatch, tmp_path):
        p = self.provisioner(monkeypatch, tmp_path)

        p._install_static_ffmpeg()

        assert p.downloaded == [watchdog._STATIC_FFMPEG_URLS["win32"][0]]

    def test_every_mirror_down_is_one_provision_error_naming_them(self, monkeypatch, tmp_path):
        urls = watchdog._STATIC_FFMPEG_URLS["win32"]
        p = self.provisioner(monkeypatch, tmp_path, failing=set(urls))

        with pytest.raises(watchdog.ProvisionError) as info:
            p._install_static_ffmpeg()

        assert all(url in str(info.value) for url in urls)

    def test_every_platform_with_a_source_has_a_backup(self):
        # macOS had one source; when evermeet.cx went 410, every Mac setup failed.
        for platform in ("linux", "win32", "darwin"):
            assert len(watchdog._STATIC_FFMPEG_URLS[platform]) >= 2, platform

    def test_every_platform_ends_on_a_host_that_keeps_its_files(self):
        # Every macOS source of 26.3.119 was a single host; one 410 stranded it.
        for platform in ("win32", "darwin"):
            assert watchdog._STATIC_FFMPEG_URLS[platform][-1].startswith("https://files.pythonhosted.org/"), platform

    def test_macos_sources_are_arm64(self):
        # Apple Silicon only: an x86_64 binary needs Rosetta, which a fresh Mac lacks.
        assert all("arm64" in url for url in watchdog._STATIC_FFMPEG_URLS["darwin"])


class TestExtractFfmpeg:
    """Each archive shape a source ships lands as a runnable binary named ffmpeg."""

    def test_a_bare_gzip_is_the_binary_itself(self, monkeypatch, tmp_path):
        import gzip
        monkeypatch.setattr(watchdog, "IS_WINDOWS", False)
        archive = tmp_path / "ffmpeg-darwin-arm64.gz"
        archive.write_bytes(gzip.compress(b"binary"))
        dest = tmp_path / "bin"
        dest.mkdir()

        make_provisioner(monkeypatch)._extract_ffmpeg(str(archive), str(dest))

        assert (dest / "ffmpeg").read_bytes() == b"binary"
        assert os.access(dest / "ffmpeg", os.X_OK)

    def test_a_zip_with_ffmpeg_at_its_root(self, monkeypatch, tmp_path):
        import zipfile
        monkeypatch.setattr(watchdog, "IS_WINDOWS", False)
        archive = tmp_path / "ffmpeg.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ffmpeg", b"binary")
        dest = tmp_path / "bin"
        dest.mkdir()

        make_provisioner(monkeypatch)._extract_ffmpeg(str(archive), str(dest))

        assert (dest / "ffmpeg").read_bytes() == b"binary"

    @pytest.mark.parametrize(("windows", "member", "name"), [
        (False, "imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1", "ffmpeg"),
        (True, "imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe", "ffmpeg.exe"),
    ])
    def test_a_pypi_wheel_yields_its_versioned_binary(self, monkeypatch, tmp_path, windows, member, name):
        import zipfile
        monkeypatch.setattr(watchdog, "IS_WINDOWS", windows)
        archive = tmp_path / "imageio_ffmpeg-0.6.0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("imageio_ffmpeg/binaries/README.md", b"readme")
            zf.writestr(member, b"binary")
        dest = tmp_path / "bin"
        dest.mkdir()

        make_provisioner(monkeypatch)._extract_ffmpeg(str(archive), str(dest))

        assert sorted(os.listdir(dest)) == [name]
        assert (dest / name).read_bytes() == b"binary"

    def test_a_tar_gz_is_not_taken_for_a_bare_gzip(self, monkeypatch, tmp_path):
        import tarfile
        monkeypatch.setattr(watchdog, "IS_WINDOWS", False)
        payload = tmp_path / "ffmpeg"
        payload.write_bytes(b"binary")
        archive = tmp_path / "ffmpeg-static.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname="ffmpeg-7.0-static/ffmpeg")
        dest = tmp_path / "bin"
        dest.mkdir()

        make_provisioner(monkeypatch)._extract_ffmpeg(str(archive), str(dest))

        assert (dest / "ffmpeg").read_bytes() == b"binary"


class TestStagedDownload:
    """A transfer that dies midway must not leave `dest` looking complete."""

    class _Body(io.BytesIO):
        """A response that hands over its bytes, or drops the connection."""

        def __init__(self, fail):
            super().__init__(b"archive-bytes")
            self.fail = fail

        def read(self, *args):
            if self.fail:
                raise OSError("connection reset")
            return super().read(*args)

    def run(self, monkeypatch, tmp_path, *, fail):
        body = self._Body(fail)
        monkeypatch.setattr(watchdog.urllib.request, "urlopen", lambda *a, **k: body)
        p = make_provisioner(monkeypatch)
        dest = tmp_path / "ffmpeg.zip"
        watchdog.Provisioner._download_file(p, "https://example.invalid/ffmpeg.zip", str(dest))
        return dest

    def test_a_complete_transfer_lands_at_dest(self, monkeypatch, tmp_path):
        dest = self.run(monkeypatch, tmp_path, fail=False)
        assert dest.read_bytes() == b"archive-bytes"
        assert not (tmp_path / "ffmpeg.zip.part").exists()

    def test_an_interrupted_transfer_leaves_nothing(self, monkeypatch, tmp_path):
        with pytest.raises(OSError):
            self.run(monkeypatch, tmp_path, fail=True)
        assert list(tmp_path.iterdir()) == []


class TestDiskSpace:
    """How much free space setup asks for, and what it says when there isn't any.

    A flat 8 GB refused a CPU-only Windows machine (STT-2Q: 4.5 GB free) that the
    CPU build of torch fits in, and let a GPU machine with 8-9 GB start an install
    that needs ~9 GB before any model.
    """

    GB = 1024 ** 3

    def run_step(self, monkeypatch, tmp_path, *, free_gb, nvidia, plat="win32", machine="AMD64"):
        p = make_provisioner(monkeypatch)
        p._has_nvidia = lambda: nvidia
        monkeypatch.setattr(watchdog, "DATA_DIR", str(tmp_path / ".stt"))
        monkeypatch.setattr(watchdog.sys, "platform", plat)
        monkeypatch.setattr(watchdog.platform, "machine", lambda: machine)
        monkeypatch.setattr(watchdog.shutil, "disk_usage",
                            lambda path: type("U", (), {"free": int(free_gb * self.GB)})())
        p._step_disk()
        return p

    def test_a_cpu_machine_with_room_for_the_cpu_build_passes(self, monkeypatch, tmp_path):
        """The STT-2Q machine: 4.5 GB free, no NVIDIA card."""
        self.run_step(monkeypatch, tmp_path, free_gb=4.5, nvidia=False)

    def test_a_cpu_machine_below_the_cpu_need_is_refused(self, monkeypatch, tmp_path):
        with pytest.raises(watchdog.InsufficientDiskError):
            self.run_step(monkeypatch, tmp_path, free_gb=2.0, nvidia=False)

    def test_a_gpu_machine_needs_room_for_the_cuda_build(self, monkeypatch, tmp_path):
        """8 GB used to pass, then ran out partway through installing ~9 GB."""
        with pytest.raises(watchdog.InsufficientDiskError):
            self.run_step(monkeypatch, tmp_path, free_gb=8.5, nvidia=True)

    def test_a_gpu_machine_with_room_passes(self, monkeypatch, tmp_path):
        self.run_step(monkeypatch, tmp_path, free_gb=20, nvidia=True)

    def test_linux_x86_without_a_card_still_gets_the_cuda_build(self, monkeypatch, tmp_path):
        """PyPI's Linux x86_64 torch pulls the nvidia-* wheels with or without a GPU."""
        with pytest.raises(watchdog.InsufficientDiskError):
            self.run_step(monkeypatch, tmp_path, free_gb=6, nvidia=False, plat="linux", machine="x86_64")

    @pytest.mark.parametrize("plat,machine", [("darwin", "arm64"), ("linux", "aarch64"), ("win32", "AMD64")])
    def test_cpu_torch_platforms(self, plat, machine):
        assert not watchdog._installs_cuda_torch(False, plat, machine)

    def test_an_unreadable_disk_is_not_a_refusal(self, monkeypatch, tmp_path):
        p = make_provisioner(monkeypatch)
        p._has_nvidia = lambda: False
        monkeypatch.setattr(watchdog, "DATA_DIR", str(tmp_path / ".stt"))

        def boom(path):
            raise OSError("unreadable")
        monkeypatch.setattr(watchdog.shutil, "disk_usage", boom)
        p._step_disk()

    def test_a_first_run_measures_the_drive_the_data_dir_will_be_on(self, monkeypatch, tmp_path):
        seen = []
        p = make_provisioner(monkeypatch)
        p._has_nvidia = lambda: False
        monkeypatch.setattr(watchdog, "DATA_DIR", str(tmp_path / ".stt"))
        monkeypatch.setattr(watchdog.shutil, "disk_usage",
                            lambda path: seen.append(path) or type("U", (), {"free": 100 * self.GB})())
        p._step_disk()
        assert seen == [str(tmp_path)]

    def test_the_message_says_what_to_free_and_where(self):
        msg = watchdog._disk_shortfall_message(4.5, 12.0, "C:", cuda=True)
        assert "Free at least 7.5 GB on C:" in msg
        assert "Retry" in msg and "GPU" in msg

    def test_windows_names_the_drive(self):
        assert watchdog._disk_label(r"C:\Users\x\.stt", "win32") == "C:"

    def test_a_full_disk_is_still_a_provision_error(self):
        """Every existing handler catches ProvisionError; this must not slip past them."""
        assert issubclass(watchdog.InsufficientDiskError, watchdog.ProvisionError)
        assert not issubclass(watchdog.InsufficientDiskError, watchdog.UnsupportedPlatformError)


class TestDiskSpaceReporting:
    def test_every_machine_groups_into_one_issue(self):
        a = watchdog.InsufficientDiskError(watchdog._disk_shortfall_message(4.5, 12, "C:", True))
        b = watchdog.InsufficientDiskError(watchdog._disk_shortfall_message(1.2, 4, "D:", False))
        assert watchdog._provision_fingerprint(a) == watchdog._provision_fingerprint(b)

    def test_it_is_reported_as_a_warning(self):
        assert watchdog._provision_level(watchdog.InsufficientDiskError("x")) == "warning"

    def test_other_failures_keep_their_level(self):
        assert watchdog._provision_level(watchdog.ProvisionError("x")) is None

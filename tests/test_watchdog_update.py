"""AutoUpdater._apply_update: zip-slip rejection, success path, swap rollback.

Runs the real update code against a file:// zipball in an isolated temp
SOURCE_DIR/DATA_DIR, with the process manager, state, and dependency
installer stubbed out — the closest thing to an end-to-end updater test
that doesn't need GitHub or a service restart.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt.watchdog as watchdog  # noqa: E402  (imports cleanly: stdlib + optional certifi only)


class StubPM:
    def __init__(self):
        self.calls = []

    def stop(self, timeout=None):
        self.calls.append("stop")

    def start(self):
        self.calls.append("start")


class StubState:
    def __init__(self):
        self.values = {}

    def set(self, **kwargs):
        self.values.update(kwargs)


@pytest.fixture
def updater(tmp_path, monkeypatch):
    source_dir = tmp_path / "app"
    data_dir = tmp_path / "data"
    source_dir.mkdir()
    data_dir.mkdir()
    (source_dir / "old.txt").write_text("old content")
    monkeypatch.setattr(watchdog, "SOURCE_DIR", str(source_dir))
    monkeypatch.setattr(watchdog, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(watchdog, "VERSION_FILE", str(source_dir / "VERSION"))
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        lambda self, log=None: None)

    upd = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
    upd.pm = StubPM()
    upd.state = StubState()
    return upd, source_dir


def make_zipball(path, files):
    """GitHub-style source zip: one top-level directory."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(f"repo-abc123/{name}", content)
    return path.as_uri()


def test_successful_update_applies_files_and_version(updater, tmp_path):
    upd, source_dir = updater
    url = make_zipball(tmp_path / "u.zip", {"new.txt": "new content", "old.txt": "updated"})

    upd._apply_update("9.9.9", url)

    assert (source_dir / "new.txt").read_text(encoding="utf-8") == "new content"
    assert (source_dir / "old.txt").read_text(encoding="utf-8") == "updated"
    assert watchdog.read_version() == "9.9.9"
    assert "Updated to 9.9.9" in upd.state.values["last_update_result"]
    assert upd.pm.calls == ["stop", "start"], "app must be stopped for the swap and restarted after"


def test_zip_slip_is_rejected_before_stopping_the_app(updater, tmp_path):
    upd, source_dir = updater
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("repo-abc/ok.txt", "fine")
        zf.writestr("../escape.txt", "evil")

    upd._apply_update("9.9.9", zip_path.as_uri())

    assert not (tmp_path / "escape.txt").exists()
    assert (source_dir / "old.txt").read_text(encoding="utf-8") == "old content", "no files may change"
    assert watchdog.read_version() != "9.9.9", "version must not be recorded"
    assert "failed" in upd.state.values["last_update_result"].lower()
    assert upd.pm.calls == [], "rejected before the app is ever stopped"


def test_swap_failure_restores_previous_version(updater, tmp_path, monkeypatch):
    upd, source_dir = updater
    url = make_zipball(tmp_path / "u.zip",
                       {"a.txt": "a", "boom.txt": "b", "old.txt": "updated"})

    real_move = shutil.move

    def failing_move(src, dst, *a, **kw):
        # Fail only when placing the staged "boom.txt" into SOURCE (not on
        # backup parking or rollback restores, which move other paths).
        if os.path.basename(str(src)) == "boom.txt" and "extracted" in str(src):
            raise OSError("disk full")
        return real_move(src, dst, *a, **kw)

    monkeypatch.setattr(watchdog.shutil, "move", failing_move)

    upd._apply_update("9.9.9", url)

    assert (source_dir / "old.txt").read_text(encoding="utf-8") == "old content", "old version must be restored"
    assert not (source_dir / "a.txt").exists(), "partially applied items must be rolled back"
    assert not (source_dir / "boom.txt").exists()
    assert watchdog.read_version() != "9.9.9", "failed update must not record the new version"
    assert "restored" in upd.state.values["last_update_result"].lower()
    assert upd.pm.calls == ["stop", "start"], "app must still be restarted after a failed swap"


def test_dep_install_failure_restores_previous_version(updater, tmp_path, monkeypatch):
    upd, source_dir = updater
    url = make_zipball(tmp_path / "u.zip", {"old.txt": "updated", "new.txt": "n"})

    def boom(self, log=None):
        raise watchdog.ProvisionError("uv exploded")

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", boom)

    upd._apply_update("9.9.9", url)

    assert (source_dir / "old.txt").read_text(encoding="utf-8") == "old content", "old version must be restored"
    assert not (source_dir / "new.txt").exists()
    assert watchdog.read_version() != "9.9.9"
    assert upd.pm.calls == ["stop", "start"]


def test_preserved_venv_is_not_touched(updater, tmp_path):
    upd, source_dir = updater
    venv = source_dir / ".venv"
    venv.mkdir()
    (venv / "marker.txt").write_text("keep me")
    url = make_zipball(tmp_path / "u.zip", {".venv/evil.txt": "nope", "a.txt": "a"})

    upd._apply_update("9.9.9", url)

    assert (venv / "marker.txt").read_text(encoding="utf-8") == "keep me"
    assert not (venv / "evil.txt").exists(), ".venv content from the archive must be skipped"
    assert (source_dir / "a.txt").exists()
    assert watchdog.read_version() == "9.9.9"


# --- 'main' channel: branch-tracking detection and apply ---------------------
# Real bare origin + clone (pattern from test_self_update.py) so the actual
# git fetch/reset/rollback code runs; PM, state, and deps stay stubbed.

import subprocess  # noqa: E402

needs_git = pytest.mark.skipif(not shutil.which("git"), reason="git not available")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def git_updater(tmp_path, monkeypatch):
    """(updater, seed, clone): clone is the managed SOURCE_DIR checkout."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True, text=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(origin), str(seed)],
                   check=True, capture_output=True, text=True)
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    (seed / "file.txt").write_text("v1\n")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "push", "-u", "origin", "main")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)],
                   check=True, capture_output=True, text=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(watchdog, "SOURCE_DIR", str(clone))
    monkeypatch.setattr(watchdog, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(watchdog, "VERSION_FILE", str(clone / "VERSION"))
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        lambda self, log=None: None)

    upd = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
    upd.pm = StubPM()
    upd.state = StubState()
    upd._pending_update = None
    return upd, seed, clone


def _advance_origin(seed, content="v2\n"):
    (seed / "file.txt").write_text(content)
    _git(seed, "commit", "-am", "advance")
    _git(seed, "push", "origin", "main")


@needs_git
def test_main_channel_up_to_date(git_updater):
    upd, _seed, _clone = git_updater

    upd._check_for_branch_update()

    assert upd._pending_update is None
    assert upd.state.values["last_update_result"].startswith("Up to date (main @")


@needs_git
def test_main_channel_detects_and_applies_update(git_updater):
    upd, seed, clone = git_updater
    _advance_origin(seed)

    upd._check_for_branch_update()
    assert upd._pending_update == (watchdog.AutoUpdater._BRANCH_TARGET, None, {})
    assert "Update available: main @" in upd.state.values["last_update_result"]

    upd.apply_pending_update()

    assert (clone / "file.txt").read_text(encoding="utf-8") == "v2\n"
    assert _head(clone) == _head(seed)
    assert upd.pm.calls == ["stop", "start"]
    assert not (clone / "VERSION").exists(), "branch mode must not write the VERSION file"
    assert "Updated to" in upd.state.values["last_update_result"]


@needs_git
def test_main_channel_rollback_on_dep_failure(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    prev = _head(clone)
    _advance_origin(seed)
    upd._check_for_branch_update()

    calls = {"n": 0}

    def flaky_deps(self, log=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("dep fail")

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", flaky_deps)
    upd.apply_pending_update()

    assert _head(clone) == prev, "failed dep install must roll the checkout back"
    assert "rolled back" in upd.state.values["last_update_result"]
    assert upd.pm.calls == ["stop", "start"]


@needs_git
def test_check_for_update_defaults_to_main_channel(git_updater, monkeypatch):
    upd, seed, _clone = git_updater
    monkeypatch.setattr(watchdog, "load_config", lambda: {})  # no channel configured
    _advance_origin(seed)

    upd.check_for_update()  # must route to branch flow, no Releases API call

    assert upd._pending_update == (watchdog.AutoUpdater._BRANCH_TARGET, None, {})


@needs_git
def test_stable_channel_still_uses_releases(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    _advance_origin(seed)
    monkeypatch.setattr(watchdog, "load_config",
                        lambda: {"watchdog": {"update_channel": "stable"}})
    monkeypatch.setattr(upd, "get_latest_release", lambda channel: (None, None, {}))

    upd.check_for_update()

    assert upd._pending_update is None
    assert upd.state.values["last_update_result"] == "No releases yet"
    assert _head(clone) != _head(seed), "stable channel must not track main"


# --- permanently unsupported hardware ----------------------------------------
# An Intel Mac can never resolve requirements.txt (PyTorch's last macOS x86_64
# wheel was 2.2.2), so an update attempt is a guaranteed stop/reset/fail/roll
# back/restart cycle. The updater must not start one.

_INTEL = ("This Mac has an Intel processor. STT needs PyTorch, which no longer "
          "publishes Intel-Mac builds, so its dependencies cannot be installed here.")


@pytest.fixture
def intel(monkeypatch):
    monkeypatch.setattr(watchdog, "_unsupported_platform_reason", lambda: _INTEL)
    monkeypatch.setattr(watchdog, "load_config", lambda: {})
    monkeypatch.setattr(watchdog, "_sentry_capture", lambda e: None)


@needs_git
def test_unsupported_platform_pauses_updates(git_updater, intel):
    upd, seed, clone = git_updater
    prev = _head(clone)
    _advance_origin(seed)

    upd.check_for_update()

    assert upd._pending_update is None
    assert upd.state.values["last_update_result"].startswith("Updates paused")
    assert upd.pm.calls == [], "the server must not be stopped for an update that cannot install"
    assert _head(clone) == prev, "nothing may be fetched or reset"


@needs_git
def test_apply_pending_update_refuses_on_an_unsupported_platform(git_updater, intel):
    # The control-channel command and the GUI's Update Now apply what a previous
    # check left pending, so the apply path carries the guard too.
    upd, seed, clone = git_updater
    prev = _head(clone)
    _advance_origin(seed)
    upd._pending_update = (watchdog.AutoUpdater._BRANCH_TARGET, None, {})

    upd.apply_pending_update()

    assert upd.pm.calls == []
    assert _head(clone) == prev


@needs_git
def test_the_pause_is_reported_once(git_updater, intel, monkeypatch):
    upd, _seed, _clone = git_updater
    captured = []
    monkeypatch.setattr(watchdog, "_sentry_capture", captured.append)

    for _ in range(3):
        upd.check_for_update()

    assert len(captured) == 1, "an hourly check must not file an hourly report"
    assert isinstance(captured[0], watchdog.UnsupportedPlatformError)
    assert "Intel" in str(captured[0])


# --- provisioning failure reporting ------------------------------------------

import types  # noqa: E402


def test_provisioning_failure_is_captured_to_sentry(monkeypatch):
    captured = []
    fake = types.ModuleType("sentry_sdk")
    fake.capture_exception = lambda e: captured.append(e)
    fake.flush = lambda timeout=None: captured.append("flushed")
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

    boom = watchdog.ProvisionError("step 3/8 failed: uv not found")

    def raise_boom(self):
        raise boom

    monkeypatch.setattr(watchdog.Provisioner, "run", raise_boom)

    # max_attempts bounds the retry loop (headless retries forever by default
    # so unattended kiosks self-heal); sentry must be captured exactly once.
    assert watchdog._run_provisioning_headless(max_attempts=1) is False
    assert captured == [boom, "flushed"], "failure must be captured then flushed"


def test_sentry_capture_without_sdk_does_not_raise(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # import raises ImportError
    watchdog._sentry_capture(Exception("x"))  # must be a silent no-op


# --- git-less installs: the default 'main' channel must still update ---------
# A zipball-provisioned SOURCE (no .git) used to log "Main channel needs a git
# checkout" forever, so a headless box that cannot have git silently froze on
# the version it was installed with.


def test_gitless_main_channel_falls_back_to_releases(updater, monkeypatch):
    upd, source_dir = updater  # fixture SOURCE_DIR has no .git
    upd._pending_update = None
    (source_dir / "VERSION").write_text("1.0.0\n")
    monkeypatch.setattr(watchdog, "load_config",
                        lambda: {"watchdog": {"update_channel": "main"}})
    monkeypatch.setattr(upd, "_check_for_branch_update",
                        lambda: pytest.fail("branch tracking needs a git checkout"))
    seen = {}

    def fake_release(channel):
        seen["channel"] = channel
        return "v1.2.3", "https://example.invalid/z.zip", {}

    monkeypatch.setattr(upd, "get_latest_release", fake_release)

    upd.check_for_update()

    assert seen["channel"] == "stable"
    assert upd._pending_update == ("1.2.3", "https://example.invalid/z.zip", {})


def test_gitless_main_channel_applies_the_release(updater, tmp_path, monkeypatch):
    upd, source_dir = updater
    upd._pending_update = None
    (source_dir / "VERSION").write_text("1.0.0\n")
    url = make_zipball(tmp_path / "u.zip", {"old.txt": "updated", "VERSION": "1.2.3\n"})
    monkeypatch.setattr(watchdog, "load_config",
                        lambda: {"watchdog": {"update_channel": "main"}})
    monkeypatch.setattr(upd, "get_latest_release", lambda channel: ("v1.2.3", url, {}))

    upd.check_and_update()

    assert (source_dir / "old.txt").read_text(encoding="utf-8") == "updated"
    assert watchdog.read_version() == "1.2.3"
    assert upd.pm.calls == ["stop", "start"]


@needs_git
def test_main_channel_keeps_branch_tracking_with_a_checkout(git_updater, monkeypatch):
    upd, seed, _clone = git_updater
    monkeypatch.setattr(watchdog, "load_config",
                        lambda: {"watchdog": {"update_channel": "main"}})
    monkeypatch.setattr(upd, "get_latest_release",
                        lambda channel: pytest.fail("must not call the Releases API"))
    _advance_origin(seed)

    upd.check_for_update()

    assert upd._pending_update == (watchdog.AutoUpdater._BRANCH_TARGET, None, {})


# --- archive updates must refresh config templates --------------------------
# config/ is never swapped wholesale (live settings live in DATA_DIR, and a
# source install may keep files here), but its tracked *.default.json templates
# have to move forward or new settings never get their defaults.


def test_archive_update_refreshes_config_templates(updater, tmp_path):
    upd, source_dir = updater
    cfg = source_dir / "config"
    cfg.mkdir()
    (cfg / "config.default.json").write_text('{"old": true}')
    (cfg / "config.json").write_text('{"user": "settings"}')
    url = make_zipball(tmp_path / "u.zip", {
        "config/config.default.json": '{"new": true}',
        "config/service_phases.default.json": '{"added": true}',
        "a.txt": "a",
    })

    upd._apply_update("9.9.9", url)

    assert (cfg / "config.default.json").read_text(encoding="utf-8") == '{"new": true}'
    assert (cfg / "service_phases.default.json").read_text(encoding="utf-8") == '{"added": true}'
    assert (cfg / "config.json").read_text(encoding="utf-8") == '{"user": "settings"}', \
        "anything not a template must survive the swap"
    assert watchdog.read_version() == "9.9.9"


def test_config_template_rollback_on_failed_update(updater, tmp_path, monkeypatch):
    upd, source_dir = updater
    cfg = source_dir / "config"
    cfg.mkdir()
    (cfg / "config.default.json").write_text('{"old": true}')
    url = make_zipball(tmp_path / "u.zip", {
        "config/config.default.json": '{"new": true}',
        "old.txt": "updated",
    })

    def boom(self, log=None):
        raise watchdog.ProvisionError("uv exploded")

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", boom)

    upd._apply_update("9.9.9", url)

    assert (cfg / "config.default.json").read_text(encoding="utf-8") == '{"old": true}'
    assert (source_dir / "old.txt").read_text(encoding="utf-8") == "old content"
    assert watchdog.read_version() != "9.9.9"


# --- a dependency install that failed once ------------------------------------
# A dependency failure is usually a property of the machine, not of the moment
# (the reported case: a macOS x86_64 venv against a torch pin with no Intel-Mac
# wheel). Retrying the same commit on every boot and every 1am costs a
# stop/reset/rollback/restart each time and can never succeed.

_UV_RESOLUTION_FAILURE = (
    "command failed (1): uv pip install -r requirements.txt — last output: "
    "× No solution found when resolving dependencies: | ╰─▶ Because torch==2.8.0 "
    "has no wheels with a matching platform tag (e.g., `macosx_26_0_x86_64`) and "
    "you require torch==2.8.0, we can conclude that your requirements are "
    "unsatisfiable."
)


def _branch_update(upd):
    """One full 'main'-channel update cycle: check, then apply."""
    upd._check_for_branch_update()
    upd.apply_pending_update()


def _fail_deps(message):
    def deps(self, log=None):
        raise watchdog.ProvisionError(message)
    return deps


@needs_git
def test_a_failed_commit_is_not_retried_on_the_next_launch(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    prev = _head(clone)
    _advance_origin(seed)
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        _fail_deps(_UV_RESOLUTION_FAILURE))

    _branch_update(upd)
    assert _head(clone) == prev
    assert upd.pm.calls == ["stop", "start"], "the first attempt is a real attempt"

    upd.pm.calls.clear()
    _branch_update(upd)

    assert upd.pm.calls == [], "the second attempt must not stop the server"
    assert _head(clone) == prev
    assert "skipped" in upd.state.values["last_update_result"]


@needs_git
def test_a_newer_commit_clears_the_block(git_updater, monkeypatch):
    upd, seed, clone = git_updater
    _advance_origin(seed)
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        _fail_deps(_UV_RESOLUTION_FAILURE))
    _branch_update(upd)

    # The next release is exactly what might carry the fix, so it must be tried.
    _advance_origin(seed, "v3\n")
    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only",
                        lambda self, log=None: None)
    upd.pm.calls.clear()
    upd._dep_block_reported = False

    _branch_update(upd)

    assert _head(clone) == _head(seed)
    assert (clone / "file.txt").read_text(encoding="utf-8") == "v3\n"
    assert upd.pm.calls == ["stop", "start"]
    assert not (Path(watchdog.DATA_DIR) / ".dep-failure").exists(), \
        "a clean install must not leave the next update blocked"


@needs_git
def test_a_resolution_failure_skips_the_repair_reinstall(git_updater, monkeypatch):
    # uv that fails while *resolving* never reaches the venv, so there is nothing
    # to repair — and the repair fails the same way a moment later, reporting a
    # second error for one cause.
    upd, seed, _clone = git_updater
    _advance_origin(seed)
    attempts = []

    def deps(self, log=None):
        attempts.append(1)
        raise watchdog.ProvisionError(_UV_RESOLUTION_FAILURE)

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", deps)
    _branch_update(upd)

    assert len(attempts) == 1, "the venv was never touched; do not reinstall over it"
    assert upd.pm.calls == ["stop", "start"], "the server still comes back up"


@needs_git
def test_a_mid_install_failure_still_repairs_the_venv(git_updater, monkeypatch):
    # The opposite case: uv resolved and was replacing packages when it died, so
    # the venv may now be half-way between two versions and must be reinstalled.
    upd, seed, _clone = git_updater
    _advance_origin(seed)
    attempts = []

    def deps(self, log=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise watchdog.ProvisionError("command failed (1): uv — last output: "
                                          "error: failed to unpack wheel: disk full")

    monkeypatch.setattr(watchdog.Provisioner, "install_deps_only", deps)
    _branch_update(upd)

    assert len(attempts) == 2, "a venv that may be half-updated must be reinstalled"


@pytest.mark.parametrize("message", [
    "× No solution found when resolving dependencies",
    "torch==2.8.0 has no wheels with a matching platform tag",
    "your requirements are unsatisfiable",
])
def test_resolution_failures_are_recognised(message):
    assert watchdog._is_resolution_failure(message)


@pytest.mark.parametrize("message", [
    "error: failed to unpack wheel: disk full",
    "Connection reset by peer",
    "",
])
def test_other_failures_are_not_mistaken_for_resolution_failures(message):
    assert not watchdog._is_resolution_failure(message)

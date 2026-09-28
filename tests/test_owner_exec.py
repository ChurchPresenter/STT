"""Running git and uv as the checkout's owner under a root service (stt/owner_exec.py).

Nothing here needs root: whether the process is root is a parameter, the passwd lookup
and the runner are injected, and chown is recorded rather than performed.
"""

import os
import subprocess
from types import SimpleNamespace

import pytest

from stt import owner_exec
from stt.owner_exec import Owner, foreign_owner, owner_popen_kwargs, reclaim, run

ME = os.getuid() if hasattr(os, "getuid") else 1000
AI = Owner(ME, os.getgid() if hasattr(os, "getgid") else 1000, "ai", "/home/ai")

posix = pytest.mark.skipif(not hasattr(os, "lchown"), reason="POSIX ownership only")


def result(code, out=""):
    return subprocess.CompletedProcess(["x"], code, out, "")


class Recorder:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


@pytest.fixture(autouse=True)
def fresh_reclaim_cache():
    owner_exec._reclaimed.clear()
    yield
    owner_exec._reclaimed.clear()


class TestForeignOwner:
    def test_not_root_runs_as_itself(self, tmp_path):
        assert foreign_owner(str(tmp_path), euid=1000, lookup=lambda u: AI) is None

    @posix
    def test_root_in_a_users_directory_finds_the_user(self, tmp_path):
        # tmp_path belongs to whoever runs the tests, which is never root in CI.
        if ME == 0:
            pytest.skip("needs a non-root owner")
        assert foreign_owner(str(tmp_path), euid=0, lookup=lambda u: AI) == AI

    @posix
    def test_a_root_owned_directory_stays_root(self):
        assert foreign_owner("/", euid=0, lookup=lambda u: AI) is None

    @posix
    def test_a_data_dir_root_created_is_judged_by_its_home(self, tmp_path, monkeypatch):
        data = tmp_path / ".stt"
        data.mkdir()
        real = os.stat

        def fake_stat(p, *a, **kw):
            st = real(p, *a, **kw)
            if os.path.abspath(p) == str(data):
                return SimpleNamespace(st_uid=0)
            return SimpleNamespace(st_uid=4242 if os.path.abspath(p) == str(tmp_path) else st.st_uid)
        monkeypatch.setattr(owner_exec.os, "stat", fake_stat)
        seen = []
        foreign_owner(str(data), euid=0, via_parent=True, lookup=lambda u: seen.append(u) or AI)
        assert seen == [4242]
        # Without via_parent the root-created dir is taken at its word.
        assert foreign_owner(str(data), euid=0, lookup=lambda u: AI) is None

    @posix
    def test_never_reaches_up_to_the_filesystem_root(self):
        assert foreign_owner("/tmp-owner-exec-missing", euid=0, via_parent=True,
                             lookup=lambda u: AI) is None


class TestRun:
    def test_no_foreign_owner_is_a_plain_call(self):
        rec = Recorder(result(0))
        run(["git", "status"], owned_by="/x", owner=None, runner=rec, capture_output=True)
        # foreign_owner is consulted for real here; the test process is not root.
        assert rec.calls == [{"capture_output": True}]

    def test_success_as_the_owner_is_the_answer(self):
        rec = Recorder(result(0, "ok"))
        r = run(["git", "status"], owned_by="/x", owner=AI, runner=rec, text=True)
        assert r.stdout == "ok" and len(rec.calls) == 1
        kw = rec.calls[0]
        assert kw["user"] == AI.uid and kw["group"] == AI.gid
        assert kw["env"]["HOME"] == "/home/ai" and kw["env"]["USER"] == "ai"
        assert kw["text"] is True

    def test_failure_as_the_owner_falls_back_to_root(self):
        rec = Recorder(result(1), result(0, "root"))
        r = run(["git", "pull"], owned_by="/x", owner=AI, runner=rec)
        assert r.stdout == "root"
        assert "user" not in rec.calls[1] and "env" not in rec.calls[1]

    def test_when_both_fail_the_owners_error_is_kept(self):
        # Root's error is usually git refusing a repo it does not own, which would hide
        # the real reason a pull was rejected.
        rec = Recorder(result(1, "diverged"), result(128, "dubious ownership"))
        assert run(["git", "pull"], owned_by="/x", owner=AI, runner=rec).stdout == "diverged"

    def test_a_child_that_cannot_drop_privileges_falls_back(self):
        rec = Recorder(PermissionError(1, "Operation not permitted"), result(0, "root"))
        assert run(["uv", "pip"], owned_by="/x", owner=AI, runner=rec).stdout == "root"

    def test_a_timeout_is_not_retried(self):
        rec = Recorder(subprocess.TimeoutExpired("git", 5))
        with pytest.raises(subprocess.TimeoutExpired):
            run(["git", "fetch"], owned_by="/x", owner=AI, runner=rec)
        assert len(rec.calls) == 1

    def test_the_named_dirs_are_reclaimed_first(self, monkeypatch):
        seen = []
        monkeypatch.setattr(owner_exec, "reclaim", lambda d, o: seen.append((d, o)))
        run(["git", "status"], owned_by="/x", owner=AI, reclaim_dirs=("/repo",),
            runner=Recorder(result(0)))
        assert seen == [("/repo", AI)]

    def test_a_callers_env_is_kept_under_the_owners_home(self):
        rec = Recorder(result(0))
        run(["uv"], owned_by="/x", owner=AI, runner=rec, env={"PATH": "/opt/bin", "HOME": "/root"})
        assert rec.calls[0]["env"] == {"PATH": "/opt/bin", "HOME": "/home/ai",
                                       "USER": "ai", "LOGNAME": "ai"}


@posix
class TestReclaim:
    def tree(self, tmp_path):
        (tmp_path / ".git" / "objects" / "ab").mkdir(parents=True)
        (tmp_path / ".git" / "objects" / "ab" / "cdef").write_text("x")
        (tmp_path / "link").symlink_to("/etc/passwd")
        return tmp_path

    def test_everything_not_the_owners_is_handed_back(self, tmp_path, monkeypatch):
        root = self.tree(tmp_path)
        changed = []
        monkeypatch.setattr(owner_exec.os, "lchown", lambda p, u, g: changed.append(p))
        other = AI._replace(uid=AI.uid + 1)  # everything on disk looks foreign to it
        n = reclaim(str(root), other)
        assert n == len(changed) == 6  # root, .git, objects, ab, cdef, link
        assert str(root / "link") in changed  # the link itself, never /etc/passwd

    def test_what_the_owner_already_has_is_left_alone(self, tmp_path, monkeypatch):
        root = self.tree(tmp_path)
        monkeypatch.setattr(owner_exec.os, "lchown", lambda *a: pytest.fail("chowned"))
        assert reclaim(str(root), AI) == 0

    def test_each_path_is_walked_once_per_process(self, tmp_path, monkeypatch):
        root = self.tree(tmp_path)
        monkeypatch.setattr(owner_exec.os, "lchown", lambda *a: None)
        other = AI._replace(uid=AI.uid + 1)
        assert reclaim(str(root), other) > 0
        assert reclaim(str(root), other) == 0

    def test_a_missing_path_is_nothing_to_do(self, tmp_path):
        assert reclaim(str(tmp_path / "nope"), AI) == 0


def test_popen_kwargs_carry_the_owners_groups():
    kw = owner_popen_kwargs(AI)
    assert kw["user"] == AI.uid and kw["group"] == AI.gid
    if hasattr(os, "getgrouplist"):
        assert isinstance(kw.get("extra_groups", []), list)


class TestCallers:
    """The three places that write into the checkout go through it."""

    def test_self_update_git_runs_as_the_owner(self, tmp_path, monkeypatch):
        from stt import self_update
        monkeypatch.setattr(owner_exec, "foreign_owner", lambda *a, **k: AI)
        monkeypatch.setattr(owner_exec, "reclaim", lambda *a, **k: 0)
        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(kw) or result(0))
        self_update._git(str(tmp_path), "status")
        assert calls and calls[0].get("user") == AI.uid

    def test_watchdog_updater_git_falls_back_before_checking(self, monkeypatch):
        from stt import watchdog
        monkeypatch.setattr(owner_exec, "foreign_owner", lambda *a, **k: AI)
        monkeypatch.setattr(owner_exec, "reclaim", lambda *a, **k: 0)
        outcomes = [result(1), result(0, "root")]
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: outcomes.pop(0))
        updater = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
        # check=True must judge the root retry, not the owner's failed first attempt.
        assert updater._git("fetch", check=True).stdout == "root"

    def test_watchdog_provisioner_retries_git_as_root(self, monkeypatch, tmp_path):
        from stt import watchdog
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(tmp_path))
        monkeypatch.setattr(owner_exec, "foreign_owner", lambda *a, **k: AI)
        monkeypatch.setattr(owner_exec, "reclaim", lambda *a, **k: 0)
        monkeypatch.setattr(watchdog, "_which", lambda name: None)
        attempts = []

        class FakeProc:
            def __init__(self, cmd, **kw):
                attempts.append(kw.get("user"))
                self.stdout = iter([])
                self._code = 1 if kw.get("user") is not None else 0

            def wait(self):
                return self._code

        monkeypatch.setattr(watchdog.subprocess, "Popen", FakeProc)
        p = watchdog.Provisioner.__new__(watchdog.Provisioner)
        p.log = lambda msg: None
        assert p._run(["git", "status"]) == 0
        assert attempts == [AI.uid, None]

    def test_watchdog_provisioner_leaves_other_tools_alone(self, monkeypatch, tmp_path):
        from stt import watchdog
        monkeypatch.setattr(watchdog, "SOURCE_DIR", str(tmp_path))
        monkeypatch.setattr(owner_exec, "foreign_owner", lambda *a, **k: pytest.fail("asked"))
        monkeypatch.setattr(watchdog, "_which", lambda name: None)

        class FakeProc:
            def __init__(self, cmd, **kw):
                assert "user" not in kw
                self.stdout = iter([])

            def wait(self):
                return 0

        monkeypatch.setattr(watchdog.subprocess, "Popen", FakeProc)
        p = watchdog.Provisioner.__new__(watchdog.Provisioner)
        p.log = lambda msg: None
        assert p._run(["sudo", "-n", "apt-get", "install", "git"]) == 0

"""stt.optional_deps: does the venv have what the live config asks it to do?

The bug this module exists for: a rebuilt venv dropped llama-cpp-python, which
requirements.txt deliberately omits, so nothing noticed. The config still said
provider="local", the LLM never loaded, and fallback="skip" returned every caption
untranslated with HTTP 200. These tests pin the decision (which deps a config asks
for) and the startup contract (never raise, never block, never retry a known-failing
build on every restart).

No venv, no network, no uv: the install is a stubbed subprocess.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt.optional_deps as od  # noqa: E402
import stt.self_update as self_update  # noqa: E402


def _config(method="llm", provider="local"):
    return {"live_translation": {"translation_method": method, "llm": {"provider": provider}}}


def _venv(tmp_path):
    """A venv skeleton with an interpreter file where the module expects one."""
    bindir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    exe = bindir / ("python.exe" if os.name == "nt" else "python3")
    exe.write_text("")
    return exe


# ─── which deps a config asks for ────────────────────────────────────

def test_local_llm_provider_asks_for_llama_cpp():
    wanted = od.wanted_deps(_config(provider="local"))
    assert [d.module for d in wanted] == ["llama_cpp"]


@pytest.mark.parametrize("method,provider", [
    ("llm", "endpoint"),   # the model belongs to another machine
    ("nllb", "local"),     # NMT path; the llm block is inert
    ("madlad", "local"),
])
def test_other_translation_setups_ask_for_nothing(method, provider):
    assert od.wanted_deps(_config(method, provider)) == []


def test_absent_and_malformed_config_ask_for_nothing():
    assert od.wanted_deps({}) == []
    # A config whose shape breaks the predicate must not take the server down with it.
    assert od.wanted_deps({"live_translation": "not-a-mapping"}) == []


def test_missing_deps_reports_only_what_is_absent():
    config = _config()
    assert missing_names(config, installed=False) == ["llama_cpp"]
    assert missing_names(config, installed=True) == []


def missing_names(config, installed):
    return [d.module for d in od.missing_deps(config, lambda _m: installed)]


# ─── skip marker ─────────────────────────────────────────────────────

def test_skip_marker_roundtrips_and_ignores_comments():
    text = od.format_skip_marker({"llama-cpp-python>=0.3.0", "other==1.0"})
    assert text.splitlines()[0].startswith("#")
    assert od.parse_skip_marker(text) == {"llama-cpp-python>=0.3.0", "other==1.0"}
    assert od.parse_skip_marker("") == set()


# ─── config reading ──────────────────────────────────────────────────

def test_read_config_survives_missing_and_malformed(tmp_path):
    assert od.read_config(str(tmp_path)) == {}          # no file at all
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "config.json").write_text("{not json")
    assert od.read_config(str(tmp_path)) == {}
    (cfgdir / "config.json").write_text(json.dumps(_config()))
    assert od.read_config(str(tmp_path)) == _config()


def test_read_config_rejects_non_object_json(tmp_path):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "config.json").write_text("[1, 2, 3]")
    assert od.read_config(str(tmp_path)) == {}


# ─── install command ─────────────────────────────────────────────────

def test_install_command_targets_the_venv_interpreter():
    cmd = od.install_command("/bin/uv", "/srv/.venv/bin/python3", "llama-cpp-python>=0.3.0")
    assert cmd == ["/bin/uv", "pip", "install", "--python",
                   "/srv/.venv/bin/python3", "llama-cpp-python>=0.3.0"]


# ─── ensure() ────────────────────────────────────────────────────────

@pytest.fixture
def uv(monkeypatch, tmp_path):
    """A findable uv, and a subprocess.run that records calls instead of running them."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, fake_run.returncode, "", fake_run.stderr)

    fake_run.returncode = 0
    fake_run.stderr = ""
    monkeypatch.setattr(self_update, "find_uv", lambda _d: "/bin/uv")
    monkeypatch.setattr(od.subprocess, "run", fake_run)
    return fake_run, calls


def _data_dir(tmp_path, config):
    d = tmp_path / "data"
    (d / "config").mkdir(parents=True)
    (d / "config" / "config.json").write_text(json.dumps(config))
    return str(d)


def test_nothing_missing_installs_nothing(tmp_path, uv):
    _fake, calls = uv
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config(provider="endpoint"))
    assert od.ensure(str(tmp_path), data, echo=lambda _m: None) == 0
    assert calls == []


def test_missing_dep_is_installed(tmp_path, uv, monkeypatch):
    _fake, calls = uv
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config())
    assert od.ensure(str(tmp_path), data, echo=lambda _m: None) == 1
    assert len(calls) == 1
    assert calls[0][:3] == ["/bin/uv", "pip", "install"]
    assert calls[0][-1] == "llama-cpp-python>=0.3.0"
    # A clean run leaves no skip marker behind.
    assert not (tmp_path / ".venv" / od.SKIP_MARKER_NAME).exists()


def test_no_venv_is_a_no_op(tmp_path, uv, monkeypatch):
    _fake, calls = uv
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    data = _data_dir(tmp_path, _config())   # no .venv created
    assert od.ensure(str(tmp_path), data, echo=lambda _m: None) == 0
    assert calls == []


def test_failed_install_is_recorded_and_not_retried(tmp_path, uv, monkeypatch):
    fake, calls = uv
    fake.returncode = 1
    fake.stderr = "no matching distribution"
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config())

    assert od.ensure(str(tmp_path), data, echo=lambda _m: None) == 0
    marker = tmp_path / ".venv" / od.SKIP_MARKER_NAME
    assert od.parse_skip_marker(marker.read_text()) == {"llama-cpp-python>=0.3.0"}

    # The whole point of the marker: the next restart does not pay for it again.
    calls.clear()
    assert od.ensure(str(tmp_path), data, echo=lambda _m: None) == 0
    assert calls == []


def test_clearing_the_marker_allows_a_retry(tmp_path, uv, monkeypatch):
    fake, calls = uv
    fake.returncode = 1
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config())
    od.ensure(str(tmp_path), data, echo=lambda _m: None)

    marker = tmp_path / ".venv" / od.SKIP_MARKER_NAME
    marker.unlink()
    fake.returncode = 0
    calls.clear()
    assert od.ensure(str(tmp_path), data, echo=lambda _m: None) == 1
    assert len(calls) == 1


def test_a_later_success_clears_a_stale_marker(tmp_path, uv, monkeypatch):
    fake, _calls = uv
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config())
    marker = tmp_path / ".venv" / od.SKIP_MARKER_NAME
    marker.write_text(od.format_skip_marker({"some-other-package"}))

    fake.returncode = 0
    od.ensure(str(tmp_path), data, echo=lambda _m: None)
    # The dep we installed is gone from the marker; the unrelated entry survives.
    assert od.parse_skip_marker(marker.read_text()) == {"some-other-package"}


def test_missing_uv_reports_the_manual_command(tmp_path, monkeypatch):
    monkeypatch.setattr(self_update, "find_uv", lambda _d: "")
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config())
    messages = []
    assert od.ensure(str(tmp_path), data, echo=messages.append) == 0
    assert any("uv pip install" in m for m in messages)


def test_a_timeout_does_not_propagate(tmp_path, monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 900)

    monkeypatch.setattr(self_update, "find_uv", lambda _d: "/bin/uv")
    monkeypatch.setattr(od.subprocess, "run", boom)
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    _venv(tmp_path)
    data = _data_dir(tmp_path, _config())
    messages = []
    assert od.ensure(str(tmp_path), data, timeout=900, echo=messages.append) == 0
    assert any("900" in m for m in messages)


# ─── CLI ─────────────────────────────────────────────────────────────

def test_check_exits_nonzero_when_a_dep_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(od, "module_installed", lambda _n: False)
    data = _data_dir(tmp_path, _config())
    assert od.main(["--check", "--data-dir", data]) == 1
    assert "llama-cpp-python" in capsys.readouterr().out


def test_check_exits_zero_when_satisfied(tmp_path, monkeypatch):
    monkeypatch.setattr(od, "module_installed", lambda _n: True)
    data = _data_dir(tmp_path, _config())
    assert od.main(["--check", "--data-dir", data]) == 0


def test_ensure_mode_always_exits_zero(tmp_path, monkeypatch):
    """The startup contract: a dependency problem never stops the server coming up."""
    def boom(*a, **k):
        raise RuntimeError("uv exploded")

    monkeypatch.setattr(od, "ensure", boom)
    data = _data_dir(tmp_path, _config())
    assert od.main(["--repo-dir", str(tmp_path), "--data-dir", data]) == 0


# ─── the live probe ──────────────────────────────────────────────────
#
# The counterpart to the startup check: what a long-running server does when the
# package goes missing (or comes back) under it. Written against the real incident —
# 124 captions of a live service went out untranslated, one log line each, and the
# only thing that noticed was a human reading server.log the next day.


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _probe(answers, clock=None, **kw):
    """A probe over a list of answers, one consumed per actual filesystem check."""
    seen = {"invalidated": 0}
    calls = iter(answers)
    last = [answers[-1]]

    def installed(_name):
        try:
            last[0] = next(calls)
        except StopIteration:
            pass
        return last[0]

    p = od.RuntimeProbe("pretend_module", clock=clock or _Clock(),
                        is_installed=installed,
                        invalidate=lambda: seen.__setitem__("invalidated",
                                                            seen["invalidated"] + 1),
                        **kw)
    return p, seen


def test_present_module_is_probed_once_and_then_trusted():
    """An importable module does not stop being importable; re-probing is pure cost."""
    probe, seen = _probe([True, False, False])
    assert [probe.available() for _ in range(4)] == [True, True, True, True]
    assert seen["invalidated"] == 0
    assert probe.take_report() is None


def test_absence_is_reported_once_not_once_per_caption():
    probe, _ = _probe([False])
    assert probe.available() is False
    assert probe.take_report() == ("missing", 0)
    for _ in range(50):
        probe.available()
    assert probe.take_report() is None


def test_absence_is_not_re_probed_before_the_retry_window():
    clock = _Clock()
    probe, seen = _probe([False, True], clock=clock, retry_seconds=30)
    assert probe.available() is False
    clock.now = 29.0
    assert probe.available() is False       # still inside the window: no filesystem work
    assert seen["invalidated"] == 0


def test_a_package_installed_while_running_is_picked_up():
    """The whole reason for invalidate_caches: find_spec cannot see it otherwise."""
    clock = _Clock()
    probe, seen = _probe([False, True], clock=clock, retry_seconds=30)
    assert probe.available() is False
    clock.now = 31.0
    assert probe.available() is True
    assert seen["invalidated"] == 1


def test_recovery_reports_how_many_calls_were_served_degraded():
    clock = _Clock()
    probe, _ = _probe([False, True], clock=clock, retry_seconds=30)
    for _ in range(7):
        probe.available()                   # one real check, then six cached refusals
    assert probe.take_report() == ("missing", 0)
    assert probe.degraded_calls() == 7
    clock.now = 31.0
    assert probe.available() is True
    assert probe.take_report() == ("restored", 7)
    assert probe.degraded_calls() == 0


def test_a_probe_that_raises_reads_as_absent_rather_than_propagating():
    """It is asked once per caption; raising there would take the caption with it."""
    def boom(_name):
        raise RuntimeError("broken finder")

    probe = od.RuntimeProbe("pretend_module", is_installed=boom, invalidate=lambda: None)
    assert probe.available() is False
    assert probe.take_report() == ("missing", 0)


def test_invalidate_failure_does_not_break_the_probe():
    clock = _Clock()

    def boom():
        raise RuntimeError("cache is stuck")

    answers = iter([False, True])
    probe = od.RuntimeProbe("pretend_module", clock=clock, retry_seconds=1,
                            is_installed=lambda _n: next(answers), invalidate=boom)
    assert probe.available() is False
    clock.now = 2.0
    assert probe.available() is True


def test_the_real_probe_agrees_with_module_installed():
    """Defaults wired to the real thing: a stdlib module is present, a nonsense one is not."""
    assert od.RuntimeProbe("json").available() is True
    assert od.RuntimeProbe("stt_module_that_does_not_exist").available() is False

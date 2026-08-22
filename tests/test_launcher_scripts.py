"""The launcher and installer scripts, asserted by reading them.

Nothing executes these: CI runs on Windows but only ruff/mypy/pytest
(.github/workflows/test.yml), and the packaged-installer job builds Inno Setup rather
than running install.ps1. That blind spot is where a fresh-install report found four
separate bugs, each of which had been shipping silently:

* stop_server.bat compared against the literal text "!errorlevel!" for want of
  EnableDelayedExpansion, so it killed nothing and said "[OK] Server stopped."
* its window-title fallback matched a title the launchers never set
* every launcher read config/config.json from the checkout, where it does not exist —
  the live file is in the data dir — so the port was always the 8080 fallback
* the installers told the operator to edit that same nonexistent checkout config, and
  to open port 80 when the shipped default is 8080

Each is a one-line regression away from returning, and none of them fails a test suite
that only runs Python. So the scripts are pinned as text.
"""

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def read(name):
    return (REPO / name).read_text(encoding="utf-8")


LAUNCHERS = ["start_server.bat", "restart_server.bat", "start_server.sh",
             "restart_server.sh", "stop_server.sh"]


class TestPortComesFromTheDataDir:
    """The live config is in STT_DATA_DIR or ~/.stt, never in the checkout."""

    @pytest.mark.parametrize("name", LAUNCHERS)
    def test_the_data_dir_is_resolved(self, name):
        body = read(name)
        assert "STT_DATA_DIR" in body, f"{name} must resolve the data dir the app uses"
        assert ".stt" in body

    @pytest.mark.parametrize("name", LAUNCHERS)
    def test_the_checkout_config_is_not_read(self, name):
        # The exact form of the old bug: a path relative to the repo, which is
        # gitignored except for the templates and so is never present.
        assert "open('config/config.json')" not in read(name)

    @pytest.mark.parametrize("name", LAUNCHERS)
    def test_the_port_is_still_read_from_the_right_key(self, name):
        # It was never the key that was wrong, only the path — don't "fix" that too.
        assert "'web_server'" in read(name)


class TestStopServerBat:
    def test_delayed_expansion_is_enabled(self):
        # Without this every "if !errorlevel!" inside a for loop is a string compare
        # against "!errorlevel!", which is never equal, so the kill never runs.
        body = read("stop_server.bat")
        head = "\n".join(body.splitlines()[:3]).lower()
        assert "setlocal enabledelayedexpansion" in head

    def test_wmic_is_not_relied_on(self):
        # Removed from Windows 11 24H2: the command-line lookup silently matched
        # nothing there, so the delayed-expansion fix alone would still be a no-op.
        # Comments may name it — explaining why it went is worth keeping — so this
        # reads the executable lines only.
        for name in ("stop_server.bat", "restart_server.bat"):
            code = [line for line in read(name).lower().splitlines()
                    if not line.strip().startswith("rem")]
            assert not any("wmic" in line for line in code), f"{name} still calls wmic"

    def test_the_success_message_is_not_unconditional(self):
        # It used to print "[OK] Server stopped." whether or not anything was killed,
        # which is how the bug survived: the script reported success every time.
        body = read("stop_server.bat")
        assert "No server process was running" in body


class TestWindowTitle:
    TITLE = "STT Server"

    @pytest.mark.parametrize("name", ["start_server.bat", "restart_server.bat"])
    def test_the_launcher_sets_a_real_title(self, name):
        assert f'start "{self.TITLE}"' in read(name), \
            'start "" leaves the title as the exe path, which no filter can match'

    @pytest.mark.parametrize("name", ["stop_server.bat", "restart_server.bat"])
    def test_the_stop_filter_matches_that_title(self, name):
        assert f'WINDOWTITLE eq {self.TITLE}*' in read(name)


class TestInstallerText:
    """What the installer tells the operator has to be true, or it costs them hours."""

    @pytest.mark.parametrize("name", ["install.ps1", "install.sh"])
    def test_no_localhost_80(self, name):
        # The shipped default port is 8080; 80 needs privileges and is not what a
        # stock install binds.
        assert "localhost:80\"" not in read(name)
        assert "localhost:80 " not in read(name)

    @pytest.mark.parametrize("name", ["install.ps1", "install.sh"])
    def test_the_named_config_is_the_one_the_server_reads(self, name):
        body = read(name)
        assert "$INSTALL_DIR/config/config.json" not in body
        assert "$INSTALL_DIR\\config\\config.json" not in body
        assert ".stt" in body and "config.json" in body

    def test_the_shipped_port_matches_what_the_installers_print(self):
        import json
        with open(REPO / "config" / "config.default.json", encoding="utf-8") as fh:
            port = json.load(fh)["web_server"]["port"]
        assert port == 8080
        for name in ("install.ps1", "install.sh"):
            assert f"localhost:{port}" in read(name)


class TestInstallShRefusesIntelMacs:
    """PyTorch ships no macOS x86_64 wheels after 2.2.2, so the pinned torch in
    requirements.txt cannot resolve on an Intel Mac. The check has to sit in
    detect_os: detect_gpu runs from install_python_deps, by which point Python,
    ffmpeg and a venv are already installed for a machine that can never run."""

    def test_the_arch_gate_exits(self):
        body = read("install.sh")
        detect_os = body.split("detect_os() {", 1)[1].split("\n}\n", 1)[0]
        assert 'if [ "$ARCH" != "arm64" ]' in detect_os
        assert "exit 1" in detect_os.split('!= "arm64"', 1)[1]

    def test_the_message_names_the_supported_hardware(self):
        body = read("install.sh")
        assert "Apple Silicon Mac (M1 or newer)" in body

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


def code_lines(name):
    """The executable lines of a .bat: comments may name what was removed and why."""
    return [ln for ln in read(name).splitlines() if not ln.strip().lower().startswith("rem")]


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
    def test_the_port_comes_from_the_shared_module(self, name):
        # stt/server_port.py reads web_server.port; a private one-liner per script is how
        # five copies each got the sudo case wrong in the same way.
        assert "-m stt.server_port" in read(name)

    @pytest.mark.parametrize("name", LAUNCHERS)
    def test_home_is_not_guessed_inline(self, name):
        # expanduser('~') under sudo is /root: a box serving port 80 as its own user was
        # reported, and port-killed, as 8080.
        assert "expanduser('~')" not in read(name)


class TestSudo:
    """The Linux scripts run as root; the server they manage usually does not."""

    def test_a_sudo_started_server_keeps_the_users_data_dir(self):
        body = read("start_server.sh")
        sudo_lines = [ln for ln in body.splitlines() if ln.strip().startswith("sudo ") and "speech_to_text.py" in ln]
        assert sudo_lines and all('STT_DATA_DIR="$DATA_DIR"' in ln for ln in sudo_lines)
        # STT_DATA_DIR alone means "watchdog-managed" to the server, which would switch
        # off its self-update and make Restart exit for a watchdog that is not there.
        assert all("STT_MANAGED=0" in ln for ln in sudo_lines)

    def test_a_root_fallback_start_keeps_the_users_data_dir(self):
        starts = [ln for ln in read("restart_server.sh").splitlines()
                  if "nohup" in ln and "speech_to_text.py" in ln]
        assert starts and all('STT_DATA_DIR="$DATA_DIR"' in ln and "STT_MANAGED=0" in ln for ln in starts)

    @pytest.mark.parametrize("name", ["restart_server.sh", "stop_server.sh"])
    def test_root_owned_files_are_handed_back(self, name):
        body = read(name)
        assert 'chown -R "$SUDO_USER' in body
        # Only inside the invoking user's home — never a chown of an arbitrary override.
        assert "pw_dir" in body

    @pytest.mark.parametrize("name", ["start_server.sh", "restart_server.sh"])
    def test_the_dependency_check_reads_the_same_config(self, name):
        assert '--data-dir "$DATA_DIR"' in read(name)


class TestStartedMessage:
    @pytest.mark.parametrize("name", ["start_server.sh", "restart_server.sh"])
    def test_the_port_is_claimed_only_once_it_answers(self, name):
        body = read(name)
        ok = [ln for ln in body.splitlines() if "[OK]" in ln and "Server started" in ln]
        # The only [OK] lines left are the two inside report_started, after its probe.
        assert len(ok) == 2, ok
        probe = body.index('curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/"')
        assert all(body.index(ln) > probe or "expected on port" in ln for ln in ok)


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


class TestRootNeverOwnsTheCheckout:
    """A root-run update or service leaves root-owned files the user's updater cannot touch."""

    def test_update_pulls_and_syncs_as_the_checkout_owner(self):
        body = read("update_server.sh")
        assert "as_owner git -C" in body
        assert 'as_owner "$uv_bin" pip install' in body
        assert 'sudo -u "$CHECKOUT_OWNER"' in body

    @pytest.mark.parametrize("name", ["start_server.sh", "restart_server.sh", "stop_server.sh"])
    def test_root_run_helpers_write_no_bytecode(self, name):
        body = read(name)
        helpers = [ln for ln in body.splitlines() if " -m stt." in ln and not ln.lstrip().startswith("#")]
        assert helpers and all(" -B -m stt." in ln for ln in helpers), helpers

    @pytest.mark.parametrize("name", ["install.sh", "deploy/stt-watchdog.service"])
    def test_services_run_as_a_user_with_the_bind_capability(self, name):
        body = read(name)
        assert "User=root" not in body
        assert "AmbientCapabilities=CAP_NET_BIND_SERVICE" in body


class TestWindowsLaunchers:
    def test_update_runs_from_a_copy(self):
        # cmd reads a .bat by byte offset; git pull rewriting it mid-run executes garbage.
        code = code_lines("update_server.bat")
        first_copy = next(i for i, ln in enumerate(code) if "--from-copy" in ln)
        first_pull = next(i for i, ln in enumerate(code) if ln.strip().startswith("git pull"))
        assert first_copy < first_pull

    def test_start_detects_a_running_server_by_command_line(self):
        code = code_lines("start_server.bat")
        assert not any("tasklist" in ln for ln in code)
        assert any("CommandLine -like '*speech_to_text*'" in ln for ln in code)

    def test_restart_success_means_the_port_answers(self):
        body = read("restart_server.bat")
        verify = body.split("Verify started")[1]
        assert "TcpClient" in verify and "findstr /I \"python\"" not in verify

    def test_watchdog_bat_expands_python_at_run_time(self):
        body = read("start_watchdog.bat")
        assert "EnableDelayedExpansion" in body
        assert '"!PYTHON_BIN!"' in body and '"%PYTHON_BIN%"' not in body

    def test_scheduled_task_has_no_time_limit(self):
        # The default ExecutionTimeLimit is 72h: the server was killed after three days.
        assert "-ExecutionTimeLimit ([TimeSpan]::Zero)" in read("install.ps1")


class TestWatchdogLogs:
    """The watchdog writes watchdog.log itself, in ~/.stt/logs, in every mode."""

    @pytest.mark.parametrize("name", ["start_watchdog.sh", "start_watchdog.bat", "start_watchdog.ps1"])
    def test_logs_are_pointed_at_the_data_dir(self, name):
        body = read(name)
        assert ".stt" in body
        assert "$SCRIPT_DIR/logs" not in body and "%SCRIPT_DIR%logs" not in body
        assert "Join-Path $ScriptDir \"logs\"" not in body

    def test_stdout_does_not_duplicate_the_log_file(self):
        body = read("start_watchdog.sh")
        assert '>> "$LOG_DIR/watchdog.log"' not in body
        assert "watchdog.stdout.log" in body

    def test_double_start_check_does_not_need_nc(self):
        body = read("start_watchdog.sh")
        assert "nc -z" not in body and "connect_ex(('127.0.0.1',57337))" in body

    def test_plist_logs_to_a_directory_the_installer_creates(self):
        plist = read("deploy/com.stt.watchdog.plist")
        assert "INSTALL_DIR/logs" not in plist.split("-->")[-1]
        assert "LOG_DIR/watchdog.stdout.log" in plist
        install = read("install.sh")
        assert 'mkdir -p "$LOG_DIR"' in install and 's|LOG_DIR|$LOG_DIR|g' in install


WINDOWS_SCRIPTS = sorted(p.name for p in REPO.glob("*.bat")) + sorted(p.name for p in REPO.glob("*.ps1"))
BATCH = sorted(p.name for p in REPO.glob("*.bat"))


class TestWindowsParsing:
    """Found on a fresh Windows 11 PC: three ways a script died before doing anything."""

    @pytest.mark.parametrize("name", WINDOWS_SCRIPTS)
    def test_ascii_only(self, name):
        # Windows PowerShell 5.1 reads a BOM-less .ps1 as cp1252: an em dash's last byte
        # became a closing quote, install.ps1 failed to parse, and start_watchdog.ps1
        # silently swallowed its else branch. cmd reads .bat in the OEM code page.
        (REPO / name).read_bytes().decode("ascii")

    @pytest.mark.parametrize("name", BATCH)
    def test_no_unescaped_paren_inside_a_block(self, name):
        # cmd parses a whole if (...) block first, so a bare ")" in an echo or rem line
        # ends it early and the script dies at startup, whichever branch would have run.
        depth, bad = 0, []
        for n, line in enumerate(read(name).splitlines(), 1):
            t = line.strip()
            low = t.lower()
            if depth > 0 and (low.startswith("echo") or low.startswith("rem")):
                body = t.replace("^(", "").replace("^)", "")
                if "(" in body or ")" in body:
                    bad.append(f"{n}: {t}")
            opens = 1 if t.endswith("(") and not low.startswith("rem") else 0
            if t.startswith(")"):
                depth -= 1
            depth += opens
        assert not bad, bad

    @pytest.mark.parametrize("name", BATCH)
    def test_no_caret_pipe_inside_a_quoted_powershell_command(self, name):
        # Inside the double quotes cmd already leaves | alone, so ^| reached PowerShell
        # as a literal caret; 2^>nul hid the error and every process lookup returned 0.
        bad = [ln for ln in code_lines(name) if '-Command "' in ln and "^|" in ln]
        assert not bad, bad

    @pytest.mark.parametrize("name", BATCH)
    def test_no_timeout(self, name):
        # timeout exits at once with redirected stdin, as when the server's restart
        # button runs restart_server.bat, so every wait was skipped.
        assert not any("timeout /t" in ln.lower() for ln in code_lines(name))

    def test_restart_probe_is_bounded_by_the_clock(self):
        verify = read("restart_server.bat").split("Verify started")[1]
        assert "Stopwatch" in verify and "TotalSeconds -lt 30" in verify

    def test_the_gpu_name_is_printed_not_returned(self):
        body = read("install.ps1")
        start = body.index("function Detect-Gpu")
        block = body[start:body.index("\n}", start)]
        smi = [ln for ln in block.splitlines() if "--query-gpu=name" in ln]
        assert smi and smi[0].rstrip().endswith("|")
        assert "Write-Host" in block.split("--query-gpu=name")[1].splitlines()[1]


def test_windows_scripts_check_out_with_crlf():
    # cmd can fail to find a goto label in an LF-only batch file (restart_server.bat's
    # wait loop), and a zip download or core.autocrlf=false would otherwise give it one.
    attrs = read(".gitattributes").splitlines()
    assert "*.bat text eol=crlf" in attrs and "*.ps1 text eol=crlf" in attrs


@pytest.mark.parametrize("name", ["stop_server.bat", "restart_server.bat"])
def test_server_lookups_kill_the_whole_tree(name):
    # A server's multiprocessing workers never mention speech_to_text, so killing only
    # the matched process left them running (seven built up on a test PC).
    lookups = [ln for ln in code_lines(name) if "'python.exe'" in ln and "*speech_to_text*" in ln]
    assert lookups and all("taskkill.exe /T /F /PID" in ln for ln in lookups)
    assert not any("Stop-Process" in ln for ln in lookups)


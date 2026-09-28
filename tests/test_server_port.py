"""Which config and port the launch scripts report (stt/server_port.py)."""

import json
import os
import subprocess
import sys

import pytest

from stt.server_port import DEFAULT_PORT, configured_port, data_dir

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_config(root, cfg):
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    with open(os.path.join(root, "config", "config.json"), "w", encoding="utf-8") as f:
        f.write(cfg if isinstance(cfg, str) else json.dumps(cfg))


class TestDataDir:
    def test_the_override_wins(self, tmp_path):
        env = {"STT_DATA_DIR": str(tmp_path), "SUDO_USER": "ai"}
        assert data_dir(env, euid=0, home="/root", home_of=lambda u: "/home/ai") == str(tmp_path)

    def test_a_plain_user_gets_their_own_home(self):
        assert data_dir({}, euid=1000, home="/home/ai") == os.path.join("/home/ai", ".stt")

    def test_under_sudo_it_is_the_invoking_users_config(self):
        # The case that printed 8080 on a box serving port 80: restart_server.sh must be
        # root, so ~ was /root and the file the server actually reads was never opened.
        got = data_dir({"SUDO_USER": "ai"}, euid=0, home="/root", home_of=lambda u: "/home/" + u)
        assert got == os.path.join("/home/ai", ".stt")

    def test_root_without_sudo_is_root(self):
        assert data_dir({}, euid=0, home="/root") == os.path.join("/root", ".stt")

    def test_sudo_user_is_ignored_when_not_root(self):
        # SUDO_USER survives into a shell that sudo -u'd to someone else; only root's
        # own home is the wrong one to read.
        got = data_dir({"SUDO_USER": "ai"}, euid=1001, home="/home/other",
                       home_of=lambda u: "/home/ai")
        assert got == os.path.join("/home/other", ".stt")

    def test_an_unknown_sudo_user_falls_back_to_home(self):
        got = data_dir({"SUDO_USER": "ghost"}, euid=0, home="/root", home_of=lambda u: None)
        assert got == os.path.join("/root", ".stt")


class TestConfiguredPort:
    def test_reads_web_server_port(self, tmp_path):
        write_config(tmp_path, {"web_server": {"port": 80}})
        assert configured_port(str(tmp_path)) == 80

    def test_a_string_port_is_coerced(self, tmp_path):
        write_config(tmp_path, {"web_server": {"port": "8188"}})
        assert configured_port(str(tmp_path)) == 8188

    @pytest.mark.parametrize("cfg", [
        {}, {"web_server": {}}, {"web_server": {"port": "eighty"}},
        {"web_server": {"port": 0}}, {"web_server": {"port": 70000}},
        {"web_server": "nope"}, [], "{not json",
    ])
    def test_anything_unreadable_is_the_shipped_default(self, tmp_path, cfg):
        write_config(tmp_path, cfg)
        assert configured_port(str(tmp_path)) == DEFAULT_PORT

    def test_a_missing_config_is_the_shipped_default(self, tmp_path):
        assert configured_port(str(tmp_path)) == DEFAULT_PORT

    def test_the_default_is_what_the_template_ships(self):
        with open(os.path.join(REPO, "config", "config.default.json"), encoding="utf-8") as f:
            assert json.load(f)["web_server"]["port"] == DEFAULT_PORT


class TestCli:
    def run(self, tmp_path, *args):
        env = dict(os.environ, STT_DATA_DIR=str(tmp_path), PYTHONPATH=REPO)
        return subprocess.run([sys.executable, "-m", "stt.server_port", *args], env=env,
                              capture_output=True, text=True, check=True).stdout.strip()

    def test_prints_the_port(self, tmp_path):
        write_config(tmp_path, {"web_server": {"port": 80}})
        assert self.run(tmp_path) == "80"

    def test_prints_the_data_dir(self, tmp_path):
        assert self.run(tmp_path, "--data-dir") == str(tmp_path)

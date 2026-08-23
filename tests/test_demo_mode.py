"""Demo mode: the switch, the data root, session discovery, and the state shims."""

from __future__ import annotations

import json
import os
import socket

import pytest

from stt import demo_mode


# --- the switch ------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_enabled_accepts_truthy_spellings(value):
    assert demo_mode.enabled({demo_mode.ENV_FLAG: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_enabled_rejects_everything_else(value):
    assert demo_mode.enabled({demo_mode.ENV_FLAG: value}) is False


def test_enabled_defaults_to_off_when_unset():
    assert demo_mode.enabled({}) is False


# --- the data root ---------------------------------------------------------


def _bundle(tmp_path, *default_names):
    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    for name in default_names:
        (bundle / "config" / name).write_text(json.dumps({"seeded": name}))
    return str(bundle)


def test_data_dir_is_never_the_real_install(tmp_path):
    root = demo_mode.data_dir(str(tmp_path))
    assert os.path.basename(root) != ".stt"
    assert root != os.path.join(str(tmp_path), ".stt")


def test_prepare_data_dir_seeds_templates_and_makes_the_tree(tmp_path):
    bundle = _bundle(tmp_path, "config.default.json", "word_highlighting.default.json")
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    for sub in ("config", "models", "logs", "_AUTOMATIC_BACKUP", demo_mode.SESSIONS_DIR_NAME):
        assert os.path.isdir(os.path.join(root, sub)), sub
    assert os.path.isfile(os.path.join(root, "config", "config.default.json"))
    assert os.path.isfile(os.path.join(root, "config", "word_highlighting.default.json"))


def test_prepare_data_dir_copies_only_templates(tmp_path):
    bundle = _bundle(tmp_path, "config.default.json")
    # A live config sitting in the bundle must not be treated as a template.
    (tmp_path / "bundle" / "config" / "config.json").write_text("{}")

    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    assert not os.path.exists(os.path.join(root, "config", "config.json"))


def test_prepare_data_dir_wipes_a_previous_run(tmp_path):
    bundle = _bundle(tmp_path, "config.default.json")
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))
    stale = os.path.join(root, "config", "config.json")
    with open(stale, "w") as handle:
        handle.write('{"web_server": {"port": 1}}')

    demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    assert not os.path.exists(stale)


def test_prepare_data_dir_touches_nothing_outside_its_root(tmp_path):
    bundle = _bundle(tmp_path, "config.default.json")
    real_install = tmp_path / ".stt"
    real_install.mkdir()
    (real_install / "config.json").write_text('{"real": true}')

    demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    assert (real_install / "config.json").read_text() == '{"real": true}'


# --- config overlay --------------------------------------------------------


def test_apply_overlay_merges_deeply_and_overwrites_leaves():
    config = {"web_server": {"port": 8080, "host": "0.0.0.0", "keep": 1}, "other": 2}

    merged = demo_mode.apply_overlay(config, {"web_server": {"port": 8099}})

    assert merged["web_server"] == {"port": 8099, "host": "0.0.0.0", "keep": 1}
    assert merged["other"] == 2


def test_apply_overlay_replaces_a_dict_with_a_scalar_and_vice_versa():
    config = {"a": {"nested": 1}, "b": 5}

    merged = demo_mode.apply_overlay(config, {"a": 9, "b": {"nested": 2}})

    assert merged == {"a": 9, "b": {"nested": 2}}


def test_apply_overlay_deep_copies_so_callers_cannot_alias():
    overlay = {"web_server": {"settings_ip_whitelist": []}}
    config = demo_mode.apply_overlay({}, overlay)

    config["web_server"]["settings_ip_whitelist"].append("10.0.0.1")

    assert overlay["web_server"]["settings_ip_whitelist"] == []


def test_missing_overlay_paths_reports_dotted_paths():
    overlay = {"web_server": {"port": 1, "nope": 2}, "gone": 3}
    template = {"web_server": {"port": 8080}}

    assert sorted(demo_mode.missing_overlay_paths(overlay, template)) == ["gone", "web_server.nope"]


def test_overlay_only_names_settings_the_real_server_reads():
    """The overlay configures the shipped product; a renamed key must fail loudly here."""
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "config", "config.default.json")
    with open(template_path, encoding="utf-8") as handle:
        template = json.load(handle)

    missing = demo_mode.missing_overlay_paths(demo_mode.config_overlay(8099), template)

    assert missing == [], f"overlay sets settings absent from config.default.json: {missing}"


# --- session discovery -----------------------------------------------------


def _session(directory, name):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write("")
    return path


def test_discover_prefers_a_drop_in_beside_the_executable(tmp_path):
    bundle = str(tmp_path / "bundle")
    exe_dir = str(tmp_path / "exe")
    root = str(tmp_path / "root")
    _session(os.path.join(bundle, "demo"), "demo.db")
    dropped = _session(os.path.join(exe_dir, demo_mode.SESSIONS_DIR_NAME), "2026-08-02_090000.db")

    assert demo_mode.discover_session(bundle, exe_dir, root) == dropped


def test_discover_falls_back_to_the_bundled_recording(tmp_path):
    bundle = str(tmp_path / "bundle")
    bundled = _session(os.path.join(bundle, "demo"), "demo.db")

    assert demo_mode.discover_session(bundle, str(tmp_path / "exe"), str(tmp_path / "root")) == bundled


def test_discover_returns_none_when_there_is_nothing_to_play(tmp_path):
    assert demo_mode.discover_session(str(tmp_path / "bundle"), None, str(tmp_path / "root")) is None


def test_discover_picks_the_newest_drop_in(tmp_path):
    exe_dir = str(tmp_path / "exe")
    sessions = os.path.join(exe_dir, demo_mode.SESSIONS_DIR_NAME)
    _session(sessions, "2026-08-02_090000.db")
    newest = _session(sessions, "2026-08-09_110000.db")

    picked = demo_mode.discover_session(str(tmp_path / "bundle"), exe_dir, str(tmp_path / "root"))

    assert picked == newest


def test_explicit_session_wins_over_everything(tmp_path):
    bundle = str(tmp_path / "bundle")
    _session(os.path.join(bundle, "demo"), "demo.db")
    chosen = _session(str(tmp_path / "elsewhere"), "chosen.db")

    assert demo_mode.discover_session(bundle, None, str(tmp_path), explicit=chosen) == chosen


def test_explicit_session_that_does_not_exist_resolves_to_nothing(tmp_path):
    """Better to report no recording than to silently play a different one."""
    bundle = str(tmp_path / "bundle")
    _session(os.path.join(bundle, "demo"), "demo.db")

    picked = demo_mode.discover_session(bundle, None, str(tmp_path),
                                        explicit=str(tmp_path / "missing.db"))

    assert picked is None


def test_a_real_installs_archive_is_ignored_unless_asked_for(tmp_path):
    home = str(tmp_path)
    archive = os.path.join(home, ".stt", "_AUTOMATIC_BACKUP", "2026", "08")
    real = _session(archive, "2026-08-02_090000.db")
    bundle = str(tmp_path / "bundle")
    bundled = _session(os.path.join(bundle, "demo"), "demo.db")
    root = str(tmp_path / "root")

    assert demo_mode.discover_session(bundle, None, root, home=home) == bundled
    assert demo_mode.discover_session(bundle, None, root, use_local_sessions=True, home=home) == real


def test_find_sessions_skips_directories_that_do_not_exist(tmp_path):
    assert demo_mode.find_sessions([str(tmp_path / "nope")]) == []


# --- state shims -----------------------------------------------------------


def test_local_manager_dict_supports_what_the_monolith_does_to_state():
    state = demo_mode.LocalManager().dict({"running": False, "db_name": None})

    state["running"] = True
    state["session_id"] = "2026-08-02_090000"
    state.update({"status": "running"})

    assert state["running"] is True
    assert state.get("missing", "fallback") == "fallback"
    assert state["status"] == "running"
    assert "db_name" in state


def test_local_manager_copies_rather_than_aliases():
    initial = {"running": False}
    state = demo_mode.LocalManager().dict(initial)

    state["running"] = True

    assert initial["running"] is False


def test_local_manager_list_round_trips():
    values = demo_mode.LocalManager().list([1, 2])
    values.append(3)
    assert values == [1, 2, 3]


def test_local_queue_supports_the_monoliths_queue_api():
    q = demo_mode.local_queue()

    assert q.empty()
    q.put({"command": "start"})
    assert q.qsize() == 1
    assert q.get_nowait() == {"command": "start"}

    q.put_nowait({"command": "stop"})
    assert q.get(timeout=0.01) == {"command": "stop"}


def test_local_queue_get_nowait_raises_empty_like_an_mpqueue():
    import queue as queue_mod

    with pytest.raises(queue_mod.Empty):
        demo_mode.local_queue().get_nowait()


# --- launching -------------------------------------------------------------


def test_pick_port_skips_a_port_that_is_actually_bound():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        busy = taken.getsockname()[1]

        assert demo_mode.port_is_free(busy) is False
        assert demo_mode.pick_port(start=busy, limit=4) != busy


def test_pick_port_returns_the_start_port_when_it_is_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]

    assert demo_mode.pick_port(start=free, limit=1) == free


def test_pick_port_falls_back_to_start_when_nothing_is_free(monkeypatch):
    monkeypatch.setattr(demo_mode, "port_is_free", lambda *a, **k: False)

    assert demo_mode.pick_port(start=9000, limit=3) == 9000


# --- the config the demo runs on -------------------------------------------


def _template(tmp_path, extra=None):
    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    template = {"analytics": {"endpoint": "https://example.invalid/api/ping"},
                "web_server": {"port": 8080, "host": "0.0.0.0",
                               "password_auth": {"enabled": True, "password": "admin"},
                               "settings_ip_whitelist": ["127.0.0.1"], "access_token": ""},
                "audio": {"autostart": True, "device": 0},
                "live_translation": {"enabled": False, "tts": {"enabled": True}},
                "service_phase": {"enabled": False},
                "sermon_summary": {"enabled": True},
                "crash_reporting": {"enabled": True, "sentry_enabled": True},
                "analytics": {"endpoint": "https://example.invalid/ping"},
                "auto_update": {"enabled": True}}
    if extra:
        template.update(extra)
    (bundle / "config" / "config.default.json").write_text(json.dumps(template))
    return str(bundle)


def test_write_config_applies_the_demo_settings_over_the_template(tmp_path):
    bundle = _template(tmp_path)
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    path = demo_mode.write_config(root, bundle, 8099)

    with open(path) as handle:
        config = json.load(handle)
    assert config["web_server"]["port"] == 8099
    assert config["web_server"]["host"] == "0.0.0.0"   # reachable from the network
    assert config["web_server"]["password_auth"]["enabled"] is False
    assert config["audio"]["autostart"] is False
    assert config["auto_update"]["enabled"] is False


def test_write_config_keeps_settings_the_demo_does_not_care_about(tmp_path):
    bundle = _template(tmp_path)
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    with open(demo_mode.write_config(root, bundle, 8099)) as handle:
        config = json.load(handle)

    assert config["audio"]["device"] == 0
    assert config["web_server"]["password_auth"]["password"] == "admin"


def test_a_demo_reports_that_it_ran_and_that_it_crashed(tmp_path):
    """The two deliberate exceptions to "a demo reaches nothing". Both are tagged as
    a demo at the source so the collector can count a trial as a trial."""
    bundle = _template(tmp_path)
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    with open(demo_mode.write_config(root, bundle, 8099)) as handle:
        config = json.load(handle)

    assert config["analytics"]["endpoint"]            # inherited, not blanked
    assert config["crash_reporting"]["sentry_enabled"] is True
    assert config["crash_reporting"]["enabled"] is True


def test_a_demo_never_updates_itself(tmp_path):
    bundle = _template(tmp_path)
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    with open(demo_mode.write_config(root, bundle, 8099)) as handle:
        config = json.load(handle)

    assert config["auto_update"]["enabled"] is False


# --- counting a trial once ------------------------------------------------


def test_the_anonymous_id_survives_the_launch_wipe(tmp_path):
    """Without this, one person opening the demo five times reads as five trials."""
    bundle = _template(tmp_path)
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))
    demo_mode.write_config(root, bundle, 8099, install_id="stable-id-1234")

    carried = demo_mode.read_install_id(home=str(tmp_path))
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))
    with open(demo_mode.write_config(root, bundle, 8099, install_id=carried)) as handle:
        config = json.load(handle)

    assert carried == "stable-id-1234"
    assert config["analytics"]["install_id"] == "stable-id-1234"


def test_a_first_run_has_no_previous_id_to_carry(tmp_path):
    assert demo_mode.read_install_id(home=str(tmp_path)) is None


def test_an_unreadable_config_does_not_break_the_launch(tmp_path):
    root = demo_mode.data_dir(str(tmp_path))
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    with open(os.path.join(root, "config", "config.json"), "w") as handle:
        handle.write("{not json")

    assert demo_mode.read_install_id(home=str(tmp_path)) is None


def test_write_config_lands_where_the_server_looks_for_it(tmp_path):
    bundle = _template(tmp_path)
    root = demo_mode.prepare_data_dir(bundle, home=str(tmp_path))

    path = demo_mode.write_config(root, bundle, 8099)

    assert path == os.path.join(root, "config", "config.json")


# --- finding the drop-in folder --------------------------------------------


def test_running_from_source_has_no_executable_directory(monkeypatch):
    monkeypatch.delattr("sys.frozen", raising=False)

    assert demo_mode.executable_dir() is None


def test_a_one_dir_build_looks_beside_its_executable(monkeypatch, tmp_path):
    exe = tmp_path / "STT-Demo" / "STT-Demo"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(exe))

    assert demo_mode.executable_dir() == str(exe.parent)


def test_a_mac_app_looks_beside_the_bundle_not_inside_it(monkeypatch, tmp_path):
    """A signed bundle must not be written into, so the folder belongs next to it."""
    exe = tmp_path / "STT Demo.app" / "Contents" / "MacOS" / "STT-Demo"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(exe))

    assert demo_mode.executable_dir() == str(tmp_path)


# --- reachable from the network --------------------------------------------


def test_the_demo_binds_the_network_not_just_loopback():
    """So it can be opened on a phone or a second screen. This is why the outbound
    guards in stt/demo_guard.py are not optional."""
    assert demo_mode.config_overlay(8099)["web_server"]["host"] == "0.0.0.0"


def test_the_banner_names_the_local_url():
    banner = demo_mode.startup_banner(8099, lan=None)

    assert "http://127.0.0.1:8099/" in banner


def test_the_banner_names_the_network_url_and_says_there_is_no_password():
    banner = demo_mode.startup_banner(8099, lan="192.0.2.55")

    assert "http://192.0.2.55:8099/" in banner
    assert "password" in banner.lower()


def test_the_banner_stays_quiet_about_the_network_when_there_is_none():
    banner = demo_mode.startup_banner(8099, lan=None)

    assert "password" not in banner.lower()


def test_the_lan_address_is_never_loopback_and_never_resolves_anything():
    address = demo_mode.lan_address()

    assert address is None or not address.startswith("127.")

"""Deciding whether an update is available (stt/watchdog.py).

The watchdog updates the deployed machine unattended, on a schedule. Two
mistakes here are expensive and quiet: deciding an update exists when it does
not restarts a machine that was fine, and deciding none exists when one does
leaves a fix undeployed indefinitely. Neither shows up until someone notices
the version is wrong.

Nothing is fetched or restarted: the release lookup, config, version read and
state are all stubbed.
"""

import pytest

from stt import watchdog


class _State:
    """Stands in for the shared state object; records what was set."""

    def __init__(self):
        self.values = {}

    def set(self, **kw):
        self.values.update(kw)


@pytest.fixture
def updater(monkeypatch):
    """An AutoUpdater with the network, config and version read replaced."""
    u = watchdog.AutoUpdater.__new__(watchdog.AutoUpdater)
    u.state = _State()
    u._pending_update = "carried over from a previous check"
    u.branch_checks = []
    u._check_for_branch_update = lambda: u.branch_checks.append(True)

    monkeypatch.setattr(watchdog, "load_config", lambda: {"watchdog": {"update_channel": "stable"}})
    monkeypatch.setattr(watchdog, "read_version", lambda: "26.1.100")
    u.releases = ("v26.1.100", "https://example/zip", {})
    u.get_latest_release = lambda channel: (
        u.releases if not isinstance(u.releases, Exception) else (_ for _ in ()).throw(u.releases))
    return u


def _channel(monkeypatch, name):
    monkeypatch.setattr(watchdog, "load_config",
                        lambda: {"watchdog": {"update_channel": name}})


class TestChannelRouting:
    def test_the_main_channel_tracks_the_branch_instead_of_releases(self, updater, monkeypatch):
        _channel(monkeypatch, "main")
        updater.check_for_update()
        assert updater.branch_checks == [True]

    def test_an_absent_channel_defaults_to_main(self, updater, monkeypatch):
        monkeypatch.setattr(watchdog, "load_config", lambda: {})
        updater.check_for_update()
        assert updater.branch_checks == [True], "the default must not silently be 'stable'"

    def test_the_stable_channel_asks_the_releases_api(self, updater, monkeypatch):
        _channel(monkeypatch, "stable")
        updater.check_for_update()
        assert updater.branch_checks == []


class TestVersionComparison:
    def test_a_newer_release_becomes_pending(self, updater):
        updater.releases = ("v26.1.200", "https://example/200.zip", {"a": 1})
        updater.check_for_update()
        assert updater._pending_update == ("26.1.200", "https://example/200.zip", {"a": 1})
        assert "26.1.200" in updater.state.values["last_update_result"]

    def test_the_same_version_clears_any_pending_update(self, updater):
        """A pending update from an earlier check must not survive being current."""
        updater.check_for_update()
        assert updater._pending_update is None
        assert "Up to date" in updater.state.values["last_update_result"]

    def test_an_older_release_is_not_offered_as_an_update(self, updater):
        # A yanked release, or a stable channel behind the installed build.
        updater.releases = ("v26.0.1", "https://example/old.zip", {})
        updater.check_for_update()
        assert updater._pending_update is None

    def test_the_v_prefix_is_not_part_of_the_version(self, updater):
        updater.releases = ("v26.1.100", "https://example/z", {})
        updater.check_for_update()
        assert updater._pending_update is None, "'v26.1.100' and '26.1.100' are one version"

    def test_versions_compare_numerically_not_as_text(self, updater, monkeypatch):
        # "26.1.9" > "26.1.100" as strings; the wrong answer skips a real update.
        monkeypatch.setattr(watchdog, "read_version", lambda: "26.1.9")
        updater.releases = ("v26.1.100", "https://example/z", {})
        updater.check_for_update()
        assert updater._pending_update is not None


class TestFailuresLeaveTheMachineAlone:
    def test_a_failed_lookup_is_recorded_and_nothing_is_pending(self, updater):
        updater.releases = RuntimeError("connection reset")
        updater.check_for_update()
        assert "Check failed" in updater.state.values["last_update_result"]
        assert "connection reset" in updater.state.values["last_update_result"]

    def test_a_failed_lookup_does_not_raise(self, updater):
        updater.releases = RuntimeError("boom")
        updater.check_for_update()  # would take the scheduler thread down

    def test_a_repo_with_no_releases_is_not_an_error(self, updater):
        updater.releases = (None, None, None)
        updater.check_for_update()
        assert updater.state.values["last_update_result"] == "No releases yet"

    def test_the_check_time_is_always_recorded(self, updater):
        """The dashboard shows this; a stuck timestamp is how a dead scheduler shows."""
        updater.releases = RuntimeError("offline")
        updater.check_for_update()
        assert updater.state.values.get("last_update_check")


class TestParseVersionUnderpinsThis:
    @pytest.mark.parametrize("older,newer", [
        ("26.1.9", "26.1.100"),
        ("26.1.100", "26.2.0"),
        ("1.0.0", "26.1.1"),
        ("26.1.100", "26.1.100.1"),
    ])
    def test_ordering(self, older, newer):
        assert watchdog.parse_version(older) < watchdog.parse_version(newer)

    def test_equal_versions_are_equal(self):
        assert watchdog.parse_version("v26.1.1") == watchdog.parse_version("26.1.1")

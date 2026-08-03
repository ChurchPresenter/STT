"""Config persistence, upload validation, and version helpers (stt/config_utils.py)."""

import re
import datetime
import json
import os

import pytest

from stt.config_utils import (
    _atomic_write_json,
    _merge_missing_keys,
    compute_display_version,
    restore_config_from_template,
    validate_file,
    is_known_timezone,
    resolve_timezone,
    system_timezone,
)


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "cfg.json"
        _atomic_write_json(str(path), {"a": 1, "b": {"c": 2}})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": {"c": 2}}

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text('{"old": true}')
        _atomic_write_json(str(path), {"new": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}

    def test_no_temp_litter_on_success(self, tmp_path):
        _atomic_write_json(str(tmp_path / "cfg.json"), {"a": 1})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cfg.json"]

    def test_failure_leaves_original_and_no_litter(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text('{"old": true}')
        with pytest.raises(TypeError):
            _atomic_write_json(str(path), {"bad": object()})
        assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cfg.json"]

    def test_ensure_ascii_toggle(self, tmp_path):
        path = tmp_path / "cfg.json"
        _atomic_write_json(str(path), {"w": "señor"}, ensure_ascii=False)
        assert "señor" in path.read_text(encoding="utf-8")
        _atomic_write_json(str(path), {"w": "señor"})
        assert "se\\u00f1or" in path.read_text(encoding="utf-8")


class TestMergeMissingKeys:
    def test_adds_missing_keys(self):
        dst = {"a": 1}
        assert _merge_missing_keys(dst, {"a": 9, "b": 2}) is True
        assert dst == {"a": 1, "b": 2}

    def test_never_overwrites_existing(self):
        dst = {"a": "user-set", "nested": {"x": False}}
        _merge_missing_keys(dst, {"a": "template", "nested": {"x": True, "y": 1}})
        assert dst["a"] == "user-set"
        assert dst["nested"] == {"x": False, "y": 1}

    def test_type_mismatch_left_alone(self):
        # User set a scalar where the template has a dict: keep the user's value
        dst = {"a": 5}
        assert _merge_missing_keys(dst, {"a": {"sub": 1}}) is False
        assert dst == {"a": 5}

    def test_no_change_returns_false(self):
        dst = {"a": 1, "b": {"c": 2}}
        assert _merge_missing_keys(dst, {"a": 0, "b": {"c": 9}}) is False

    def test_added_values_are_deep_copies(self):
        src = {"b": {"list": [1, 2]}}
        dst = {}
        _merge_missing_keys(dst, src)
        dst["b"]["list"].append(3)
        assert src["b"]["list"] == [1, 2]


class TestRestoreConfigFromTemplate:
    def test_copies_template(self, tmp_path):
        template = tmp_path / "config.default.json"
        template.write_text('{"fresh": true}')
        target = tmp_path / "config.json"
        assert restore_config_from_template(str(template), str(target)) is True
        assert json.loads(target.read_text(encoding="utf-8")) == {"fresh": True}

    def test_missing_template_returns_false(self, tmp_path):
        target = tmp_path / "config.json"
        assert restore_config_from_template(str(tmp_path / "nope.json"), str(target)) is False
        assert not target.exists()


class FakeUpload:
    def __init__(self, filename):
        self.filename = filename


class TestValidateFile:
    def test_no_file(self):
        assert validate_file(None) == (False, "No file selected")
        assert validate_file(FakeUpload("")) == (False, "No file selected")

    def test_supported_audio_and_video(self):
        assert validate_file(FakeUpload("sermon.mp3")) == (True, None)
        assert validate_file(FakeUpload("Service.MP4")) == (True, None)

    def test_unsupported_extension(self):
        ok, err = validate_file(FakeUpload("notes.txt"))
        assert ok is False
        assert "txt" in err

    def test_no_extension(self):
        ok, _err = validate_file(FakeUpload("plainfile"))
        assert ok is False


class TestComputeDisplayVersion:
    def test_commits_since_tag_folded_into_patch(self):
        assert compute_display_version("26.1.2-17-g398f75e", "398f75e", "26.1.2") == "26.1.19-398f75e"

    def test_exact_tag_passthrough(self):
        assert compute_display_version("26.1.3", "abc1234", "26.1.3") == "26.1.3"

    def test_non_semver_describe_passthrough(self):
        assert compute_display_version("v-weird-tag", "abc", "1.0") == "v-weird-tag"

    def test_frozen_build_with_commit(self):
        assert compute_display_version("", "abc1234", "26.1.2") == "26.1.2-abc1234"

    def test_fallback_to_version_file(self):
        assert compute_display_version("", "", "26.1.2") == "26.1.2"

    def test_monotonic_across_a_release(self):
        # one commit after the 26.1.3 tag must sort above the tag itself
        assert compute_display_version("26.1.3-1-gaaaa111", "", "x") == "26.1.4-aaaa111"


def _has_tz_database() -> bool:
    """Whether this machine can load an IANA zone name at all.

    Windows ships no system tz database, so zoneinfo needs the `tzdata` package
    (in requirements.txt for runtime and requirements-dev.txt for CI). Without it
    the feature genuinely cannot work, and these assertions would fail for an
    environment reason rather than a code one — so say that plainly instead.
    """
    return is_known_timezone("America/New_York")


needs_tzdata = pytest.mark.skipif(
    not _has_tz_database(),
    reason="no IANA tz database on this machine; install tzdata",
)


class TestResolveTimezone:
    """The timezone the transcript is stamped in.

    This setting existed in the API and in config for a long time while
    get_configured_timezone() ignored it entirely and always returned the machine's
    own zone — so it silently did nothing, even after the restart the save endpoint
    told the operator to perform.
    """

    UTC = datetime.timezone.utc
    FIXED = datetime.timezone(datetime.timedelta(hours=5), "TEST")

    def test_auto_uses_the_system_zone(self):
        tz, note = resolve_timezone({"mode": "auto", "value": "Asia/Tokyo"}, self.FIXED)
        assert tz is self.FIXED
        assert note is None

    def test_missing_config_is_auto(self):
        for cfg in (None, {}, {"value": "Asia/Tokyo"}):
            tz, note = resolve_timezone(cfg, self.FIXED)
            assert tz is self.FIXED and note is None

    @needs_tzdata
    def test_a_named_zone_is_actually_used(self):
        # The whole point: a configured zone must reach the returned tzinfo.
        tz, note = resolve_timezone({"mode": "manual", "value": "America/New_York"}, self.FIXED)
        assert note is None
        assert tz is not self.FIXED
        assert "New_York" in str(tz)

    @needs_tzdata
    def test_the_zone_changes_the_wall_clock(self):
        # Proves it resolves to a real zone rather than a label.
        moment = datetime.datetime(2026, 7, 1, 12, 0, tzinfo=self.UTC)
        ny, _ = resolve_timezone({"mode": "manual", "value": "America/New_York"}, self.UTC)
        tokyo, _ = resolve_timezone({"mode": "manual", "value": "Asia/Tokyo"}, self.UTC)
        assert moment.astimezone(ny).hour != moment.astimezone(tokyo).hour

    def test_an_unknown_zone_falls_back_and_explains(self):
        # Holds with or without a tz database — the point is that it never raises.
        # Timestamps are written on every row; a typo must not stop a service.
        tz, note = resolve_timezone({"mode": "manual", "value": "Mars/Olympus_Mons"}, self.FIXED)
        assert tz is self.FIXED
        assert note and "Mars/Olympus_Mons" in note

    def test_a_mode_with_no_value_falls_back_and_explains(self):
        tz, note = resolve_timezone({"mode": "manual", "value": "   "}, self.FIXED)
        assert tz is self.FIXED
        assert note and "no value" in note

    def test_mode_is_case_and_space_insensitive(self):
        tz, note = resolve_timezone({"mode": "  AUTO  "}, self.FIXED)
        assert tz is self.FIXED and note is None

    @needs_tzdata
    def test_a_value_is_trimmed(self):
        tz, note = resolve_timezone({"mode": "manual", "value": "  Asia/Tokyo  "}, self.FIXED)
        assert note is None and "Tokyo" in str(tz)

    def test_it_always_returns_a_usable_tzinfo(self):
        for cfg in (None, {}, {"mode": "manual"}, {"mode": "x", "value": "nope"}):
            tz, _ = resolve_timezone(cfg, self.FIXED)
            assert datetime.datetime.now(tz).utcoffset() is not None


class TestIsKnownTimezone:
    """Guards the save endpoint so a bad name is rejected, not silently ignored."""

    @needs_tzdata
    def test_real_zones(self):
        for z in ("America/New_York", "Europe/Kyiv", "UTC", "Asia/Tokyo"):
            assert is_known_timezone(z)

    def test_nonsense_is_rejected(self):
        for z in ("Mars/Olympus_Mons", "EST5EDT_typo", "not a zone"):
            assert not is_known_timezone(z)

    def test_blank_is_rejected(self):
        for z in ("", "   ", None):
            assert not is_known_timezone(z)

    @needs_tzdata
    def test_whitespace_is_trimmed_before_checking(self):
        assert is_known_timezone("  America/New_York  ")


class TestSystemTimezone:
    def test_returns_a_concrete_tzinfo(self):
        tz = system_timezone()
        assert datetime.datetime.now(tz).utcoffset() is not None


class TestShippedTemplateHasNoCredential:
    """config.default.json must not ship a usable password.

    It did: `"password": "admin"` with auth enabled, which meant every
    installation of this software shared one password, published in a public
    repository. The server already had the better behaviour — an empty value makes
    it generate a random 12-character password at startup, save it, and print it to
    the console — and the shipped default was the only thing standing in its way.
    """

    PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "config.default.json")

    @pytest.fixture
    def template(self):
        with open(self.PATH, encoding="utf-8") as fh:
            return json.load(fh)

    def test_the_password_ships_empty(self, template):
        password = template["web_server"]["password_auth"]["password"]
        assert password == "", (
            "a non-empty default is the same password on every install, in a public "
            "repository — leave it empty so each install generates its own")

    def test_no_other_credential_field_ships_a_value(self, template):
        # Same rule, applied to every secret-shaped key rather than the one that
        # happened to be wrong.
        secretish = re.compile(r"pass|secret|token|api_key|credential", re.I)
        offenders = []

        def walk(node, path=""):
            if not isinstance(node, dict):
                return
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if isinstance(value, dict):
                    walk(value, here)
                elif (isinstance(value, str) and value.strip()
                        and not key.startswith("_") and secretish.search(key)):
                    offenders.append(here)

        walk(template)
        assert not offenders, f"shipped credential values: {offenders}"

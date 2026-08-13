"""The anonymous live-map ping (stt/livemap.py).

Two events share one URL shape: app_start says an install is running, and
transcription_start says it is captioning a service. The collector reads both, so the
shape is a contract — the byte-compatibility case below is the one that matters most.
"""

import pytest

from stt.livemap import (
    EVENT_APP_START,
    EVENT_TRANSCRIPTION_START,
    build_ping_url,
    ensure_install_id,
    numeric_version,
    os_name_for_platform,
    ping_fields_from_config,
)

ENDPOINT = "https://stt.churchpresenter.org/api/ping"


class TestOsName:
    @pytest.mark.parametrize("platform,expected", [
        ("darwin", "macos"), ("win32", "windows"), ("linux", "linux"),
    ])
    def test_the_three_platforms_the_map_groups_by(self, platform, expected):
        assert os_name_for_platform(platform) == expected

    @pytest.mark.parametrize("platform", ["freebsd12", "", "  ", None])
    def test_an_unrecognised_platform_still_counts_as_an_install(self, platform):
        # The map counts installs; dropping the ping over an unknown platform would
        # lose the install entirely, which is the worse answer.
        assert os_name_for_platform(platform) == "linux"


class TestNumericVersion:
    def test_the_commit_suffix_is_stripped(self):
        # The collector rejects anything beyond dotted numerics, and the commit
        # travels in its own parameter.
        assert numeric_version("26.1.22-gc588d29", "26.1.0") == "26.1.22"

    def test_an_already_numeric_version_passes_through(self):
        assert numeric_version("26.1.22", "") == "26.1.22"

    def test_a_blank_display_version_falls_back(self):
        # Frozen builds have no git describe, so the VERSION file is all there is.
        assert numeric_version("", "26.1.22") == "26.1.22"
        assert numeric_version(None, "26.1.22") == "26.1.22"

    def test_with_nothing_at_all_the_ping_is_still_worth_sending(self):
        assert numeric_version("", "") == "unknown"
        assert numeric_version(None, None) == "unknown"


class TestBuildPingUrl:
    @pytest.mark.parametrize("endpoint", ["", "   ", None])
    def test_a_blank_endpoint_is_the_kill_switch(self, endpoint):
        # None must mean "make no request", not "post somewhere else": this is how an
        # operator or a fork opts out.
        assert build_ping_url(endpoint, event=EVENT_APP_START,
                              os_name="linux", version="26.1.22") is None

    def test_the_transcription_url_is_the_historical_one_plus_event(self):
        # The collector has received this exact string for as long as the ping has
        # existed. Changing anything but the appended field would break it.
        url = build_ping_url(ENDPOINT, event=EVENT_TRANSCRIPTION_START,
                             os_name="macos", version="26.1.22",
                             transcribe_lang="ru", translate_lang="en",
                             commit="c588d29", offloaded=True)
        historical = (ENDPOINT + "?os=macos&version=26.1.22"
                      "&transcribe_lang=ru&translate_lang=en"
                      "&commit=c588d29&offloaded=1")
        assert url == historical + "&event=transcription_start"

    def test_the_app_start_url_carries_only_what_exists_at_boot(self):
        # No session has started, so claiming a language would invent one.
        url = build_ping_url(ENDPOINT, event=EVENT_APP_START,
                             os_name="linux", version="26.1.22", commit="c588d29")
        assert url == (ENDPOINT + "?os=linux&version=26.1.22"
                       "&commit=c588d29&event=app_start")

    def test_optional_fields_are_omitted_rather_than_blank(self):
        url = build_ping_url(ENDPOINT, event=EVENT_APP_START,
                             os_name="linux", version="26.1.22")
        assert "commit=" not in url
        assert "offloaded=" not in url
        assert "transcribe_lang=" not in url

    def test_offloaded_appears_only_when_true(self):
        url = build_ping_url(ENDPOINT, event=EVENT_TRANSCRIPTION_START,
                             os_name="linux", version="26.1.22", offloaded=False)
        assert "offloaded" not in url

    def test_values_are_percent_encoded(self):
        # Previously interpolated raw: a value with a space or an ampersand produced a
        # malformed URL, and one with '&' could forge a parameter.
        url = build_ping_url(ENDPOINT, event=EVENT_TRANSCRIPTION_START,
                             os_name="linux", version="26.1.22 beta",
                             transcribe_lang="ru&offloaded=1")
        assert "version=26.1.22%20beta" in url
        assert "transcribe_lang=ru%26offloaded%3D1" in url
        assert url.count("offloaded=1") == 0, "an injected parameter must not survive"

    def test_the_event_is_always_last(self):
        # So a collector reading the older shape sees the historical prefix intact.
        for event in (EVENT_APP_START, EVENT_TRANSCRIPTION_START):
            url = build_ping_url(ENDPOINT, event=event, os_name="linux",
                                 version="26.1.22", commit="abc")
            assert url.endswith("&event=" + event)


class TestPingFieldsFromConfig:
    def test_translation_off_reports_none(self):
        fields = ping_fields_from_config({"live_translation": {"enabled": False,
                                                              "target_language": "en"}})
        assert fields["translate_lang"] == "none"

    def test_translation_on_without_a_target_reports_unknown(self):
        # Distinct from "none": translation is switched on but has nowhere to go, and
        # the map should not read that as translation being off.
        fields = ping_fields_from_config({"live_translation": {"enabled": True,
                                                               "target_language": "  "}})
        assert fields["translate_lang"] == "unknown"

    def test_translation_on_reports_the_target(self):
        fields = ping_fields_from_config({"live_translation": {"enabled": True,
                                                               "target_language": "ru"}})
        assert fields["translate_lang"] == "ru"

    def test_transcribe_language_defaults_to_auto(self):
        assert ping_fields_from_config({})["transcribe_lang"] == "auto"
        assert ping_fields_from_config({"audio": {"language": ""}})["transcribe_lang"] == "auto"

    def test_offloaded_needs_both_the_flag_and_an_endpoint(self):
        def offloaded(remote):
            return ping_fields_from_config({"live_translation": {"remote": remote}})["offloaded"]
        assert offloaded({"enabled": True, "endpoint": "http://192.168.2.52:8080"}) is True
        assert offloaded({"enabled": True, "endpoint": ""}) is False
        assert offloaded({"enabled": False, "endpoint": "http://192.168.2.52:8080"}) is False

    def test_a_config_missing_whole_sections_does_not_raise(self):
        # This runs on the way to a fire-and-forget ping; a half-written config must
        # never be able to break a transcription start.
        fields = ping_fields_from_config({})
        assert fields == {"transcribe_lang": "auto", "translate_lang": "none",
                          "offloaded": False}


class TestEnsureInstallId:
    def test_an_id_is_generated_once_and_reported_as_new(self):
        analytics = {"install_id": ""}
        iid, changed = ensure_install_id(analytics, lambda: "fixed-id")
        assert (iid, changed) == ("fixed-id", True)
        assert analytics["install_id"] == "fixed-id"

    def test_an_existing_id_is_reused_without_a_config_write(self):
        # changed=False is what stops every boot rewriting the config file.
        analytics = {"install_id": "already-here"}
        assert ensure_install_id(analytics, lambda: "new") == ("already-here", False)

    @pytest.mark.parametrize("stored", ["", "   ", None])
    def test_a_blank_id_counts_as_absent(self, stored):
        analytics = {"install_id": stored}
        iid, changed = ensure_install_id(analytics, lambda: "fixed-id")
        assert (iid, changed) == ("fixed-id", True)

    def test_a_missing_key_is_handled(self):
        analytics = {}
        assert ensure_install_id(analytics, lambda: "fixed-id") == ("fixed-id", True)

    def test_the_rest_of_the_analytics_section_survives(self):
        # It is the live config's own section: rebuilding it would drop the endpoint
        # and the comment keys with it.
        analytics = {"endpoint": ENDPOINT, "_endpoint_comment": "...", "install_id": ""}
        ensure_install_id(analytics, lambda: "fixed-id")
        assert analytics["endpoint"] == ENDPOINT
        assert analytics["_endpoint_comment"] == "..."

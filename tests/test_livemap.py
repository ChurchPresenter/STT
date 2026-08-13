"""The anonymous live-map ping (stt/livemap.py).

Two events share one URL shape: app_start says an install is running, and
transcription_start says it is captioning a service. The collector reads both, so the
shape is a contract — the byte-compatibility case below is the one that matters most.
"""

import pytest

from stt.livemap import (
    EVENT_APP_START,
    EVENT_TRANSCRIPTION_START,
    arch_label,
    build_ping_url,
    ensure_install_id,
    gpu_label,
    install_fields_from_config,
    numeric_version,
    os_name_for_platform,
    os_version_label,
    ping_fields_from_config,
    transcription_model_label,
    translation_model_label,
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

    def test_the_historical_parameters_keep_their_names_values_and_order(self):
        # The collector has parsed this prefix for as long as the ping has existed.
        # Fields added since go after `commit`, and `event` stays last, so nothing the
        # collector already reads moves or changes.
        url = build_ping_url(ENDPOINT, event=EVENT_TRANSCRIPTION_START,
                             os_name="macos", version="26.1.22",
                             transcribe_lang="ru", translate_lang="en",
                             commit="c588d29", offloaded=True)
        historical = (ENDPOINT + "?os=macos&version=26.1.22"
                      "&transcribe_lang=ru&translate_lang=en"
                      "&commit=c588d29&offloaded=1")
        assert url == historical + "&event=transcription_start"

    def test_the_descriptive_fields_sit_between_commit_and_offloaded(self):
        url = build_ping_url(ENDPOINT, event=EVENT_TRANSCRIPTION_START,
                             os_name="linux", version="26.1.22", commit="c588d29",
                             os_version="Ubuntu 24.04.1 LTS", arch="x86_64",
                             gpu="NVIDIA GeForce RTX 4060", stt_model="large-v3",
                             mt_model="nllb:facebook/nllb-200-distilled-600M",
                             offloaded=True)
        assert url == (ENDPOINT + "?os=linux&version=26.1.22&commit=c588d29"
                       "&os_version=Ubuntu%2024.04.1%20LTS&arch=x86_64"
                       "&gpu=NVIDIA%20GeForce%20RTX%204060&stt_model=large-v3"
                       "&mt_model=nllb%3Afacebook%2Fnllb-200-distilled-600M"
                       "&offloaded=1&event=transcription_start")

    @pytest.mark.parametrize("field", ["os_version", "arch", "gpu", "stt_model", "mt_model"])
    def test_a_blank_descriptive_field_is_omitted(self, field):
        # An absent stt_model means "this install has never chosen one" — a real
        # answer, and one a blank value would hide.
        url = build_ping_url(ENDPOINT, event=EVENT_APP_START, os_name="linux",
                             version="26.1.22", **{field: ""})
        assert field + "=" not in url

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


class TestOsVersionLabel:
    def test_each_platform_reads_its_own_source(self):
        raw = dict(mac_ver="15.5", win_ver="10.0.26100",
                   distro="Ubuntu 24.04.1 LTS", kernel="6.8.0-45-generic")
        assert os_version_label("darwin", **raw) == "15.5"
        assert os_version_label("win32", **raw) == "10.0.26100"
        assert os_version_label("linux", **raw) == "Ubuntu 24.04.1 LTS"

    def test_linux_falls_back_to_the_kernel_without_os_release(self):
        # A container image or a stripped install may have no /etc/os-release.
        assert os_version_label("linux", kernel="6.8.0-45-generic") == "6.8.0-45-generic"

    def test_an_unknown_platform_is_treated_as_linux(self):
        # Mirrors os_name_for_platform, which counts anything unrecognised as linux.
        assert os_version_label("freebsd12", distro="FreeBSD 14") == "FreeBSD 14"

    def test_nothing_known_reports_nothing(self):
        # "" is what makes build_ping_url omit the field rather than send an empty one.
        assert os_version_label("linux") == ""
        assert os_version_label("darwin", mac_ver="   ") == ""

    def test_padding_and_newlines_are_collapsed(self):
        # /etc/os-release values arrive quoted and padded.
        assert os_version_label("linux", distro=" Ubuntu\n 24.04 LTS ") == "Ubuntu 24.04 LTS"

    def test_an_absurd_value_is_truncated_rather_than_sent_whole(self):
        assert len(os_version_label("linux", distro="x" * 500)) == 40


class TestArchLabel:
    def test_the_platform_value_passes_through_lowercased(self):
        assert arch_label("X86_64") == "x86_64"

    def test_arm_names_are_not_normalised(self):
        # macOS says arm64 where Linux says aarch64 for the same silicon, and which
        # one an install reports is itself a signal.
        assert arch_label("arm64") == "arm64"
        assert arch_label("aarch64") == "aarch64"

    def test_an_unknown_architecture_reports_nothing(self):
        assert arch_label("") == ""
        assert arch_label(None) == ""


class TestGpuLabel:
    def test_a_vendor_name_is_kept_intact(self):
        assert gpu_label("NVIDIA GeForce RTX 4060") == "NVIDIA GeForce RTX 4060"

    def test_a_cpu_only_box_reports_nothing(self):
        # Blank is the answer, not a missing one: it says this install runs on CPU.
        assert gpu_label("") == ""
        assert gpu_label(None) == ""

    def test_padding_is_collapsed_and_length_capped(self):
        assert gpu_label("  Apple  M2 Pro\n") == "Apple M2 Pro"
        assert len(gpu_label("x" * 500)) == 40


class TestTranscriptionModelLabel:
    def test_a_whisper_size_is_reported_plainly(self):
        assert transcription_model_label(
            {"model": {"type": "whisper", "whisper": {"model": "large-v3"}}}) == "large-v3"

    def test_a_hugging_face_id_is_prefixed(self):
        assert transcription_model_label(
            {"model": {"type": "huggingface",
                       "huggingface": {"model_id": "distil-whisper/distil-large-v3"}}}
        ) == "hf:distil-whisper/distil-large-v3"

    def test_a_custom_model_reports_its_architecture_and_never_its_path(self):
        label = transcription_model_label(
            {"model": {"type": "custom",
                       "custom": {"model_type": "wav2vec2",
                                  "model_path": "/home/pastor-smith/models/mine"}}})
        assert label == "custom:wav2vec2"

    def test_a_fresh_install_with_no_model_reports_nothing(self):
        # Ships empty on purpose, and an absent field is how the map sees that.
        assert transcription_model_label({"model": {"type": "whisper", "whisper": {"model": ""}}}) == ""

    def test_a_config_missing_the_section_does_not_raise(self):
        assert transcription_model_label({}) == ""


class TestTranslationModelLabel:
    def test_translation_off_reports_nothing(self):
        assert translation_model_label(
            {"live_translation": {"enabled": False, "translation_model": "facebook/nllb-200"}}) == ""

    @pytest.mark.parametrize("method", ["nllb", "madlad"])
    def test_an_nmt_checkpoint_is_prefixed_by_its_engine(self, method):
        # The two engines are not comparable; unprefixed they would share a column.
        assert translation_model_label(
            {"live_translation": {"enabled": True, "translation_method": method,
                                  "translation_model": "facebook/nllb-200-distilled-600M"}}
        ) == method + ":facebook/nllb-200-distilled-600M"

    def test_an_llm_reports_its_model_name(self):
        assert translation_model_label(
            {"live_translation": {"enabled": True, "translation_method": "llm",
                                  "llm": {"model": "gemma-3-12b-it"}}}) == "llm:gemma-3-12b-it"

    def test_a_local_gguf_reports_its_filename_not_its_path(self):
        for path in ("/home/pastor-smith/models/gemma-3-12b-it-Q4_K_M.gguf",
                     r"C:\Users\pastor-smith\models\gemma-3-12b-it-Q4_K_M.gguf"):
            assert translation_model_label(
                {"live_translation": {"enabled": True, "translation_method": "llm",
                                      "llm": {"model": "", "gguf_file": path}}}
            ) == "llm:gemma-3-12b-it-Q4_K_M.gguf"

    def test_an_engine_with_no_model_chosen_still_names_the_engine(self):
        assert translation_model_label(
            {"live_translation": {"enabled": True, "translation_method": "llm", "llm": {}}}) == "llm"
        assert translation_model_label({"live_translation": {"enabled": True}}) == "nllb"

    def test_a_config_missing_the_section_does_not_raise(self):
        assert translation_model_label({}) == ""


class TestInstallFieldsNeverLeakSecrets:
    def test_no_endpoint_key_or_path_reaches_the_url(self):
        # The load-bearing case: an endpoint URL identifies the operator's own
        # infrastructure, and a model path routinely contains their name.
        config = {
            "model": {"type": "custom",
                      "custom": {"model_type": "whisper",
                                 "model_path": "/home/pastor-smith/models/mine"}},
            "live_translation": {
                "enabled": True, "translation_method": "llm",
                "remote": {"enabled": True, "endpoint": "http://192.168.2.52:8080"},
                "llm": {"model": "", "endpoint": "http://192.168.2.52:11434",
                        "api_key": "sk-secret-token",
                        "gguf_file": "gemma.gguf",
                        "gguf_path": "/home/pastor-smith/models/gemma.gguf"},
            },
        }
        url = build_ping_url(ENDPOINT, event=EVENT_APP_START, os_name="linux",
                             version="26.1.22", **install_fields_from_config(config))
        for secret in ("pastor-smith", "sk-secret-token", "192.168.2.52", "11434", "/home/"):
            assert secret not in url
        assert "stt_model=custom%3Awhisper" in url
        assert "mt_model=llm%3Agemma.gguf" in url

    def test_both_events_can_carry_them(self):
        # They describe the install, not a session, so app_start reports them too.
        assert set(install_fields_from_config({})) == {"stt_model", "mt_model"}


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

"""Stripping a recording machine's identity out of captured API responses."""

from __future__ import annotations

from stt import demo_redact


# --- key-name deny list ----------------------------------------------------


def test_credential_fields_are_blanked_not_removed():
    """A page that renders the field must still find it."""
    payload = {"smb_username": "avteam", "smb_password": "hunter2", "enabled": True}

    cleaned = demo_redact.redact(payload)

    assert cleaned == {"smb_username": "", "smb_password": "", "enabled": True}


def test_blanking_preserves_the_value_type():
    payload = {"access_token": "abc", "token_count": 7, "token_list": ["a"],
               "token_map": {"k": "v"}, "token_enabled": True}

    cleaned = demo_redact.redact(payload)

    assert cleaned == {"access_token": "", "token_count": 0, "token_list": [],
                       "token_map": {}, "token_enabled": False}


def test_the_deny_list_matches_on_substrings_and_case():
    payload = {"SMB_Password": "x", "sentry_dsn": "https://k@o.ingest.io/1",
               "remote_endpoint": "http://192.168.2.52:8080", "install_id": "abc-123"}

    cleaned = demo_redact.redact(payload)

    assert all(value == "" for value in cleaned.values())


def test_extra_keys_can_be_denied_by_the_caller():
    payload = {"harmless": "keep", "custom_field": "drop"}

    cleaned = demo_redact.redact(payload, extra_keys=["custom_field"])

    assert cleaned == {"harmless": "keep", "custom_field": ""}


def test_ordinary_fields_survive_untouched():
    payload = {"success": True, "count": 3, "models": ["small", "large-v3"],
               "status": "stopped"}

    assert demo_redact.redact(payload) == payload


# --- free-text rewriting ---------------------------------------------------


def test_a_home_directory_becomes_a_neutral_one():
    payload = {"checkpoint_path": "/Users/avteam/.stt/panns_data/Cnn14.pth"}

    cleaned = demo_redact.redact(payload)

    assert cleaned["checkpoint_path"] == "/Users/demo/.stt/panns_data/Cnn14.pth"
    assert "avteam" not in cleaned["checkpoint_path"]


def test_a_linux_home_directory_is_rewritten_too():
    assert demo_redact.redact_text("/home/operator/.stt/models") == "/Users/demo/.stt/models"


def test_a_windows_home_directory_is_rewritten():
    cleaned = demo_redact.redact_text(r"C:\Users\Operator\.stt\models")

    assert "Operator" not in cleaned
    assert "demo" in cleaned


def test_an_smb_share_stops_naming_a_real_server():
    cleaned = demo_redact.redact_text("//nas.church.local/recordings/2026")

    assert "nas.church.local" not in cleaned
    assert cleaned.startswith("//demo-share/")


def test_a_lan_address_is_replaced_but_loopback_is_kept():
    """Loopback is generic and pages behave differently without it."""
    cleaned = demo_redact.redact_text("peer 192.168.2.52, local 127.0.0.1")

    assert "192.168.2.52" not in cleaned
    assert "127.0.0.1" in cleaned


def test_an_email_address_is_replaced():
    assert "@example.com" in demo_redact.redact_text("contact avteam@church.org")


def test_the_recording_machines_hostname_is_replaced_case_insensitively():
    cleaned = demo_redact.redact_text("host MacStudio-AV reports OK", hostname="macstudio-av")

    assert "MacStudio-AV" not in cleaned
    assert demo_redact.PLACEHOLDER_HOST in cleaned


def test_redaction_reaches_into_nested_structures():
    payload = {"config": {"targets": [{"destination_path": "/Users/avteam/out",
                                       "smb_password": "x"}]}}

    cleaned = demo_redact.redact(payload)
    target = cleaned["config"]["targets"][0]

    assert target["destination_path"] == "/Users/demo/out"
    assert target["smb_password"] == ""


def test_redaction_does_not_mutate_the_input():
    payload = {"checkpoint_path": "/Users/avteam/x", "smb_password": "secret"}

    demo_redact.redact(payload)

    assert payload == {"checkpoint_path": "/Users/avteam/x", "smb_password": "secret"}


def test_non_string_leaves_pass_through():
    payload = {"a": None, "b": 1.5, "c": False, "d": []}

    assert demo_redact.redact(payload) == payload


# --- the review pass -------------------------------------------------------


def test_residual_flags_are_empty_once_a_payload_is_clean():
    payload = {"checkpoint_path": "/Users/avteam/.stt/x", "smb_password": "s"}

    cleaned = demo_redact.redact(payload)

    assert demo_redact.residual_flags(cleaned) == []


def test_residual_flags_name_what_survived_and_where():
    payload = {"config": {"note": "ask avteam@church.org or 192.168.2.52"}}

    flags = demo_redact.residual_flags(payload)

    assert any("config.note" in flag and "email" in flag for flag in flags)
    assert any("config.note" in flag and "192.168.2.52" in flag for flag in flags)


def test_residual_flags_report_a_hostname_that_slipped_through():
    flags = demo_redact.residual_flags({"msg": "from MacStudio-AV"}, hostname="macstudio-av")

    assert any("hostname" in flag for flag in flags)


def test_residual_flags_walk_lists():
    flags = demo_redact.residual_flags({"items": [{"p": "/Users/avteam/x"}]})

    assert any("items[0].p" in flag for flag in flags)


def test_loopback_is_not_flagged():
    assert demo_redact.residual_flags({"url": "http://127.0.0.1:8080"}) == []


# --- identifying dict keys -------------------------------------------------


def test_an_address_used_as_a_dict_key_is_redacted():
    """A map keyed by peer address carries the addresses in its keys."""
    payload = {"per_host_limits": {"192.168.2.62": 80, "127.0.0.1": 10}}

    cleaned = demo_redact.redact(payload)
    keys = set(cleaned["per_host_limits"])

    assert "192.168.2.62" not in keys
    assert "127.0.0.1" in keys


def test_two_keys_redacting_to_the_same_text_both_survive():
    payload = {"peers": {"192.168.2.62": 1, "10.0.0.5": 2}}

    cleaned = demo_redact.redact(payload)

    assert len(cleaned["peers"]) == 2
    assert sorted(cleaned["peers"].values()) == [1, 2]


def test_a_hostname_used_as_a_dict_key_is_redacted():
    cleaned = demo_redact.redact({"hosts": {"MacStudio-AV": 1}}, hostname="macstudio-av")

    assert "MacStudio-AV" not in cleaned["hosts"]


def test_residual_flags_inspect_dict_keys_too():
    flags = demo_redact.residual_flags({"peers": {"192.168.2.62": 1}})

    assert any("(key)" in flag and "192.168.2.62" in flag for flag in flags)


def test_ordinary_keys_are_left_alone():
    payload = {"models": {"small": 1, "large-v3": 2}}

    assert demo_redact.redact(payload) == payload

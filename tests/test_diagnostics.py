"""The diagnostic report: what it must contain, and what it must never leak.

The redaction rules carry the risk here. A caption is verbatim congregation
speech, so the log filter is deny-by-default and these tests exist to keep it
that way when someone adds a new print to the monolith.
"""

from __future__ import annotations

import pytest

from stt import diagnostics


# --- log filtering: deny by default ---------------------------------------


def test_an_unvetted_tag_is_dropped_rather_than_guessed_at():
    lines = ["[NEW-FEATURE] he said unto them, whatsoever"]
    assert diagnostics.filter_log_lines(lines) == []


@pytest.mark.parametrize("tag", [
    "LIVE-TRANSLATION", "TRANS-DBG", "SRT", "SRT-TRANSLATION", "SERMON",
    "LLM-TRANSLATE", "LLM-LOCAL", "TRANSLATION", "REMOTE_TRANSLATE", "HTML",
    "TTS", "SESSION-META",
])
def test_every_tag_on_the_text_path_stays_out(tag):
    """These print what somebody actually said."""
    assert tag not in diagnostics.LOG_TAGS
    assert diagnostics.filter_log_lines([f"[{tag}] and he spoke to the congregation"]) == []


def test_operational_lines_are_kept():
    lines = [
        "[INIT] Step 3/5: Loading model (whisper, backend=faster-whisper)...",
        "[OK] Model loaded successfully: whisper",
        "[ERROR] Model loading failed: unable to open file",
    ]
    assert diagnostics.filter_log_lines(lines) == lines


def test_timestamped_lines_are_matched_on_their_tag():
    line = "[2026-09-04 06:12:01.123] [INIT] Step 2/5: Trying audio device: mic"
    assert diagnostics.filter_log_lines([line]) == [line]


def test_a_worker_traceback_survives_even_though_it_has_no_tag():
    lines = [
        "Process Process-3:",
        "Traceback (most recent call last):",
        '  File "speech_to_text.py", line 20855, in main',
        "RuntimeError: unable to open model file",
    ]
    assert diagnostics.filter_log_lines(lines) == lines


def test_an_untagged_caption_line_is_dropped():
    assert diagnostics.filter_log_lines(["and the peace of God be with you all"]) == []


def test_home_directories_are_redacted_out_of_kept_lines():
    line = r"[OK] Database initialized: C:\Users\Oluwatobi Owolabi\.stt\session.db"
    out = diagnostics.filter_log_lines([line])[0]
    assert "Oluwatobi" not in out
    assert ".stt" in out, "the useful part of the path survives"


def test_an_absurdly_long_line_is_truncated_rather_than_trusted():
    out = diagnostics.filter_log_lines(["[INFO] " + "x" * 5000])[0]
    assert len(out) < 500
    assert out.endswith("…[truncated]")


def test_only_the_tail_is_kept_because_a_report_is_read_backwards():
    lines = [f"[INFO] line {i}" for i in range(50)]
    out = diagnostics.filter_log_lines(lines, limit=5)
    assert out == [f"[INFO] line {i}" for i in range(45, 50)]


def test_blank_lines_are_dropped():
    assert diagnostics.filter_log_lines(["", "   ", "[OK] fine"]) == ["[OK] fine"]


# --- config: allowlist ----------------------------------------------------


def test_secrets_never_reach_the_report():
    config = {
        "transcription": {"model": "large-v3"},
        "pairing": {"token": "super-secret-token"},
        "translation": {"api_key": "sk-live-abcdef", "enabled": True},
        "file_mover": {"smb_password": "hunter2", "smb_user": "ada"},
        "analytics": {"install_id": "752c46ee-9261"},
    }
    summary = diagnostics.summarise_config(config)
    blob = repr(summary)

    assert "super-secret-token" not in blob
    assert "sk-live-abcdef" not in blob
    assert "hunter2" not in blob
    assert "752c46ee" not in blob
    assert summary["translation.enabled"] is True


def test_a_new_setting_beside_a_secret_is_not_reported_by_accident():
    summary = diagnostics.summarise_config({"pairing": {"token": "x", "new_knob": 7}})
    assert summary == {}


def test_absent_settings_are_skipped_not_reported_as_null():
    summary = diagnostics.summarise_config({"audio": {"energy_threshold": 100}})
    assert summary == {"audio.energy_threshold": 100}


def test_a_model_path_containing_a_username_is_redacted():
    summary = diagnostics.summarise_config(
        {"translation": {"model": "/Users/ada/models/nllb"}}
    )
    assert "ada" not in summary["translation.model"]


def test_lookup_survives_a_non_mapping_midway():
    assert diagnostics.summarise_config({"audio": "not-a-dict"}) == {}


# --- model health: the question issue #8 needed answered -------------------


def test_a_complete_model_is_reported_ok():
    health = diagnostics.check_model_dir("faster-whisper-large-v3", [
        ("model.bin", 3_090_000_000),
        ("config.json", 2_000),
        ("tokenizer.json", 2_400_000),
        ("vocabulary.json", 1_000_000),
    ])
    assert health.ok is True
    assert health.missing == ()
    assert "Looks complete." in health.notes


def test_a_truncated_download_is_caught():
    """The exact shape that leaves the UI on STARTING for ever."""
    health = diagnostics.check_model_dir("faster-whisper-large-v3", [
        ("model.bin", 400_000_000),  # died part-way through a 3 GB transfer
        ("config.json", 2_000),
        ("tokenizer.json", 2_400_000),
        ("vocabulary.json", 1_000_000),
    ])
    assert health.truncated is True
    assert health.ok is False
    assert any("Repair" in note for note in health.notes)


def test_a_missing_tokenizer_is_named_as_the_thing_that_hangs_the_start():
    health = diagnostics.check_model_dir("faster-whisper-large-v3", [
        ("model.bin", 3_090_000_000),
        ("config.json", 2_000),
    ])
    assert health.ok is False
    assert "tokenizer.json" in health.missing
    assert any("Repair" in note for note in health.notes)


def test_a_directory_with_no_weights_is_not_a_downloaded_model():
    health = diagnostics.check_model_dir("faster-whisper-small", [("config.json", 2_000)])
    assert health.ok is False
    assert health.weight_bytes == 0
    assert any("not really downloaded" in note for note in health.notes)


def test_an_unknown_model_name_is_not_guessed_as_truncated():
    health = diagnostics.check_model_dir("faster-whisper-custom-thing", [
        ("model.bin", 1_000),
        ("config.json", 1), ("tokenizer.json", 1), ("vocabulary.json", 1),
    ])
    assert health.truncated is False, "no size hint means no opinion"
    assert health.ok is True


def test_safetensors_layouts_are_recognised_as_weights():
    health = diagnostics.check_model_dir("nllb-200-distilled-600M", [
        ("model.safetensors", 2_400_000_000),
        ("config.json", 2_000),
        ("tokenizer.json", 1_000),
    ], family="transformers")
    assert health.ok is True


# --- assembly -------------------------------------------------------------


def _report(**overrides):
    base = dict(
        versions={"app": "26.3.16"},
        platform={"os": "windows"},
        hardware={"ram_gb": 8},
        config={"audio": {"energy_threshold": 100}},
        models=[diagnostics.check_model_dir("faster-whisper-base", [
            ("model.bin", 145_000_000), ("config.json", 1),
            ("tokenizer.json", 1), ("vocabulary.json", 1)])],
        audio_devices=["Microphone (Conexant ISST Audio)"],
        transcription_state={"status": "starting", "running": False, "error": None},
        log_lines=["[INIT] Step 3/5: Loading model"],
    )
    base.update(overrides)
    return diagnostics.build_report(**base)


def test_the_current_caption_is_not_copied_out_of_the_shared_state():
    report = _report(transcription_state={
        "status": "running",
        "current_text": "and he said unto them",
        "last_transcription": "peace be with you",
    })
    blob = repr(report["transcription_state"])
    assert "said unto them" not in blob
    assert "peace be with you" not in blob
    assert report["transcription_state"]["status"] == "running"


def test_the_report_says_it_is_not_sent_anywhere():
    assert "not sent anywhere" in _report()["generated"]


def test_text_rendering_includes_every_section():
    text = diagnostics.format_report_text(_report())
    for heading in ("Versions", "Platform", "Hardware", "Settings",
                    "Models on disk", "Audio devices", "Transcription state", "Log"):
        assert heading in text


def test_a_bad_model_is_visibly_marked_in_the_text():
    report = _report(models=[diagnostics.check_model_dir(
        "faster-whisper-large-v3", [("model.bin", 1_000)])])
    text = diagnostics.format_report_text(report)
    assert "[BAD ]" in text


def test_rendering_an_empty_report_does_not_explode():
    text = diagnostics.format_report_text({})
    assert "STT diagnostic report" in text


def test_the_log_section_states_the_privacy_rule():
    assert "captions are never included" in diagnostics.format_report_text(_report())


# --- filename -------------------------------------------------------------


def test_filename_is_safe_on_windows():
    name = diagnostics.report_filename("2026-09-04 06:12:01")
    assert ":" not in name and " " not in name
    assert name.startswith("stt-diagnostic-") and name.endswith(".txt")


# --- which directories are models at all ----------------------------------


@pytest.mark.parametrize("dir_name,files", [
    (".hf_cache", ["version.txt", "some.lock"]),
    ("tts", []),
    ("panns_data", ["readme.md"]),
])
def test_a_cache_or_parent_directory_is_not_judged_as_a_model(dir_name, files):
    """Reporting a healthy install's cache folder as corrupt invites damage."""
    assert diagnostics.infer_family(dir_name, files) is None


def test_a_gguf_directory_is_recognised_and_needs_no_companions():
    family = diagnostics.infer_family(
        "unsloth--gemma-4-12B-it-GGUF", ["gemma-4-12b-it-Q4_K_M.gguf"])
    assert family == "gguf"

    health = diagnostics.check_model_dir(
        "unsloth--gemma-4-12B-it-GGUF",
        [("gemma-4-12b-it-Q4_K_M.gguf", 7_300_000_000)],
        family=family,
    )
    assert health.ok is True
    assert health.missing == ()


def test_a_faster_whisper_directory_is_recognised_by_name():
    assert diagnostics.infer_family("faster-whisper-large-v3", ["model.bin"]) == "faster-whisper"


def test_an_openai_whisper_checkpoint_is_recognised():
    assert diagnostics.infer_family("whisper-medium", ["medium.pt"]) == "whisper"


def test_a_transformers_directory_is_recognised():
    assert diagnostics.infer_family(
        "facebook--nllb-200", ["model.safetensors", "config.json"]) == "transformers"


def test_the_largest_shard_is_the_one_whose_size_is_judged():
    health = diagnostics.check_model_dir("facebook--nllb-200", [
        ("model-00001-of-00002.safetensors", 2_000_000_000),
        ("model-00002-of-00002.safetensors", 900_000_000),
        ("config.json", 1_000), ("tokenizer.json", 1_000),
    ], family="transformers")
    assert health.weight_bytes == 2_000_000_000
    assert health.ok is True


# --- agreement with the loader's own predicate -----------------------------
#
# The bug these exist for: this module used to demand "vocabulary.json" exactly,
# while faster-whisper globs "vocabulary.*" and real Systran repos ship
# "vocabulary.txt". A working model was reported broken, which invites an
# operator to delete something that was fine.

def test_a_model_shipping_vocabulary_txt_is_not_condemned():
    health = diagnostics.check_model_dir("faster-whisper-large-v3", [
        ("model.bin", 3_090_000_000),
        ("config.json", 2_000),
        ("tokenizer.json", 2_400_000),
        ("vocabulary.txt", 900_000),
    ])
    assert health.ok is True, "vocabulary.txt satisfies the requirement"
    assert health.missing == ()


def test_either_vocabulary_spelling_satisfies_the_requirement():
    for name in ("vocabulary.txt", "vocabulary.json"):
        health = diagnostics.check_model_dir("faster-whisper-small", [
            ("model.bin", 490_000_000), ("config.json", 1),
            ("tokenizer.json", 1), (name, 1),
        ])
        assert health.ok is True, f"{name} should be accepted"


def test_a_missing_vocabulary_is_reported_as_either_spelling():
    health = diagnostics.check_model_dir("faster-whisper-small", [
        ("model.bin", 490_000_000), ("config.json", 1), ("tokenizer.json", 1),
    ])
    assert health.missing == ("vocabulary.txt or vocabulary.json",)


def test_the_required_list_is_imported_rather_than_restated():
    """Two modules answering 'is this loadable?' is the bug; one list is the fix."""
    from stt import model_files

    assert diagnostics.REQUIRED_COMPANIONS is model_files.REQUIRED_BY_FAMILY, (
        "the report must ask model_files, not keep a second opinion"
    )


@pytest.mark.parametrize("files", [
    ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
    ["model.bin", "config.json", "tokenizer.json", "vocabulary.json"],
    ["model.bin", "config.json", "tokenizer.json"],
    ["model.bin", "config.json"],
    ["config.json"],
])
def test_the_report_and_the_loader_agree_about_a_real_directory(tmp_path, files):
    """Agreement is the property worth asserting, not either verdict alone."""
    from stt import model_files

    model_dir = tmp_path / "faster-whisper-small"
    model_dir.mkdir()
    for name in files:
        (model_dir / name).write_bytes(b"x" * 1000)

    status = model_files.faster_whisper_status(str(model_dir))
    entries = [(f, 1000) for f in files]
    health = diagnostics.check_model_dir(
        "faster-whisper-small", entries, status=status)

    assert health.ok is status.complete


# --- the disk-aware status wins -------------------------------------------


def test_the_manifest_backed_verdict_overrides_the_size_guess():
    """A directory the loader accepts must not be called truncated by a table."""
    from stt.model_files import DirStatus

    health = diagnostics.check_model_dir(
        "faster-whisper-large-v3",
        [("model.bin", 12_345), ("config.json", 1),
         ("tokenizer.json", 1), ("vocabulary.txt", 1)],
        status=DirStatus("complete", []),
    )
    assert health.ok is True, "the size hint must not overrule the real check"
    assert health.truncated is False


def test_a_status_reporting_missing_files_is_believed():
    from stt.model_files import DirStatus

    health = diagnostics.check_model_dir(
        "faster-whisper-large-v3",
        [("model.bin", 3_090_000_000)],
        status=DirStatus("incomplete", ["tokenizer.json"]),
    )
    assert health.ok is False
    assert health.missing == ("tokenizer.json",)
    assert any("Repair" in note for note in health.notes)


def test_a_truncated_weights_file_is_not_listed_twice():
    from stt.model_files import DirStatus

    health = diagnostics.check_model_dir(
        "faster-whisper-large-v3", [("model.bin", 1_000)],
        status=DirStatus("incomplete", ["model.bin"]),
    )
    assert health.missing == (), "the weights are described by the truncation note"
    assert health.truncated is True


# --- the pair a production report condemned --------------------------------
#
# Verbatim from a diagnostic report pulled off a machine whose translation was
# working. Both entries were reported BAD:
#
#   [BAD ] google--madlad400-3b-mt
#          No weights file — this model is not really downloaded.
#   [BAD ] google--madlad400-3b-mt-ct2-int8_float16
#          Missing tokenizer.json — ...
#
# Converting a model leaves the weights in the -ct2- sibling, and the original
# HuggingFace directory is often stripped afterwards to reclaim several GB — a
# workflow the app endorses. The conversion has no tokenizer because
# ctranslate2.Translator never reads one; it comes from the parent directory.


def _madlad_pair(tmp_path):
    hf = tmp_path / "google--madlad400-3b-mt"
    ct2 = tmp_path / "google--madlad400-3b-mt-ct2-int8_float16"
    hf.mkdir()
    ct2.mkdir()
    for name in ("config.json", "tokenizer.json", "special_tokens_map.json"):
        (hf / name).write_bytes(b"x" * 1400)
    (ct2 / "model.bin").write_bytes(b"x" * 2_700_000)
    (ct2 / "config.json").write_bytes(b"x" * 300)
    (ct2 / "shared_vocabulary.json").write_bytes(b"x" * 4000)
    return hf, ct2


def _judge(tmp_path, model_dir):
    from stt import model_disk, model_files

    names = sorted(p.name for p in tmp_path.iterdir())
    entries = [(f.name, f.stat().st_size) for f in model_dir.iterdir() if f.is_file()]
    family = diagnostics.infer_family(model_dir.name, [n for n, _ in entries])
    return diagnostics.check_model_dir(
        model_dir.name, entries, family=family,
        status=model_files.dir_status(str(model_dir), family),
        sibling_weights=bool(model_disk.ct2_variant_names(names, model_dir.name)),
    )


def test_a_stripped_hf_directory_beside_its_conversion_is_healthy(tmp_path):
    hf, _ = _madlad_pair(tmp_path)
    health = _judge(tmp_path, hf)

    assert health.ok is True, "the conversion holds the weights; this is a supported state"
    assert any("conversion" in note for note in health.notes)
    assert not any("not really downloaded" in note for note in health.notes)


def test_a_ct2_conversion_is_not_asked_for_a_tokenizer(tmp_path):
    _, ct2 = _madlad_pair(tmp_path)
    health = _judge(tmp_path, ct2)

    assert health.ok is True
    assert health.missing == ()
    assert "tokenizer.json" not in repr(health.notes), (
        "ctranslate2.Translator never reads a tokenizer; it comes from the parent"
    )


def test_a_conversion_is_recognised_by_name_before_its_contents(tmp_path):
    """It has model.bin like faster-whisper and config.json like transformers."""
    assert diagnostics.infer_family(
        "google--madlad400-3b-mt-ct2-int8_float16",
        ["model.bin", "config.json", "shared_vocabulary.json"]) == "ct2"


def test_a_stripped_hf_directory_with_no_conversion_is_still_broken(tmp_path):
    """The sibling is what makes it legitimate — without one it really is empty."""
    hf = tmp_path / "google--madlad400-3b-mt"
    hf.mkdir()
    (hf / "config.json").write_bytes(b"x" * 1400)

    health = _judge(tmp_path, hf)
    assert health.ok is False
    assert any("not really downloaded" in note for note in health.notes)


def test_the_report_agrees_with_the_predicate_that_was_already_right(tmp_path):
    """model_presence has known this all along; the report just never asked."""
    from stt import model_disk

    hf, ct2 = _madlad_pair(tmp_path)
    for model_dir in (hf, ct2):
        presence = model_disk.model_presence(str(tmp_path), model_dir.name)
        assert _judge(tmp_path, model_dir).ok is presence.downloaded


def test_the_faster_whisper_tokenizer_rule_stays_in_its_own_family(tmp_path):
    """That rule exists for an unkillable HTTP fetch peculiar to faster-whisper."""
    from stt import model_files

    assert "tokenizer.json" in model_files.REQUIRED_BY_FAMILY["faster-whisper"]
    for family in ("ct2", "transformers", "gguf", "whisper"):
        assert "tokenizer.json" not in model_files.REQUIRED_BY_FAMILY[family]


@pytest.mark.parametrize("dir_name,files", [
    ("en_US-lessac-medium", ["en_US-lessac-medium.onnx"]),
    ("panns", ["Cnn14_mAP=0.431.pth"]),
])
def test_tts_and_panns_directories_are_no_longer_invisible(dir_name, files):
    """.onnx and .pth were not recognised as weights, so these classified as None."""
    assert diagnostics.infer_family(dir_name, files) is not None

"""Guards on the demo's PyInstaller spec.

These are static assertions over the spec source rather than a build, because the two
failure modes they catch — a missing dynamic import, an ML library sneaking into the
bundle — otherwise surface only after a four-minute build, or worse, only when
someone runs the shipped artifact.
"""

from __future__ import annotations

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "packaging", "demo.spec")
BUILD = os.path.join(ROOT, "packaging", "build.py")
RTHOOK = os.path.join(ROOT, "packaging", "demo_rthook.py")


@pytest.fixture(scope="module")
def spec():
    with open(SPEC, encoding="utf-8") as handle:
        return handle.read()


def test_the_spec_exists():
    assert os.path.isfile(SPEC)
    assert os.path.isfile(RTHOOK)


def test_the_runtime_hook_is_what_makes_the_build_a_demo(spec):
    """Without it the artifact would start a real server and look for a microphone."""
    assert "demo_rthook.py" in spec
    with open(RTHOOK, encoding="utf-8") as handle:
        hook = handle.read()
    assert "STT_DEMO" in hook


@pytest.mark.parametrize("module", [
    "torch", "torchaudio", "transformers", "faster_whisper", "whisper",
    "ctranslate2", "panns_inference", "huggingface_hub", "numpy",
    "speech_recognition", "silero_vad", "edge_tts", "onnxruntime",
])
def test_every_ml_library_is_excluded(module, spec):
    """The excludes are what keep the download small and prove the demo cannot
    reach an inference engine."""
    assert f'"{module}"' in spec.split("excludes=[", 1)[1].split("]", 1)[0]


@pytest.mark.parametrize("module", [
    "flask", "flask_socketio", "engineio", "socketio", "jinja2", "werkzeug",
])
def test_the_web_stack_is_not_excluded(module, spec):
    excludes = spec.split("excludes=[", 1)[1].split("]", 1)[0]
    assert f'"{module}"' not in excludes


def test_the_engineio_async_driver_is_named_explicitly(spec):
    """engineio picks its driver by dynamic import, which PyInstaller cannot see.
    Without this the demo serves pages and then fails every WebSocket."""
    hidden = spec.split("hiddenimports=[", 1)[1].split("]", 1)[0]
    assert "engineio.async_drivers.threading" in hidden


@pytest.mark.parametrize("module", [
    "stt.demo_mode", "stt.demo_playback", "stt.demo_api", "stt.demo_fixtures",
])
def test_the_demo_modules_are_named_explicitly(module, spec):
    """The monolith reaches them through names PyInstaller's analysis cannot follow."""
    hidden = spec.split("hiddenimports=[", 1)[1].split("]", 1)[0]
    assert module in hidden


def test_the_pages_and_assets_are_bundled(spec):
    for asset in ('"templates"', '"static"', '.default.json'):
        assert asset in spec


def test_the_build_refuses_to_guess_which_recording_to_ship(spec):
    """A session database is congregation speech until someone decides otherwise."""
    assert "STT_DEMO_DB" in spec
    assert "SystemExit" in spec


def test_a_live_config_is_never_bundled(spec):
    """It carries the build machine's settings, and possibly its credentials."""
    assert 'endswith(".default.json")' in spec


def test_the_demo_does_not_ask_for_the_microphone(spec):
    """It never opens an audio device; asking would be a lie, and macOS would prompt."""
    assert "NSMicrophoneUsageDescription" not in spec.replace(
        "# No NSMicrophoneUsageDescription", "")


def test_the_demo_has_its_own_identity(spec):
    """So installing it can never replace a real STT."""
    assert 'bundle_identifier="com.stt.demo"' in spec
    assert 'name="STT-Demo"' in spec


def test_the_build_script_offers_the_demo_target():
    with open(BUILD, encoding="utf-8") as handle:
        build = handle.read()

    assert "--demo" in build
    assert "demo.spec" in build
    # Building one target must not delete the other's artifacts.
    assert 'shutil.rmtree(os.path.join(ROOT, "dist"))' not in build

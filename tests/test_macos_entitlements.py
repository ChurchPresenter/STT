"""The signed macOS app must be allowed to ask for the microphone.

A hardened-runtime app without com.apple.security.device.audio-input is refused
the microphone silently: no prompt, no entry under Privacy > Microphone, and
every device still lists and opens but delivers zeros. A Mac user hit exactly
that; the entitlement was missing from every signed build until then.
"""

import plistlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTITLEMENTS = REPO / "installer" / "macos" / "entitlements.plist"
BUILD_WORKFLOW = REPO / ".github" / "workflows" / "build.yml"


def test_the_app_may_use_the_microphone():
    with ENTITLEMENTS.open("rb") as f:
        assert plistlib.load(f).get("com.apple.security.device.audio-input") is True


def test_the_signed_build_applies_these_entitlements():
    workflow = BUILD_WORKFLOW.read_text(encoding="utf-8")
    assert "--entitlements installer/macos/entitlements.plist" in workflow
    assert "--options runtime" in workflow


def test_the_app_says_why_it_asks():
    """The entitlement allows asking; this is the text of the prompt itself."""
    assert "NSMicrophoneUsageDescription" in (REPO / "packaging" / "watchdog.spec").read_text(encoding="utf-8")

"""The wheels-only install policy, and the scripts that must mirror it.

The bug this module exists for: on a macOS 11 Apple Silicon Mac, llvmlite's
arm64 wheel tag had moved to macosx_12_0, so nothing matched. uv did not
backtrack to the last release that fits — it fell back to the source
distribution, tried to compile LLVM bindings with no toolchain present, and
failed. Unattended setup then retried the same doomed install for hours.

Two things are pinned here: that the policy names the packages that actually
bit, and that install.sh/install.ps1 still pass the same list. The scripts
cannot import the module (they run before the checkout exists), so drift
between them is invisible until someone's install breaks.
"""

import pathlib
import re

import pytest

from stt import wheel_policy

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_the_packages_that_broke_installs_are_covered():
    for package in ("llvmlite", "numba", "scikit-learn", "scipy"):
        assert package in wheel_policy.WHEEL_ONLY_PACKAGES


def test_pure_python_packages_are_not_listed():
    """Forbidding a source build for an sdist-only package makes it
    uninstallable. Only compiled packages that publish wheels broadly belong."""
    for package in ("panns-inference", "openai-whisper", "edge-tts", "supertonic"):
        assert package not in wheel_policy.WHEEL_ONLY_PACKAGES


def test_the_flag_is_a_single_comma_separated_value():
    args = wheel_policy.only_binary_args()
    assert args[0] == "--only-binary"
    assert args[1].split(",") == list(wheel_policy.WHEEL_ONLY_PACKAGES)
    assert " " not in args[1]


@pytest.mark.parametrize("script", ["install.sh", "install.ps1"])
def test_the_installers_mirror_the_policy(script):
    body = (REPO / script).read_text(encoding="utf-8")
    match = re.search(r'ONLY_BINARY\s*=\s*"([^"]+)"', body)
    assert match, f"{script} no longer defines ONLY_BINARY"
    assert match.group(1) == wheel_policy.only_binary_value()


@pytest.mark.parametrize("script", ["install.sh", "install.ps1"])
def test_every_install_invocation_passes_the_flag(script):
    """A branch that forgets the flag is the one that reintroduces the bug:
    the macOS branch of install.sh is exactly where it would have mattered."""
    body = (REPO / script).read_text(encoding="utf-8")
    installs = [line for line in body.splitlines()
                if re.search(r"pip install .*-r ", line)]
    assert installs
    for line in installs:
        assert "--only-binary" in line, line

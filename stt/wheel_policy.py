"""Which packages must never be built from source on a user's machine.

uv, like pip, prefers a wheel but silently falls back to the source
distribution when no wheel matches the platform. On a developer's laptop that
is a slow install; on a church PC provisioned unattended it is a hard failure,
because the machine has no compiler and nobody is watching.

The failure that produced this module: a macOS 11 (Big Sur) Apple Silicon Mac.
``llvmlite`` moved its arm64 wheel tag from ``macosx_11_0`` to ``macosx_12_0``
at 0.48, so on that machine the newest llvmlite had no usable wheel. uv did not
fall back to 0.47, which does — it fell back to the *sdist*, tried to compile
LLVM bindings with no toolchain present, and failed. First-run setup then
retried the same doomed install every ten minutes for hours.

Pinning an old llvmlite would have fixed that one machine by holding every
other install back, and would not have stopped the next package from doing the
same thing (``scikit-learn``, which librosa also pulls, moved its arm64 tag the
same way). Refusing the source fallback is the general fix: uv must then
backtrack to a version that *does* publish a wheel for this machine, so a
modern Mac still gets the newest build and an older one quietly gets the last
release that supports it.

Only compiled scientific packages are listed. Each publishes wheels broadly, so
forbidding a source build costs nothing; pure-Python dependencies are often
sdist-only and must stay buildable.

The installer scripts (install.sh, install.ps1) apply the same list in shell,
before this package can be imported. This module is the source of truth they
mirror, and tests/test_wheel_policy.py asserts they have not drifted.
"""

from __future__ import annotations

from typing import List, Tuple

#: Compiled packages that must come as wheels or not at all. All are transitive
#: dependencies of librosa (itself required by panns-inference/torchlibrosa),
#: which is where every source-build failure seen in the field has come from.
WHEEL_ONLY_PACKAGES: Tuple[str, ...] = (
    "llvmlite",
    "numba",
    "numpy",
    "scikit-learn",
    "scipy",
    "soxr",
)


def only_binary_value() -> str:
    """The comma-separated package list, as uv/pip expect it after the flag."""
    return ",".join(WHEEL_ONLY_PACKAGES)


def only_binary_args() -> List[str]:
    """The arguments to splice into a uv/pip command line."""
    return ["--only-binary", only_binary_value()]

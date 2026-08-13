"""Which PyTorch wheel index a machine's GPU actually needs.

PyTorch's CUDA 12.8 Windows/Linux builds dropped the older architectures. On a Pascal
card (GTX 10-series, compute capability 6.1) a cu128 install still reports
``torch.cuda.is_available() == True`` — and then every kernel launch fails with "no
kernel image is available for execution on the device". The installers picked cu128 on
the mere presence of nvidia-smi, so that card was a broken install with a working-looking
probe. A field report on a GTX 1080 Ti found it the hard way.

The rule is a compute-capability threshold: 7.0 (Volta) is where cu128's architecture
list begins, so anything below it gets cu126, whose list still carries sm_61.

An unknown capability returns None — no index, i.e. the default PyPI wheel — because
guessing is what caused the bug. A CPU wheel is slow and obvious; a CUDA wheel with no
kernels for the card is fast to install and fails only once a service is running.

The installer scripts (install.sh, install.ps1) must apply the same threshold in shell,
before this package can be imported. This module is the source of truth they mirror.
"""

from __future__ import annotations

from typing import Optional

CU128_INDEX = "https://download.pytorch.org/whl/cu128"
CU126_INDEX = "https://download.pytorch.org/whl/cu126"

# Volta. cu128 ships sm_70 and up; sm_61 (Pascal) and older need the cu126 wheels.
MIN_CU128_CAPABILITY = 7.0


def parse_compute_capability(raw: Optional[str]) -> Optional[float]:
    """``nvidia-smi --query-gpu=compute_cap`` output as a float, or None.

    None means "could not tell", which every caller must treat as "do not choose a
    CUDA index" rather than as a default. Multi-GPU output is one line per card, and
    the *lowest* capability governs: a wheel set has to run on every card in the box,
    and the weakest one decides which kernels must exist.
    """
    if raw is None:
        return None
    caps = []
    for line in str(raw).splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            caps.append(float(text))
        except ValueError:
            continue  # a header, an error message, "N/A" on an old driver
    return min(caps) if caps else None


def torch_index_url(compute_cap: Optional[str], *, has_gpu: bool = True) -> Optional[str]:
    """The ``--extra-index-url`` to install torch with, or None for plain PyPI.

    ``has_gpu`` False short-circuits to None: no NVIDIA card, no CUDA wheels. With a
    card but an unreadable capability the answer is also None, which installs the
    default wheel — CPU on Windows, CUDA-bundled on Linux — and is the conservative
    end of a choice that cannot be verified.
    """
    if not has_gpu:
        return None
    capability = parse_compute_capability(compute_cap)
    if capability is None:
        return None
    return CU128_INDEX if capability >= MIN_CU128_CAPABILITY else CU126_INDEX


def index_args(compute_cap: Optional[str], *, has_gpu: bool = True) -> list:
    """The index arguments to splice into a pip/uv command line, possibly empty."""
    url = torch_index_url(compute_cap, has_gpu=has_gpu)
    return ["--extra-index-url", url] if url else []

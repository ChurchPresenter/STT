"""Which torch wheel index a GPU needs (stt/cuda_index.py).

The bug this exists to prevent: the installers appended the cu128 index whenever
nvidia-smi was present. cu128 carries no sm_61 kernels, so a GTX 1080 Ti installed
cleanly, reported torch.cuda.is_available() == True, and then failed every kernel
launch with "no kernel image is available for execution on the device" — a broken
install with a working-looking probe, found in the field rather than here.
"""

import pytest

from stt.cuda_index import (
    CU126_INDEX,
    CU128_INDEX,
    index_args,
    parse_compute_capability,
    torch_index_url,
)


class TestParseComputeCapability:
    @pytest.mark.parametrize("raw,expected", [
        ("6.1", 6.1), ("8.9\n", 8.9), ("  7.5  ", 7.5),
    ])
    def test_a_single_card(self, raw, expected):
        assert parse_compute_capability(raw) == expected

    def test_the_lowest_card_governs(self):
        # The wheels have to run on every card in the box, so the weakest one decides
        # which kernels must exist.
        assert parse_compute_capability("8.9\n6.1\n") == 6.1

    @pytest.mark.parametrize("raw", [None, "", "   ", "N/A", "Failed to initialize NVML"])
    def test_unreadable_output_is_not_a_number(self, raw):
        assert parse_compute_capability(raw) is None

    def test_junk_lines_are_skipped_not_fatal(self):
        assert parse_compute_capability("compute_cap\n8.6\n") == 8.6


class TestTorchIndexUrl:
    @pytest.mark.parametrize("cap", ["7.0", "7.5", "8.6", "8.9", "9.0"])
    def test_volta_and_newer_get_cu128(self, cap):
        assert torch_index_url(cap) == CU128_INDEX

    @pytest.mark.parametrize("cap", ["6.1", "6.0", "5.2", "3.7"])
    def test_pre_volta_gets_cu126(self, cap):
        # 6.1 is the GTX 1080 Ti from the report.
        assert torch_index_url(cap) == CU126_INDEX

    def test_the_threshold_is_inclusive_at_volta(self):
        assert torch_index_url("7.0") == CU128_INDEX
        assert torch_index_url("6.9") == CU126_INDEX

    def test_no_gpu_means_no_index(self):
        assert torch_index_url("8.9", has_gpu=False) is None

    @pytest.mark.parametrize("cap", [None, "", "N/A"])
    def test_an_unreadable_capability_never_guesses(self, cap):
        # None installs the default wheel. Guessing cu128 is exactly what broke the
        # Pascal box: a CPU build is slow and obvious, a CUDA build with no kernels
        # for the card looks perfect until the first service.
        assert torch_index_url(cap) is None

    def test_a_mixed_box_is_served_by_the_weakest_card(self):
        assert torch_index_url("8.9\n6.1") == CU126_INDEX


class TestIndexArgs:
    def test_it_splices_into_a_command_line(self):
        assert index_args("8.9") == ["--extra-index-url", CU128_INDEX]

    def test_nothing_to_splice_when_undecidable(self):
        assert index_args(None) == []
        assert index_args("8.9", has_gpu=False) == []

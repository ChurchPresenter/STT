"""stt/fd_limit.py: raising launchd's 256-file soft limit without ever lowering it."""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

import pytest

from stt import fd_limit

INF = -1  # stand-in for RLIM_INFINITY in the fake


class FakeResource:
    RLIMIT_NOFILE = 8
    RLIM_INFINITY = INF

    def __init__(self, soft: int, hard: int, reject_above: Optional[int] = None, get_error: bool = False) -> None:
        self.soft = soft
        self.hard = hard
        self.reject_above = reject_above
        self.get_error = get_error
        self.calls: List[Tuple[int, int]] = []

    def getrlimit(self, which: int) -> Tuple[int, int]:
        assert which == self.RLIMIT_NOFILE
        if self.get_error:
            raise OSError("no")
        return self.soft, self.hard

    def setrlimit(self, which: int, limits: Tuple[int, int]) -> None:
        self.calls.append(limits)
        if self.reject_above is not None and limits[0] > self.reject_above:
            raise ValueError("not allowed")
        self.soft, self.hard = limits


def test_raises_launchd_default_to_target() -> None:
    res = FakeResource(256, INF)
    assert fd_limit.raise_nofile_limit(resource_mod=res) == (256, 10240)
    assert res.soft == 10240
    assert res.hard == INF  # hard limit untouched


def test_capped_by_hard_limit() -> None:
    res = FakeResource(256, 1024)
    assert fd_limit.raise_nofile_limit(resource_mod=res) == (256, 1024)
    assert res.hard == 1024


def test_never_lowers_a_higher_soft_limit() -> None:
    res = FakeResource(65536, INF)
    assert fd_limit.raise_nofile_limit(resource_mod=res) == (65536, 65536)
    assert res.calls == []


def test_infinite_soft_limit_is_left_alone() -> None:
    res = FakeResource(INF, INF)
    assert fd_limit.raise_nofile_limit(resource_mod=res) == (INF, INF)
    assert res.calls == []


def test_falls_back_when_kernel_rejects_target() -> None:
    res = FakeResource(256, INF, reject_above=5000)
    assert fd_limit.raise_nofile_limit(resource_mod=res) == (256, 4096)


def test_total_refusal_leaves_limit_unchanged() -> None:
    res = FakeResource(256, INF, reject_above=100)
    assert fd_limit.raise_nofile_limit(resource_mod=res) == (256, 256)
    assert res.soft == 256


def test_unreadable_limit_returns_none() -> None:
    res = FakeResource(256, INF, get_error=True)
    assert fd_limit.raise_nofile_limit(resource_mod=res) is None
    assert fd_limit.current_nofile_limit(resource_mod=res) is None


def test_no_resource_module_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fd_limit, "_load_resource", lambda: None)
    assert fd_limit.raise_nofile_limit() is None
    assert fd_limit.current_nofile_limit() is None


def test_current_limit_reports_soft_and_hides_infinity() -> None:
    assert fd_limit.current_nofile_limit(resource_mod=FakeResource(256, INF)) == 256
    assert fd_limit.current_nofile_limit(resource_mod=FakeResource(INF, INF)) is None


@pytest.mark.skipif(sys.platform == "win32", reason="no /dev/fd on Windows")
def test_open_fd_count_tracks_new_descriptors() -> None:
    before = fd_limit.open_fd_count()
    assert before is not None
    r, w = os.pipe()
    try:
        assert fd_limit.open_fd_count() == before + 2
    finally:
        os.close(r)
        os.close(w)


def test_open_fd_count_unknown_directory(tmp_path: Any) -> None:
    assert fd_limit.open_fd_count(str(tmp_path / "missing")) is None

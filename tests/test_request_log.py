"""Unit tests for stt.request_log — the size-capped SQLite access log."""

import os

import pytest

from stt.request_log import (
    KIND_ACTION,
    KIND_CONNECTION,
    KIND_HTTP,
    SOURCE_API,
    SOURCE_SOCKET,
    SOURCE_WEB,
    RequestLog,
    classify_source,
)


@pytest.fixture()
def log(tmp_path):
    store = RequestLog(os.path.join(tmp_path, "log.db"))
    yield store
    store.close()


def test_classify_source_api_vs_web():
    assert classify_source("/api/config") == SOURCE_API
    assert classify_source("/api/logs") == SOURCE_API
    assert classify_source("/") == SOURCE_WEB
    assert classify_source("/server-settings") == SOURCE_WEB
    # A path that merely contains "api" but isn't under the prefix is web.
    assert classify_source("/rapid") == SOURCE_WEB


def test_classify_source_custom_prefix():
    assert classify_source("/v1/config", api_prefix="/v1/") == SOURCE_API
    assert classify_source("/api/config", api_prefix="/v1/") == SOURCE_WEB


def test_log_and_query_roundtrip(log):
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, method="GET", path="/",
            status=200, ip="127.0.0.1", user_agent="curl", duration_ms=1.5)
    rows = log.query()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == SOURCE_WEB
    assert row["kind"] == KIND_HTTP
    assert row["method"] == "GET"
    assert row["path"] == "/"
    assert row["status"] == 200
    assert row["ip"] == "127.0.0.1"
    assert row["duration_ms"] == 1.5
    assert isinstance(row["ts"], float)


def test_query_newest_first(log):
    for i in range(5):
        log.log(source=SOURCE_API, kind=KIND_HTTP, path=f"/api/{i}", ts=float(i))
    rows = log.query()
    paths = [r["path"] for r in rows]
    assert paths == ["/api/4", "/api/3", "/api/2", "/api/1", "/api/0"]


def test_query_filters(log):
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/")
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/config")
    log.log(source=SOURCE_SOCKET, kind=KIND_CONNECTION, detail="connect sid=abc")
    log.log(source=SOURCE_SOCKET, kind=KIND_ACTION, detail="approve_staged")

    assert len(log.query(source=SOURCE_API)) == 1
    assert log.query(source=SOURCE_API)[0]["path"] == "/api/config"
    assert len(log.query(source=SOURCE_SOCKET)) == 2
    assert len(log.query(kind=KIND_ACTION)) == 1
    assert len(log.query(source=SOURCE_SOCKET, kind=KIND_CONNECTION)) == 1


def test_query_search_matches_path_ip_and_detail(log):
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/translation", ip="10.0.0.5")
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="192.168.1.9")
    log.log(source=SOURCE_SOCKET, kind=KIND_ACTION, detail="submit_correction")

    assert len(log.query(search="translation")) == 1
    assert len(log.query(search="192.168")) == 1
    assert len(log.query(search="correction")) == 1
    assert len(log.query(search="nothing-here")) == 0


def test_query_limit(log):
    for i in range(10):
        log.log(source=SOURCE_WEB, kind=KIND_HTTP, path=f"/{i}", ts=float(i))
    assert len(log.query(limit=3)) == 3


def test_pruning_caps_row_count(tmp_path):
    store = RequestLog(os.path.join(tmp_path, "log.db"), max_rows=10, prune_every=5)
    try:
        for i in range(50):
            store.log(source=SOURCE_WEB, kind=KIND_HTTP, path=f"/{i}", ts=float(i))
        # Never exceeds the cap by more than one prune window.
        assert store.count() <= 10
        # The newest row is retained.
        assert store.query(limit=1)[0]["path"] == "/49"
    finally:
        store.close()


def test_clear(log):
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/")
    assert log.count() == 1
    log.clear()
    assert log.count() == 0
    assert log.query() == []


def test_log_never_raises_after_close(tmp_path):
    store = RequestLog(os.path.join(tmp_path, "log.db"))
    store.close()
    # Logging against a closed connection must be swallowed, not raised.
    store.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/")


def test_persistence_across_reopen(tmp_path):
    db = os.path.join(tmp_path, "log.db")
    store = RequestLog(db)
    store.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/x")
    store.close()

    reopened = RequestLog(db)
    try:
        assert reopened.count() == 1
        assert reopened.query()[0]["path"] == "/api/x"
    finally:
        reopened.close()


def _seed(log, *, ts, status=200, duration_ms=1.0, path="/api/x"):
    log.log(source=SOURCE_API, kind=KIND_HTTP, method="GET", path=path,
            status=status, duration_ms=duration_ms, ts=ts)


def test_stats_empty_window_is_zeros(log):
    s = log.stats(300, now=1000.0)
    assert s["requests"] == 0
    assert s["error_count"] == 0
    assert s["error_rate"] == 0.0
    assert s["server_error_count"] == 0
    assert s["server_error_rate"] == 0.0
    assert s["client_error_count"] == 0
    assert s["client_error_rate"] == 0.0
    assert s["req_per_min"] == 0.0
    assert s["p50_ms"] is None
    assert s["p95_ms"] is None


def test_stats_counts_rate_and_errors(log):
    now = 10_000.0
    # 6 requests inside a 60s window: 2 errors (500, 404), 4 ok.
    _seed(log, ts=now - 5, status=200, duration_ms=10)
    _seed(log, ts=now - 4, status=200, duration_ms=20)
    _seed(log, ts=now - 3, status=200, duration_ms=30)
    _seed(log, ts=now - 2, status=200, duration_ms=40)
    _seed(log, ts=now - 1, status=500, duration_ms=50)
    _seed(log, ts=now - 1, status=404, duration_ms=60)
    # One request OUTSIDE the window must be excluded.
    _seed(log, ts=now - 120, status=200, duration_ms=999)

    s = log.stats(60, now=now)
    assert s["requests"] == 6
    assert s["error_count"] == 2
    assert s["error_rate"] == round(2 / 6, 3)
    # split: the 500 is a server error, the 404 a client error
    assert s["server_error_count"] == 1
    assert s["server_error_rate"] == round(1 / 6, 3)
    assert s["client_error_count"] == 1
    assert s["client_error_rate"] == round(1 / 6, 3)
    assert s["req_per_min"] == 6.0  # 6 requests over a 60s window
    # durations 10..60 sorted; nearest-rank p50 -> index round(0.5*5)=2 -> 30
    assert s["p50_ms"] == 30.0
    # p95 -> index round(0.95*5)=5 -> 60
    assert s["p95_ms"] == 60.0


def test_stats_4xx_only_is_no_server_error(log):
    # A window full of 4xx (e.g. a non-whitelisted viewer polling a gated
    # endpoint) must report zero server errors, so the health tile stays green.
    now = 20_000.0
    _seed(log, ts=now - 3, status=200, duration_ms=10)
    _seed(log, ts=now - 2, status=403, duration_ms=20)
    _seed(log, ts=now - 1, status=403, duration_ms=30)
    _seed(log, ts=now - 1, status=404, duration_ms=40)

    s = log.stats(60, now=now)
    assert s["error_count"] == 3            # total >= 400
    assert s["server_error_count"] == 0     # no 5xx
    assert s["server_error_rate"] == 0.0
    assert s["client_error_count"] == 3     # all 4xx


def test_stats_ignores_null_durations_in_percentiles(log):
    now = 5000.0
    _seed(log, ts=now - 1, status=200, duration_ms=10)
    # a socket-style row with no duration
    log.log(source=SOURCE_SOCKET, kind=KIND_CONNECTION, ts=now - 1)
    s = log.stats(60, now=now)
    assert s["requests"] == 2      # both counted as requests
    assert s["p50_ms"] == 10.0     # only the timed one feeds the percentile


def test_query_exclude_paths(log):
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/logs", status=200)
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/health", status=200)
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/translate", status=200)
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/server-settings", status=200)
    # a socket action row has no path — must survive the exclusion
    log.log(source=SOURCE_SOCKET, kind=KIND_ACTION, detail="start")

    rows = log.query(exclude_paths=["/api/logs", "/api/health"])
    paths = sorted(r["path"] for r in rows if r["path"])
    assert paths == ["/api/translate", "/server-settings"]
    # the null-path socket row is still returned
    assert any(r["path"] is None for r in rows)

    # no exclusion returns everything
    assert len(log.query()) == 5


def test_synchronous_is_normal_not_full(tmp_path):
    """An access log must not fsync on every request.

    WAL alone leaves synchronous at FULL. With a display client and the health page
    polling, a service is thousands of fsyncs on the same disk the session audio is being
    written to. NORMAL is what the session database itself uses; the worst case it admits
    is losing the last few log rows on a power cut.
    """
    log = RequestLog(str(tmp_path / "access.db"))
    try:
        mode = log._conn.execute("PRAGMA synchronous").fetchone()[0]
        assert mode == 1, f"expected NORMAL (1), got {mode}"
        assert log._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        log.close()


def test_query_filters_by_exact_ip(log):
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="192.168.2.62")
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/logs", ip="192.168.2.52")
    # A prefix of a logged address must not match — the filter is exact.
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/health", ip="192.168.2.620")
    rows = log.query(ip="192.168.2.62")
    assert [r["path"] for r in rows] == ["/"]


def test_query_combines_ip_with_other_filters(log):
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="10.0.0.1")
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/config", ip="10.0.0.1")
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/config", ip="10.0.0.2")
    rows = log.query(ip="10.0.0.1", source=SOURCE_API)
    assert len(rows) == 1
    assert rows[0]["path"] == "/api/config"


def test_distinct_ips_counts_and_orders_by_traffic(log):
    for _ in range(3):
        log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="10.0.0.1", ts=5.0)
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="10.0.0.2", ts=9.0)
    rows = log.distinct_ips()
    assert [r["ip"] for r in rows] == ["10.0.0.1", "10.0.0.2"]
    assert rows[0]["count"] == 3
    assert rows[0]["last_ts"] == 5.0
    assert rows[1]["last_ts"] == 9.0


def test_distinct_ips_skips_rows_without_an_ip(log):
    log.log(source=SOURCE_SOCKET, kind=KIND_CONNECTION, path="connect")
    log.log(source=SOURCE_SOCKET, kind=KIND_ACTION, path="start", ip="")
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="10.0.0.1")
    assert [r["ip"] for r in log.distinct_ips()] == ["10.0.0.1"]


def test_distinct_ips_honours_exclude_paths(log):
    # An address that only ever polled the dashboard is not a client when
    # polling is hidden.
    log.log(source=SOURCE_API, kind=KIND_HTTP, path="/api/health", ip="10.0.0.9")
    log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip="10.0.0.1")
    rows = log.distinct_ips(exclude_paths=["/api/health"])
    assert [r["ip"] for r in rows] == ["10.0.0.1"]


def test_distinct_ips_respects_limit(log):
    for i in range(10):
        log.log(source=SOURCE_WEB, kind=KIND_HTTP, path="/", ip=f"10.0.0.{i}")
    assert len(log.distinct_ips(limit=4)) == 4

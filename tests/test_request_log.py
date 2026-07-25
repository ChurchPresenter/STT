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

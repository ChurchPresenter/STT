"""A refused request repeated by one client must not fill the access log.

The log is capped by row count, so a page polling an endpoint it is not allowed
to reach evicts real traffic — one display client did exactly that for a week.
The browser-side guard stops the traffic once the page reloads; this keeps the
log usable in the meantime.

The hook is extracted from the monolith and run against a stub namespace (see
tests/conftest.py) — the module cannot be imported, and CI installs no Flask.
"""

from conftest import extract_definitions
from stt import repeat_filter as repeat_filter_mod

A_IP = "192.168.2.90"


def build(*, first_n=3, cooldown=600, clock=None):
    ns = extract_definitions(
        "speech_to_text.py", ["_refusal_log_verdict"],
        extra_globals={
            "request": type("R", (), {"method": "GET"})(),
            "_repeat_filter": repeat_filter_mod,
            "_REFUSAL_STATUSES": (401, 403, 404),
            "_refusal_log_filter": repeat_filter_mod.RepeatSuppressor(
                first_n=first_n, cooldown_seconds=cooldown, clock=clock),
        },
    )
    return ns["_refusal_log_verdict"]


def test_a_successful_request_is_always_logged():
    verdict = build()
    for _ in range(100):
        assert verdict("/api/service-phase", 200, A_IP).log is True


def test_a_client_error_that_is_not_a_refusal_is_always_logged():
    # A 400 differs run to run and is what someone reads to work out why.
    verdict = build()
    for _ in range(100):
        assert verdict("/api/transcription/start", 400, A_IP).log is True


def test_a_repeated_refusal_is_thinned():
    verdict = build(first_n=3)
    logged = sum(1 for _ in range(60) if verdict("/api/service-phase", 403, A_IP).log)
    assert logged == 3


def test_the_thinning_is_per_client_and_path():
    verdict = build(first_n=1)
    assert verdict("/api/service-phase", 403, A_IP).log is True
    assert verdict("/api/service-phase", 403, A_IP).log is False
    # A different machine hitting the same wall is its own story...
    assert verdict("/api/service-phase", 403, "192.168.2.133").log is True
    # ...as is the same machine hitting a different one.
    assert verdict("/api/logs", 403, A_IP).log is True


def test_the_heartbeat_row_says_how_many_it_stands_for():
    class Clock:
        now = 1000.0

        def __call__(self):
            return self.now

    clock = Clock()
    verdict = build(first_n=1, cooldown=600, clock=clock)
    verdict("/api/service-phase", 403, A_IP)
    for _ in range(59):
        verdict("/api/service-phase", 403, A_IP)
    clock.now += 600
    row = verdict("/api/service-phase", 403, A_IP)
    assert row.log is True
    assert row.suppressed == 59


def test_a_404_from_a_peer_on_an_older_build_is_thinned_too():
    # The other flood in the log: the paired machine polling an endpoint this
    # build does not have yet, for as long as the version gap lasts.
    verdict = build(first_n=3)
    logged = sum(1 for _ in range(400) if verdict("/api/service-phase", 404, "192.168.2.52").log)
    assert logged == 3

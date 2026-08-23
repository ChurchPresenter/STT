"""The demo makes no outbound requests, and this is what proves it stays that way.

The audit test at the bottom reads the real monolith, so a choke point that loses its
guard fails the suite rather than shipping a demo that can be used as somebody else's
network client.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from stt import demo_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONOLITH = os.path.join(ROOT, "speech_to_text.py")


def audit(source, expected=("target",)):
    return demo_guard.audit_choke_points(textwrap.dedent(source), expected)


# --- the message -----------------------------------------------------------


def test_the_refusal_explains_itself_and_names_what_was_refused():
    message = demo_guard.blocked_message("a paired translation server")

    assert "demo" in message.lower()
    assert "a paired translation server" in message


def test_every_refusal_reads_the_same_way():
    first = demo_guard.blocked_message("downloads")
    second = demo_guard.blocked_message("downloads")

    assert first == second


# --- what the audit accepts ------------------------------------------------


def test_a_guard_as_the_first_statement_passes():
    assert audit("""
        def target():
            if DEMO:
                return None
            reach_the_network()
    """) == []


def test_a_guard_after_the_docstring_passes():
    assert audit('''
        def target():
            """Does a thing."""
            if DEMO:
                return None
            reach_the_network()
    ''') == []


def test_a_compound_condition_still_counts_as_a_guard():
    assert audit("""
        def target():
            if DEMO and not override:
                return None
            reach_the_network()
    """) == []


def test_a_raising_guard_counts_too():
    assert audit("""
        def target():
            if DEMO:
                raise RuntimeError("no")
            reach_the_network()
    """) == []


def test_an_async_function_is_audited_the_same_way():
    assert audit("""
        async def target():
            if DEMO:
                return None
            await reach_the_network()
    """) == []


# --- what the audit rejects ------------------------------------------------


def test_an_unguarded_function_fails():
    failures = audit("""
        def target():
            reach_the_network()
    """)

    assert len(failures) == 1
    assert "target" in failures[0]


def test_a_guard_placed_after_the_work_fails():
    """A check that runs once the request is built stops nothing."""
    failures = audit("""
        def target():
            response = reach_the_network()
            if DEMO:
                return None
            return response
    """)

    assert len(failures) == 1


def test_a_guard_on_something_other_than_demo_fails():
    failures = audit("""
        def target():
            if disabled:
                return None
            reach_the_network()
    """)

    assert len(failures) == 1


def test_a_choke_point_that_was_renamed_away_is_reported():
    failures = audit("""
        def something_else():
            if DEMO:
                return None
    """)

    assert len(failures) == 1
    assert "not found" in failures[0]


def test_every_named_choke_point_is_checked():
    failures = audit("""
        def first():
            reach_the_network()

        def second():
            if DEMO:
                return None
    """, expected=("first", "second", "third"))

    assert len(failures) == 2   # first unguarded, third missing


# --- against the real monolith ---------------------------------------------


@pytest.fixture(scope="module")
def monolith():
    with open(MONOLITH, encoding="utf-8") as handle:
        return handle.read()


def test_the_demo_cannot_reach_the_network_through_any_known_door(monolith):
    """The property this whole module exists for.

    If this fails, a demo — which binds the network and has no password — has regained
    a way to make outbound requests on behalf of whoever can reach it.
    """
    failures = demo_guard.audit_choke_points(monolith)

    assert failures == [], "unguarded outbound paths in demo mode:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("name", demo_guard.CHOKE_POINTS)
def test_each_choke_point_still_exists_in_the_monolith(name, monolith):
    assert f"def {name}(" in monolith


def test_the_download_path_refuses_in_demo_mode(monkeypatch, tmp_path):
    """stt/downloads fronts every download, including the cloudflared fetch that
    /api/tunnel/start would otherwise perform."""
    from stt import demo_mode, downloads

    monkeypatch.setenv(demo_mode.ENV_FLAG, "1")

    with pytest.raises(RuntimeError) as excinfo:
        downloads.download_url_to_file("https://example.invalid/x", str(tmp_path / "x"))

    assert "demo" in str(excinfo.value).lower()
    assert not os.path.exists(tmp_path / "x")


def test_the_download_path_is_untouched_outside_demo_mode(monkeypatch):
    from stt import demo_mode

    monkeypatch.delenv(demo_mode.ENV_FLAG, raising=False)

    assert demo_mode.enabled() is False


# --- the real functions, extracted and run ---------------------------------


def test_the_peer_door_refuses_in_demo_mode():
    """Exercised on the monolith's actual _peer_request, not a copy of it.

    This is the door the whole /api/remote-translation family, the pairing handshake
    and the offload path all go through.
    """
    from conftest import extract_definitions

    sent = []

    class _Session:
        def request(self, method, url, **kwargs):
            sent.append(url)
            raise AssertionError("a demo must not reach a paired machine")

    ns = extract_definitions("speech_to_text.py", ["_peer_request"], {
        "DEMO": True,
        "_demo_guard": demo_guard,
        "_get_remote_http_session": _Session,
        "_remote_auth_headers": dict,
    })

    with pytest.raises(RuntimeError) as excinfo:
        ns["_peer_request"]("GET", "http://192.0.2.9:8080", "/api/translation/status")

    assert "demo" in str(excinfo.value).lower()
    assert sent == []


# --- the two deliberate exceptions -----------------------------------------


def test_the_livemap_ping_says_it_is_a_demo(monolith):
    """A trial is not an install. If src stops being sent the collector counts a
    stranger trying the demo as a church running the application."""
    assert 'src="demo" if DEMO else' in monolith


def test_the_ping_and_sentry_are_not_treated_as_blocked_doors():
    """They are the exceptions; listing them here would assert the opposite."""
    assert "_send_livemap_ping" not in demo_guard.CHOKE_POINTS
    assert "_init_sentry" not in demo_guard.CHOKE_POINTS


def test_sentry_events_carry_a_demo_tag(monolith):
    """So a packaging bug in the demo never counts against the real application."""
    assert 'set_tag("demo"' in monolith

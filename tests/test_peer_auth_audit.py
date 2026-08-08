"""The paired-peer calls in the monolith must all go through one door.

The regression this guards is specific: Machine B stops accepting Machine A's
address once it has issued A a token, so a call site that omits the header is
refused — but only on a pairing that has upgraded, which is production and not
a test rig. Two call sites had drifted that way and read as "the peer is
unreachable" while the peer was healthy.
"""

import os

import pytest

from stt.peer_auth_audit import Finding, describe, direct_peer_calls

MONOLITH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "speech_to_text.py")


def test_no_peer_call_in_the_monolith_bypasses_the_door():
    """The invariant. speech_to_text.py is read as text, never imported — it has
    import-time side effects that make it unimportable from a test."""
    if not os.path.exists(MONOLITH):
        pytest.skip("speech_to_text.py not present")
    with open(MONOLITH, encoding="utf-8") as handle:
        findings = direct_peer_calls(handle.read())
    assert findings == [], (
        "These calls reach the paired peer without going through _peer_request, "
        "so they carry no bearer token and a paired Machine B will refuse them:\n"
        + describe(findings)
    )


def test_flags_a_bare_get_to_a_peer_endpoint():
    source = 'def poll():\n    return _req.get(endpoint + "/api/health", timeout=3)\n'
    assert [f.func for f in direct_peer_calls(source)] == ["poll"]


def test_flags_a_concatenated_post_to_a_translate_endpoint():
    source = (
        'def start(ep):\n'
        '    return _req.post(ep.rstrip("/") + "/api/translate/preload", timeout=10)\n'
    )
    findings = direct_peer_calls(source)
    assert len(findings) == 1
    assert findings[0].endpoint == "//api/translate/preload"  # rstrip("/") arg, then the path


def test_ignores_a_call_routed_through_the_door():
    """_peer_request is a plain call, not an attribute call, so it is not an
    HTTP send and never counts as a bypass."""
    source = 'def start(ep):\n    return _peer_request("POST", ep, "/api/translate/preload")\n'
    assert direct_peer_calls(source) == []


def test_ignores_a_non_peer_url():
    source = 'def ping():\n    return _req.get(url + "/api/telemetry", timeout=10)\n'
    assert direct_peer_calls(source) == []


def test_ignores_route_decorators_naming_a_peer_path():
    """@app.route("/api/health") declares this machine's own endpoint. Only
    request-sending attribute calls count."""
    source = '@app.route("/api/health")\ndef health():\n    return "ok"\n'
    assert direct_peer_calls(source) == []


def test_honours_the_allowlist():
    source = 'def _peer_request(m, ep, path):\n    return s.request(m, ep + "/api/translate")\n'
    assert direct_peer_calls(source, allowed=["_peer_request"]) == []
    assert [f.func for f in direct_peer_calls(source, allowed=[])] == ["_peer_request"]


def test_attributes_a_call_to_its_innermost_function():
    source = (
        'def outer():\n'
        '    def inner():\n'
        '        return _req.get(ep + "/api/health")\n'
        '    return inner\n'
    )
    assert [f.func for f in direct_peer_calls(source)] == ["inner"]


def test_finds_an_fstring_url():
    source = 'def probe(host):\n    return _req.get(f"http://{host}/api/translate/preload")\n'
    assert [f.func for f in direct_peer_calls(source)] == ["probe"]


def test_findings_are_sorted_by_line():
    source = (
        'def a():\n'
        '    _req.get(ep + "/api/health")\n'
        '    _req.post(ep + "/api/translate/preload")\n'
    )
    findings = direct_peer_calls(source)
    assert [f.line for f in findings] == sorted(f.line for f in findings)
    assert len(findings) == 2


def test_describe_names_the_line_and_function():
    text = describe([Finding(42, "/api/health", "poll")])
    assert "42" in text and "poll" in text and "/api/health" in text

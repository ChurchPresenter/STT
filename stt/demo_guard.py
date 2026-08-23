"""The demo reaches the network in exactly two ways, and here is what enforces that.

A demo is handed to strangers and, since it binds the network, is reachable by anyone
who can see the port — with its authentication deliberately switched off. So it must
not be usable as somebody else's network client: no model downloads, no calls to a
paired machine, no LLM endpoint, no cloud text-to-speech, and no port scanning.

The two exceptions are deliberate and are *not* listed below: the live-map ping and
Sentry. A demo reports that it ran and that it crashed, each tagged as a demo at the
source, because a trial on a machine we will never see is the only feedback a
downloadable artifact can give. Everything else stays shut.

The first attempt at that property was a list of intercepted routes, and a list is
exactly the wrong shape for it: the routes that were forgotten (a cloudflared download,
an SMB mount taken straight from a request body) were the ones that mattered. So the
property is enforced at the few functions every outbound request actually passes
through. A route nobody remembered still cannot reach the network, because the door it
would have to use is shut.

:func:`audit_choke_points` is what keeps that true over time — the same technique
``stt/peer_auth_audit`` uses to enforce that every peer call goes through one door. It
reads the monolith and reports any choke point that has lost its guard, so the test
suite fails rather than a demo quietly regaining network access.

Stdlib-only, pure, side-effect free.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple, Union

#: Functions in speech_to_text.py that must refuse to act when DEMO is set. Between
#: them they front every outbound request the web process can make.
#:
#: Adding a new way to reach the network means adding it here too — the audit only
#: checks what it is told about, and cannot know about a door it has not been shown.
CHOKE_POINTS: Tuple[str, ...] = (
    # Deliberately NOT here: _send_livemap_ping and Sentry. A demo reports that it ran
    # and that it crashed, both tagged as a demo at the source (src="demo" on the ping,
    # a "demo" tag on the Sentry event) so the collector counts a trial as a trial. They
    # are the only two exceptions, and everything below is what keeps them the only two.
    #
    # Every call to a paired offload machine, and the whole /api/remote-translation
    # proxy family behind it.
    "_peer_request",
    # An arbitrary OpenAI-compatible URL, sent an Authorization header and caption text.
    "_translate_via_llm",
    # A live call to Microsoft's voice list; reachable on a language switch even with
    # text-to-speech switched off.
    "get_edge_tts_voices",
    # Tries five ports against a supplied hostname — a port scanner with a config write
    # at the end of it.
    "_probe_remote_port",
)

#: The guard the audit looks for at the top of each choke point.
GUARD_NAME = "DEMO"

_MESSAGE = ("Not available in the demo: it makes no outbound network requests. "
            "Run the full application to use {what}.")


def blocked_message(what: str) -> str:
    """One explanation, so every refusal reads the same wherever it surfaces."""
    return _MESSAGE.format(what=what)


def _guards_on_demo(node: ast.AST) -> bool:
    """Whether this statement is an ``if DEMO:`` (or ``if DEMO and ...``) guard."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.BoolOp):
        return any(isinstance(value, ast.Name) and value.id == GUARD_NAME
                   for value in test.values)
    return isinstance(test, ast.Name) and test.id == GUARD_NAME


def _leading_statements(body: List[ast.stmt]) -> List[ast.stmt]:
    """A function's statements with a leading docstring skipped."""
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


def audit_choke_points(source: str,
                       expected: Tuple[str, ...] = CHOKE_POINTS) -> List[str]:
    """Names in ``expected`` that are missing, or whose guard is not the first thing
    they do.

    An empty list means the demo cannot reach the network through any known door.

    The guard must come *first*: a check placed after the request has been built, or
    after a config read that itself reaches out, does not stop anything. Requiring
    position rather than mere presence is what makes this an enforcement rather than
    a reminder.
    """
    _Func = Union[ast.FunctionDef, ast.AsyncFunctionDef]
    tree = ast.parse(source)
    found: Dict[str, _Func] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, node)

    failures: List[str] = []
    for name in expected:
        func = found.get(name)
        if func is None:
            failures.append(f"{name}: not found in the monolith")
            continue
        statements = _leading_statements(func.body)
        if not statements or not _guards_on_demo(statements[0]):
            failures.append(f"{name}: no 'if {GUARD_NAME}:' guard as its first statement")
    return failures

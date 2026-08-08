"""Keep every call to the paired translation peer going through one door.

Machine A offloads translation to Machine B, and B authenticates A by a bearer
token. Once B has issued that token, A's *address* stops being a way in — an
un-authenticated request from a machine that holds a token is refused outright
(see :mod:`stt.pair_tokens`). So a call site that forgets the Authorization
header does not degrade, it fails, and it fails only for the pairings that have
upgraded — which is exactly the deployment you cannot reproduce on a laptop.

Two of them did forget, sat beside call sites that did not, and the result read
as "the remote machine is unreachable" while the machine was healthy and one
line above was talking to it happily.

The monolith now funnels all of it through ``_peer_request``, which attaches the
token and re-claims one when B has forgotten us. This module is what keeps it
that way: it parses the source and reports any call that reaches a peer endpoint
without going through that door. It is a source audit rather than a runtime
check because the failure is a call site that is never exercised in testing —
the point is to catch it in the file, not in production.

Stdlib-only and free of Flask, so it can be tested directly and run in CI with
no ML dependencies installed.
"""

import ast
from typing import List, NamedTuple, Optional, Sequence

#: Path fragments that only ever appear in a call to the paired peer.
PEER_ENDPOINT_MARKERS = (
    "/api/translate",          # covers /api/translate and /api/translate/*
    "/api/translation/status",
    "/api/health",
)

#: Attribute names that mean "send an HTTP request" on a requests module or
#: Session. ``request`` is included because ``_peer_request`` itself uses it.
_REQUEST_METHODS = frozenset({"get", "post", "put", "patch", "delete", "request"})

#: Functions allowed to reach a peer endpoint directly, and why. Each is a case
#: the door cannot serve without biting its own tail.
ALLOWED_FUNCTIONS = {
    # The door itself.
    "_peer_request",
    # Mints the token the door attaches. Routing it through the door would mean
    # a refusal triggers a claim that triggers a refusal.
    "_claim_remote_pair_token",
    # Port discovery: tries candidate ports to find where the peer listens, so
    # there is no resolved endpoint yet and nothing to authenticate against.
    "_probe_remote_port",
}


class Finding(NamedTuple):
    """One call that reaches a peer endpoint without going through the door."""

    line: int
    endpoint: str
    func: str

    def describe(self) -> str:
        """One line an operator or a failing test can read."""
        return f"line {self.line}: {self.func}() calls {self.endpoint} directly"


def _string_parts(node: ast.AST) -> List[str]:
    """Every literal string inside an expression, in source order.

    URLs here are built by concatenation (``endpoint.rstrip("/") + "/api/health"``)
    or by f-string, so the literal fragments have to be gathered out of the
    expression tree rather than read off a single constant. Order matters: a
    marker can straddle two fragments, and ``ast.walk`` is breadth-first, which
    would reassemble ``ep.rstrip("/") + "/api/health"`` in the wrong order.
    ``iter_child_nodes`` yields fields in source order, so recurse by hand.
    """
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    parts: List[str] = []
    for child in ast.iter_child_nodes(node):
        parts.extend(_string_parts(child))
    return parts


def _peer_endpoint(node: ast.AST, markers: Sequence[str] = PEER_ENDPOINT_MARKERS) -> Optional[str]:
    """The peer path this expression addresses, or None if it addresses no peer."""
    joined = "".join(_string_parts(node))
    for marker in markers:
        if marker in joined:
            return joined
    return None


def _called_method(node: ast.Call) -> Optional[str]:
    """The attribute name being called (``x.post(...)`` -> ``"post"``)."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _url_argument(node: ast.Call, method: str) -> Optional[ast.AST]:
    """The argument holding the URL.

    ``session.request(method, url)`` puts it second; every other verb
    (``get``/``post``/…) puts it first.
    """
    index = 1 if method == "request" else 0
    return node.args[index] if len(node.args) > index else None


def direct_peer_calls(source: str, allowed: Optional[Sequence[str]] = None) -> List[Finding]:
    """Every HTTP call to a peer endpoint that bypasses ``_peer_request``.

    ``allowed`` overrides :data:`ALLOWED_FUNCTIONS` — the tests use it to check
    the allowlist is honoured without depending on the real one.

    Returns findings sorted by line. An empty list is the invariant: every peer
    call goes through the one door that attaches the token.
    """
    permitted = set(ALLOWED_FUNCTIONS if allowed is None else allowed)
    tree = ast.parse(source)
    findings: List[Finding] = []

    def visit(node: ast.AST, enclosing: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call) and enclosing not in permitted:
                method = _called_method(child)
                if method in _REQUEST_METHODS:
                    url = _url_argument(child, method)
                    endpoint = _peer_endpoint(url) if url is not None else None
                    if endpoint is not None:
                        findings.append(Finding(child.lineno, endpoint, enclosing))
            visit(child, enclosing)

    visit(tree, "<module>")
    return sorted(findings)


def describe(findings: Sequence[Finding]) -> str:
    """The findings as a block of text, for a test's assertion message."""
    return "\n".join(f.describe() for f in findings)

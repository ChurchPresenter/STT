"""The control-endpoint documentation must name routes that exist.

Documentation for an integration contract is the deliverable, not a description of one: a
Companion button is configured once from a copy-pasted URL and then trusted for years.
Documentation that rots is worse than none, because the person following it has no way to
tell it is wrong except by pressing the button during a service.

So every path the page names is checked against the routes the server actually registers,
and every parameter it lists is checked against the ones the handler actually reads.
"""

import ast
import os
import re

import pytest

DOC = "docs/control-endpoints.md"
SOURCE = "speech_to_text.py"

# Paths mentioned in the page, in the two forms it writes them: as a bare route in a
# sentence and inside a copy-paste URL.
_PATH = re.compile(r"`?(?:https?://STT-HOST:\d+)?(/api/[a-z0-9/_-]+)")


@pytest.fixture(scope="module")
def documented():
    with open(DOC, encoding="utf-8") as handle:
        text = handle.read()
    return text, sorted({m.group(1).rstrip("/") for m in _PATH.finditer(text)})


@pytest.fixture(scope="module")
def registered():
    with open(SOURCE, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    routes = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (isinstance(func, ast.Attribute) and func.attr == "route"):
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                routes.add(str(decorator.args[0].value).rstrip("/"))
    return routes


def test_the_page_exists_and_names_some_routes(documented):
    _, paths = documented
    assert os.path.exists(DOC)
    assert len(paths) >= 4, "the page should document more than one endpoint"


def test_every_documented_route_exists(documented, registered):
    _, paths = documented
    missing = [p for p in paths if p not in registered]
    assert not missing, (
        "documented but not registered: %s. A button configured from this page would 404 "
        "during a service." % ", ".join(missing))


def test_the_control_link_is_documented(documented):
    _, paths = documented
    assert "/api/control/phase-mark" in paths


def test_the_documented_parameters_are_the_ones_read(documented):
    """A parameter table nobody checks drifts from the handler within one refactor."""
    text, _ = documented
    with open(SOURCE, encoding="utf-8") as handle:
        source = handle.read()
    table = text.split("| Parameter | Meaning |", 1)[1].split("\n\n", 1)[0]
    named = {row.split("|")[1].strip().strip("`")
             for row in table.splitlines() if row.startswith("| `")}
    assert named, "the parameter table should not be empty"
    for parameter in named:
        assert 'data.get("%s")' % parameter in source or '"%s"' % parameter in source, (
            "%s is documented but nothing reads it" % parameter)


def test_the_token_requirement_is_stated(documented):
    # The one thing a reader must not have to discover by being refused.
    text, _ = documented
    assert "key=" in text and "access token" in text.lower()


def test_the_tunnel_caveat_is_stated(documented):
    # ?key= is local-network only, and a remote button would fail in a way nothing explains.
    text, _ = documented
    assert "tunnel" in text.lower()

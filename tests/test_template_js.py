"""Every template's inline JavaScript must parse.

A page's script tags are concatenated into one scope by the browser, so a single
duplicate `const` anywhere in a 2600-line block is a parse error that stops the
*whole* page's JavaScript from running — no settings load, no status, no buttons.
That is invisible to the Python suite: the run that shipped exactly this defect
was 806 tests green.

Checked with `node --check`, which is present on the CI runner. Skipped when node
is unavailable so a local run without it still passes; the gate is CI.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = sorted((REPO / "templates").glob("*.html"))

# Inline scripts only — a src= tag is a vendored library, not ours to parse.
_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)
_JINJA_EXPR = re.compile(r"\{\{.*?\}\}", re.S)
_JINJA_STMT = re.compile(r"\{%.*?%\}", re.S)


def _inline_js(path):
    """The page's inline JS, with Jinja placeholders replaced by valid literals."""
    html = path.read_text(encoding="utf-8")
    js = "\n".join(_SCRIPT.findall(html))
    js = _JINJA_EXPR.sub("null", js)
    return _JINJA_STMT.sub("", js)


pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is needed to parse JavaScript")


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_inline_javascript_parses(path):
    js = _inline_js(path)
    if not js.strip():
        pytest.skip("no inline JavaScript")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(js)
        tmp = f.name
    try:
        result = subprocess.run(["node", "--check", tmp],
                                capture_output=True, text=True, timeout=60)
    finally:
        Path(tmp).unlink(missing_ok=True)
    assert result.returncode == 0, (
        f"{path.name} inline JavaScript does not parse — the whole page's script "
        f"would fail to run:\n{result.stderr}")


def test_templates_are_actually_being_checked():
    """Guard against the parametrization silently collapsing to nothing."""
    assert len(TEMPLATES) >= 8, "expected the template set to be found"


# Read out of the HTML attribute rather than by parsing the JavaScript around it. A general
# "called but never defined" checker was tried and rejected: stripping string literals
# desynchronises on a regex literal like /[&<>"']/g, after which it loses whole regions and
# reports functions as missing that are plainly there. A gate that cries wolf gets turned off.
#
# This one cannot: an on* attribute names a page-local function, so a name with no definition
# is always a fault. It is the shape the damage takes when a careless edit removes a span of
# script — the markup keeps pointing at what is no longer there, and the button does nothing
# but log to a console nobody has open.
_HANDLER = re.compile(r"""\bon[a-z]+\s*=\s*["']([^"']+)["']""")
_CALLED = re.compile(r"(?<![.\w$])([a-zA-Z_$][\w$]*)\s*\(")
_DEFINED = re.compile(
    r"(?:function\s+(\w+)\s*\()"
    r"|(?:(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function\b|\())"
    # corrections.html hangs its handlers off window on purpose, so they survive the page
    # being re-rendered; without this they read as undefined.
    r"|(?:window\.(\w+)\s*=)")

# Provided by the browser or by base.html rather than by the page under test.
_PROVIDED = {"toggleTheme", "refreshSysreqBanner", "showAlert", "return", "this", "event"}

# A handler may hold a whole statement — file.html keys off `if (event.key === 'Enter')` —
# so control flow and the browser's own functions are not calls to page code.
_NOT_CALLS = {"if", "for", "while", "switch", "return", "typeof", "catch", "function",
              "setTimeout", "setInterval", "confirm", "alert", "prompt", "fetch",
              "parseInt", "parseFloat", "encodeURIComponent", "decodeURIComponent"}


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_inline_handlers_name_a_function_the_page_defines(path):
    src = path.read_text(encoding="utf-8")
    js = _inline_js(path)
    defined = {a or b or c for a, b, c in _DEFINED.findall(js)} | _PROVIDED
    # base.html defines the shared ones every page inherits.
    base = REPO / "templates" / "base.html"
    if path != base:
        defined |= {a or b or c for a, b, c in _DEFINED.findall(_inline_js(base))}

    missing = set()
    for handler in _HANDLER.findall(src):
        for name in _CALLED.findall(handler):
            if name in _NOT_CALLS:
                continue
            if name not in defined:
                missing.add(name)

    assert not missing, (
        f"{path.name} has on* handlers calling {sorted(missing)}, which its script does not "
        f"define — the markup outlived the code behind it")


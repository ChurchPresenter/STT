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

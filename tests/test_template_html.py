"""Every template's HTML must nest correctly.

A stray </div> does not fail to render — the browser silently reparents
everything after it. That is how the translation page ended up with its Save
button escaping the settings panel and overlapping the controls above it: the
page still loaded, the tests still passed, and only the layout showed it.

Templates are checked with Jinja blocks, <script> and <style> stripped, since
those legitimately contain markup-like text.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = sorted((REPO / "templates").glob("*.html"))

# Elements with no closing tag; not tracked on the stack.
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}

# Elements whose closing tag HTML makes optional. Not used in these templates
# in a way that matters, but excluded so a legitimate omission is not an error.
OPTIONAL_CLOSE = {"li", "option", "p", "td", "th", "tr", "thead", "tbody"}


class _Nesting(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID and tag not in OPTIONAL_CLOSE:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in VOID or tag in OPTIONAL_CLOSE:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}> at line {self.getpos()[0]}")
            return
        top, line = self.stack[-1]
        if top == tag:
            self.stack.pop()
        else:
            self.errors.append(
                f"</{tag}> at line {self.getpos()[0]} closes <{top}> opened at line {line}")
            self.stack.pop()


def _markup_only(path):
    """Template text with Jinja, script and style removed."""
    src = path.read_text(encoding="utf-8")
    src = re.sub(r"\{%.*?%\}", "", src, flags=re.S)
    src = re.sub(r"<script.*?</script>", "", src, flags=re.S)
    return re.sub(r"<style.*?</style>", "", src, flags=re.S)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_tags_are_balanced_and_correctly_nested(path):
    checker = _Nesting()
    checker.feed(_markup_only(path))
    unclosed = [f"<{tag}> opened at line {line} is never closed"
                for tag, line in checker.stack]
    problems = checker.errors + unclosed
    assert not problems, (
        f"{path.name} has broken markup — the browser will reparent the rest of "
        f"the page rather than fail:\n  " + "\n  ".join(problems[:10]))


def test_templates_are_actually_being_checked():
    assert len(TEMPLATES) >= 8

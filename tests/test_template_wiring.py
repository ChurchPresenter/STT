"""Templates must not reference handlers or element ids that do not exist.

Neither failure raises anything. An onclick naming a function that was renamed
gives a silent console error when the user clicks; a getElementById for an id
that no longer exists returns null and throws deep inside a handler. Both survive
a full Python suite and a rendering page — this session shipped one of each
(a call to showModelCategory, which never existed, caught by grep rather than by
a test).

Templates extend base.html, so definitions there count as available everywhere.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO / "templates"
TEMPLATES = sorted(TEMPLATE_DIR.glob("*.html"))

_INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)
_HANDLER = re.compile(r'\bon(?:click|change|input|submit|dblclick)\s*=\s*"\s*([A-Za-z_$][\w$]*)\s*\(')
_GET_BY_ID = re.compile(r"""getElementById\(\s*['"]([\w-]+)['"]\s*\)""")
_ID_ATTR = re.compile(r"""\bid\s*=\s*['"]([\w-]+)['"]""")

# Every way this codebase defines a callable a handler can reach.
_DEFINITION_PATTERNS = (
    re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\("),
    re.compile(r"\basync\s+function\s+([A-Za-z_$][\w$]*)\s*\("),
    re.compile(r"\bwindow\.([A-Za-z_$][\w$]*)\s*="),
    re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\(|[A-Za-z_$][\w$]*\s*=>)"),
)

# Browser globals a handler may legitimately name.
_BUILTINS = {"alert", "confirm", "print", "open", "close", "reload", "history"}


def _inline_js(path):
    return "\n".join(_INLINE_SCRIPT.findall(path.read_text(encoding="utf-8")))


def _defined_names(js):
    names = set()
    for pattern in _DEFINITION_PATTERNS:
        names.update(pattern.findall(js))
    return names


BASE_JS = _inline_js(TEMPLATE_DIR / "base.html") if (TEMPLATE_DIR / "base.html").exists() else ""
BASE_NAMES = _defined_names(BASE_JS) | _BUILTINS


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_inline_handlers_are_defined(path):
    html = path.read_text(encoding="utf-8")
    js = _inline_js(path)
    available = _defined_names(js) | BASE_NAMES
    missing = sorted({name for name in _HANDLER.findall(html) if name not in available})
    assert not missing, (
        f"{path.name} has onclick/onchange handlers naming functions that are not "
        f"defined: {missing}. Clicking gives a silent console error.")


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_element_ids_referenced_from_js_exist(path):
    """getElementById targets must appear somewhere in the page.

    Deliberately loose: the id only has to appear in the file's text, so ids
    inside innerHTML template literals count. A reference to an id that appears
    nowhere at all is the real defect — a rename that missed a call site.
    """
    html = path.read_text(encoding="utf-8")
    js = _inline_js(path)
    # id="..." anywhere in the file, which includes ids inside the innerHTML
    # template literals the pages build — but NOT bare words in JS, or every
    # getElementById argument would vouch for itself.
    present = set(_ID_ATTR.findall(html))
    missing = sorted({el for el in _GET_BY_ID.findall(js) if el not in present})
    assert not missing, (
        f"{path.name} calls getElementById for ids that appear nowhere in the "
        f"page: {missing}. The call returns null and throws in the handler.")


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_element_ids_are_unique(path):
    """A duplicate id makes getElementById silently return the wrong element."""
    html = path.read_text(encoding="utf-8")
    # Only ids in real attributes, not ones built inside JS strings.
    markup = _INLINE_SCRIPT.sub("", html)
    ids = _ID_ATTR.findall(markup)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"{path.name} defines these ids more than once: {dupes}"


def test_templates_are_actually_being_checked():
    assert len(TEMPLATES) >= 8


def test_pollfetch_users_load_the_script_that_defines_it():
    """A page calling pollFetch must actually have it.

    It is defined in static/poll-fetch.js rather than inline, because index.html
    is the one page that does not extend base.html — and it is also the page most
    likely to be left open on a machine that is not whitelisted, which is what
    pollFetch exists for. A page that calls it without loading it throws a
    ReferenceError that stops the rest of that page's JavaScript.
    """
    tag = "poll-fetch.js"
    for path in TEMPLATES:
        html = path.read_text(encoding="utf-8")
        if "pollFetch" not in html:
            continue
        extends_base = '{% extends "base.html" %}' in html
        loads_it = tag in html
        base_loads_it = tag in (TEMPLATE_DIR / "base.html").read_text(encoding="utf-8")
        assert loads_it or (extends_base and base_loads_it), (
            f"{path.name} calls pollFetch but never loads {tag}")

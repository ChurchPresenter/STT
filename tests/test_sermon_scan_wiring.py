"""The scan must ask before it writes, and only the operator may ask twice.

Two structural pins, in the spirit of stt/demo_guard.audit_choke_points. The rule itself is
unit-tested in test_sermon_scan_action.py; what cannot be tested there is that the scan
actually consults it, because the scan lives in a function too large and too entangled to
import. Both failures below are silent — the summariser simply starts duplicating work
again, and nothing reports it until an operator finds two summaries of one sermon.

The second pin is the subtle one. ``manual`` and ``ignore_settle`` look interchangeable and
are not: the end-of-session catch-up skips the settle window too, and it is an automatic
path. Deriving one from the other would re-summarise every sermon at the end of every
service.
"""

import ast

import pytest

SCAN = "_sermon_summary_scan"
DECIDER = "_sermon_scan_action"


@pytest.fixture(scope="module")
def module():
    source = open("speech_to_text.py", encoding="utf-8").read()
    return source, ast.parse(source)


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in speech_to_text.py")


def test_the_scan_asks_the_shared_rule(module):
    source, tree = module
    body = ast.get_source_segment(source, _function(tree, SCAN)) or ""
    assert f"{DECIDER}(" in body, (
        f"{SCAN} no longer calls {DECIDER}. Every boundary nudge would queue a fresh "
        "summary of preaching that already has one."
    )


def test_the_scan_can_still_report_a_change(module):
    source, tree = module
    body = ast.get_source_segment(source, _function(tree, SCAN)) or ""
    assert "_sermon_note_range_change(" in body
    assert "_sermon_range_change(" in body


def test_the_queue_put_is_reachable(module):
    # A pin asserting restraint must prove the restrained thing still exists.
    source, tree = module
    body = ast.get_source_segment(source, _function(tree, SCAN)) or ""
    assert "_sermon_queue.put(" in body


def test_manual_is_a_parameter_of_its_own(module):
    _, tree = module
    args = _function(tree, SCAN).args
    names = [a.arg for a in args.args + args.kwonlyargs]
    assert "manual" in names, f"{SCAN} lost its manual parameter"
    assert "ignore_settle" in names


def test_manual_defaults_to_off(module):
    _, tree = module
    args = _function(tree, SCAN).args
    defaults = dict(zip(args.args[-len(args.defaults):], args.defaults)) if args.defaults else {}
    node = next((v for k, v in defaults.items() if k.arg == "manual"), None)
    assert isinstance(node, ast.Constant) and node.value is False, (
        "manual must default to False, or every automatic caller becomes an operator request"
    )


def test_only_the_operators_route_passes_manual(module):
    """Exactly one caller may set it, and it must be the route behind the button.

    The other two callers are the phase tick and the end-of-session catch-up. Both are
    automatic, and the catch-up already passes ignore_settle=True — which is precisely why
    manual has to be asked for separately.
    """
    _, tree = module
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == SCAN):
            continue
        passes_manual = any(kw.arg == "manual" and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True for kw in node.keywords)
        callers.append((node.lineno, passes_manual))

    assert len(callers) >= 3, f"expected the tick, the catch-up and the route; found {callers}"
    manual_callers = [line for line, passes in callers if passes]
    assert len(manual_callers) == 1, (
        f"{len(manual_callers)} callers pass manual=True (lines {manual_callers}); "
        "only /api/sermon-summary/generate may."
    )

    enclosing = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno <= manual_callers[0]:
            end = getattr(node, "end_lineno", node.lineno)
            if end >= manual_callers[0]:
                enclosing = node.name
    assert enclosing == "generate_sermon_summary", (
        f"manual=True is passed from {enclosing!r}, not the operator's route"
    )

"""The live loop must never store an untranslated caption as its own translation.

This is a structural pin, in the spirit of stt/demo_guard.audit_choke_points and
stt/peer_auth_audit: the rule it protects is one line of code away from being undone, and
the failure is silent. With ``remote.fallback = "skip"`` the translator hands back the
source text, and writing that to the database takes the row out of every set that would
have repaired it — ``translated_text IS NULL`` is what the loop, the backfill and the
replay harness all select on. The caption is then Russian in an English SRT, permanently,
and nothing anywhere reports a problem. One measured service lost 30 captions that way.

The loop is a single enormous function that cannot be imported or executed here, so the
guarantee is asserted against its parse tree instead: the database write must sit inside
the branch that decided the caption was really translated.
"""

import ast

import pytest

PUMP = "emit_translated_entries"
UPDATE = "UPDATE transcriptions SET translated_text"
GUARD = "_persist_it"


@pytest.fixture(scope="module")
def pump():
    source = open("speech_to_text.py", encoding="utf-8").read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == PUMP:
            return source, node
    raise AssertionError(f"{PUMP} not found in speech_to_text.py")


def _guarded_blocks(source, fn):
    """Every `if _persist_it:` body in the pump, as source text."""
    return [ast.get_source_segment(source, node) or ""
            for node in ast.walk(fn)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            and node.test.id == GUARD]


def test_the_pump_still_writes_translations(pump):
    # A pin that asserts the absence of something must first prove it is looking in the
    # right place, or deleting the write would make it pass.
    source, fn = pump
    assert UPDATE in (ast.get_source_segment(source, fn) or "")


def test_the_database_write_is_guarded(pump):
    source, fn = pump
    blocks = _guarded_blocks(source, fn)
    assert blocks, f"no `if {GUARD}:` branch in {PUMP} — the persist guard is gone"
    assert any(UPDATE in block for block in blocks), (
        f"the `{UPDATE}...` write in {PUMP} is no longer inside `if {GUARD}:`. "
        "An untranslated caption would be stored as its own translation and never retried."
    )


def test_the_guard_is_decided_before_it_is_used(pump):
    _, fn = pump
    stores, loads = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == GUARD:
            (stores if isinstance(node.ctx, ast.Store) else loads).append(node.lineno)
    assert stores, f"{GUARD} is never assigned in {PUMP}"
    assert min(stores) < min(loads), f"{GUARD} is read before it is decided"


def test_the_decision_comes_from_the_tested_helper(pump):
    # The rule itself lives in stt/translation_attempts.persist_decision, where it is unit
    # tested; the pump must ask it rather than re-deriving the condition inline.
    source, fn = pump
    body = ast.get_source_segment(source, fn) or ""
    assert "_persist_decision(" in body


def test_a_failed_caption_is_recorded_for_retry(pump):
    # Leaving the row NULL is only half of it: without recording the attempt the caption
    # is retried every 0.5s cycle, spending a full remote timeout each time.
    source, fn = pump
    body = ast.get_source_segment(source, fn) or ""
    assert "_live_translate_attempts.record_failure(" in body
    assert "_live_translate_attempts.should_attempt(" in body

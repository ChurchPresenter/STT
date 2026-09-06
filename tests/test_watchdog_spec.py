"""Guards on the watchdog's PyInstaller spec.

The frozen binary runs its **bundled** copy of stt/watchdog.py before it hands off
to the git checkout (``_maybe_handoff_to_source``). So every ``stt.*`` module that
watchdog.py imports at module scope has to be inside the bundle: a miss is a
binary that dies at startup, on every install, before it can reach the source that
would have worked.

PyInstaller's static analysis is not trusted for these — ``stt.crash_reports`` and
``stt.wheel_policy`` are listed by hand in the spec, which is the previous
author's judgement on that question, and this test simply keeps the list honest.

Static assertions over the sources rather than a build, for the same reason
tests/test_demo_spec.py gives: otherwise this surfaces only after a slow build,
or only when somebody runs the shipped artifact.
"""

from __future__ import annotations

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "packaging", "watchdog.spec")
WATCHDOG = os.path.join(ROOT, "stt", "watchdog.py")


@pytest.fixture(scope="module")
def spec():
    with open(SPEC, encoding="utf-8") as handle:
        return handle.read()


def _stt_imports_at_module_scope():
    """Every ``stt.*`` module watchdog.py imports before it can hand off.

    Walks the module body and the bodies of module-level ``try``/``if`` blocks,
    which is where the real ones live — they sit in a try/except that retries with
    sys.path patched for source installs.
    """
    with open(WATCHDOG, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    found = set()

    def visit(body):
        for node in body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "stt":
                    found.update(f"stt.{alias.name}" for alias in node.names)
                elif node.module.startswith("stt."):
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith("stt."))
            elif isinstance(node, (ast.Try, ast.If)):
                visit(node.body)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
                visit(node.orelse)
                visit(getattr(node, "finalbody", []))

    visit(tree.body)
    return found


def test_the_spec_and_the_watchdog_both_exist():
    assert os.path.isfile(SPEC)
    assert os.path.isfile(WATCHDOG)


def test_watchdog_imports_at_least_the_modules_we_know_about():
    """A sanity check on the walker itself, so a silent zero cannot pass."""
    found = _stt_imports_at_module_scope()
    assert "stt.crash_reports" in found
    assert len(found) >= 2


def test_every_stt_module_the_bundled_watchdog_imports_is_bundled(spec):
    """The bundled copy runs first; anything it imports must be in the bundle."""
    missing = sorted(
        module for module in _stt_imports_at_module_scope()
        if f'"{module}"' not in spec and f"'{module}'" not in spec
    )
    assert not missing, (
        f"stt/watchdog.py imports {missing} at module scope, but the spec does not "
        f"list them in hiddenimports. The frozen binary imports them before it can "
        f"hand off to source, so it would die at startup."
    )


def test_the_job_object_is_bundled(spec):
    """Named explicitly: it is the newest of these and the easiest to forget."""
    assert '"stt.win_job"' in spec


def test_the_spec_freezes_the_watchdog_and_not_the_server(spec):
    """The .exe is a bootstrapper. The server runs from the checkout, in the venv."""
    assert "watchdog.py" in spec
    assert "speech_to_text.py" not in spec

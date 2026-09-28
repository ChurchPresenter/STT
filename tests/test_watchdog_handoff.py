"""The source watchdog runs on its own stt package, inside a frozen process.

A frozen bootstrapper hands off by exec'ing the checkout's stt/watchdog.py in its
own process, where ``stt`` is already imported as the bundled package. That bundle
is whatever the installed binary was built with, so a module added since —
``fd_limit`` in the field — was not in it, the hand-off failed with "cannot import
name 'fd_limit' from 'stt'", and the machine stayed on bundled watchdog code.

Reproduced in a child interpreter so the evicted/reloaded ``stt`` never touches
this test process's own imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHDOG = os.path.join(ROOT, "stt", "watchdog.py")


def _run(tmp_path, body):
    """Run ``body`` in a child that has an older, smaller ``stt`` already imported."""
    bundle = tmp_path / "bundle"
    (bundle / "stt").mkdir(parents=True)
    (bundle / "stt" / "__init__.py").write_text("BUNDLED = True\n", encoding="utf-8")
    (bundle / "stt" / "crash_reports.py").write_text("OLD_COPY = True\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    script = textwrap.dedent(f"""
        import importlib.util, sys
        sys.path[:] = [p for p in sys.path if p not in ("", {ROOT!r})]
        sys.path.insert(0, {str(bundle)!r})
        import stt, stt.crash_reports
        assert stt.BUNDLED and stt.crash_reports.OLD_COPY
        spec = importlib.util.spec_from_file_location("stt_watchdog_source", {WATCHDOG!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    """) + textwrap.dedent(body)
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)


def test_source_watchdog_loads_over_an_older_bundled_stt(tmp_path):
    result = _run(tmp_path, """
        print(mod._fd_limit.__name__)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stt.fd_limit"


def test_its_stt_modules_come_from_the_checkout_not_the_bundle(tmp_path):
    result = _run(tmp_path, """
        import stt, stt.crash_reports
        print(stt.__file__)
        print(hasattr(stt.crash_reports, "OLD_COPY"), hasattr(stt.crash_reports, "scrub_event"))
    """)
    assert result.returncode == 0, result.stderr
    pkg_file, flags = result.stdout.strip().splitlines()
    assert os.path.dirname(os.path.abspath(pkg_file)) == os.path.join(ROOT, "stt")
    assert flags == "False True"


def test_a_normal_import_is_left_alone():
    """Tests and the server import stt.watchdog through the package they already have."""
    import stt
    from stt import watchdog

    assert sys.modules["stt"] is stt
    assert watchdog._fd_limit is sys.modules["stt.fd_limit"]

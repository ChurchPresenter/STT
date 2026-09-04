"""The live page's "no audio" warning, executed rather than grepped.

A muted mic, an unplugged cable, or another application holding the device makes
ffmpeg produce nothing; the capture loop restarts it every 10 seconds and used to
do so entirely in the log, so the page looked normally idle while nothing was
being recorded. `audio_stalled` is the worker saying that is happening, and
updateAudioBanner is what an operator actually sees.

The logic lives in a template, so it is extracted and run under node with a stub
DOM — the same idea as conftest.extract_definitions for the monolith's Python.
node is already this suite's gate for template JS (test_template_js.py) and is
skipped the same way when it is absent.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "templates" / "index.html"
HEALTH = REPO / "templates" / "health.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def extract_function(source, name):
    """The full text of `function <name>(...) { ... }`, by brace matching.

    A regex cannot do this: the body contains braces, template literals and
    nested functions. Brace counting is crude but exact enough for a function
    that is known to be well-formed — test_template_js.py already asserts the
    whole file parses.
    """
    start = source.index(f"function {name}(")
    depth = 0
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


#: A DOM small enough to reason about and real enough for this function: the
#: three elements it touches, recording what was done to each.
_STUB_DOM = """
function makeEl(id) {
  return {
    id: id,
    style: {display: ''},
    textContent: '',
    title: '',
    classes: [],
    classList: {
      _o: null,
      add: function () { for (const c of arguments) if (!this._o.classes.includes(c)) this._o.classes.push(c); },
      remove: function () { for (const c of arguments) this._o.classes = this._o.classes.filter(x => x !== c); },
    },
  };
}
const ELS = {};
for (const id of ['audio-status-container', 'audio-status-dot', 'audio-status-text']) {
  ELS[id] = makeEl(id);
  ELS[id].classList._o = ELS[id];
}
const document = {getElementById: (id) => ELS[id] || null};
"""


def run_banner(state, preset_display="flex"):
    """Call updateAudioBanner(state) under node; return the resulting DOM."""
    fn = extract_function(INDEX.read_text(encoding="utf-8"), "updateAudioBanner")
    script = (
        _STUB_DOM
        + f"ELS['audio-status-container'].style.display = {json.dumps(preset_display)};\n"
        + fn
        + f"\nupdateAudioBanner({json.dumps(state)});\n"
        + "console.log(JSON.stringify({"
        "  display: ELS['audio-status-container'].style.display,"
        "  text: ELS['audio-status-text'].textContent,"
        "  title: ELS['audio-status-text'].title,"
        "  classes: ELS['audio-status-dot'].classes}));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


RUNNING_STALLED = {"running": True, "status": "running", "audio_stalled": True}


class TestShown:
    def test_a_stalled_microphone_during_a_run_is_shown(self):
        dom = run_banner(RUNNING_STALLED, preset_display="none")
        assert dom["display"] == "flex"

    def test_it_reads_as_an_error_not_a_warning(self):
        """Nothing is being recorded. That is not a degraded state."""
        assert "error" in run_banner(RUNNING_STALLED)["classes"]

    def test_the_wording_owns_the_delay(self):
        """The worker only sets the flag after ffmpeg's own 10s no-data
        threshold, so the banner lags reality. Saying "no audio" flat would
        promise an immediacy it does not have."""
        assert "10s" in run_banner(RUNNING_STALLED)["text"]

    def test_the_tooltip_says_what_to_check(self):
        title = run_banner(RUNNING_STALLED)["title"].lower()
        assert "connected" in title and "another application" in title

    def test_no_stale_state_class_survives(self):
        """The dot is shared markup; a leftover class from a previous state
        would paint the error the wrong colour."""
        classes = run_banner(RUNNING_STALLED)["classes"]
        assert classes == ["error"], classes


class TestHidden:
    @pytest.mark.parametrize("state,why", [
        ({"running": True, "status": "running", "audio_stalled": False}, "audio is flowing"),
        ({"running": False, "status": "stopped", "audio_stalled": True}, "a stopped session has no mic to be quiet"),
        ({"running": False, "status": "starting"}, "the flag is absent during startup"),
        ({"running": True, "status": "running"}, "an older server sends no such field"),
        ({}, "empty state object"),
        (None, "the poll failed and passed nothing"),
    ])
    def test_the_banner_stays_hidden(self, state, why):
        assert run_banner(state)["display"] == "none", why

    def test_it_clears_a_banner_that_was_showing(self):
        """Recovery has to be visible too: a mic that comes back must take the
        warning with it, not leave it up for the rest of the service."""
        recovered = {"running": True, "status": "running", "audio_stalled": False}
        assert run_banner(recovered, preset_display="flex")["display"] == "none"


class TestHealthPage:
    """The health dashboard's own treatment of the same flag.

    A dead microphone reads 0/100 there, which is indistinguishable from a quiet
    room — the number looks like silence rather than failure.
    """

    def test_the_tile_distinguishes_no_signal_from_silence(self):
        source = HEALTH.read_text(encoding="utf-8")
        assert "audio_stalled" in source
        assert "No signal" in source

    def test_the_card_is_flagged_so_it_is_coloured(self):
        """setStatus on c-audio is what turns the value and meter red; without
        it the tile would read "No signal" in the normal colour."""
        source = HEALTH.read_text(encoding="utf-8")
        assert 'setStatus("c-audio", "error")' in source

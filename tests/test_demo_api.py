"""What the demo answers instead of loading a model, downloading one, or translating."""

from __future__ import annotations

import pytest

from stt import demo_api


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def state(clock):
    return demo_api.State(now=clock, translations={"Мир вам": "Peace be with you"})


def get(path, state, **args):
    return demo_api.intercept("GET", path, args, {}, state)


def post(path, state, body=None, player=None):
    return demo_api.intercept("POST", path, {}, body or {}, state, player=player)


# --- the boundary ----------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/config",
    "/api/transcription/status",
    "/api/transcription/start",
    "/api/service-phase/sessions",
    "/api/logs",
    "/api/word-highlighting/words",
    "/api/translation/settings",   # config-backed; must not be caught by /api/translate
    "/api/translation/status",
])
def test_the_real_server_still_handles_its_own_routes(path, state):
    assert demo_api.intercept("GET", path, {}, {}, state) is None


@pytest.mark.parametrize("path", [
    "/api/models/list",
    "/api/tts/voices",
    "/api/translate",
    "/api/llm/summarize",
    "/api/calibration/status",
    "/api/audio-devices",
])
def test_anything_that_could_load_a_model_is_intercepted(path, state):
    assert demo_api.is_intercepted(path) is True


def test_an_endpoint_added_later_fails_closed(state):
    """Deny-by-default: a new /api/models/ route must not reach a view that imports torch."""
    payload, status = get("/api/models/invented-next-year", state)

    assert status == 200
    assert payload["success"] is False


def test_a_new_route_outside_the_intercepted_families_is_left_alone(state):
    assert demo_api.intercept("GET", "/api/something-new", {}, {}, state) is None


# --- recorded catalogues ---------------------------------------------------


@pytest.mark.parametrize("path", [
    "/api/models/list",
    "/api/models/faster-whisper/list",
    "/api/models/nllb-list",
    "/api/tts/voices",
    "/api/tts/models",
    "/api/audio-devices",
])
def test_catalogues_come_back_populated(path, state):
    payload, status = get(path, state)

    assert status == 200
    assert payload.get("success") is not False
    assert payload, f"{path} returned nothing for the page to render"


def test_a_catalogue_is_copied_so_callers_cannot_corrupt_the_fixture(state):
    first, _ = get("/api/models/list", state)
    first["poisoned"] = True

    second, _ = get("/api/models/list", state)

    assert "poisoned" not in second


def test_reading_a_catalogue_does_not_change_state(state):
    get("/api/models/list", state)
    get("/api/tts/voices", state)

    assert state.installed == set()
    assert state.downloads == {}
    assert state.overrides == {}


# --- downloads -------------------------------------------------------------


def test_a_download_starts_progresses_and_completes(state, clock):
    started, _ = post("/api/models/download", state, {"model": "large-v3"})
    assert started["success"] is True

    midway, _ = get("/api/models/download-status", state)
    assert midway["status"] == "downloading"
    assert 0 <= midway["progress"] < 100

    clock.advance(demo_api.DOWNLOAD_SECONDS / 2)
    later, _ = get("/api/models/download-status", state)
    assert later["progress"] > midway["progress"]

    clock.advance(demo_api.DOWNLOAD_SECONDS)
    done, _ = get("/api/models/download-status", state)
    assert done["status"] == "completed"
    assert done["progress"] == 100


def test_progress_never_goes_backwards(state, clock):
    post("/api/models/download", state, {"model": "large-v3"})

    seen = []
    for _ in range(10):
        clock.advance(1.5)
        seen.append(get("/api/models/download-status", state)[0]["progress"])

    assert seen == sorted(seen)
    assert seen[-1] == 100


def test_a_finished_download_shows_up_as_installed(state, clock):
    post("/api/models/download", state, {"model": "large-v3"})
    clock.advance(demo_api.DOWNLOAD_SECONDS + 1)
    get("/api/models/download-status", state)

    local, _ = get("/api/models/local", state)
    names = {m.get("name") for m in local["models"]}

    assert "large-v3" in names


def test_nothing_is_downloading_before_anything_was_asked_for(state):
    payload, _ = get("/api/models/download-status", state)

    assert payload["downloading"] is False
    assert payload["status"] == "idle"


def test_a_download_can_be_cancelled_midway(state, clock):
    post("/api/models/download", state, {"model": "large-v3"})
    clock.advance(2.0)

    post("/api/models/cancel-download", state)
    clock.advance(demo_api.DOWNLOAD_SECONDS)

    payload, _ = get("/api/models/download-status", state)
    assert payload["downloading"] is False
    assert "large-v3" not in state.installed


@pytest.mark.parametrize("path", demo_api.DOWNLOAD_PATHS)
def test_every_download_family_starts_a_job(path, state):
    payload, _ = post(path, state, {"model": "something"})

    assert payload["success"] is True
    assert state.downloads


def test_a_model_can_be_removed_again(state, clock):
    post("/api/models/download", state, {"model": "large-v3"})
    clock.advance(demo_api.DOWNLOAD_SECONDS + 1)
    get("/api/models/download-status", state)

    post("/api/models/remove", state, {"model": "large-v3"})

    assert "large-v3" not in state.installed


# --- translation -----------------------------------------------------------


def test_translating_a_line_from_the_service_returns_what_it_really_produced(state):
    payload, _ = post("/api/translate", state, {"text": "Мир вам"})

    assert payload["success"] is True
    assert payload["translation"] == "Peace be with you"


def test_translating_something_else_explains_the_limit_without_erroring(state):
    payload, status = post("/api/translate", state, {"text": "something unrelated"})

    assert status == 200            # a toast, not an error page
    assert payload["success"] is False
    assert payload["error"]


def test_translating_nothing_is_refused(state):
    payload, _ = post("/api/translate", state, {"text": "   "})

    assert payload["success"] is False


# --- things that cannot be demonstrated ------------------------------------


@pytest.mark.parametrize("path", sorted(demo_api.UNAVAILABLE))
def test_unavailable_actions_answer_with_a_message_not_an_error_status(path, state):
    payload, status = post(path, state)

    assert status == 200
    assert payload["success"] is False
    assert payload["error"]


def test_pairing_a_second_machine_is_refused(state):
    payload, status = post("/api/translate/pair/request", state, {"host": "10.0.0.5"})

    assert status == 200
    assert payload["success"] is False


# --- settings writes inside the intercepted families -----------------------


def test_choosing_a_model_is_remembered_by_the_matching_read(state):
    post("/api/models/manager", state, {"live_model": {"whisper": {"model": "large-v3"}}})

    payload, _ = get("/api/models/manager", state)

    assert payload["live_model"]["whisper"]["model"] == "large-v3"


def test_a_write_with_no_body_still_succeeds(state):
    payload, _ = post("/api/tts/settings", state, {})

    assert payload["success"] is True


# --- canned work product ---------------------------------------------------


def test_a_summary_comes_back_written(state):
    payload, _ = post("/api/llm/summarize", state, {"text": "..."})

    assert payload["success"] is True
    assert payload["summary"]


def test_the_sermon_summary_generator_never_reaches_an_llm(state):
    assert demo_api.is_intercepted("/api/sermon-summary/generate") is True
    payload, _ = post("/api/sermon-summary/generate", state, {})
    assert payload["success"] is True


def test_calibration_reports_plausible_results(state):
    payload, _ = get("/api/calibration/results", state)

    assert payload["success"] is True
    assert payload["noise_floor_db"] < payload["speech_peak_db"]


# --- playback control ------------------------------------------------------


class FakePlayer:
    def __init__(self) -> None:
        self.running = False
        self.db_path = None
        self.speed = 1.0
        self.restarted = False

    def begin_session(self):
        self.running = True

    def end_session(self):
        self.running = False

    def restart(self):
        self.restarted = True

    def set_speed(self, value):
        self.speed = value

    def elapsed_s(self):
        return 12.5


def test_playback_can_be_paused_and_resumed(state):
    player = FakePlayer()

    post("/api/demo/control", state, {"action": "play"}, player=player)
    assert player.running is True

    post("/api/demo/control", state, {"action": "pause"}, player=player)
    assert player.running is False


def test_playback_speed_can_be_changed(state):
    player = FakePlayer()

    post("/api/demo/control", state, {"action": "speed", "value": 2.0}, player=player)

    assert player.speed == 2.0


def test_a_bad_speed_is_refused_rather_than_crashing(state):
    player = FakePlayer()

    payload, status = post("/api/demo/control", state, {"action": "speed", "value": "fast"},
                           player=player)

    assert status == 200
    assert payload["success"] is False


def test_an_unknown_playback_action_is_refused(state):
    payload, _ = post("/api/demo/control", state, {"action": "rewind"},
                      player=FakePlayer())

    assert payload["success"] is False


def test_playback_status_reports_where_the_service_has_reached(state):
    payload, _ = get("/api/demo/status", state)
    assert payload["success"] is False          # no player wired in

    payload, _ = demo_api.intercept("GET", "/api/demo/status", {}, {}, state,
                                    player=FakePlayer())
    assert payload["elapsed_s"] == 12.5


# --- routes that reach the network or spawn a process ----------------------


@pytest.mark.parametrize("path", [
    "/api/tunnel/start",
    "/api/tunnel/stop",
    "/api/tunnel/settings",
    "/api/tunnel/status",
    "/api/file-mover/test",
    "/api/file-mover/browse-remote",
    "/api/file-mover/configure",
    "/api/file-mover/status",
    "/api/remote-translation/status",
    "/api/remote-translation/pair/request",
])
def test_routes_that_escape_the_machine_are_refused(path, state):
    """No network guard can stop these: the tunnel spawns a binary it downloaded, and
    file-mover shells out to `net use` / `sudo mount -t cifs` with a path and
    credentials taken straight from the request body."""
    payload, status = post(path, state, {"destination_path": "//example.invalid/share"})

    assert status == 200
    assert payload["success"] is False
    assert payload["error"]


def test_starting_a_tunnel_cannot_publish_the_demo(state):
    """A tunnel would put a demo with no password on a public URL."""
    assert demo_api.intercept("POST", "/api/tunnel/start", {}, {}, state) is not None


def test_the_tunnel_binary_path_cannot_be_set(state):
    """It is stored verbatim and later executed, so this is code execution."""
    payload, _ = post("/api/tunnel/settings", state, {"binary": "/bin/sh"})

    assert payload["success"] is False


def test_the_local_file_manager_still_works(state):
    """file-manager is local browsing and must keep working; only file-mover is blocked.
    The names differ by three letters."""
    for path in ("/api/file-manager/list", "/api/file-manager/download",
                 "/api/file-manager/session-meta"):
        assert demo_api.intercept("GET", path, {}, {}, state) is None


def test_the_client_side_peer_proxies_are_covered(state):
    """/api/translate/pair guards the server half of pairing; these are the client
    half and sit under a different prefix, which is how they were missed."""
    assert demo_api.is_intercepted("/api/remote-translation/pair/confirm") is True
    assert demo_api.is_intercepted("/api/remote-translation/status") is True


def test_a_new_route_under_a_blocked_prefix_fails_closed(state):
    payload, _ = post("/api/tunnel/invented-later", state)

    assert payload["success"] is False


@pytest.mark.parametrize("path", ["/api/tunnel/status", "/api/file-mover/status"])
def test_a_blocked_family_still_renders_its_page(path, state):
    """The refusal belongs on the button, not on the whole screen: a status read
    performs no action, so it answers from the recording."""
    payload, status = get(path, state)

    assert status == 200
    assert payload.get("success") is not False


def test_a_blocked_family_with_no_recorded_read_still_refuses(state):
    payload, _ = get("/api/tunnel/some-unrecorded-read", state)

    assert payload["success"] is False


def test_the_refusal_says_which_capability_was_refused(state):
    tunnel, _ = post("/api/tunnel/start", state)
    mover, _ = post("/api/file-mover/test", state)

    assert tunnel["error"] != mover["error"]
    assert "public address" in tunnel["error"]
    assert "network share" in mover["error"]

"""evaluate_start: turning a start that never finished into one that failed."""

import pathlib

import pytest

from stt import start_watch

STALL = 600.0


def starting(stage="3/5 loading model", at=1000.0, **extra):
    state = {"status": "starting", "init_stage": stage, "init_stage_at": at}
    state.update(extra)
    return state


class TestNotOurBusiness:
    """Only a start in progress is judged; everything else is left alone."""

    @pytest.mark.parametrize("status", ["running", "stopped", "stopping", "error", "restarting"])
    def test_other_statuses_are_never_touched(self, status):
        state = starting(at=0.0)
        state["status"] = status
        # Both failure conditions hold — a dead worker and an ancient stamp —
        # and it still must not fire, because the state is not "starting".
        assert start_watch.evaluate_start(state, worker_alive=False, now=99999.0, stall_seconds=STALL) is None

    def test_healthy_start_passes(self):
        assert start_watch.evaluate_start(
            starting(at=1000.0), worker_alive=True, now=1005.0, stall_seconds=STALL
        ) is None


class TestDeadWorker:
    def test_dead_worker_is_reported(self):
        verdict = start_watch.evaluate_start(
            starting(), worker_alive=False, now=1005.0, stall_seconds=STALL
        )
        assert verdict is not None
        assert "worker process is gone" in verdict.message

    def test_dead_worker_names_the_stage_it_died_on(self):
        verdict = start_watch.evaluate_start(
            starting(stage="2/5 opening audio device"), worker_alive=False, now=1001.0, stall_seconds=STALL
        )
        assert "2/5 opening audio device" in verdict.message

    def test_dead_worker_beats_the_deadline(self):
        """A worker that exited one second in must not wait out the stall clock."""
        verdict = start_watch.evaluate_start(
            starting(at=1000.0), worker_alive=False, now=1001.0, stall_seconds=STALL
        )
        assert verdict is not None


class TestStall:
    def test_stage_stale_past_the_deadline_is_reported(self):
        verdict = start_watch.evaluate_start(
            starting(at=1000.0), worker_alive=True, now=1000.0 + STALL + 1, stall_seconds=STALL
        )
        assert verdict is not None
        assert "3/5 loading model" in verdict.error

    def test_exactly_at_the_deadline_is_still_healthy(self):
        assert start_watch.evaluate_start(
            starting(at=1000.0), worker_alive=True, now=1000.0 + STALL, stall_seconds=STALL
        ) is None

    def test_a_stage_that_keeps_advancing_never_stalls(self):
        """The whole point of a per-stage clock: a slow load is not a wedged one.

        Five steps, each taking most of the deadline, total far past it.
        """
        now = 0.0
        for step in range(1, 6):
            at = now
            now += STALL - 1
            state = starting(stage=f"{step}/5 step", at=at)
            assert start_watch.evaluate_start(
                state, worker_alive=True, now=now, stall_seconds=STALL
            ) is None

    def test_stalled_minutes_are_reported(self):
        verdict = start_watch.evaluate_start(
            starting(at=0.0), worker_alive=True, now=1500.0, stall_seconds=STALL
        )
        assert "25 minutes" in verdict.message

    def test_zero_deadline_switches_the_stall_check_off(self):
        assert start_watch.evaluate_start(
            starting(at=0.0), worker_alive=True, now=99999.0, stall_seconds=0
        ) is None


class TestMissingFields:
    """A start must never be killed because a field was absent."""

    def test_no_timestamp_is_not_a_stall(self):
        state = {"status": "starting", "init_stage": "3/5 loading model"}
        assert start_watch.evaluate_start(
            state, worker_alive=True, now=99999.0, stall_seconds=STALL
        ) is None

    def test_non_numeric_timestamp_is_not_a_stall(self):
        state = starting(at="soon")
        assert start_watch.evaluate_start(
            state, worker_alive=True, now=99999.0, stall_seconds=STALL
        ) is None

    def test_a_dead_worker_with_no_stage_still_reports(self):
        verdict = start_watch.evaluate_start(
            {"status": "starting"}, worker_alive=False, now=10.0, stall_seconds=STALL
        )
        assert verdict is not None
        assert "initialising" in verdict.message


class TestStageOf:
    @pytest.mark.parametrize("stage", [None, "", "   ", 5])
    def test_unusable_stages_fall_back(self, stage):
        assert start_watch.stage_of({"init_stage": stage}) == "initialising"

    def test_stage_is_stripped(self):
        assert start_watch.stage_of({"init_stage": "  4/5 opening database  "}) == "4/5 opening database"


class TestRefusalBeforeSpawn:
    """Refusing a bad model at the route, not inside the worker.

    The loader has always refused an incomplete directory, but only after the
    start route spawned a process that re-imported the whole monolith. On Windows
    that is the difference between an instant error and minutes of nothing.
    """

    def test_a_complete_model_is_not_refused(self):
        assert start_watch.refusal_reason("large-v3", "complete", "") is None

    def test_an_incomplete_model_is_refused_and_names_what_is_missing(self):
        reason = start_watch.refusal_reason(
            "large-v3", "incomplete", "tokenizer.json")
        assert reason is not None
        assert "large-v3" in reason
        assert "tokenizer.json" in reason
        assert "Repair" in reason, "the message must name the remedy that exists"

    def test_a_model_that_was_never_downloaded_is_told_to_download_not_repair(self):
        reason = start_watch.refusal_reason("large-v3", "absent", "")
        assert reason is not None
        assert "not downloaded" in reason
        assert "Repair" not in reason, "there is nothing to repair yet"

    def test_a_fresh_install_with_nothing_selected_is_not_refused(self):
        """The setup banner explains this far better than an error on Start."""
        for name in ("", "   "):
            assert start_watch.refusal_reason(name, "absent", "") is None

    def test_the_refusal_matches_the_loader_s_own_wording(self):
        """One fault must not be described two ways.

        The loader raises its own message from inside the worker. If these drift,
        an operator who reads both concludes there are two problems.
        """
        remedy = "Open the Model Manager and press Repair on this model."
        source = pathlib.Path("speech_to_text.py").read_text(encoding="utf-8")

        assert remedy in source, "the loader stopped offering Repair; this must follow"
        assert remedy in start_watch.refusal_reason("m", "incomplete", "tokenizer.json")

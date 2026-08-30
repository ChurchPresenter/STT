"""Deferring sermon summaries around live services, without losing the work or the files.

The two hazards pinned down here: queueing a summary mid-service (which makes the caption
model contend with a congregation's captions), and letting a deferred end-of-session run
outlive the delivery hold it depends on — the run would resume into a session database
that had already been delivered and deleted locally.
"""

from stt.sermon_scheduling import (
    DEFAULT_PAUSE_CEILING_SECONDS,
    HOLD_TRANSCRIBING,
    defer_scan,
    finalise_expired,
    hold_reason,
    pause_fields,
    working_seconds,
)

NOW = 10_000.0


class TestDeferScan:
    def test_the_tick_is_deferred_while_a_service_runs(self):
        assert defer_scan(enabled=True, ignore_settle=False, transcription_running=True,
                          defer_while_live=True) is True

    def test_the_tick_proceeds_once_the_service_stops(self):
        assert defer_scan(enabled=True, ignore_settle=False, transcription_running=False,
                          defer_while_live=True) is False

    def test_the_end_of_session_catch_up_is_never_deferred(self):
        # It is the mechanism the deferral relies on; deferring it would lose the work.
        assert defer_scan(enabled=True, ignore_settle=True, transcription_running=True,
                          defer_while_live=True) is False

    def test_an_operators_request_is_never_deferred(self):
        # Same path as the catch-up: asking for a summary is asking for one. The worker
        # still holds it until the service ends; the scan itself records it.
        assert defer_scan(enabled=False, ignore_settle=True, transcription_running=True,
                          defer_while_live=True) is False

    def test_the_disabled_summariser_still_skips_the_tick(self):
        assert defer_scan(enabled=False, ignore_settle=False, transcription_running=False,
                          defer_while_live=True) is True

    def test_defer_while_live_off_restores_mid_service_queueing(self):
        assert defer_scan(enabled=True, ignore_settle=False, transcription_running=True,
                          defer_while_live=False) is False


class TestHoldReason:
    def test_a_running_session_holds_the_worker(self):
        assert hold_reason(transcription_running=True, transcription_starting=False,
                           defer_while_live=True) == HOLD_TRANSCRIBING

    def test_a_starting_session_holds_it_too(self):
        # The running flag flips a moment after Start; a chunk begun in that window is one
        # the new service pays for.
        assert hold_reason(transcription_running=False, transcription_starting=True,
                           defer_while_live=True) == HOLD_TRANSCRIBING

    def test_an_idle_machine_does_not(self):
        assert hold_reason(transcription_running=False, transcription_starting=False,
                           defer_while_live=True) is None

    def test_the_end_of_session_window_is_exactly_when_it_may_run(self):
        # The session has stopped, nothing new has started: this is what the whole
        # deferral defers to.
        assert hold_reason(transcription_running=False, transcription_starting=False,
                           defer_while_live=True) is None

    def test_defer_while_live_off_never_holds(self):
        assert hold_reason(transcription_running=True, transcription_starting=True,
                           defer_while_live=False) is None


class TestPauseFields:
    def test_entering_a_pause_stamps_the_start(self):
        assert pause_fields(now=NOW, paused_at=0.0, paused_total=0.0,
                            pausing=True) == (NOW, 0.0)

    def test_staying_paused_does_not_restart_the_clock(self):
        # The worker polls every few seconds while held.
        assert pause_fields(now=NOW + 30.0, paused_at=NOW, paused_total=0.0,
                            pausing=True) == (NOW, 0.0)

    def test_leaving_a_pause_accumulates_it(self):
        assert pause_fields(now=NOW + 30.0, paused_at=NOW, paused_total=5.0,
                            pausing=False) == (0.0, 35.0)

    def test_leaving_without_a_pause_is_a_no_op(self):
        assert pause_fields(now=NOW, paused_at=0.0, paused_total=12.0,
                            pausing=False) == (0.0, 12.0)

    def test_pauses_accumulate_across_several_services(self):
        paused_at, total = pause_fields(now=100.0, paused_at=0.0, paused_total=0.0,
                                        pausing=True)
        paused_at, total = pause_fields(now=200.0, paused_at=paused_at,
                                        paused_total=total, pausing=False)
        paused_at, total = pause_fields(now=500.0, paused_at=paused_at,
                                        paused_total=total, pausing=True)
        paused_at, total = pause_fields(now=800.0, paused_at=paused_at,
                                        paused_total=total, pausing=False)
        assert (paused_at, total) == (0.0, 400.0)


class TestWorkingSeconds:
    def test_a_run_that_never_paused_counts_all_of_it(self):
        assert working_seconds(now=NOW + 60.0, since=NOW, paused_at=0.0,
                               paused_total=0.0) == 60.0

    def test_finished_pauses_are_excluded(self):
        assert working_seconds(now=NOW + 100.0, since=NOW, paused_at=0.0,
                               paused_total=40.0) == 60.0

    def test_an_open_pause_is_excluded_as_it_accrues(self):
        assert working_seconds(now=NOW + 100.0, since=NOW, paused_at=NOW + 40.0,
                               paused_total=0.0) == 40.0

    def test_it_never_goes_negative(self):
        assert working_seconds(now=NOW, since=NOW + 50.0, paused_at=0.0,
                               paused_total=0.0) == 0.0


class TestFinaliseExpired:
    def test_a_run_that_never_paused_behaves_as_before(self):
        # Regression pin on the original since-only rule.
        assert finalise_expired(now=NOW + 1799.0, since=NOW, paused_at=0.0,
                                paused_total=0.0, deadline=1800.0) is False
        assert finalise_expired(now=NOW + 1800.0, since=NOW, paused_at=0.0,
                                paused_total=0.0, deadline=1800.0) is True

    def test_time_spent_standing_aside_does_not_count(self):
        # An hour parked for a new service, 100s of actual work: not stuck.
        assert finalise_expired(now=NOW + 3700.0, since=NOW, paused_at=0.0,
                                paused_total=3600.0, deadline=1800.0) is False

    def test_an_open_pause_does_not_count_either(self):
        assert finalise_expired(now=NOW + 3700.0, since=NOW, paused_at=NOW + 100.0,
                                paused_total=0.0, deadline=1800.0) is False

    def test_a_genuinely_stuck_run_still_expires(self):
        assert finalise_expired(now=NOW + 5000.0, since=NOW, paused_at=0.0,
                                paused_total=3000.0, deadline=1800.0) is True

    def test_the_wall_clock_ceiling_ends_an_endless_hold(self):
        # Services back to back all day: the files matter more than the summary.
        assert finalise_expired(now=NOW + DEFAULT_PAUSE_CEILING_SECONDS, since=NOW,
                                paused_at=NOW + 10.0, paused_total=0.0,
                                deadline=1800.0) is True

    def test_below_the_ceiling_a_paused_run_survives(self):
        assert finalise_expired(now=NOW + DEFAULT_PAUSE_CEILING_SECONDS - 1.0, since=NOW,
                                paused_at=NOW + 10.0, paused_total=0.0,
                                deadline=1800.0) is False

    def test_the_ceiling_can_be_switched_off(self):
        assert finalise_expired(now=NOW + 1_000_000.0, since=NOW, paused_at=NOW + 1.0,
                                paused_total=0.0, deadline=1800.0,
                                max_total=0.0) is False

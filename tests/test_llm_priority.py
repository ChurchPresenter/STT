"""When the sermon summariser must stand aside for live captions.

The deployment these rules exist for: two machines share one model. One transcribes a
service and offloads both its captions and its sermon summaries to the other. The
summarising machine is not itself transcribing, so anything it infers from its own
transcription state is blind to the captions it is serving — ``TestServingMachine`` pins
that case down, and ``test_a_silent_stretch_of_a_live_service_still_defers`` pins down the
one that makes caption traffic alone an insufficient signal.
"""

from stt.llm_priority import (
    DEFAULT_QUIET_SECONDS,
    DEFAULT_STALE_SECONDS,
    KIND_CAPTION,
    KIND_HEARTBEAT,
    WAIT_LOCAL_BACKLOG,
    WAIT_PEER_CAPTIONS,
    WAIT_PEER_SESSION,
    WAIT_TRANSCRIBING,
    PeerActivity,
    hold_poll_seconds,
    local_pump_busy,
    quiet_window,
    stale_window,
    summariser_wait_reason,
)

NOW = 1_000_000.0


class TestPeerActivity:
    def test_nothing_recorded_reads_as_never(self):
        activity = PeerActivity()
        assert activity.last_seen(KIND_HEARTBEAT) == 0.0
        assert activity.last_client(KIND_HEARTBEAT) is None

    def test_a_recorded_signal_is_read_back(self):
        activity = PeerActivity()
        activity.record("192.168.2.62", KIND_HEARTBEAT, NOW)
        assert activity.last_seen(KIND_HEARTBEAT) == NOW
        assert activity.last_client(KIND_HEARTBEAT) == "192.168.2.62"

    def test_the_kinds_are_tracked_apart(self):
        activity = PeerActivity()
        activity.record("192.168.2.62", KIND_HEARTBEAT, NOW)
        assert activity.last_seen(KIND_CAPTION) == 0.0

    def test_the_most_recent_signal_wins(self):
        activity = PeerActivity()
        activity.record("192.168.2.62", KIND_CAPTION, NOW)
        activity.record("192.168.2.70", KIND_CAPTION, NOW + 5.0)
        assert activity.last_seen(KIND_CAPTION) == NOW + 5.0
        assert activity.last_client(KIND_CAPTION) == "192.168.2.70"

    def test_an_out_of_order_record_does_not_rewind_the_clock(self):
        activity = PeerActivity()
        activity.record("192.168.2.70", KIND_CAPTION, NOW + 5.0)
        activity.record("192.168.2.62", KIND_CAPTION, NOW)
        assert activity.last_seen(KIND_CAPTION) == NOW + 5.0

    def test_snapshot_reports_ages(self):
        activity = PeerActivity()
        activity.record("192.168.2.62", KIND_HEARTBEAT, NOW - 12.0)
        assert activity.snapshot(NOW) == {KIND_HEARTBEAT: 12.0}

    def test_forget_clears_one_kind_or_all(self):
        activity = PeerActivity()
        activity.record("a", KIND_CAPTION, NOW)
        activity.record("a", KIND_HEARTBEAT, NOW)
        activity.forget(KIND_CAPTION)
        assert activity.last_seen(KIND_CAPTION) == 0.0
        assert activity.last_seen(KIND_HEARTBEAT) == NOW
        activity.forget()
        assert activity.snapshot(NOW) == {}


class TestLocalPumpBusy:
    def test_a_fresh_non_zero_count_is_busy(self):
        assert local_pump_busy(3, NOW - 1.0, NOW, 60.0) is True

    def test_a_fresh_zero_count_is_not(self):
        assert local_pump_busy(0, NOW - 1.0, NOW, 60.0) is False

    def test_a_stale_count_is_not_believed(self):
        assert local_pump_busy(9, NOW - 300.0, NOW, 60.0) is False

    def test_never_published_is_not_busy(self):
        # The initial state on a machine whose pump has never run.
        assert local_pump_busy(0, 0.0, NOW, 60.0) is False

    def test_the_window_covers_a_pump_cycle_blocked_on_a_slow_peer(self):
        # A cycle that spends 45s waiting on three 15s remote calls publishes only at its
        # start; the default window must still call that a live backlog. Under the old
        # 5s window this returned False exactly when captions were most backed up.
        assert local_pump_busy(5, NOW - 45.0, NOW, DEFAULT_STALE_SECONDS) is True

    def test_a_garbled_count_is_not_busy(self):
        assert local_pump_busy(None, NOW, NOW, 60.0) is False

    def test_a_garbled_timestamp_reads_as_ancient(self):
        # Anything unreadable must mean "no recent report", never "reported just now".
        assert local_pump_busy(5, "never", NOW, 60.0) is False
        assert summariser_wait_reason(now=NOW, last_heartbeat_at="never") is None


class TestSummariserWaitReason:
    def test_an_idle_machine_may_proceed(self):
        assert summariser_wait_reason(now=NOW) is None

    def test_a_live_session_here_holds_the_summariser(self):
        assert summariser_wait_reason(now=NOW, transcribing=True) == WAIT_TRANSCRIBING

    def test_a_peers_heartbeat_holds_the_summariser(self):
        reason = summariser_wait_reason(now=NOW, last_heartbeat_at=NOW - 5.0)
        assert reason == WAIT_PEER_SESSION

    def test_a_silent_stretch_of_a_live_service_still_defers(self):
        # Music and prayer: no caption offloaded for two minutes, but the heartbeat keeps
        # arriving. Caption traffic alone would have called this idle and taken the model.
        reason = summariser_wait_reason(now=NOW, last_heartbeat_at=NOW - 18.0,
                                        last_caption_at=NOW - 120.0)
        assert reason == WAIT_PEER_SESSION

    def test_recent_caption_traffic_holds_the_summariser(self):
        reason = summariser_wait_reason(now=NOW, last_caption_at=NOW - 3.0)
        assert reason == WAIT_PEER_CAPTIONS

    def test_a_local_backlog_holds_the_summariser(self):
        reason = summariser_wait_reason(now=NOW, local_pending_count=4,
                                        local_pending_at=NOW - 1.0)
        assert reason == WAIT_LOCAL_BACKLOG

    def test_a_quiet_peer_releases_it(self):
        reason = summariser_wait_reason(now=NOW, last_heartbeat_at=NOW - 600.0,
                                        last_caption_at=NOW - 600.0)
        assert reason is None

    def test_the_quiet_window_boundary_is_exclusive(self):
        assert summariser_wait_reason(now=NOW, last_heartbeat_at=NOW - 45.0,
                                      quiet_seconds=45.0) is None
        assert summariser_wait_reason(now=NOW, last_heartbeat_at=NOW - 44.9,
                                      quiet_seconds=45.0) == WAIT_PEER_SESSION

    def test_defer_while_live_off_restores_the_old_narrow_behaviour(self):
        # Only the local-backlog arm survives, which is exactly what the code did before.
        reason = summariser_wait_reason(now=NOW, transcribing=True,
                                        last_heartbeat_at=NOW - 1.0,
                                        last_caption_at=NOW - 1.0,
                                        defer_while_live=False)
        assert reason is None
        reason = summariser_wait_reason(now=NOW, local_pending_count=2,
                                        local_pending_at=NOW - 1.0,
                                        defer_while_live=False)
        assert reason == WAIT_LOCAL_BACKLOG

    def test_pause_on_backlog_off_leaves_the_live_arms_alone(self):
        assert summariser_wait_reason(now=NOW, local_pending_count=4,
                                      local_pending_at=NOW - 1.0,
                                      pause_on_backlog=False) is None
        assert summariser_wait_reason(now=NOW, transcribing=True,
                                      pause_on_backlog=False) == WAIT_TRANSCRIBING

    def test_both_switches_off_never_waits(self):
        reason = summariser_wait_reason(now=NOW, transcribing=True,
                                        last_heartbeat_at=NOW,
                                        local_pending_count=9, local_pending_at=NOW,
                                        defer_while_live=False, pause_on_backlog=False)
        assert reason is None

    def test_a_clock_that_stepped_backwards_does_not_report_idle(self):
        # A timestamp in the future is "just now", not "ancient".
        assert summariser_wait_reason(now=NOW, last_heartbeat_at=NOW + 300.0) == WAIT_PEER_SESSION


class TestServingMachine:
    """The machine holding the model, which is the one that must get this right.

    It is not transcribing and its own pump never publishes, so its transcription state
    and its pending count both say "idle" while it is translating a service's captions.
    Only what arrives over the wire knows.
    """

    def test_it_defers_to_the_service_it_is_serving(self):
        activity = PeerActivity()
        activity.record("192.168.2.62", KIND_HEARTBEAT, NOW - 3.0)
        activity.record("192.168.2.62", KIND_CAPTION, NOW - 2.0)
        reason = summariser_wait_reason(
            now=NOW, transcribing=False,
            last_heartbeat_at=activity.last_seen(KIND_HEARTBEAT),
            last_caption_at=activity.last_seen(KIND_CAPTION),
            local_pending_count=0, local_pending_at=0.0,
            quiet_seconds=DEFAULT_QUIET_SECONDS)
        assert reason == WAIT_PEER_SESSION

    def test_it_proceeds_once_that_service_has_stopped(self):
        # The heartbeat stops with the transcription, so the window clears on its own.
        activity = PeerActivity()
        activity.record("192.168.2.62", KIND_HEARTBEAT, NOW - 120.0)
        activity.record("192.168.2.62", KIND_CAPTION, NOW - 118.0)
        reason = summariser_wait_reason(
            now=NOW, transcribing=False,
            last_heartbeat_at=activity.last_seen(KIND_HEARTBEAT),
            last_caption_at=activity.last_seen(KIND_CAPTION),
            quiet_seconds=DEFAULT_QUIET_SECONDS)
        assert reason is None


class TestWindows:
    def test_a_configured_window_is_used(self):
        assert quiet_window(90) == 90.0
        assert stale_window(120) == 120.0

    def test_a_window_below_the_heartbeat_interval_is_raised(self):
        # A 5s window clears between two pings of a service that is still running, which
        # is the one way the deferral fails without saying so.
        assert quiet_window(5) == 25.0

    def test_absurd_windows_are_clamped(self):
        assert quiet_window(100000) == 900.0
        assert stale_window(0) == 5.0

    def test_a_missing_or_garbled_setting_falls_back(self):
        assert quiet_window(None) == DEFAULT_QUIET_SECONDS
        assert quiet_window("soon") == DEFAULT_QUIET_SECONDS
        assert stale_window(None) == DEFAULT_STALE_SECONDS


class TestHoldPoll:
    def test_the_first_check_is_prompt_and_logged(self):
        interval, should_log = hold_poll_seconds(0.0)
        assert interval == 1.0
        assert should_log is True

    def test_it_backs_off_once_the_wait_is_real(self):
        interval, _ = hold_poll_seconds(30.0)
        assert interval == 5.0

    def test_it_reports_once_a_minute_rather_than_every_poll(self):
        assert hold_poll_seconds(60.0)[1] is True
        assert hold_poll_seconds(65.0)[1] is False
        assert hold_poll_seconds(120.0)[1] is True

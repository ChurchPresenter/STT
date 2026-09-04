"""The transcription worker's fatal-exception path.

The bug these cover: a worker exception reached neither Sentry (multiprocessing
bypasses sys.excepthook) nor the UI (only three init steps wrote an error state),
so the page reported STARTING forever and no event was ever filed.
"""

from __future__ import annotations

import sys
import threading

import pytest

from stt import worker_crash


def _raise(exc):
    """Return ``exc`` with a real traceback attached, as the handler will see it."""
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001 - deliberately re-raised below
        return caught


class Recorder:
    """A callable that remembers its calls, optionally blowing up."""

    def __init__(self, boom=False):
        self.calls = []
        self.boom = boom

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.boom:
            raise RuntimeError("handler exploded")


# --- reporting ------------------------------------------------------------


def test_fatal_exception_reaches_sentry_the_ui_and_the_log():
    capture, flush, set_state, log = Recorder(), Recorder(), Recorder(), Recorder()

    outcome = worker_crash.report_worker_crash(
        _raise(ValueError("model dir is empty")),
        capture=capture, flush=flush, set_state=set_state, log=log,
    )

    assert outcome == worker_crash.CrashOutcome(
        reported=True, flushed=True, state_written=True, quiet=False
    )
    assert len(capture.calls) == 1
    assert set_state.calls[0][0][0] == "ValueError: model dir is empty"
    assert log.calls, "the crash must always be logged"


def test_flush_follows_capture_so_the_event_survives_process_exit():
    order = []
    worker_crash.report_worker_crash(
        _raise(RuntimeError("boom")),
        capture=lambda exc: order.append("capture"),
        flush=lambda: order.append("flush"),
    )
    assert order == ["capture", "flush"]


def test_ui_state_is_written_before_the_blocking_flush():
    order = []
    worker_crash.report_worker_crash(
        _raise(RuntimeError("boom")),
        capture=lambda exc: order.append("capture"),
        flush=lambda: order.append("flush"),
        set_state=lambda msg: order.append("state"),
    )
    assert order.index("state") < order.index("flush")


def test_nothing_is_flushed_when_reporting_is_disabled():
    flush = Recorder()
    outcome = worker_crash.report_worker_crash(_raise(RuntimeError("boom")), flush=flush)

    assert outcome.reported is False
    assert outcome.flushed is False
    assert flush.calls == [], "no event was queued, so there is nothing to wait for"


def test_traceback_is_logged_not_just_the_message():
    log = Recorder()
    worker_crash.report_worker_crash(_raise(ValueError("deep")), log=log)

    logged = "\n".join(call[0][0] for call in log.calls)
    assert "Traceback" in logged
    assert "test_worker_crash.py" in logged


# --- the crash handler may never raise ------------------------------------


def test_a_failing_state_write_still_lets_the_report_through():
    capture, flush = Recorder(), Recorder()

    outcome = worker_crash.report_worker_crash(
        _raise(RuntimeError("boom")),
        capture=capture, flush=flush, set_state=Recorder(boom=True),
    )

    assert outcome.state_written is False
    assert outcome.reported is True and outcome.flushed is True


def test_a_failing_capture_still_lets_the_ui_learn():
    set_state = Recorder()

    outcome = worker_crash.report_worker_crash(
        _raise(RuntimeError("boom")),
        capture=Recorder(boom=True), set_state=set_state,
    )

    assert outcome.reported is False
    assert outcome.state_written is True
    assert set_state.calls, "the operator must not be left on STARTING"


@pytest.mark.parametrize("failing", ["capture", "flush", "set_state", "log"])
def test_no_callable_can_raise_out_of_the_crash_path(failing):
    kwargs = {name: Recorder() for name in ("capture", "flush", "set_state", "log")}
    kwargs[failing] = Recorder(boom=True)

    # Must not raise: an exception here would replace the real fault.
    worker_crash.report_worker_crash(_raise(RuntimeError("boom")), **kwargs)


# --- shutdown is not a crash ----------------------------------------------


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(0)])
def test_stopping_the_worker_files_no_issue_and_sets_no_error(exc):
    capture, set_state = Recorder(), Recorder()

    outcome = worker_crash.report_worker_crash(
        _raise(exc), capture=capture, set_state=set_state,
    )

    assert outcome.quiet is True
    assert capture.calls == [], "a stop must not appear on the dashboard"
    assert set_state.calls == [], "a stop must not overwrite the stopping state"


def test_quiet_exits_are_still_logged():
    log = Recorder()
    worker_crash.report_worker_crash(_raise(KeyboardInterrupt()), log=log)
    assert log.calls


# --- error text -----------------------------------------------------------


def test_an_empty_exception_message_still_names_the_type():
    assert worker_crash.format_state_error(KeyError()) == "KeyError"


def test_error_text_keeps_the_message_when_there_is_one():
    assert worker_crash.format_state_error(OSError("disk full")) == "OSError: disk full"


def test_banner_distinguishes_the_handled_path_from_multiprocessing_s_own():
    banner = worker_crash.format_crash_banner("worker", RuntimeError("boom"))
    assert banner.startswith("[FATAL]")
    assert "worker" in banner and "RuntimeError" in banner
    assert "Process Process-" not in banner


def test_missing_dependency_message_names_every_missing_package():
    msg = worker_crash.missing_dependency_message(["numpy", "SpeechRecognition"])
    assert "numpy and SpeechRecognition" in msg
    assert "requirements.txt" in msg


# --- excepthook chaining --------------------------------------------------


def test_chaining_keeps_the_previously_installed_hook():
    order = []
    chained = worker_crash.chain_excepthook(
        lambda *a: order.append("ours"), lambda *a: order.append("sentry")
    )

    chained(ValueError, ValueError("x"), None)

    assert order == ["ours", "sentry"], "Sentry's hook must still run after ours"


def test_our_hook_blowing_up_does_not_cost_us_sentry_s():
    order = []
    chained = worker_crash.chain_excepthook(Recorder(boom=True), lambda *a: order.append("sentry"))

    chained(ValueError, ValueError("x"), None)

    assert order == ["sentry"]


def test_chaining_onto_nothing_is_fine():
    ours = Recorder()
    worker_crash.chain_excepthook(ours, None)(ValueError, ValueError("x"), None)
    assert len(ours.calls) == 1


def test_a_hook_is_never_chained_onto_itself():
    ours = Recorder()
    worker_crash.chain_excepthook(ours, ours)(ValueError, ValueError("x"), None)
    assert len(ours.calls) == 1, "chaining a hook onto itself would recurse"


def test_chaining_passes_threading_excepthook_s_single_argument_through():
    seen = []
    chained = worker_crash.chain_excepthook(lambda args: seen.append(args), None)

    hook_args = threading.ExceptHookArgs(
        (ValueError, ValueError("x"), None, threading.current_thread())
    )
    chained(hook_args)

    assert seen[0].exc_type is ValueError


def test_chaining_passes_sys_excepthook_s_three_arguments_through():
    seen = []
    chained = worker_crash.chain_excepthook(lambda *a: seen.append(a), None)

    chained(ValueError, ValueError("x"), None)

    assert seen[0][0] is ValueError
    assert isinstance(seen[0][1], ValueError)


def test_the_real_sys_excepthook_can_be_wrapped_without_losing_it():
    original = sys.excepthook
    try:
        sys.excepthook = worker_crash.chain_excepthook(Recorder(), original)
        assert sys.excepthook is not original
        sys.excepthook(ValueError, ValueError("x"), None)
    finally:
        sys.excepthook = original

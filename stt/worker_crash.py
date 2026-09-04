"""Why a dying transcription worker has to report itself, and how.

The transcription worker is a ``multiprocessing.Process``. That one fact breaks
two assumptions the rest of the app makes about error reporting.

**Sentry never sees it.** ``sentry_sdk.init`` reports uncaught exceptions by
installing ``sys.excepthook``. CPython's ``multiprocessing.process.BaseProcess.
_bootstrap`` wraps the child's ``run()`` in a bare ``except:`` that writes to
stderr via ``traceback.print_exc()`` and **never calls ``sys.excepthook``**. So
an exception in the worker reached no dashboard, and the docstring on
``_init_sentry`` claiming "both report" was simply wrong. The worker therefore
has to catch its own fatal exception and hand it to Sentry explicitly — which
also means flushing, because the process is a moment from exiting and the
background transport would otherwise be killed with the event still queued.

**The UI never learns.** Only three init steps write ``status="error"`` back to
the shared state. Anything else — or any exception after init — left the state
exactly as ``/api/transcription/start`` wrote it, so the page reported STARTING
forever. The same handler fixes that: whatever kills the worker, the operator
sees a failure rather than a spinner that never resolves.

The two are one action ("the worker is dying; say so, everywhere") and are done
together here so neither can be added without the other.

Separately, ``install_crash_diagnostics`` used to *replace* ``sys.excepthook``
with its own printer. In the processes where the hook does fire that displaced
Sentry's hook instead of running alongside it, silently turning off reporting in
the main process too. ``chain_excepthook`` is the fix: run ours, then whatever
was installed before.

Everything here takes its IO as callables. Nothing imports sentry_sdk, touches a
Manager dict, or prints on its own — the caller supplies those, and every one of
them is allowed to fail without taking down the crash path, because a crash
handler that raises loses the original exception.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, List, NamedTuple, Optional

#: Exceptions that mean "the operator asked us to stop", not "we broke". They are
#: never reported to Sentry and never set an error state.
_QUIET_EXCEPTIONS = (KeyboardInterrupt, SystemExit)


class CrashOutcome(NamedTuple):
    """What the crash path managed to do, for logging and for tests.

    Each flag is "this step ran to completion", so a partial report is visible
    rather than being flattened into a single success/failure.
    """

    reported: bool  #: the exception reached the capture callable
    flushed: bool  #: the transport was flushed after capture
    state_written: bool  #: the shared state now says "error"
    quiet: bool  #: a shutdown signal, deliberately not reported


def is_quiet_exit(exc: BaseException) -> bool:
    """True when ``exc`` is a shutdown request rather than a fault.

    ``SystemExit`` counts: argparse raises it on a bad argument, and while that
    *is* a bug, it arrives during shutdown too. Treating it as quiet costs one
    unreported misconfiguration; treating it as a crash would file an issue every
    time the worker is stopped.
    """
    return isinstance(exc, _QUIET_EXCEPTIONS)


def format_crash_banner(role: str, exc: BaseException) -> str:
    """The single line that precedes the traceback in the log.

    Kept distinct from multiprocessing's own ``Process Process-N:`` header so a
    log can be grepped for the worker's *handled* fatal path specifically.
    """
    return f"[FATAL] {role} process is exiting on an unhandled {type(exc).__name__}: {exc}"


def format_state_error(exc: BaseException) -> str:
    """The short error string the UI shows for a fatal worker exception.

    The type name is included because the message alone is routinely empty
    (``KeyError``, ``RuntimeError()``), which would render as a blank error and
    look like yet another silent failure.
    """
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def report_worker_crash(
    exc: BaseException,
    *,
    role: str = "worker",
    capture: Optional[Callable[[BaseException], Any]] = None,
    flush: Optional[Callable[[], Any]] = None,
    set_state: Optional[Callable[[str], Any]] = None,
    log: Optional[Callable[[str], Any]] = None,
) -> CrashOutcome:
    """Announce a fatal worker exception to the log, to Sentry, and to the UI.

    Ordering is deliberate. The log line goes first, because it is the only
    channel that cannot fail for network reasons and the only one available on an
    install with reporting switched off. The UI state is written *before* the
    flush, since flushing blocks for as long as the transport takes and an
    operator staring at STARTING should not wait on Sentry to find out the run is
    dead.

    Every callable is optional and every one is individually guarded: an install
    with ``sentry_enabled: false`` passes no ``capture``, and a ``set_state`` that
    raises (the Manager dict's owning process may already be gone) must not
    prevent the flush. Returns what actually happened rather than raising.
    """
    quiet = is_quiet_exit(exc)

    if log is not None:
        _safely(log, format_crash_banner(role, exc))
        for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
            _safely(log, line.rstrip("\n"))

    if quiet:
        # A stop is not a crash: no Sentry event, and no error state to clear the
        # operator's own "stopping" out from under them.
        return CrashOutcome(reported=False, flushed=False, state_written=False, quiet=True)

    state_written = set_state is not None and _safely(set_state, format_state_error(exc))
    reported = capture is not None and _safely(capture, exc)
    # Only worth blocking on a flush if something was actually queued.
    flushed = reported and flush is not None and _safely(flush)

    return CrashOutcome(
        reported=reported,
        flushed=flushed,
        state_written=state_written,
        quiet=False,
    )


def chain_excepthook(
    handler: Callable[..., Any],
    previous: Optional[Callable[..., Any]],
) -> Callable[..., Any]:
    """Run ``handler``, then the hook that was installed before it.

    Written for ``sys.excepthook`` and ``threading.excepthook`` alike, so the
    arguments are passed through untouched rather than named — the two hooks have
    different signatures (three arguments vs one args object).

    ``handler`` raising must not cost us ``previous``: the previous hook is
    usually Sentry's, and losing it is exactly the bug this exists to fix. A
    ``previous`` of ``None`` (or the same object as ``handler``, which would
    recurse) is simply skipped.
    """

    def _chained(*args: Any, **kwargs: Any) -> None:
        _safely(handler, *args, **kwargs)
        if previous is not None and previous is not handler:
            _safely(previous, *args, **kwargs)

    return _chained


def _safely(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """Call ``fn``, swallowing anything it raises. True when it returned normally.

    The crash path runs while the process is already failing; an exception
    escaping any step here would replace the real fault with a misleading one.
    """
    try:
        fn(*args, **kwargs)
        return True
    except BaseException:  # noqa: BLE001 - a crash handler may not raise
        return False


def missing_dependency_message(missing: List[str]) -> str:
    """The error text for a worker that cannot run because deps are absent.

    Lives here so the same wording reaches the log, Sentry and the UI. Previously
    this was raised before the crash handler was installed, which meant the one
    failure with a genuinely actionable fix was also the one nobody ever saw.
    """
    names = " and ".join(missing)
    return (
        f"Transcription requires {names}, which is not installed. "
        "Reinstall the dependencies with install.sh "
        "(or 'uv pip install -r requirements.txt')."
    )

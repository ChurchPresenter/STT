"""Turning a start that never finished into a start that failed.

``/api/transcription/start`` hands the work to a separate process and returns at
once, writing ``status="starting"``. Only the worker ever writes
``status="running"``, and only after audio, model, VAD and database init have all
succeeded. Between those two writes there was no signal of any kind.

A crash in that window is now caught and reported (see :mod:`stt.worker_crash`).
A **hang** is not, and cannot be: it raises nothing, so it reaches no handler.
The observed one is a faster-whisper model directory missing ``tokenizer.json``,
which sends the loader into an HTTP fetch with no timeout we can set — the UI
then sits on STARTING for ever, with no path back to a truthful state except an
operator pressing Stop or Force Reset. Nothing polled the worker's liveness on a
timer either, so a worker that had simply died looked identical.

This module is the missing judgement, kept pure so it can be tested without a
Manager dict, a process, or Flask. The caller supplies the state, whether the
process is alive, and the clock.

**Why the deadline is per-stage, not total.** A cold load of large-v3 on a slow
disk legitimately takes minutes, and a fixed total budget either kills that or is
so generous it never fires. The worker instead stamps which init step it is on;
the clock restarts at every step. A slow start keeps moving and is left alone; a
wedged one stops moving and is reported, naming the step it stopped at.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional

#: The state value the worker never leaves when it hangs.
STARTING = "starting"

#: State keys the worker stamps as it moves through init. Kept here so the
#: monolith's writer and this reader cannot drift apart.
STAGE_KEY = "init_stage"
STAGE_AT_KEY = "init_stage_at"

#: Stamped by the start route before it puts the command on the queue, so the
#: window between "the operator pressed Start" and "the worker picked it up" is
#: also covered by the deadline.
STAGE_REQUESTED = "0/5 start requested"


class StartVerdict(NamedTuple):
    """The error state a stuck start should be corrected to.

    Shaped as the fields the caller writes into ``transcription_state`` so the
    reaper has no formatting decisions left to make.
    """

    error: str  #: short cause, shown in the UI's error tooltip
    message: str  #: the operator-facing sentence


def stage_of(state: Mapping[str, Any]) -> str:
    """The init step the worker last reported, or a usable stand-in.

    A state with no stage at all is one written before this existed (or by a
    path that forgot to stamp it); "initialising" is honest there and keeps the
    message readable rather than printing ``None``.
    """
    stage = state.get(STAGE_KEY)
    if isinstance(stage, str) and stage.strip():
        return stage.strip()
    return "initialising"


def evaluate_start(
    state: Mapping[str, Any],
    *,
    worker_alive: bool,
    now: float,
    stall_seconds: float,
) -> Optional[StartVerdict]:
    """Judge a start in progress. ``None`` means "still healthy, leave it alone".

    Only ever fires while the status is ``starting``: a running, stopping or
    already-errored transcription is not this function's business, and a worker
    that is alive and progressing is not either.

    Two ways a start is over without having said so:

    * **the worker is gone** — it exited without reaching its own crash handler
      (killed, ``os._exit``, a segfault in a native model loader), so nothing
      wrote an error state;
    * **nothing has moved for too long** — the wedge case, reported against the
      stage it stopped on so the log points somewhere.

    A missing or unparseable timestamp is treated as "not stalled". Reporting a
    healthy start as dead because a field was absent would be a worse failure
    than the one being fixed: it would stop a real service mid-sentence.
    """
    if state.get("status") != STARTING:
        return None

    if not worker_alive:
        return StartVerdict(
            error="The transcription worker exited during startup.",
            message=(
                f"Startup failed while {stage_of(state)} — the worker process is gone. "
                "Check the server log, then press Start again."
            ),
        )

    if stall_seconds <= 0:
        return None  # deadline switched off

    stamped = state.get(STAGE_AT_KEY)
    if not isinstance(stamped, (int, float)):
        return None
    if now - float(stamped) <= stall_seconds:
        return None

    stalled_for = int(now - float(stamped))
    return StartVerdict(
        error=f"Startup stalled at {stage_of(state)}.",
        message=(
            f"No progress for {stalled_for // 60} minutes while {stage_of(state)}. "
            "The model files may be incomplete — check the Model Manager, then press Start again."
        ),
    )

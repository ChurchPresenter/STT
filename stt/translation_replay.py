"""Replay a past service's captions through the translator and score the result.

Every translation change so far has been argued from a handful of captions read by
eye. That is how a change that fixes four captions and breaks six gets shipped: the
four are the ones you went looking for. A service database is a far better witness —
it holds every caption the model actually saw, with what it actually returned — and
re-running a candidate setting over one turns "this prompt reads better" into a count.

The measurement this exists to make honest is a *comparison*. A single run's numbers
mean little in isolation: what matters is which captions changed against the run
before it, and in which direction. So :func:`compare` is the real output, and
:func:`shipped_run` makes the baseline free — the captions a service already produced
are a run, no model required, so a candidate always has something to be scored against.

Scoring reuses :mod:`stt.llm_translate` rather than reimplementing it. A harness with
its own copy of the validation rules measures its copy, not the thing being shipped.

Stdlib-only and model-free: the caller injects ``translate_fn``, so this module is
unit-testable without a GPU, a server, or a network.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from stt.llm_translate import check_translation, word_count

# A caption's source text and what the service shipped for it.
#
# ``shipped_latency_ms`` is derived from the two timestamps the row carries rather
# than measured: translation_ts_ms is written when the translated text lands, so the
# difference includes queueing behind other captions. It is a service-level number,
# not a per-call one, and is reported separately from a replay's own timing for that
# reason.


@dataclass(frozen=True)
class CaptionPair:
    row_id: int
    timestamp: str
    source: str
    shipped: Optional[str]
    shipped_latency_ms: Optional[int]
    # Which engine produced ``shipped``, for sessions recorded since rows carry it.
    # Older databases leave this None, and a comparison against them has to be read
    # with that in mind: a caption a rejection sent to the NMT model is stored the
    # same way as one the LLM produced, so scoring an LLM candidate against them
    # compares it to a different model on exactly the rows the two disagree about.
    shipped_engine: Optional[str] = None
    shipped_model: Optional[str] = None


@dataclass(frozen=True)
class ReplayResult:
    """One caption's outcome under one setting."""

    pair: CaptionPair
    raw: Optional[str]
    accepted: Optional[str]
    reason: Optional[str]
    elapsed_ms: Optional[float]

    @property
    def ok(self) -> bool:
        return self.accepted is not None


@dataclass
class Summary:
    """Aggregate counters for a single run."""

    label: str
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    by_reason: Dict[str, int] = field(default_factory=dict)
    # Output-to-source word ratio over captions with a source of at least
    # ``ratio_min_source_words``, where the ratio starts to mean something.
    ratio_p10: float = 0.0
    ratio_p50: float = 0.0
    ratio_p90: float = 0.0
    short_outputs: int = 0
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None


@dataclass
class Comparison:
    """What moved between two runs, matched by row id."""

    before: Summary
    after: Summary
    identical: int = 0
    changed: int = 0
    fixed: List[ReplayResult] = field(default_factory=list)
    broken: List[ReplayResult] = field(default_factory=list)
    both_rejected: int = 0
    only_in_one: int = 0


def _read_only_uri(db_path: str) -> str:
    """A sqlite URI that cannot write, journal, or create the file.

    ``immutable=1`` rather than ``mode=ro``: a service database is opened while its
    WAL may still hold committed frames, and mode=ro would create the -shm sidecar
    next to a file this tool has no business modifying. immutable skips the WAL
    entirely, so a session recorded today reads back as of its last checkpoint and
    the operator's directory is left exactly as it was found.
    """
    return "file:%s?immutable=1" % urllib.parse.quote(db_path)


def load_session_pairs(db_path: str, *, min_source_words: int = 1) -> List[CaptionPair]:
    """The captions a service translated, oldest first.

    Mirrors the filter the caption path itself applies — final rows, not denied,
    non-empty — so a replay sees the same population the model saw. Rows the
    hallucination filter denied are excluded for the same reason they were never
    translated: they are not speech.
    """
    conn = sqlite3.connect(_read_only_uri(db_path), uri=True)
    try:
        # mt_engine/mt_model postdate most recorded sessions, so they are selected only
        # when present rather than making every older database unreadable.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(transcriptions)")}
        provenance = "mt_engine, mt_model" if {"mt_engine", "mt_model"} <= columns else "NULL, NULL"
        rows = conn.execute(
            """
            SELECT id, COALESCE(timestamp, ''), text, translated_text, ts_ms, translation_ts_ms,
                   %s
            FROM transcriptions
            WHERE COALESCE(is_final, 1) = 1
              AND COALESCE(denied, 0) = 0
              AND text IS NOT NULL AND TRIM(text) != ''
            ORDER BY id ASC
            """ % provenance
        ).fetchall()
    finally:
        conn.close()

    pairs: List[CaptionPair] = []
    for (row_id, timestamp, text, translated, ts_ms, translation_ts_ms,
         mt_engine, mt_model) in rows:
        source = (text or "").strip()
        if word_count(source) < min_source_words:
            continue
        latency: Optional[int] = None
        if ts_ms and translation_ts_ms and translation_ts_ms >= ts_ms:
            latency = int(translation_ts_ms - ts_ms)
        shipped = (translated or "").strip() or None
        pairs.append(CaptionPair(int(row_id), str(timestamp), source, shipped, latency,
                                 shipped_engine=mt_engine or None,
                                 shipped_model=mt_model or None))
    return pairs


# The settings that decide what a caption looks like, wherever the session recorded
# them. An offloaded session keeps them under mt.remote.effective.* — the translating
# box's own values, which is the only place they exist — and a local one under mt.*.
# Both are read, remote first, because on an offloaded session the local mt.llm.* keys
# describe a model that never ran.
_SETTING_KEYS = (
    ("model", ("mt.remote.effective.model", "mt.llm.model", "mt.model")),
    ("method", ("mt.remote.effective.method", "mt.method")),
    ("system_prompt", ("mt.remote.effective.llm_system_prompt", "mt.llm.system_prompt")),
    ("max_tokens", ("mt.remote.effective.llm_max_tokens", "mt.llm.max_tokens")),
    ("n_ctx", ("mt.remote.effective.llm_n_ctx", "mt.llm.n_ctx")),
    ("context_window", ("mt.remote.effective.llm_context_window", "mt.context_window")),
    ("retry_on_reject", ("mt.remote.effective.llm_retry_on_reject", "mt.llm.retry_on_reject")),
    ("fallback", ("mt.remote.effective.llm_fallback", "mt.llm.fallback")),
    ("target_language", ("mt.target_language",)),
    ("app_commit", ("app.commit",)),
)


def session_settings(db_path: str) -> Dict[str, str]:
    """What the session says it was translated with, or {} if it did not record it.

    A replay configured by whatever the operator typed is a replay that can disagree
    with the session for reasons nobody sees: the wrong prompt, a different quant, a
    context window that was never on. The session knows; ask it. An empty result is
    itself the answer — a session that recorded nothing cannot be replayed faithfully,
    and a report should say so rather than imply a comparison it cannot support.
    """
    conn = sqlite3.connect(_read_only_uri(db_path), uri=True)
    try:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "session_meta" not in names:
            return {}
        stored = dict(conn.execute("SELECT key, value FROM session_meta"))
    finally:
        conn.close()

    out: Dict[str, str] = {}
    for label, candidates in _SETTING_KEYS:
        for key in candidates:
            value = stored.get(key)
            if value not in (None, ""):
                out[label] = str(value)
                break
    return out


def settings_mismatch(session: Mapping[str, str], candidate: Mapping[str, str]) -> List[str]:
    """Settings the session and the run about to happen disagree on.

    Only compares keys the caller supplies, so an intentional change — a new prompt,
    a different context window — is named rather than hidden, and the operator sees
    exactly what they are varying instead of trusting they remembered.
    """
    differences = []
    for key, value in candidate.items():
        recorded = session.get(key)
        if recorded is not None and str(value) != recorded:
            differences.append("%s: session=%r, this run=%r" % (key, recorded, str(value)))
    return differences


def resolve_model(pair: CaptionPair, settings: Mapping[str, str]) -> Optional[str]:
    """The model that produced ``pair``, resolving the row's NULL to the session's.

    A row carries a model name only when it stopped matching what the session
    recorded at start — a hot reload — so NULL is not "unknown", it is "what the
    session says". Reading the column alone would report a mid-session change as
    the only attributable caption in the service and the rest as unattributed,
    which is exactly backwards.
    """
    if pair.shipped_model:
        return pair.shipped_model
    if pair.shipped is None:
        return None
    return settings.get("model") or None


def model_breakdown(pairs: Sequence[CaptionPair],
                    settings: Mapping[str, str]) -> Dict[str, int]:
    """How many captions each model produced, once NULLs are resolved.

    More than one entry means the model changed while the service was running, and
    a comparison that treats the session as one configuration is measuring two.
    """
    counts: Dict[str, int] = {}
    for pair in pairs:
        model = resolve_model(pair, settings)
        if model:
            counts[model] = counts.get(model, 0) + 1
    return counts


def engine_breakdown(pairs: Sequence[CaptionPair]) -> Dict[str, int]:
    """How many captions each engine produced, for a session that recorded it.

    Unlike the model, the engine is stored on every row: it genuinely varies caption
    to caption, because a rejected caption is translated by a different engine than
    the one beside it. There is nothing redundant to leave out.

    An empty result means the session predates per-row provenance, which is worth
    saying out loud in a report rather than showing as a clean-looking zero: it is
    the difference between "no caption fell back" and "we cannot tell".
    """
    counts: Dict[str, int] = {}
    for pair in pairs:
        if pair.shipped is None or not pair.shipped_engine:
            continue
        counts[pair.shipped_engine] = counts.get(pair.shipped_engine, 0) + 1
    return counts


def shipped_run(pairs: Sequence[CaptionPair], target_lang: str) -> List[ReplayResult]:
    """Score what the service actually shipped, without translating anything.

    The baseline every candidate is measured against, and free: these captions were
    already generated. Scoring them with today's rules also answers a question worth
    asking on its own — how many of a past service's captions would the *current*
    validator have rejected, had it been deployed at the time.
    """
    results: List[ReplayResult] = []
    for pair in pairs:
        if pair.shipped is None:
            results.append(ReplayResult(pair, None, None, "untranslated", None))
            continue
        accepted, reason = check_translation(pair.shipped, pair.source, target_lang)
        latency = float(pair.shipped_latency_ms) if pair.shipped_latency_ms is not None else None
        results.append(ReplayResult(pair, pair.shipped, accepted, reason, latency))
    return results


def replay(pairs: Sequence[CaptionPair], translate_fn: Callable[[str], Optional[str]],
           target_lang: str, *,
           on_progress: Optional[Callable[[int, int], None]] = None,
           clock: Callable[[], float] = time.perf_counter) -> List[ReplayResult]:
    """Run every caption through ``translate_fn`` and score the output.

    ``translate_fn`` takes the source text and returns the model's raw reply, or None
    if the call failed — the same contract the caption path works to, so a failure
    here scores as a rejection rather than vanishing from the denominator. ``clock``
    is injected so tests can time a run without waiting through one.
    """
    results: List[ReplayResult] = []
    total = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        started = clock()
        try:
            raw = translate_fn(pair.source)
        except Exception as exc:  # a dead endpoint mid-run should not lose the work done so far
            print("[REPLAY] caption %d failed: %s" % (pair.row_id, exc))
            raw = None
        elapsed_ms = (clock() - started) * 1000.0
        accepted, reason = check_translation(raw, pair.source, target_lang)
        results.append(ReplayResult(pair, raw, accepted, reason, elapsed_ms))
        if on_progress is not None:
            on_progress(index, total)
    return results


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Empty input is 0.0 rather than an exception."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def summarize(results: Sequence[ReplayResult], label: str, *,
              ratio_min_source_words: int = 8, short_ratio: float = 0.45) -> Summary:
    """Aggregate one run.

    ``short_ratio`` counts outputs far shorter than their source. Nothing in the
    validator rejects those today — the length rules only bound growth — and they are
    the silent failure this harness was built to size: a full sentence answered with
    "Thank you." is fluent, correctly scripted, number-clean, and wrong.
    """
    summary = Summary(label=label, total=len(results))
    ratios: List[float] = []
    latencies: List[float] = []
    for result in results:
        if result.ok:
            summary.accepted += 1
        else:
            summary.rejected += 1
            key = result.reason or "unknown"
            summary.by_reason[key] = summary.by_reason.get(key, 0) + 1
        if result.elapsed_ms is not None:
            latencies.append(result.elapsed_ms)
        text = result.accepted
        if text is None:
            continue
        src_words = word_count(result.pair.source)
        if src_words >= ratio_min_source_words:
            ratio = word_count(text) / float(src_words)
            ratios.append(ratio)
            if ratio < short_ratio:
                summary.short_outputs += 1

    summary.ratio_p10 = round(_percentile(ratios, 0.10), 3)
    summary.ratio_p50 = round(_percentile(ratios, 0.50), 3)
    summary.ratio_p90 = round(_percentile(ratios, 0.90), 3)
    if latencies:
        summary.latency_p50_ms = round(_percentile(latencies, 0.50), 1)
        summary.latency_p90_ms = round(_percentile(latencies, 0.90), 1)
    return summary


def compare(before: Sequence[ReplayResult], after: Sequence[ReplayResult], *,
            before_label: str = "before", after_label: str = "after") -> Comparison:
    """What the change did, caption by caption.

    ``fixed`` and ``broken`` hold the captions that crossed the accept/reject line, in
    each direction — the two lists an operator actually has to read before deciding.
    A change is only worth shipping if ``broken`` is empty or understood; a net
    improvement that quietly breaks a caption type is still a regression on Sunday.
    """
    index_after = {result.pair.row_id: result for result in after}
    comparison = Comparison(
        before=summarize(before, before_label),
        after=summarize(after, after_label),
    )
    for prior in before:
        later = index_after.pop(prior.pair.row_id, None)
        if later is None:
            comparison.only_in_one += 1
            continue
        if prior.ok and later.ok:
            if (prior.accepted or "").strip() == (later.accepted or "").strip():
                comparison.identical += 1
            else:
                comparison.changed += 1
        elif later.ok and not prior.ok:
            comparison.fixed.append(later)
        elif prior.ok and not later.ok:
            comparison.broken.append(later)
        else:
            comparison.both_rejected += 1
    comparison.only_in_one += len(index_after)
    return comparison


def render_summary(summary: Summary) -> str:
    """A few lines fit for a terminal."""
    lines = [
        "%s: %d captions, %d accepted, %d rejected"
        % (summary.label, summary.total, summary.accepted, summary.rejected),
    ]
    if summary.by_reason:
        ordered = sorted(summary.by_reason.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("  rejections: " + ", ".join("%s=%d" % (k, v) for k, v in ordered))
    lines.append("  length ratio p10/p50/p90: %.2f / %.2f / %.2f   short outputs: %d"
                 % (summary.ratio_p10, summary.ratio_p50, summary.ratio_p90, summary.short_outputs))
    if summary.latency_p50_ms is not None:
        lines.append("  latency p50/p90: %.0f ms / %.0f ms"
                     % (summary.latency_p50_ms, summary.latency_p90_ms or 0.0))
    return "\n".join(lines)


def render_comparison(comparison: Comparison, *, examples: int = 5) -> str:
    """The comparison, with the captions that crossed the line spelled out."""
    lines = [render_summary(comparison.before), render_summary(comparison.after), ""]
    lines.append("fixed: %d   broken: %d   changed: %d   identical: %d   both rejected: %d"
                 % (len(comparison.fixed), len(comparison.broken), comparison.changed,
                    comparison.identical, comparison.both_rejected))
    if comparison.only_in_one:
        lines.append("unmatched rows: %d" % comparison.only_in_one)
    for title, bucket in (("BROKEN", comparison.broken), ("FIXED", comparison.fixed)):
        for result in bucket[:examples]:
            lines.append("")
            lines.append("  [%s %s] %s" % (title, result.pair.timestamp, result.pair.source[:160]))
            lines.append("     was: %s" % ((result.pair.shipped or "-")[:160]))
            lines.append("     now: %s" % ((result.accepted or result.raw or "-")[:160]))
    return "\n".join(lines)


def as_dict(results: Sequence[ReplayResult]) -> List[Dict[str, object]]:
    """Results in a shape that survives a JSON round-trip, for diffing runs later."""
    return [
        {
            "row_id": result.pair.row_id,
            "timestamp": result.pair.timestamp,
            "source": result.pair.source,
            "shipped": result.pair.shipped,
            "raw": result.raw,
            "accepted": result.accepted,
            "reason": result.reason,
            "elapsed_ms": result.elapsed_ms,
        }
        for result in results
    ]


def from_dict(rows: Sequence[Dict[str, object]]) -> List[ReplayResult]:
    """Rebuild results saved by :func:`as_dict`, so an old run stays comparable."""
    out: List[ReplayResult] = []
    for row in rows:
        row_id = row.get("row_id")
        pair = CaptionPair(
            row_id=(row_id if isinstance(row_id, int) else 0),
            timestamp=str(row.get("timestamp") or ""),
            source=str(row.get("source") or ""),
            shipped=(str(row["shipped"]) if row.get("shipped") is not None else None),
            shipped_latency_ms=None,
        )
        elapsed = row.get("elapsed_ms")
        out.append(ReplayResult(
            pair=pair,
            raw=(str(row["raw"]) if row.get("raw") is not None else None),
            accepted=(str(row["accepted"]) if row.get("accepted") is not None else None),
            reason=(str(row["reason"]) if row.get("reason") is not None else None),
            elapsed_ms=(float(elapsed) if isinstance(elapsed, (int, float)) else None),
        ))
    return out


def http_translator(endpoint: str, source_lang: str, target_lang: str, *,
                    timeout: float = 30.0) -> Callable[[str], Optional[str]]:
    """A ``translate_fn`` backed by another machine's ``POST /api/translate``.

    Two things about that endpoint matter here. The caller's IP must be in the
    server's ``live_translation.trusted_clients``, or every call returns 403. And the
    server keeps an LRU cache keyed by text: without ``return_extras`` a second run
    replays the first run's answers, which would make every comparison show a
    perfect, and entirely fictional, zero-change result.
    """
    url = endpoint.rstrip("/") + "/api/translate"

    def translate(text: str) -> Optional[str]:
        body = json.dumps({
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "return_extras": True,  # bypasses the server-side text cache
        }).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied LAN endpoint
            payload = json.loads(response.read().decode("utf-8"))
        translated = payload.get("translated_text")
        return str(translated) if translated else None

    return translate


def _load_run(path: str) -> Tuple[List[ReplayResult], str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return from_dict(payload.get("results") or []), str(payload.get("label") or path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m stt.translation_replay",
        description="Replay a service database's captions through a translator and score them.")
    parser.add_argument("db", help="session database to replay (opened read-only)")
    parser.add_argument("--endpoint", help="translating server, e.g. http://192.168.2.52:8080. "
                                           "Omitted: score what the service shipped, no calls made.")
    parser.add_argument("--source-lang", default="ru")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--label", default="", help="name for this run in the report")
    parser.add_argument("--limit", type=int, default=0, help="replay only the first N captions")
    parser.add_argument("--out", help="write the run to this JSON file for later comparison")
    parser.add_argument("--baseline", help="compare against a run saved by --out; "
                                           "default compares against what the service shipped")
    parser.add_argument("--examples", type=int, default=5,
                        help="fixed/broken captions to print per bucket")
    args = parser.parse_args(argv)

    pairs = load_session_pairs(args.db)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if not pairs:
        print("no translatable captions found in %s" % args.db)
        return 1
    print("loaded %d captions from %s" % (len(pairs), args.db))

    settings = session_settings(args.db)
    if settings:
        print("  recorded settings:")
        for key in ("method", "model", "context_window", "max_tokens", "n_ctx",
                    "retry_on_reject", "fallback", "app_commit"):
            if key in settings:
                print("    %-16s %s" % (key, settings[key]))
        prompt = settings.get("system_prompt")
        if prompt:
            print("    %-16s %d chars, %s" % ("system_prompt", len(prompt), prompt[:48] + "…"))
        else:
            print("    %-16s not recorded — the prompt that produced these captions is unknown"
                  % "system_prompt")
    else:
        print("  recorded settings: none — this session predates session_meta, so a replay\n"
              "                     cannot be checked against what actually produced it.")

    models = model_breakdown(pairs, settings)
    if len(models) > 1:
        # Two models in one service means the comparison below is measuring two
        # configurations as though they were one. Say so before the numbers.
        print("  WARNING: the model changed mid-session — %s" % ", ".join(
            "%s=%d" % (name, count) for name, count in sorted(models.items())))

    engines = engine_breakdown(pairs)
    if engines:
        print("  shipped by: " + ", ".join(
            "%s=%d" % (name, count) for name, count in sorted(engines.items())))
    else:
        print("  shipped by: not recorded — this session predates per-row provenance, so a\n"
              "              caption the LLM declined is stored exactly like one it produced.\n"
              "              Treat a comparison against it as approximate.")

    if args.baseline:
        before, before_label = _load_run(args.baseline)
    else:
        before, before_label = shipped_run(pairs, args.target_lang), "shipped"

    if not args.endpoint:
        print(render_summary(summarize(before, before_label)))
        return 0

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print("  %d/%d" % (done, total), flush=True)

    translate = http_translator(args.endpoint, args.source_lang, args.target_lang)
    after = replay(pairs, translate, args.target_lang, on_progress=progress)
    label = args.label or args.endpoint

    print()
    print(render_comparison(compare(before, after, before_label=before_label, after_label=label),
                            examples=args.examples))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"label": label, "db": args.db, "results": as_dict(after)},
                      handle, ensure_ascii=False, indent=1)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

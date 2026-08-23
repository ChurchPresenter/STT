#!/usr/bin/env python3
"""Produce the session database the demo replays.

Two sources:

  --synthetic      Generate a written service. Nobody's speech, no review needed.
                   This is what a public download should ship.

  --from-session   Scrub a real recording. Reduces risk; does not certify. Read the
                   .review.txt it writes, end to end, before the result goes anywhere.
                   Neither the source nor the result may be committed.

Usage:
    python scripts/make_demo_session.py --synthetic -o build/demo.db
    python scripts/make_demo_session.py --from-session service.db -o demo.db \\
        --names names.txt --minutes 30
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt import demo_synth  # noqa: E402


def _synthetic(args) -> int:
    rows = demo_synth.generate_rows(seed=args.seed)
    demo_synth.write(args.out, rows)
    print(f"Wrote {args.out}")
    print(f"  {len(rows)} captions, {demo_synth.duration_minutes(rows):.1f} minutes, "
          f"seed {args.seed}")
    print("  Synthetic: no recorded speech, safe to ship.")
    return 0


def _scrub(args) -> int:
    from stt import demo_scrub

    names = []
    if args.names:
        with open(args.names, encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()
                     and not line.startswith("#")]
    if not names:
        print("WARNING: no --names file given, so no personal names will be replaced.",
              file=sys.stderr)

    window = None
    if args.minutes:
        window = (0, int(args.minutes * 60_000))

    report = demo_scrub.scrub_session(
        args.from_session, args.out, demo_scrub.build_rules(names),
        window=window, require_translation=not args.allow_untranslated)

    review = args.out + ".review.txt"
    demo_scrub.write_review(review, report)

    print(f"Wrote {args.out}")
    print(f"  {report.rows_out} captions kept, {report.rows_dropped} dropped")
    print(f"  replacements: {dict(report.replacements) or 'none'}")
    print(f"  residual flags: {len(report.residual_flags)}")
    print(f"\nReview file: {review}")
    print("Read it end to end before this recording goes anywhere. The scrubber "
          "reduces risk; it does not certify.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true",
                        help="generate a written service")
    source.add_argument("--from-session", metavar="DB",
                        help="scrub an existing session database")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--seed", type=int, default=20260823,
                        help="synthetic only; the same seed gives the same service")
    parser.add_argument("--names", metavar="FILE",
                        help="scrub only; one name per line to replace")
    parser.add_argument("--minutes", type=float,
                        help="scrub only; keep only the first N minutes")
    parser.add_argument("--allow-untranslated", action="store_true",
                        help="scrub only; keep captions with no translation")
    args = parser.parse_args()

    if args.synthetic:
        return _synthetic(args)
    return _scrub(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""The declarative phase-naming engine (stt/phase_rules.py).

The first job of this suite is a conformance one. The shipped rule file has to reproduce
what the hand-written labeller did, block for block, before any new rule is allowed to
change an answer — otherwise a rewrite that quietly renames a Sunday looks like an
improvement. TestConformance runs both over the same services and compares.
"""

import json
import os

import pytest

from stt.phase_rules import Rule, apply_rules, parse_rules
from stt.service_phase import (
    MUSIC,
    QUIET,
    SPEECH,
    bin_rows,
    classify_bin,
    compile_cues,
    compile_fragment_cues,
    label_blocks,
    sum_cues,
    track_blocks,
)

MIN = 60_000
RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "config", "service_phases.default.json")

# The settings the shipped rule file encodes, so the old labeller is asked the same question.
CFG = {"sermon_min_minutes": 8, "songs_min_minutes": 3, "communion_min_hits": 12,
       "closing_max_minutes": 6}


def shipped_rules():
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return parse_rules(json.load(f))


def rows(spec, start=1_000_000, text=""):
    out = []
    for i, c in enumerate(spec):
        stype = {"M": "Music", "S": "Speaking", "_": "Quiet"}[c]
        out.append((start + i * MIN, stype, 0.9 if c == "M" else 0.0, text))
    return out


def blocks_for(spec, text="", cues=None):
    """Blocks with their cues already summed, which is the state the engine is handed."""
    b = bin_rows(rows(spec, text=text), cues=cues)
    blocks = track_blocks([classify_bin(x) for x in b], b)
    for block in blocks:
        block.cues = sum_cues(b, block)
    return blocks, b


def named(blocks):
    return [(x.kind, x.label) for x in blocks]


class TestConformance:
    """The rule file must say exactly what label_blocks said."""

    SPECS = [
        "M" * 5 + "S" * 12,
        "S" * 12 + "M" * 5 + "S" * 12,
        "S" * 4 + "M" * 6 + "S" * 12,
        "M" * 6 + "_" * 4 + "S" * 4 + "M" * 6 + "S" * 12,
        "M" * 6 + "_" * 4 + "M" * 5 + "_" * 4 + "S" * 4 + "M" * 6 + "S" * 12,
        "S" * 20,
        "M" * 20,
        "_" * 10 + "S" * 12,
        "S" * 12 + "M" * 5 + "S" * 4,
        "S" * 12 + "M" * 5 + "S" * 4 + "_" * 5,
        "M" * 3 + "_" * 4 + "S" * 4 + "M" * 6 + "_" * 4 + "M" * 6 + "S" * 12,
        "S" * 4 + "M" * 4 + "S" * 12 + "M" * 4 + "S" * 12 + "M" * 4 + "S" * 5,
    ]

    @pytest.mark.parametrize("spec", SPECS)
    def test_rule_file_reproduces_the_hand_written_labeller(self, spec):
        old_blocks, old_bins = blocks_for(spec)
        label_blocks(old_blocks, old_bins, CFG)

        new_blocks, _ = blocks_for(spec)
        apply_rules(new_blocks, shipped_rules())

        assert named(new_blocks) == named(old_blocks)

    @pytest.mark.parametrize("spec", SPECS)
    def test_confidences_match_too(self, spec):
        old_blocks, old_bins = blocks_for(spec)
        label_blocks(old_blocks, old_bins, CFG)
        new_blocks, _ = blocks_for(spec)
        apply_rules(new_blocks, shipped_rules())
        assert [b.confidence for b in new_blocks] == [b.confidence for b in old_blocks]

    def test_the_shipped_file_parses_into_rules(self):
        rules = shipped_rules()
        assert [r.name for r in rules][:3] == ["Communion", "Communion", "Sermon"]
        assert any(r.span for r in rules)


class TestMatching:
    def test_a_quiet_block_matches_nothing_and_stays_unnamed(self):
        blocks, _ = blocks_for("S" * 10 + "_" * 6 + "S" * 10)
        apply_rules(blocks, shipped_rules())
        assert all(b.label is None for b in blocks if b.kind == QUIET)

    def test_numbering_is_per_rule_and_in_order(self):
        blocks, _ = blocks_for("S" * 12 + "M" * 5 + "S" * 12 + "M" * 5)
        apply_rules(blocks, shipped_rules())
        assert [b.label for b in blocks if b.kind == SPEECH] == ["Sermon 1", "Sermon 2"]
        assert [b.label for b in blocks if b.kind == MUSIC] == ["Songs 1", "Songs 2"]

    def test_before_first_stops_applying_once_that_phase_is_seen(self):
        blocks, _ = blocks_for("S" * 4 + "M" * 6 + "S" * 12 + "M" * 6 + "S" * 4 + "M" * 6)
        apply_rules(blocks, shipped_rules())
        speech = [b.label for b in blocks if b.kind == SPEECH]
        assert speech[0] == "Opening"
        assert "Opening" not in speech[1:]

    def test_an_ongoing_block_does_not_commit_to_a_confident_name(self):
        blocks, _ = blocks_for("S" * 4)
        blocks[-1].ongoing = True
        apply_rules(blocks, shipped_rules())
        assert blocks[-1].label == "Opening" and blocks[-1].confidence == 0.3

    def test_first_sunday_raises_communion_confidence_only(self):
        cues = compile_cues({"communion": [r"причасти\w*"]})
        blocks, _ = blocks_for("S" * 12, text="причастие причастие", cues=cues)
        apply_rules(blocks, shipped_rules(), first_sunday=True)
        assert blocks[0].label == "Communion" and blocks[0].confidence == 0.8

        blocks2, _ = blocks_for("S" * 12, text="причастие причастие", cues=cues)
        apply_rules(blocks2, shipped_rules(), first_sunday=False)
        assert blocks2[0].confidence == 0.6


class TestBrokenRulesAreSkipped:
    """A typo costs its own rule, never the whole service."""

    @pytest.mark.parametrize("bad", [
        {"phases": [{"name": "", "match": {"kind": "S"}}]},
        {"phases": [{"name": "X"}]},
        {"phases": [{"match": {"kind": "S"}}]},
        {"phases": ["not a rule"]},
        {"phases": [{"name": "X", "match": "not a dict"}]},
        {},
        None,
    ])
    def test_unusable_rules_are_dropped(self, bad):
        assert parse_rules(bad) == [] or all(r.name for r in parse_rules(bad))

    def test_a_bad_rule_does_not_take_the_good_ones_with_it(self):
        rules = parse_rules({"phases": [
            {"name": "X"},
            {"name": "Sermon", "match": {"kind": "S", "min_minutes": 8}, "confidence": 0.7},
        ]})
        assert [r.name for r in rules] == ["Sermon"]

    def test_a_non_numeric_confidence_falls_back(self):
        rules = parse_rules({"phases": [
            {"name": "Sermon", "match": {"kind": "S"}, "confidence": "very"},
        ]})
        assert rules[0].confidence == 0.5


class TestCommunionSpan:
    """The case this engine was built for.

    Taken from a service that the previous detector missed completely:
    the words of institution were read over quiet distribution split by short music, so the
    communion cues fell 25-to-3 in blocks the old rule never examined. The shapes below
    reproduce that service's structure.
    """

    VERSE_1 = "В ту ночь, в которую предан был, взял хлеб и, возблагодарив, преломил"
    VERSE_2 = "Также и чашу после вечери, чаша есть новый завет в крови"

    def fragments(self):
        cfg = json.load(open(os.path.join(os.path.dirname(RULES_FILE), "config.default.json"),
                             encoding="utf-8"))["service_phase"]
        return compile_fragment_cues(cfg["cue_fragments"])

    def service(self, verse_text):
        """Speaking, then quiet distribution with short music through it, then songs."""
        parts = [
            (rows("S" * 12, start=1_000_000), ""),
            (rows("_" * 4, start=1_000_000 + 12 * MIN), verse_text),
            (rows("M" * 2, start=1_000_000 + 16 * MIN), ""),
            (rows("_" * 13, start=1_000_000 + 18 * MIN), self.VERSE_2),
            (rows("M" * 2, start=1_000_000 + 31 * MIN), ""),
            (rows("_" * 5, start=1_000_000 + 33 * MIN), ""),
            (rows("M" * 6, start=1_000_000 + 38 * MIN), ""),
        ]
        out = []
        for chunk, text in parts:
            out.extend((ts, st, mp, text) for ts, st, mp, _ in chunk)
        return out

    def analyzed(self, verse_text):
        b = bin_rows(self.service(verse_text), cues=self.fragments())
        blocks = track_blocks([classify_bin(x) for x in b], b)
        for block in blocks:
            block.cues = sum_cues(b, block)
        return blocks, apply_rules(blocks, shipped_rules())

    def test_the_verse_opens_a_communion_over_the_quiet_distribution(self):
        blocks, spans = self.analyzed(self.VERSE_1)
        assert len(spans) == 1
        span = spans[0]
        assert span.label == "Communion"
        # It starts at the quiet block the verse was read in, not at the sermon before it.
        assert blocks[span.start_index].kind == QUIET
        # And it swallows the short music and the quiet after it rather than stopping dead.
        assert span.end_index > span.start_index + 1
        assert blocks[span.end_index].kind in (QUIET, MUSIC)

    def test_the_sermon_before_it_is_untouched(self):
        blocks, _ = self.analyzed(self.VERSE_1)
        assert blocks[0].label == "Sermon 1"

    def test_a_sermon_quoting_the_verse_stays_a_sermon(self):
        # The operator's own caveat: a sermon can be *about* communion. It keeps talking,
        # so not_when catches it on length and no span forms.
        rows_ = [(ts, st, mp, self.VERSE_1 + " " + self.VERSE_2)
                 for ts, st, mp, _ in rows("S" * 20)]
        b = bin_rows(rows_, cues=self.fragments())
        blocks = track_blocks([classify_bin(x) for x in b], b)
        for block in blocks:
            block.cues = sum_cues(b, block)
        spans = apply_rules(blocks, shipped_rules())
        assert blocks[0].label == "Sermon 1"
        assert spans == []

    def test_one_fragment_alone_is_not_enough(self):
        # "the cup" in a sermon is a topic; two different lines of the formula is a reading.
        _, spans = self.analyzed("тело моё")
        assert not any(s.label == "Communion" and s.start_index == 1 for s in spans)

    def test_blocks_inside_the_span_carry_no_competing_name(self):
        blocks, spans = self.analyzed(self.VERSE_1)
        span = spans[0]
        for i in range(span.start_index + 1, span.end_index + 1):
            assert blocks[i].label is None

    def test_songs_after_the_communion_still_number(self):
        blocks, _ = self.analyzed(self.VERSE_1)
        assert blocks[-1].label == "Songs 1"


class TestOrdinalRestart:
    def test_music_before_the_anchor_is_unnumbered(self):
        blocks, _ = blocks_for("M" * 6 + "_" * 4 + "S" * 4 + "M" * 6 + "S" * 12)
        apply_rules(blocks, shipped_rules())
        assert [b.label for b in blocks if b.kind == MUSIC] == ["Music", "Songs 1"]

    def test_without_the_anchor_the_count_starts_at_the_first_block(self):
        blocks, _ = blocks_for("M" * 5 + "S" * 12 + "M" * 5)
        apply_rules(blocks, shipped_rules())
        assert [b.label for b in blocks if b.kind == MUSIC] == ["Songs 1", "Songs 2"]

"""How many sermons a service may have, and which blocks keep the name.

The case behind this: a real service produced Sermon 1 (30 min), Sermon 2 (47 min) and
Sermon 3 (14 min) where the church has two sermons. Nothing is wrong with the third block
on its own — it is speech, and it is longer than the sermon threshold — so no per-block
rule can reject it. The lengths in these tests are that service's.

Nothing here ships a number. A cap is a fact about one congregation, and the tests that
matter most are the ones asserting that an install which has said nothing about its
services is left exactly as it was.
"""

from stt.phase_rank import (
    Demotion,
    apply_limits,
    base_name,
    fallback_for,
    limit_notes,
    parse_limits,
    rank_and_limit,
)
from stt.phase_rules import parse_rules

MIN = 60_000
BASE = 1_700_000_000_000

RULES = parse_rules({"phases": [
    {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
    {"name": "Closing", "match": {"kind": "S", "max_minutes": 6, "is_last_named": True}},
    {"name": "Speaking", "match": {"kind": "S"}},
    {"name": "Songs", "number": True, "match": {"kind": "M", "min_minutes": 3}},
    {"name": "Music", "match": {"kind": "M"}},
]})


class Block:
    """Enough of stt.service_phase.Block for the ranker, which only reads attributes."""

    def __init__(self, index, label, minutes, start_minute, kind="S", ongoing=False):
        self.index = index
        self.label = label
        self.minutes = minutes
        self.start_ms = BASE + start_minute * MIN
        self.kind = kind
        self.ongoing = ongoing
        self.confidence = 0.7


def service(*specs):
    """Blocks from (label, minutes, start_minute[, ongoing]) tuples."""
    out = []
    for i, spec in enumerate(specs):
        label, minutes, start = spec[0], spec[1], spec[2]
        ongoing = spec[3] if len(spec) > 3 else False
        kind = "M" if str(label).startswith(("Songs", "Music")) else "S"
        out.append(Block(i, label, minutes, start, kind=kind, ongoing=ongoing))
    return out


def labels(blocks):
    return [b.label for b in blocks]


class TestBaseName:
    def test_an_ordinal_is_dropped(self):
        assert base_name("Sermon 2") == "Sermon"

    def test_an_unnumbered_name_survives(self):
        assert base_name("Communion") == "Communion"

    def test_nothing_reads_as_empty(self):
        assert base_name(None) == ""


class TestRankAndLimit:
    def test_the_real_service_keeps_its_two_sermons(self):
        blocks = service(("Sermon 1", 30, 13), ("Songs 2", 7, 43),
                         ("Sermon 2", 47, 50), ("Sermon 3", 14, 110))
        out = rank_and_limit(blocks, name="Sermon", max_count=2, fallback_label="Speaking")
        assert labels(blocks) == ["Sermon 1", "Songs 2", "Sermon 2", "Speaking"]
        assert [d.minutes for d in out] == [14]

    def test_survivors_renumber_in_time_order_not_rank_order(self):
        # The longest sermon is the second one; it must not become "Sermon 1".
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 47, 50), ("Sermon 3", 14, 110))
        rank_and_limit(blocks, name="Sermon", max_count=2, fallback_label="Speaking")
        assert labels(blocks) == ["Sermon 1", "Sermon 2", "Speaking"]

    def test_demoting_the_first_sermon_renumbers_the_rest(self):
        blocks = service(("Sermon 1", 10, 5), ("Sermon 2", 40, 20), ("Sermon 3", 38, 70))
        out = rank_and_limit(blocks, name="Sermon", max_count=2, fallback_label="Speaking")
        assert labels(blocks) == ["Speaking", "Sermon 1", "Sermon 2"]
        assert [d.was for d in out] == ["Sermon 1"]

    def test_a_service_within_its_limit_is_untouched(self):
        blocks = service(("Sermon 1", 39, 9), ("Songs 2", 3, 48), ("Sermon 2", 29, 51))
        assert rank_and_limit(blocks, name="Sermon", max_count=2,
                              fallback_label="Speaking") == []
        assert labels(blocks) == ["Sermon 1", "Songs 2", "Sermon 2"]

    def test_a_tie_keeps_the_earlier_block(self):
        # Two candidates of equal length must not flap between ticks.
        blocks = service(("Sermon 1", 20, 10), ("Sermon 2", 20, 40))
        out = rank_and_limit(blocks, name="Sermon", max_count=1, fallback_label="Speaking")
        assert labels(blocks) == ["Sermon 1", "Speaking"]
        assert [d.was for d in out] == ["Sermon 2"]

    def test_an_ongoing_block_is_never_demoted(self):
        # It may still be growing into the sermon it claims to be.
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 9, 60, True))
        assert rank_and_limit(blocks, name="Sermon", max_count=1,
                              fallback_label="Speaking") == []
        assert labels(blocks) == ["Sermon 1", "Sermon 2"]

    def test_a_live_block_does_not_crowd_out_a_finished_one(self):
        # The cap counts finished blocks only, so a service can be briefly over it while
        # one is still being spoken; it settles when that block ends.
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 20, 50),
                         ("Sermon 3", 9, 80, True))
        assert rank_and_limit(blocks, name="Sermon", max_count=2,
                              fallback_label="Speaking") == []
        assert labels(blocks) == ["Sermon 1", "Sermon 2", "Sermon 3"]

    def test_it_settles_once_that_block_ends(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 20, 50), ("Sermon 3", 9, 80))
        rank_and_limit(blocks, name="Sermon", max_count=2, fallback_label="Speaking")
        assert labels(blocks) == ["Sermon 1", "Sermon 2", "Speaking"]

    def test_closed_only_off_ranks_the_ongoing_block_too(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 9, 60, True))
        rank_and_limit(blocks, name="Sermon", max_count=1, fallback_label="Speaking",
                       closed_only=False)
        assert labels(blocks) == ["Sermon 1", "Speaking"]

    def test_a_zero_cap_demotes_everything(self):
        blocks = service(("Sermon 1", 30, 13))
        rank_and_limit(blocks, name="Sermon", max_count=0, fallback_label="Speaking")
        assert labels(blocks) == ["Speaking"]

    def test_a_negative_cap_is_ignored(self):
        blocks = service(("Sermon 1", 30, 13))
        assert rank_and_limit(blocks, name="Sermon", max_count=-1,
                              fallback_label="Speaking") == []

    def test_a_demoted_block_loses_its_confidence(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 14, 110))
        rank_and_limit(blocks, name="Sermon", max_count=1, fallback_label="Speaking")
        assert blocks[1].confidence == 0.3


class TestFallbackComesFromTheRules:
    def test_the_fallback_is_the_rule_vocabulary(self):
        assert fallback_for(RULES, "Sermon", "S") == "Speaking"

    def test_a_church_that_renamed_the_phase_gets_its_own_word(self):
        rules = parse_rules({"phases": [
            {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
            {"name": "Talk", "match": {"kind": "S"}},
        ]})
        assert fallback_for(rules, "Sermon", "S") == "Talk"

    def test_a_conditional_rule_is_not_a_catch_all(self):
        # Closing needs is_last_named, so it cannot be handed a block that is not last.
        rules = parse_rules({"phases": [
            {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
            {"name": "Closing", "match": {"kind": "S", "is_last_named": True}},
        ]})
        assert fallback_for(rules, "Sermon", "S") is None

    def test_a_rule_of_another_kind_is_not_used(self):
        rules = parse_rules({"phases": [
            {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
            {"name": "Music", "match": {"kind": "M"}},
        ]})
        assert fallback_for(rules, "Sermon", "S") is None

    def test_an_unnameable_block_is_left_without_a_label(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 14, 110))
        rank_and_limit(blocks, name="Sermon", max_count=1, fallback_label=None)
        assert blocks[1].label is None


class TestApplyLimits:
    def test_no_limits_is_a_no_op(self):
        # Every install that has not described its services.
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 47, 50), ("Sermon 3", 14, 110))
        assert apply_limits(blocks, RULES, None) == []
        assert apply_limits(blocks, RULES, {}) == []
        assert labels(blocks) == ["Sermon 1", "Sermon 2", "Sermon 3"]

    def test_a_configured_cap_is_applied(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 47, 50), ("Sermon 3", 14, 110))
        out = apply_limits(blocks, RULES, {"Sermon": {"max": 2}})
        assert labels(blocks) == ["Sermon 1", "Sermon 2", "Speaking"]
        assert len(out) == 1

    def test_an_explicit_fallback_wins(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 14, 110))
        apply_limits(blocks, RULES, {"Sermon": {"max": 1, "fallback": "Announcements"}})
        assert blocks[1].label == "Announcements"

    def test_several_phases_can_be_capped(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 14, 110),
                         ("Songs 1", 5, 8), ("Songs 2", 7, 43), ("Songs 3", 3, 105))
        apply_limits(blocks, RULES, {"Sermon": {"max": 1}, "Songs": {"max": 2}})
        assert blocks[1].label == "Speaking"
        assert blocks[4].label == "Music"

    def test_a_malformed_limit_costs_only_itself(self):
        blocks = service(("Sermon 1", 30, 13), ("Sermon 2", 14, 110))
        apply_limits(blocks, RULES, {"Sermon": {"max": "two"}})
        assert labels(blocks) == ["Sermon 1", "Sermon 2"]


class TestParseLimits:
    def test_a_well_formed_block_is_read(self):
        assert parse_limits({"Sermon": {"max": 2}}) == {"Sermon": {"max": 2}}

    def test_comments_are_skipped(self):
        assert parse_limits({"_comment": "x", "Sermon": {"max": 2}}) == {"Sermon": {"max": 2}}

    def test_junk_is_dropped_without_taking_the_rest(self):
        # A hand-edited file is the normal case, so a typo must cost its own setting only.
        got = parse_limits({"Sermon": {"max": 2}, "Songs": {"max": "lots"},
                            "Prayer": "nonsense", "Music": {"max": -1}})
        assert got == {"Sermon": {"max": 2}}

    def test_optional_fields_survive(self):
        got = parse_limits({"Sermon": {"max": 1, "fallback": "Talk", "closed_only": False}})
        assert got == {"Sermon": {"max": 1, "fallback": "Talk", "closed_only": False}}

    def test_anything_that_is_not_a_map_reads_as_none(self):
        assert parse_limits(None) == {} and parse_limits([1, 2]) == {}


class TestNotes:
    def test_a_demotion_says_what_it_did_and_why(self):
        note = limit_notes([Demotion(3, BASE, 14, "Sermon 3", "Speaking", "over")])[0]
        assert "Sermon 3" in note and "14" in note and "Speaking" in note

    def test_no_demotions_says_nothing(self):
        assert limit_notes([]) == []


class TestAStoppedServiceSettles:
    """The gap that survived the first cut of this, caught by replaying the real service.

    A stopped session never closes its final block: the tick that wrote the timeline ran
    while the service was running, so the last block is always ``ongoing``. Anything that
    exempts an ongoing block therefore never judges that one — and the last block is exactly
    where a spurious phase sits. The service that prompted these limits ended with a
    fourteen-minute "Sermon 3" that was really the closing, still flagged ongoing hours
    later.
    """

    def service(self):
        # This morning's shape, with the final block still marked ongoing as a stopped
        # session leaves it.
        return service_with_ongoing_tail()

    def test_while_the_service_runs_the_last_block_is_left_alone(self):
        blocks = self.service()
        assert rank_and_limit(blocks, name="Sermon", max_count=2,
                              fallback_label="Speaking", closed_only=True) == []
        assert labels(blocks)[-1] == "Sermon 3"

    def test_once_it_has_stopped_the_last_block_is_judged(self):
        blocks = self.service()
        out = rank_and_limit(blocks, name="Sermon", max_count=2,
                             fallback_label="Speaking", closed_only=False)
        assert [d.was for d in out] == ["Sermon 3"]
        assert labels(blocks) == ["Sermon 1", "Sermon 2", "Speaking"]


def service_with_ongoing_tail():
    return service(("Sermon 1", 30, 13), ("Sermon 2", 47, 50), ("Sermon 3", 14, 110, True))

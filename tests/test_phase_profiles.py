"""One church's services, in that church's own files.

The rule these tests exist to hold: nothing about a particular congregation may reach the
app. A Sunday morning service and a Wednesday evening one have different shapes, and the
only place that difference is allowed to live is a file in the operator's own config
directory. The most important tests here are the dull ones — that an install which has
described nothing behaves exactly as it did before, and that a profile file is never
overwritten.
"""

import datetime
import json
import os

from stt.phase_profiles import (
    BASE_FILENAME,
    DEFAULT_PROFILE,
    Profile,
    apply_to_profile,
    list_profiles,
    load_profile,
    merge_config,
    parse_profile,
    profile_filename,
    profile_path,
    seed_missing,
    select_profile,
    slugify_profile,
)

SUNDAY_MORNING = datetime.datetime(2026, 8, 30, 9, 54)     # the archived morning service
WEDNESDAY_EVENING = datetime.datetime(2026, 8, 26, 18, 50)  # the archived midweek service

SCHEDULE = [
    {"profile": "sunday-morning", "weekdays": [6], "from": "08:00", "to": "13:00"},
    {"profile": "midweek", "weekdays": [2], "from": "17:00", "to": "22:00"},
]

RULES = {"phases": [
    {"name": "Sermon", "number": True, "match": {"kind": "S", "min_minutes": 8}},
    {"name": "Closing", "match": {"kind": "S", "max_minutes": 6, "is_last_named": True}},
    {"name": "Speaking", "match": {"kind": "S"}},
]}


def write(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return str(path)


class TestSlug:
    def test_a_name_becomes_a_slug(self):
        assert slugify_profile("Sunday Morning") == "sunday-morning"

    def test_it_refuses_to_escape_the_directory(self):
        # The name arrives from an HTTP parameter.
        assert slugify_profile("../../etc/passwd") == "etcpasswd"
        assert slugify_profile("/absolute") == "absolute"
        assert slugify_profile("..") == ""

    def test_it_is_bounded(self):
        assert len(slugify_profile("x" * 200)) == 40

    def test_nothing_useful_reads_as_empty(self):
        assert slugify_profile("") == "" and slugify_profile("///") == ""

    def test_a_path_never_leaves_the_config_directory(self, tmp_path):
        for name in ("../../etc/passwd", "/absolute", "..", "a/b"):
            got = os.path.realpath(profile_path(str(tmp_path), name))
            assert got.startswith(os.path.realpath(str(tmp_path)) + os.sep)


class TestFilenames:
    def test_the_default_profile_keeps_the_existing_filename(self):
        # An install that already has a rule file must not need to move it.
        assert profile_filename(DEFAULT_PROFILE) == BASE_FILENAME
        assert profile_filename("") == BASE_FILENAME

    def test_a_named_profile_gets_its_own_file(self):
        assert profile_filename("Sunday Morning") == "service_phases.sunday-morning.json"

    def test_profiles_are_listed_from_disk(self, tmp_path):
        write(tmp_path / BASE_FILENAME, RULES)
        write(tmp_path / "service_phases.midweek.json", RULES)
        write(tmp_path / "unrelated.json", {})
        assert sorted(list_profiles(str(tmp_path))) == [DEFAULT_PROFILE, "midweek"]

    def test_a_missing_directory_lists_nothing(self, tmp_path):
        assert list_profiles(str(tmp_path / "nope")) == []


class TestSelectProfile:
    def test_the_two_archived_services_pick_different_profiles(self):
        # 10am Sunday and 7pm Wednesday are the operator's own example.
        assert select_profile(SCHEDULE, SUNDAY_MORNING) == "sunday-morning"
        assert select_profile(SCHEDULE, WEDNESDAY_EVENING) == "midweek"

    def test_a_service_outside_every_window_falls_back(self):
        assert select_profile(SCHEDULE, datetime.datetime(2026, 8, 30, 22, 0)) == "default"

    def test_the_right_day_at_the_wrong_hour_does_not_match(self):
        assert select_profile(SCHEDULE, datetime.datetime(2026, 8, 26, 9, 0)) == "default"

    def test_first_match_wins(self):
        schedule = [{"profile": "first", "weekdays": [6]},
                    {"profile": "second", "weekdays": [6]}]
        assert select_profile(schedule, SUNDAY_MORNING) == "first"

    def test_an_override_beats_the_schedule(self):
        # The one time a person disagrees with the calendar is the time they are right.
        assert select_profile(SCHEDULE, SUNDAY_MORNING, override="Christmas") == "christmas"

    def test_an_empty_schedule_uses_the_default(self):
        assert select_profile([], SUNDAY_MORNING) == "default"

    def test_a_window_with_no_weekday_matches_any_day(self):
        schedule = [{"profile": "evening", "from": "17:00", "to": "22:00"}]
        assert select_profile(schedule, WEDNESDAY_EVENING) == "evening"

    def test_a_malformed_entry_costs_only_itself(self):
        schedule = [{"profile": "broken", "weekdays": ["sunday"]},
                    {"profile": "midweek", "weekdays": [2]}]
        assert select_profile(schedule, WEDNESDAY_EVENING) == "midweek"

    def test_a_malformed_window_matches_nothing_rather_than_everything(self):
        schedule = [{"profile": "broken", "from": "25:99"}]
        assert select_profile(schedule, SUNDAY_MORNING) == "default"

    def test_an_entry_with_no_name_is_skipped(self):
        assert select_profile([{"weekdays": [6]}], SUNDAY_MORNING) == "default"


class TestLoadProfile:
    def test_a_named_profile_is_read(self, tmp_path):
        write(tmp_path / "service_phases.midweek.json",
              dict(RULES, service={"closing_max_minutes": 15},
                   limits={"Sermon": {"max": 2}}))
        got = load_profile(str(tmp_path), "midweek", "config/service_phases.default.json")
        assert got.name == "midweek"
        assert got.service == {"closing_max_minutes": 15}
        assert got.limits == {"Sermon": {"max": 2}}
        assert got.rules

    def test_a_missing_profile_falls_back_to_the_base_file(self, tmp_path):
        write(tmp_path / BASE_FILENAME, RULES)
        got = load_profile(str(tmp_path), "midweek", "config/service_phases.default.json")
        assert got.source.endswith(BASE_FILENAME)
        assert got.rules

    def test_nothing_on_disk_falls_back_to_the_shipped_template(self, tmp_path):
        got = load_profile(str(tmp_path), "midweek", "config/service_phases.default.json")
        assert got.source.endswith("service_phases.default.json")
        assert got.rules
        # And the shipped template says nothing about any church's services.
        assert got.service == {} and got.limits == {}

    def test_a_corrupt_profile_falls_through_rather_than_raising(self, tmp_path):
        (tmp_path / "service_phases.midweek.json").write_text("{ not json",
                                                             encoding="utf-8")
        write(tmp_path / BASE_FILENAME, RULES)
        assert load_profile(str(tmp_path), "midweek",
                            "config/service_phases.default.json").rules

    def test_the_source_says_which_file_answered(self, tmp_path):
        # An operator whose profile is missing gets the shipped rules; without this
        # nothing could tell them so.
        path = write(tmp_path / "service_phases.midweek.json", RULES)
        assert load_profile(str(tmp_path), "midweek", "x").source == path

    def test_comment_keys_are_not_settings(self, tmp_path):
        write(tmp_path / "service_phases.midweek.json",
              dict(RULES, service={"_comment": "hi", "closing_max_minutes": 15}))
        got = load_profile(str(tmp_path), "midweek", "x")
        assert got.service == {"closing_max_minutes": 15}


class TestMergeConfig:
    def test_a_profile_that_says_nothing_changes_nothing(self):
        base = {"bin_seconds": 60, "dominance": 0.35}
        assert merge_config(base, Profile("default")) == base

    def test_no_profile_at_all_changes_nothing(self):
        base = {"bin_seconds": 60}
        assert merge_config(base, None) == base

    def test_the_profile_overrides_the_service_numbers(self):
        base = {"bin_seconds": 60, "closing_max_minutes": 6}
        got = merge_config(base, parse_profile(dict(RULES, service={"closing_max_minutes": 15}),
                                               "midweek"))
        assert got["closing_max_minutes"] == 15

    def test_the_machines_own_settings_are_left_alone(self):
        # Bin width and dominance describe this machine's audio, not Sunday evening.
        base = {"bin_seconds": 60, "dominance": 0.35}
        got = merge_config(base, parse_profile(dict(RULES, service={"sermon_min_minutes": 12}),
                                               "midweek"))
        assert got["bin_seconds"] == 60 and got["dominance"] == 0.35

    def test_limits_arrive_under_their_own_key(self):
        got = merge_config({}, parse_profile(dict(RULES, limits={"Sermon": {"max": 2}}),
                                             "sunday-morning"))
        assert got["limits"] == {"Sermon": {"max": 2}}


class TestSeedMissing:
    def test_it_writes_a_profile_that_does_not_exist(self, tmp_path):
        template = write(tmp_path / "template.json", RULES)
        path = seed_missing(str(tmp_path / "cfg"), "midweek", template)
        assert path and os.path.exists(path)

    def test_it_never_overwrites(self, tmp_path):
        # The rule the word-highlighting config learned the hard way: an operator's tuning
        # must not be silently replaced by defaults.
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        mine = write(cfg / "service_phases.midweek.json", {"phases": [], "mine": True})
        template = write(tmp_path / "template.json", RULES)
        assert seed_missing(str(cfg), "midweek", template) is None
        with open(mine, encoding="utf-8") as handle:
            assert json.load(handle)["mine"] is True

    def test_a_missing_template_is_reported_rather_than_raised(self, tmp_path):
        assert seed_missing(str(tmp_path), "midweek", str(tmp_path / "nope.json")) is None


class TestApplyToProfile:
    def proposal(self, value, actionable=True):
        return {"suggested": value, "actionable": actionable}

    def test_a_threshold_lands_in_the_rule_that_owns_it(self):
        # The dead end this closes: proposing sermon_min_minutes into the config block,
        # where the detector never reads it.
        got = apply_to_profile(RULES, {"sermon_min_minutes": self.proposal(12)},
                               ["sermon_min_minutes"])
        sermon = next(p for p in got["phases"] if p["name"] == "Sermon")
        assert sermon["match"]["min_minutes"] == 12

    def test_a_closing_length_lands_in_the_closing_rule(self):
        got = apply_to_profile(RULES, {"closing_max_minutes": self.proposal(15)},
                               ["closing_max_minutes"])
        closing = next(p for p in got["phases"] if p["name"] == "Closing")
        assert closing["match"]["max_minutes"] == 15

    def test_a_cap_lands_in_limits(self):
        got = apply_to_profile(RULES, {"max_sermons": self.proposal(2)}, ["max_sermons"])
        assert got["limits"] == {"Sermon": {"max": 2}}

    def test_anything_else_lands_in_service(self):
        got = apply_to_profile(RULES, {"service_length_minutes": self.proposal(120)},
                               ["service_length_minutes"])
        assert got["service"] == {"service_length_minutes": 120}

    def test_a_proposal_not_accepted_is_ignored(self):
        got = apply_to_profile(RULES, {"sermon_min_minutes": self.proposal(12)}, [])
        sermon = next(p for p in got["phases"] if p["name"] == "Sermon")
        assert sermon["match"]["min_minutes"] == 8

    def test_a_proposal_the_evidence_does_not_support_is_ignored(self):
        got = apply_to_profile(RULES,
                               {"sermon_min_minutes": self.proposal(12, actionable=False)},
                               ["sermon_min_minutes"])
        sermon = next(p for p in got["phases"] if p["name"] == "Sermon")
        assert sermon["match"]["min_minutes"] == 8

    def test_it_does_not_mutate_what_it_was_given(self):
        original = json.loads(json.dumps(RULES))
        apply_to_profile(RULES, {"sermon_min_minutes": self.proposal(12)},
                         ["sermon_min_minutes"])
        assert RULES == original

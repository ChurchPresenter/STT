"""The output delay must measure age in the zone rows were stamped in.

Rows are written with datetime.now(configured_timezone); the delay's age check and
the staged-row backdating both have to be taken in that same zone. While
get_configured_timezone() always returned the system zone the two agreed by
accident. Once the timezone became configurable, a naive datetime.now() comparison
put every row either instantly live or permanently staged — off by exactly the
offset between the configured zone and the machine's.
"""

import datetime

import pytest

from conftest import extract_definitions


def backdate_ns(tz, delay_seconds=7):
    captured = {}

    class _Conn:
        def execute(self, sql, params=()):
            captured.setdefault("calls", []).append((sql, params))
            return self
        def fetchone(self): return None
        def commit(self): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    ns = extract_definitions(
        "speech_to_text.py", ["_backdate_staged_rows"],
        extra_globals={
            "config": {"corrections": {"output_delay": {"delay_seconds": delay_seconds}}},
            "configured_timezone": tz,
            "datetime": datetime.datetime,
            "timedelta": datetime.timedelta,
            "sqlite3": type("S", (), {"connect": staticmethod(lambda *a, **k: _Conn())})(),
            "_db_lock": __import__("threading").Lock(),
            "_open_db_writer": lambda: _Conn(),
            "_invalidate_entries_cache": lambda: None,
            "print": lambda *a, **k: None,
        })
    return ns, captured


class TestBackdateUsesConfiguredZone:
    KYIV = datetime.timezone(datetime.timedelta(hours=3), "Kyiv")
    UTC = datetime.timezone.utc

    def _stamp(self, tz, delay=7):
        ns, cap = backdate_ns(tz, delay)
        ns["_backdate_staged_rows"](seg_id=1)
        for sql, params in cap.get("calls", []):
            if "UPDATE" in sql.upper() and params:
                return params[0]
        raise AssertionError("no UPDATE was issued — the stub no longer matches the code")

    def test_the_backdated_stamp_is_in_the_configured_zone(self):
        # A row stamped 'now' in the configured zone must read as older than the
        # delay once backdated — that is the whole point of backdating.
        stamp = self._stamp(self.KYIV, delay=7)
        written_now = datetime.datetime.now(self.KYIV).replace(tzinfo=None)
        age = (written_now - datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")).total_seconds()
        assert age >= 7, f"backdated stamp is only {age:.0f}s old; it would stay staged"

    def test_it_tracks_the_zone_rather_than_the_machine(self):
        # Two very different zones must both produce a stamp that is past the window
        # in their own zone. A naive now() can only satisfy one of them.
        for tz in (self.KYIV, self.UTC, datetime.timezone(datetime.timedelta(hours=-8))):
            stamp = self._stamp(tz, delay=7)
            now_there = datetime.datetime.now(tz).replace(tzinfo=None)
            age = (now_there - datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")).total_seconds()
            assert age >= 7, f"zone {tz} produced a {age:.0f}s-old stamp"

    def test_a_longer_delay_backdates_further(self):
        short = self._stamp(self.KYIV, delay=2)
        long = self._stamp(self.KYIV, delay=30)
        assert long < short, "a 30s delay must backdate further than a 2s delay"


    def test_the_approve_all_cutoff_is_also_zone_correct(self):
        """Approving everything selects rows by `timestamp > cutoff`.

        That cutoff is compared against stamps written in the configured zone, so it
        has to be taken there too — otherwise approve-all either matches every row in
        the session or none of them.
        """
        ns, cap = backdate_ns(self.KYIV, delay_seconds=7)
        ns["_backdate_staged_rows"](seg_id=None)
        cutoff = None
        for sql, params in cap.get("calls", []):
            if "timestamp >" in sql and len(params) == 2:
                cutoff = params[1]
        assert cutoff, "no approve-all UPDATE issued"
        now_there = datetime.datetime.now(self.KYIV).replace(tzinfo=None)
        age = (now_there - datetime.datetime.strptime(cutoff, "%Y-%m-%d %H:%M:%S")).total_seconds()
        # The cutoff should sit one delay-window back from "now" in that zone.
        assert 5 <= age <= 9, f"cutoff is {age:.0f}s back, expected ~7"

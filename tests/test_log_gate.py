"""Tests for stt.log_gate.ChangeGate."""

import threading

from stt.log_gate import ChangeGate


def test_first_value_for_a_key_is_always_a_change():
    gate = ChangeGate()
    assert gate.changed("devices", 0) is True


def test_repeat_of_the_same_value_is_suppressed():
    gate = ChangeGate()
    gate.changed("devices", 0)
    assert [gate.changed("devices", 0) for _ in range(10)] == [False] * 10


def test_changed_value_reports_and_rearms():
    gate = ChangeGate()
    assert gate.changed("devices", 0) is True
    assert gate.changed("devices", 1) is True
    assert gate.changed("devices", 1) is False
    assert gate.changed("devices", 0) is True


def test_falsy_and_none_values_are_tracked_not_treated_as_absent():
    """0 must not be confused with "no value seen yet" — the device count that
    motivated this module is 0 in the steady state."""
    gate = ChangeGate()
    assert gate.changed("count", 0) is True
    assert gate.changed("count", 0) is False
    assert gate.changed("other", None) is True
    assert gate.changed("other", None) is False


def test_keys_are_independent():
    gate = ChangeGate()
    gate.changed("a", 1)
    assert gate.changed("b", 1) is True
    assert gate.changed("a", 1) is False


def test_gates_do_not_share_state():
    first, second = ChangeGate(), ChangeGate()
    first.changed("k", "v")
    assert second.changed("k", "v") is True


def test_reset_clears_one_key_or_all():
    gate = ChangeGate()
    gate.changed("a", 1)
    gate.changed("b", 2)
    gate.reset("a")
    assert gate.changed("a", 1) is True
    assert gate.changed("b", 2) is False
    gate.reset()
    assert gate.changed("b", 2) is True


def test_reset_of_unknown_key_is_a_no_op():
    gate = ChangeGate()
    gate.changed("a", 1)
    gate.reset("nope")
    assert gate.changed("a", 1) is False


def test_concurrent_callers_see_exactly_one_change_per_value():
    """Flask serves this from multiple threads; a value must not slip through
    the gate twice."""
    gate = ChangeGate()
    changes = []
    changes_lock = threading.Lock()
    start = threading.Event()

    def worker():
        start.wait()
        if gate.changed("devices", 0):
            with changes_lock:
                changes.append(1)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert sum(changes) == 1

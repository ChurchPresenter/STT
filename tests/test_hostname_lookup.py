"""Unit tests for stt.hostname_lookup — the non-blocking reverse-DNS cache."""

from stt.hostname_lookup import HostnameCache


class FakeClock:
    """Hand-advanced clock so TTL behaviour is tested without waiting."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_get_returns_none_first_then_resolved_name():
    cache = HostnameCache(resolver=lambda ip: "display.local")
    # First look at an unknown address never blocks on DNS.
    assert cache.get("192.168.2.62") is None
    assert cache.wait_idle()
    assert cache.get("192.168.2.62") == "display.local"


def test_resolver_is_called_once_per_address():
    calls = []

    def resolver(ip):
        calls.append(ip)
        return "host-" + ip

    cache = HostnameCache(resolver=resolver)
    cache.get("10.0.0.1")
    assert cache.wait_idle()
    for _ in range(5):
        assert cache.get("10.0.0.1") == "host-10.0.0.1"
    assert calls == ["10.0.0.1"]


def test_failed_lookup_is_cached_negatively():
    calls = []

    def resolver(ip):
        calls.append(ip)
        raise OSError("no PTR record")

    cache = HostnameCache(resolver=resolver, negative_ttl_seconds=600)
    cache.get("10.0.0.9")
    assert cache.wait_idle()
    assert cache.get("10.0.0.9") is None
    assert cache.get("10.0.0.9") is None
    assert calls == ["10.0.0.9"]


def test_negative_entry_is_retried_after_its_ttl():
    calls = []

    def resolver(ip):
        calls.append(ip)
        if len(calls) == 1:
            raise OSError("no PTR record")
        return "later.local"

    clock = FakeClock()
    cache = HostnameCache(resolver=resolver, negative_ttl_seconds=60, clock=clock)
    cache.get("10.0.0.5")
    assert cache.wait_idle()
    # Still inside the negative TTL: no second lookup.
    clock.advance(30)
    assert cache.get("10.0.0.5") is None
    assert calls == ["10.0.0.5"]
    # Past it: the address is tried again.
    clock.advance(100)
    cache.get("10.0.0.5")
    assert cache.wait_idle()
    assert cache.get("10.0.0.5") == "later.local"


def test_expired_positive_entry_keeps_showing_the_stale_name():
    clock = FakeClock()
    cache = HostnameCache(resolver=lambda ip: "slow.local", ttl_seconds=60, clock=clock)
    cache.prime("10.0.0.7", "old.local")
    # Past the TTL the name is still returned rather than blanking out, while
    # the refresh runs in the background.
    clock.advance(1000)
    assert cache.get("10.0.0.7") == "old.local"
    assert cache.wait_idle()
    assert cache.get("10.0.0.7") == "slow.local"


def test_hostname_is_normalised():
    cache = HostnameCache(resolver=lambda ip: "  Display.local.  ")
    cache.get("10.0.0.2")
    assert cache.wait_idle()
    assert cache.get("10.0.0.2") == "Display.local"


def test_empty_hostname_is_treated_as_unresolved():
    cache = HostnameCache(resolver=lambda ip: "")
    assert cache.resolve_now("10.0.0.3") is None


def test_get_ignores_missing_addresses():
    calls = []
    cache = HostnameCache(resolver=lambda ip: calls.append(ip) or "x")
    assert cache.get(None) is None
    assert cache.get("") is None
    assert calls == []


def test_get_many_maps_every_address():
    cache = HostnameCache(resolver=lambda ip: "host-" + ip)
    cache.get_many(["10.0.0.1", "10.0.0.2", None, ""])
    assert cache.wait_idle()
    names = cache.get_many(["10.0.0.1", "10.0.0.2"])
    assert names == {"10.0.0.1": "host-10.0.0.1", "10.0.0.2": "host-10.0.0.2"}


def test_cache_is_capped():
    clock = FakeClock()
    cache = HostnameCache(resolver=lambda ip: "host", max_entries=10, clock=clock)
    for i in range(50):
        cache.prime(f"10.0.0.{i}", "host")
        clock.advance(1)
    assert len(cache._entries) <= 10
    # The most recently primed entries are the ones kept.
    assert cache.get("10.0.0.49") == "host"

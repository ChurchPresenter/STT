"""LocalAgreement live-hypothesis stabilization (stt/hypothesis_buffer.py)."""

from stt.hypothesis_buffer import LocalAgreementBuffer


class TestStabilize:
    def test_empty_first_pass_shows_nothing(self):
        b = LocalAgreementBuffer()
        assert b.stabilize("") == ""
        # First non-empty pass has no predecessor to agree with -> nothing yet.
        assert b.stabilize("the quick brown") == ""

    def test_agreement_confirms_stable_prefix(self):
        b = LocalAgreementBuffer()
        b.stabilize("the quick brown")            # seed
        # Second pass agrees on the prefix that both share.
        assert b.stabilize("the quick brown fox") == "the quick brown"
        # Third pass confirms "fox" too (present in both prev and now).
        assert b.stabilize("the quick brown fox jumps") == "the quick brown fox"

    def test_identical_consecutive_confirms_everything(self):
        b = LocalAgreementBuffer()
        b.stabilize("hello world")
        assert b.stabilize("hello world") == "hello world"

    def test_revised_tail_word_is_never_shown(self):
        # The classic jitter case: a word gets revised before it stabilizes.
        b = LocalAgreementBuffer()
        b.stabilize("the quick brown")
        assert b.stabilize("the quick brown fox") == "the quick brown"   # 'fox' held back
        # Next pass revises 'fox' -> 'fax': it was never shown, so no rewrite.
        assert b.stabilize("the quick brown fax jumps") == "the quick brown"
        # Once 'fax jumps' repeats, it confirms.
        assert b.stabilize("the quick brown fax jumps over") == "the quick brown fax jumps"

    def test_committed_prefix_is_never_revised_on_divergence(self):
        b = LocalAgreementBuffer()
        b.stabilize("i love this")
        b.stabilize("i love this song")   # confirms "i love this"
        assert b.committed_text == "i love this"
        # A hypothesis that diverges from the shown prefix does not rewrite it.
        assert b.stabilize("i loved that song entirely") == "i love this"

    def test_reset_starts_fresh(self):
        b = LocalAgreementBuffer()
        b.stabilize("alpha beta")
        b.stabilize("alpha beta gamma")   # confirms "alpha beta"
        assert b.committed_text == "alpha beta"
        b.reset()
        assert b.committed_text == ""
        assert b.stabilize("new phrase here") == ""   # no predecessor after reset

    def test_none_text_is_safe(self):
        b = LocalAgreementBuffer()
        assert b.stabilize(None) == ""

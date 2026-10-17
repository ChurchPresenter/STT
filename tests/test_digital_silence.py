"""A blocked microphone delivers exact zeros; a quiet room never does."""

import array

import pytest

from stt.digital_silence import DEFAULT_THRESHOLD_SECONDS, DigitalSilence, is_digital_silence


def pcm(*samples):
    """16-bit signed PCM bytes, the format the capture delivers."""
    return array.array("h", samples).tobytes()


ZEROS = pcm(*([0] * 1600))
#: Room tone: a near-silent analogue input still moves the low bits.
ROOM_TONE = pcm(*([0] * 800 + [1, -1, 2, 0, -2] * 160))


class TestIsDigitalSilence:
    def test_all_zero_samples_are_silence(self):
        assert is_digital_silence(ZEROS)

    def test_one_quiet_sample_is_not(self):
        assert not is_digital_silence(pcm(*([0] * 1599 + [1])))

    def test_a_negative_sample_is_not(self):
        """-1 is all ones in two's complement: the byte test must not miss it."""
        assert not is_digital_silence(pcm(0, 0, -1, 0))

    def test_room_tone_is_not(self):
        assert not is_digital_silence(ROOM_TONE)

    def test_an_empty_buffer_is_no_evidence(self):
        assert not is_digital_silence(b"")


class TestDigitalSilence:
    def test_zeros_are_not_reported_before_the_threshold(self):
        d = DigitalSilence()
        assert not d.feed(ZEROS, now=0.0)
        assert not d.feed(ZEROS, now=DEFAULT_THRESHOLD_SECONDS - 0.1)

    def test_zeros_are_reported_at_the_threshold(self):
        d = DigitalSilence()
        d.feed(ZEROS, now=0.0)
        assert d.feed(ZEROS, now=DEFAULT_THRESHOLD_SECONDS)

    def test_a_quiet_room_is_never_reported(self):
        d = DigitalSilence()
        assert not any(d.feed(ROOM_TONE, now=float(t)) for t in range(60))

    def test_signal_arriving_clears_it_on_the_next_chunk(self):
        d = DigitalSilence()
        d.feed(ZEROS, now=0.0)
        assert d.feed(ZEROS, now=20.0)
        assert not d.feed(ROOM_TONE, now=20.5)

    def test_a_gated_feed_between_speakers_is_never_reported(self):
        """A mixer's USB/Dante feed sends exact zeros while the channel is gated
        or muted. Once the input has carried real signal, it is known to work."""
        d = DigitalSilence()
        d.feed(ROOM_TONE, now=0.0)
        assert not any(d.feed(ZEROS, now=float(t)) for t in range(1, 600))
        assert not d.silent(now=600.0)

    def test_once_cleared_it_stays_cleared(self):
        """Started early on a gated feed: warned until the first sound, then never again."""
        d = DigitalSilence()
        d.feed(ZEROS, now=0.0)
        assert d.feed(ZEROS, now=30.0)
        d.feed(ROOM_TONE, now=31.0)
        assert not d.feed(ZEROS, now=100.0)

    def test_a_new_session_starts_unproven(self):
        """A permission revoked between services must still be caught."""
        old = DigitalSilence()
        old.feed(ROOM_TONE, now=0.0)
        new = DigitalSilence()
        new.feed(ZEROS, now=0.0)
        assert new.feed(ZEROS, now=DEFAULT_THRESHOLD_SECONDS)

    def test_an_empty_chunk_changes_nothing(self):
        d = DigitalSilence()
        d.feed(ZEROS, now=0.0)
        assert d.feed(b"", now=12.0)
        assert not DigitalSilence().feed(b"", now=12.0)

    @pytest.mark.parametrize("threshold", [0.0, 3.0])
    def test_the_threshold_is_configurable(self, threshold):
        d = DigitalSilence(threshold_seconds=threshold)
        d.feed(ZEROS, now=0.0)
        assert d.silent(now=threshold)

"""Safe numeric coercion (stt/coercion.py)."""

from stt.coercion import coerce_int


class TestCoerceInt:
    def test_valid_int_passthrough(self):
        assert coerce_int(5, 0) == 5

    def test_numeric_string(self):
        assert coerce_int("7", 0) == 7

    def test_float_and_float_string(self):
        assert coerce_int(3.9, 0) == 3          # int() truncates
        assert coerce_int("4", 0) == 4
        # a float-looking string is NOT a valid int literal → default
        assert coerce_int("4.5", 99) == 99

    def test_bad_values_fall_back_to_default(self):
        assert coerce_int(None, 15) == 15
        assert coerce_int("abc", 15) == 15
        assert coerce_int("", 15) == 15
        assert coerce_int({}, 15) == 15
        assert coerce_int([], 15) == 15

    def test_clamp_low_and_high(self):
        assert coerce_int(0, 15, lo=3, hi=120) == 3      # below lo
        assert coerce_int(500, 15, lo=3, hi=120) == 120  # above hi
        assert coerce_int(50, 15, lo=3, hi=120) == 50    # in range

    def test_bounds_none_is_passthrough(self):
        assert coerce_int(-1000, 0) == -1000
        assert coerce_int(1000000, 0) == 1000000

    def test_default_is_not_re_clamped(self):
        # A bad value returns the default verbatim even if it sits outside lo/hi
        # (callers are expected to pass an in-range default).
        assert coerce_int("abc", 15, lo=100, hi=200) == 15

    def test_only_one_bound(self):
        assert coerce_int(1, 0, lo=5) == 5
        assert coerce_int(9, 0, hi=5) == 5

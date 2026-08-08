"""Safe numeric coercion (stt/coercion.py)."""

from stt.coercion import coerce_bool, coerce_float, coerce_int


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


class TestCoerceFloat:
    def test_valid_and_string(self):
        assert coerce_float(1.5, 0.0) == 1.5
        assert coerce_float("2.5", 0.0) == 2.5
        assert coerce_float(3, 0.0) == 3.0

    def test_bad_values_fall_back(self):
        assert coerce_float(None, 1.0) == 1.0
        assert coerce_float("abc", 1.0) == 1.0
        assert coerce_float({}, 1.0) == 1.0

    def test_clamp(self):
        assert coerce_float(0.05, 1.0, lo=0.1, hi=3.0) == 0.1
        assert coerce_float(5.0, 1.0, lo=0.1, hi=3.0) == 3.0
        assert coerce_float(1.5, 1.0, lo=0.1, hi=3.0) == 1.5

    def test_default_not_re_clamped(self):
        assert coerce_float("x", 1.0, lo=10.0, hi=20.0) == 1.0


class TestCoerceBool:
    def test_real_bools_pass_through(self):
        assert coerce_bool(True) is True
        assert coerce_bool(False) is False

    def test_on_spellings(self):
        for value in ("1", "true", "TRUE", " Yes ", "on", "y", "t"):
            assert coerce_bool(value) is True, value

    def test_off_spellings(self):
        for value in ("0", "false", "No", "OFF", "n", "f", ""):
            assert coerce_bool(value, default=True) is False, value

    def test_the_string_false_is_not_truthy(self):
        # The whole point: bool("false") is True in Python, and a form or query
        # string can only ever send text.
        assert coerce_bool("false") is False

    def test_numbers_follow_truthiness(self):
        assert coerce_bool(1) is True
        assert coerce_bool(0) is False
        assert coerce_bool(2.5) is True

    def test_unrecognised_returns_the_default_rather_than_guessing(self):
        assert coerce_bool("maybe", default=True) is True
        assert coerce_bool("maybe", default=False) is False
        assert coerce_bool(None, default=True) is True
        assert coerce_bool({}, default=False) is False

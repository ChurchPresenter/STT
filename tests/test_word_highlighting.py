"""The shipped Bible-book highlighting config (config/word_highlighting.default.json).

test_formatting.py covers the matching *engine*; this covers the *patterns we ship*.
Every phrase here is a real utterance from the 2026-08-02 morning/evening services,
where Russian epistle names, bare English book names, and fourteen patterns that had
lost the backslash on ``\\s`` all failed to highlight.
"""

import json
import re
from pathlib import Path

import pytest

from stt.formatting import apply_word_highlighting_server

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "word_highlighting.default.json"


@pytest.fixture(scope="module")
def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


TAG = re.compile(r'<span style="color: #[0-9a-f]{6};">|</span>')


def highlighted(text, config):
    """The coloured substrings, in plain-text terms.

    Spans nest — an entry for "Holy Spirit" wraps the phrase, then a later entry
    for "Holy" colours a word inside it — so this walks the tags with a stack
    rather than pattern-matching span pairs.
    """
    out = apply_word_highlighting_server(text, config)
    plain, spans, open_at, pos = [], [], [], 0
    for tag in TAG.finditer(out):
        plain.append(out[pos:tag.start()])
        pos = tag.end()
        length = sum(len(p) for p in plain)
        if tag.group().startswith("</"):
            spans.append((open_at.pop(), length))
        else:
            open_at.append(length)
    plain.append(out[pos:])
    flat = "".join(plain)
    return [flat[a:b] for a, b in spans]


def is_highlighted(text, config, phrase):
    """Does some span cover `phrase` exactly?"""
    return any(span.lower() == phrase.lower() for span in highlighted(text, config))


class TestRussianEpistles:
    """``Римлян[ам]?`` allowed one trailing letter, so the dative plural the
    preacher actually says — Римлян**ам** — never matched."""

    @pytest.mark.parametrize(
        "text,book",
        [
            ("Римлянам 8.15.", "Римлянам"),
            ("Римлянам 1.28 написано", "Римлянам"),
            ("Ефесянам 4.32 сказано", "Ефесянам"),
            ("И в Галатам апостол Павел пишет", "Галатам"),
            ("Филиппийцам 4.13", "Филиппийцам"),
            ("Колоссянам 3.13", "Колоссянам"),
            ("Эфесянам 2.8", "Эфесянам"),
            ("1 Фессалоникийцам 5.16", "1 Фессалоникийцам"),
            ("2 Коринфянам 5.17", "2 Коринфянам"),
            ("в послании к Коринфянам", "Коринфянам"),
            ("Евреям 8.12", "Евреям"),
        ],
    )
    def test_case_forms_highlight(self, config, text, book):
        assert is_highlighted(text, config, book)

    def test_roman_the_person_is_not_a_book(self, config):
        assert not highlighted("римлянин пришёл в город", config)


class TestEnglishBareBookNames:
    """Epistles existed only as numbered alternatives, so a book named without a
    numeral — 'the epistle to the Corinthians' — never matched."""

    @pytest.mark.parametrize(
        "text,book",
        [
            ("The Apostle Paul says in the epistle to the Corinthians", "Corinthians"),
            ("2 Corinthians 5:17", "2 Corinthians"),
            ("Romans 8:15", "Romans"),
            ("Samuel said to the people", "Samuel"),
            ("we read in Chronicles", "Chronicles"),
            ("Thessalonians reminds us", "Thessalonians"),
            ("1 Kings 18", "1 Kings"),
            ("the book of Kings", "book of Kings"),
        ],
    )
    def test_named_books_highlight(self, config, text, book):
        assert is_highlighted(text, config, book)

    def test_kings_as_a_common_noun_is_not_a_book(self, config):
        # 2026-08-02 #2223: "There is already war with them, kings are attacking
        # them" — a bare `Kings` alternative would wrongly colour this.
        assert not highlighted("kings are attacking them, and God protects them", config)


class TestMultiWordPhrases:
    """The ``\\s`` -> literal ``s`` typo, plus entry order: a generic single-word
    entry running first wraps the word in a span the phrase can no longer cross."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "Old Testament",
            "New Testament",
            "Holy Spirit",
            "Holy Ghost",
            "Second Coming",
            "Judgment Day",
            "Virgin Mary",
            "King of Kings",
            "Ветхий Завет",
            "Новый Завет",
            "Святой Дух",
            "Святое Писание",
            "Агнец Божий",
            "Второе Пришествие",
        ],
    )
    def test_whole_phrase_is_covered(self, config, phrase):
        assert is_highlighted(phrase, config, phrase)

    @pytest.mark.parametrize(
        "text,phrase",
        [
            ("Евангелие от Матфея, 7 глава", "от Матфея"),
            ("Евангелие от Иоанна, 3 глава", "от Иоанна"),
            ("Евангелие от Луки", "от Луки"),
            ("1 Петра 3", "1 Петра"),
            ("2 Иоанна 1", "2 Иоанна"),
            ("1 Коринфянам 13", "1 Коринфянам"),
        ],
    )
    def test_russian_qualified_books(self, config, text, phrase):
        assert is_highlighted(text, config, phrase)


class TestShippedConfigIsSound:
    def test_every_pattern_compiles(self, config):
        for entry in config["words"]:
            re.compile(entry["word"])

    def test_no_unescaped_s_star(self, config):
        """`Olds*Testament` — a lost backslash silently demands a literal 's'."""
        offenders = [e["word"] for e in config["words"] if re.search(r"(?<!\\)s\*", e["word"])]
        assert offenders == []

    def test_no_truncated_cyrillic_ending_classes(self, config):
        """`Римлян[ам]?` matches one letter where the case ending needs two."""
        offenders = [
            e["word"]
            for e in config["words"]
            if re.search(r"(?:Римлян|Коринфян|Колоссян|Ефесян|Галат|Филиппийц|Фессалоникийц)\[", e["word"])
        ]
        assert offenders == []

    def test_enabled_with_no_colors_suppressed(self, config):
        assert config["enabled"] is True
        assert config.get("disabled_colors") == []

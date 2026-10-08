"""Generalisation: reduce precision while keeping the analytic value.

Masking a date removes the age or tenure signal that a WHS survey exists to
measure. For most quasi-identifiers the right response is to widen the value,
not delete it:

    "12/03/1985"        -> "1985"          year only
    "I am 34"           -> "30-39"         age band
    "18 years of service"-> "10+ years of service"
    "team of three"     -> "team of 1-10"

Generalisation is orthogonal to :class:`~pii_redact.types.MaskStyle`: a policy
can mask direct identifiers and generalise quasi-identifiers in the same pass.
It is applied inside the same right-to-left rewrite, so a generalised span is
never re-detected as a new PII candidate.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .types import Category

#: Band edges for age. Bands are 10 years wide below 65 and wider above, which
#: is where the re-identification benefit stops and the analytic loss starts.
_AGE_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 17, "under 18"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 74, "65-74"),
    (75, 120, "75+"),
)

#: Service bands. Kept wide: length of service is a strong correlator of
#: seniority and therefore of pay and of who is likely to be spoken about.
_TENURE_BANDS: tuple[tuple[int, str], ...] = (
    (2, "under 2 years"),
    (5, "2-4 years"),
    (10, "5-9 years"),
    (20, "10-19 years"),
    (30, "20-29 years"),
    (40, "30-39 years"),
)

#: Team-size bands. Very small groups are the identifying case.
_TEAM_BANDS: tuple[tuple[int, str], ...] = (
    (1, "1 person"),
    (5, "2-5 people"),
    (10, "6-10 people"),
    (20, "11-20 people"),
    (50, "21-50 people"),
    (10**9, "50+ people"),
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BAND_LABEL_RE = re.compile(
    r"^(?:under\s+\d+|\d+\+|\d+\s*-\s*\d+|\d+\+?\s+years?|\d+\+?\s+people?)$"
)


def is_banded(value: str) -> bool:
    """True when ``value`` is already a band produced by this module.

    Generalisers must be idempotent: a Delta table of redacted text will be
    read and redacted again by a later job, and ``generalise_age("25-34")``
    returning ``"25-25-34"`` would corrupt the distribution every pass. Bands
    are therefore recognised and passed through unchanged.
    """
    first = value.strip().split(",", 1)[0].split(" ", 1)[0] if value.strip() else ""
    return bool(_BAND_LABEL_RE.fullmatch(first.strip()))


#: Respondents write small numbers as words more often than as digits, so both
#: forms have to parse for generalisation to fire.
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}
_WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE)


def parse_number(value: str) -> float | None:
    """First number in ``value``, as digits or as an English word."""
    match = _NUMBER_RE.search(value)
    if match is not None:
        try:
            return float(match.group())
        except ValueError:
            return None
    match = _WORD_NUMBER_RE.search(value)
    if match is not None:
        return float(_WORD_NUMBERS[match.group(1).lower()])
    return None


def _is_bare_number(value: str) -> bool:
    """True when ``value`` *starts* with the number to be generalised.

    A generaliser handed a phrase will rewrite whatever number it finds inside.
    "the only one" became "1 person" that way: the word "one" was read as a
    count. Requiring the first token to be the number rejects such phrases while
    still allowing the trailing context that gives a value its meaning, such as
    "18 years of service".
    """
    tokens = value.strip().split()
    if not tokens:
        return False
    return parse_number(tokens[0]) is not None


def generalise_age(value: str) -> str:
    """``"34"`` -> ``"30-39"``; bands and unparseable values pass through."""
    if is_banded(value) or not _is_bare_number(value):
        return value
    number = parse_number(value)
    if number is None:
        return value
    age = int(number)
    for low, high, label in _AGE_BANDS:
        if low <= age <= high:
            return label
    return value


#: Matches the unit word the band label would otherwise duplicate, so
#: "18 years of service" does not become "10-19 years years of service".
_TRAILING_UNIT_RE = re.compile(r"^\s*\+?\s*(?:years?|yrs?|months?|people|staff|workers|employees|members)\b", re.IGNORECASE)


def _replace_number(value: str, number: float, band_label: str) -> str:
    """Swap the leading number for a band label, keeping the rest of the phrase.

    The trailing unit word is dropped when the band already implies it.
    """
    match = _NUMBER_RE.search(value)
    if match is not None:
        trailing = value[match.end() :]
    else:
        word = _WORD_NUMBER_RE.search(value)
        if word is None:
            return value
        trailing = value[word.end() :]
    if _TRAILING_UNIT_RE.match(trailing):
        trailing = ""
    return f"{band_label}{trailing}"


def generalise_tenure(value: str) -> str:
    """``"18 years of service"`` -> ``"10-19 years of service"``.

    A start year is already a band, not a duration: "since 2019" is left
    alone. Treating it as a duration would produce nonsense such as "40+ years".
    """
    if is_banded(value) or not _is_bare_number(value):
        return value
    number = parse_number(value)
    if number is None:
        return value
    if 1900 <= number <= 2099 and re.fullmatch(r"\s*(?:19|20)\d{2}\s*", value):
        return value.strip()
    years = int(number)
    band = next((label for ceiling, label in _TENURE_BANDS if years < ceiling), "40+ years")
    return _replace_number(value, years, band)


def generalise_team_size(value: str) -> str:
    """``"three"`` -> ``"2-5 people"``, or ``"12"`` -> ``"11-20 people"``."""
    if is_banded(value) or not _is_bare_number(value):
        return value
    number = parse_number(value)
    if number is None:
        return value
    count = int(number)
    band = next((label for ceiling, label in _TEAM_BANDS if count <= ceiling), "50+ people")
    return _replace_number(value, count, band)


def generalise_date(value: str) -> str:
    """``"12/03/1985"`` -> ``"1985"``.

    Year alone is usually enough for a WHS trend and removes the ability to
    compute an exact age, which is the identifier.
    """
    years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", value)
    if years:
        return years[0]
    # A two-digit year such as "12/03/85" carries no more identity than the year
    # does; return it unchanged rather than guessing a century.
    return value


def generalise_year(value: str) -> str:
    """Keep a bare year, dropping any day or month around it."""
    return generalise_date(value)


#: Category -> generaliser. Add here rather than branching at call sites.
GENERALISERS: dict[Category, Callable[[str], str]] = {
    Category.AGE: generalise_age,
    Category.TENURE: generalise_tenure,
    Category.TEAM_SIZE: generalise_team_size,
    Category.DATE_AU: generalise_date,
    Category.DATE_ISO: generalise_date,
    Category.DATE: generalise_date,
}


def generalise(category: Category, value: str) -> str:
    """Widen ``value`` for ``category``, or return it unchanged.

    Unknown categories and unparseable values pass through untouched: silently
    dropping text that cannot be generalised would be worse than leaving it.
    """
    handler = GENERALISERS.get(category)
    return handler(value) if handler is not None else value


__all__ = [
    "GENERALISERS",
    "generalise",
    "generalise_age",
    "generalise_date",
    "generalise_team_size",
    "generalise_tenure",
    "generalise_year",
    "is_banded",
    "parse_number",
]
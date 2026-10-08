"""Checksum validators for Australian (and generic) identifiers.

Every validator is pure, allocation-light and side-effect free so it can be
called from hot detection loops and from tests without fixtures.
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D")


def digits_only(value: str) -> str:
    """Strip every non-digit character."""
    return _DIGITS.sub("", value)


def _weighted_sum(value: str, weights: tuple[int, ...], reduce_one: bool = False) -> int:
    """Sum ``weights[i] * value[i]``, optionally reducing each digit by one.

    ``reduce_one`` matches the ABN convention, where every digit is decremented
    before weighting. TFN and ACN weight the raw digits instead.
    """
    total = 0
    for index, char in enumerate(value):
        digit = int(char)
        if reduce_one:
            digit -= 1
        total += digit * weights[index]
    return total


def complement10(total: int) -> int:
    """The mod-10 complement, used by ACN as its check digit."""
    return (10 - total % 10) % 10


# --- Tax File Number -------------------------------------------------------
# A 9-digit number issued by the ATO: eight identifier digits plus a check
# digit. Each digit is weighted by ``_TFN_WEIGHTS`` and the weighted sum must be
# divisible by 11.
_TFN_WEIGHTS = (1, 4, 3, 7, 5, 8, 6, 9, 10)


def is_tfn(value: str) -> bool:
    """True when ``value`` is a structurally valid Australian TFN."""
    digits = digits_only(value)
    if len(digits) != 9:
        return False
    return _weighted_sum(digits, _TFN_WEIGHTS) % 11 == 0


# --- Australian Business Number -------------------------------------------
# An 11-digit identifier issued by the ABR: two leading "prefix" digits that
# encode the entity type and state, followed by the ACN. Each of the 11 digits
# is reduced by one, weighted by ``_ABN_WEIGHTS``, and the total must be
# divisible by 89. The two prefix digits are the only pair that satisfies this
# for a given ACN, which is why the checksum can be validated offline.
_ABN_WEIGHTS = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def is_abn(value: str) -> bool:
    """True when ``value`` satisfies the ABR mod-89 check."""
    digits = digits_only(value)
    if len(digits) != 11:
        return False
    return _weighted_sum(digits, _ABN_WEIGHTS, reduce_one=True) % 89 == 0


# --- Australian Company Number --------------------------------------------
# A 9-digit company identifier issued by ASIC: eight identifier digits plus a
# check digit. The first eight digits are weighted by ``_ACN_WEIGHTS`` and the
# check digit is the mod-10 complement of that sum.
_ACN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 1)


def acn_check_digit(value: str) -> int:
    """Expected check digit for the first eight digits of an ACN."""
    return complement10(_weighted_sum(value[:8], _ACN_WEIGHTS))


def is_acn(value: str) -> bool:
    """True when ``value`` satisfies the ASIC mod-10 check."""
    digits = digits_only(value)
    if len(digits) != 9:
        return False
    return acn_check_digit(digits) == int(digits[8])


# --- Medicare number --------------------------------------------------------
# A 10-digit number: nine service digits plus a check digit.
# The nine digits are weighted 1, 3, 7, 9, 1, 3, 7, 9, 1; the check digit is
# ``(10 - sum % 10) % 10``.
_MEDICARE_WEIGHTS = [1, 3, 7, 9, 1, 3, 7, 9, 1]

_MEDICARE_RE = re.compile(r"\b[1-9]\d{9}\b|\b[1-9]\d\s?\d{5}\s?\d{2}\b")


def medicare_check_digit(service_digits: str) -> int:
    """Return the expected check digit for nine Medicare service digits."""
    total = sum(int(d) * w for d, w in zip(service_digits, _MEDICARE_WEIGHTS, strict=True))
    return (10 - total % 10) % 10


def is_medicare(value: str) -> bool:
    """True when ``value`` is a 10-digit Medicare number with a valid check digit."""
    digits = digits_only(value)
    if len(digits) != 10 or digits[0] == "0":
        return False
    return medicare_check_digit(digits[:9]) == int(digits[9])


# --- Medicare enrolment reference ------------------------------------------
# A 10-digit reference issued by Services Australia, printed as
# ``N 0000 0000 0``. There is no published checksum, so detection is a shape
# test and confidence is reduced accordingly.
def is_medicare_enrolment(value: str) -> bool:
    """True when ``value`` has the shape of a Medicare enrolment reference.

    Requires a leading non-zero digit, a non-zero second digit and ten digits
    overall. This is deliberately conservative: without a checksum, a looser
    test would match arbitrary ten-digit numbers.
    """
    digits = digits_only(value)
    if len(digits) != 10:
        return False
    return digits[0] != "0" and digits[1] != "0"


# --- Bank State Branch ------------------------------------------------------
# A 6-digit routing identifier. The first two digits select the state. ``00``,
# ``97`` and ``99`` are unallocated; the other second digits are live, so the
# check stays shape-level rather than an enumeration.
_BSB_RE = re.compile(r"\b\d{6}\b")

_INVALID_BSB_PREFIXES = {"00", "97", "99"}


def is_bsb(value: str) -> bool:
    """True when ``value`` is a plausible BSB.

    No checksum exists for BSBs, so this rejects only the prefixes the RBA
    never allocates. Callers should require surrounding context (an account
    number, or a nearby routing-label) before treating a match as PII.
    """
    digits = digits_only(value)
    if len(digits) != 6:
        return False
    return digits[:2] not in _INVALID_BSB_PREFIXES


# --- Luhn (credit cards, Medicare-style numbers) --------------------------
def luhn_check(digits: str) -> bool:
    """Generic Luhn validation over a string of digits."""
    if not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# Issuer identification prefixes: the international schemes plus the eftpos,
# Co- and UnionPay ranges that appear on Australian merchant data. ``card_brand``
# resolves the network; this pattern only rejects digits that cannot start a card.
_CARD_RE = re.compile(
    r"(?:"
    r"4\d{3}"                        # Visa
    r"|5[1-5]\d{2}"                  # Mastercard
    r"|6(?:011|5\d{2})"               # Discover / UnionPay
    r"|3[47]\d{2}"                    # Amex
    r"|2(?:2[2-9]\d|[3-6]\d{2}|7[01]\d|720)"  # eftpos
    r"|30[0-9]\d|36\d{2}|38\d{2}"     # eftpos 300-series, 36, 38
    r"|56(?:12|02)|63(?:04|59)"       # eftpos BINs
    r"|979\d"                         # UnionPay
    r")"
    r"\d{8,16}"
)


def card_brand(value: str) -> str | None:
    """Return the card network for ``value``, or ``None`` if the IIN is unknown.

    Covers the schemes actually seen on Australian merchant data: Visa,
    Mastercard, American Express, eftpos (including the Co- and DIN ranges) and
    the local UnionPay entries. Kept as a prefix table rather than a large
    regex so it stays readable and testable.
    """
    digits = digits_only(value)
    if not 12 <= len(digits) <= 19:
        return None
    if digits[0] == "4":
        return "visa"
    if digits[:2] in {"51", "52", "53", "54", "55"}:
        return "mastercard"
    if digits[:2] in {"34", "37"} or digits[:4] == "6011":
        return "amex"
    if digits[:4] in {"5612", "5602", "6304", "6759"} or digits[:2] in {"62", "63"}:
        return "eftpos"
    if digits[:3] in {"300", "301", "302", "303", "304", "305", "308", "309"}:
        return "eftpos"
    if digits[:2] in {"36", "38"}:
        return "eftpos"
    if digits[:3] == "979":
        return "unionpay"
    return None


def is_credit_card(value: str) -> bool:
    """True when ``value`` has a recognised IIN and passes Luhn."""
    digits = digits_only(value)
    if not _CARD_RE.fullmatch(digits):
        return False
    return card_brand(digits) is not None and luhn_check(digits)


# --- IBAN ------------------------------------------------------------------
def iban_check(value: str) -> bool:
    """Validate an IBAN using the ISO 7064 mod-97 check."""
    compact = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def is_passport(value: str) -> bool:
    """True for an Australian passport number shape: one letter then seven digits.

    ``E`` is not issued, and the letter is not ``O`` or ``I`` to avoid
    confusing digits with letters.
    """
    compact = value.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]\d{7}", compact):
        return False
    return compact[0] not in {"E", "O", "I"}
"""Australian identifier detectors.

Two-tier design: a narrow regex finds candidates, then a checksum validator
confirms each one. This keeps false positives near zero for TFN, ABN, ACN,
Medicare and card numbers, which would otherwise match any 9-11 digit run.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .. import checksums as ck
from ..types import Category, Finding, Span
from .base import RegexDetector


class ChecksumDetector(RegexDetector):
    """Scans with a regex, keeps only spans whose checksum validates.

    Subclasses set ``candidate_pattern`` and ``validator``.
    """

    candidate_pattern: str
    category: Category
    flags = 0
    #: When False the checksum gate is advisory: matches are kept but scored low.
    require_checksum = True

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (re.compile(self.candidate_pattern, self.flags),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        for match in self.patterns()[0].finditer(text):
            start, end = match.span()
            raw = match.group()
            if not self.validator(raw):
                if self.require_checksum:
                    continue
                yield Finding(
                    span=Span(start, end),
                    category=self.category,
                    detector=self.name,
                    stage=self.stage,
                    confidence=0.4,
                    value=raw,
                    metadata=(("checksum", "failed"),),
                )
                continue
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=self.confidence_for(match),
                value=raw,
                metadata=self.metadata_for(match),
            )

    def validator(self, raw: str) -> bool:
        raise NotImplementedError


class TFNDetector(ChecksumDetector):
    """Tax File Number: nine digits, mod-11 weighted."""

    name = "tfn"
    category = Category.TFN
    candidate_pattern = r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b"

    def validator(self, raw: str) -> bool:
        return ck.is_tfn(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group())}


class ABNDetector(ChecksumDetector):
    """Australian Business Number: eleven digits, mod-89 weighted."""

    name = "abn"
    category = Category.ABN
    candidate_pattern = r"\b\d{2}[- ]?\d{3}[- ]?\d{3}[- ]?\d{3}\b"

    def validator(self, raw: str) -> bool:
        return ck.is_abn(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group())}


class ACNDetector(ChecksumDetector):
    """Australian Company Number: nine digits, mod-10 weighted."""

    name = "acn"
    category = Category.ACN
    candidate_pattern = r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b"

    def validator(self, raw: str) -> bool:
        return ck.is_acn(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group())}


class MedicareDetector(ChecksumDetector):
    """Medicare number: ten digits with a weighted check digit.

    Grouping varies in the wild (``2123 4567 82``, ``2 1234 5678 0``,
    unseparated), so the candidate pattern accepts optional separators between
    any two digits and lets the checksum decide.
    """

    name = "medicare"
    category = Category.MEDICARE
    candidate_pattern = r"(?<![\d\-])[1-9](?:[\s-]?\d){9}(?![\d\-])"

    def validator(self, raw: str) -> bool:
        return ck.is_medicare(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group())}


class MedicareEnrolmentDetector(ChecksumDetector):
    """Medicare enrolment reference: shape-only, so confidence is reduced."""

    name = "medicare_enrolment"
    category = Category.MEDICARE_ENROLMENT
    candidate_pattern = r"\b[1-9]\s\d{4}\s\d{4}\s\d\b"
    require_checksum = False
    confidence = 0.6

    def validator(self, raw: str) -> bool:
        return ck.is_medicare_enrolment(raw)

    def confidence_for(self, match: re.Match[str]) -> float:
        return self.confidence if self.validator(match.group()) else 0.4


class CreditCardDetector(ChecksumDetector):
    """Card numbers: recognised IIN plus a Luhn check.

    ``require_checksum`` is relaxed because card data is frequently tokenised
    in test fixtures; a Luhn failure is reported at low confidence rather than
    dropped outright.
    """

    name = "credit_card"
    category = Category.CREDIT_CARD
    # Optional separators between any digits, then validated by IIN + Luhn.
    candidate_pattern = (
        r"(?<![\d\-])(?:"
        r"4\d{3}"                    # Visa
        r"|5[1-5]\d{2}"              # Mastercard
        r"|6(?:011|5\d{2})"           # Discover / UnionPay
        r"|3[47]\d{2}"                # Amex
        r"|2(?:2[2-9]\d|[3-6]\d{2}|7[01]\d|720)"  # eftpos
        r"|30[0-9]\d|36\d{2}|38\d{2}"  # eftpos 300-series, 36, 38
        r"|56(?:12|02)|63(?:04|59)"   # eftpos BINs
        r"|979\d"                     # UnionPay
        r")"
        r"(?:[\s-]?\d){8,16}"
        r"(?![\d\-])"
    )
    require_checksum = False

    def validator(self, raw: str) -> bool:
        return ck.is_credit_card(raw)

    def confidence_for(self, match: re.Match[str]) -> float:
        digits = ck.digits_only(match.group())
        brand = ck.card_brand(digits)
        if brand is None:
            return 0.3
        return 1.0 if ck.luhn_check(digits) else 0.5

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        digits = ck.digits_only(match.group())
        return {
            "brand": ck.card_brand(digits) or "unknown",
            "length": len(digits),
            "luhn": ck.luhn_check(digits),
        }


class PassportDetector(RegexDetector):
    """Australian passport number: a letter followed by seven digits."""

    name = "passport"
    category = Category.PASSPORT
    pattern = r"\b(?![EIO])[A-Z]\d{7}\b"

    def patterns(self):
        return (re.compile(self.pattern),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if ck.is_passport(match.group()) else 0.5


class BSBDetector(RegexDetector):
    """Bank State Branch, gated on nearby routing context.

    A bare six-digit number is too ambiguous to redact safely, so a candidate
    is only accepted when a routing label or a following account number is
    present in the surrounding text.
    """

    name = "bsb"
    category = Category.BANK_BSB

    _CONTEXT = re.compile(
        r"(?i)\b(?:bsb|routing|rtn|bank\s*state\s*branch|sb-?no|branch)\b"
        r"|\b(?:acc(?:ount)?|a/c)\s*(?:no\.?|number|#)?"
    )
    _ACCOUNT_AFTER = re.compile(r"^\s?[- ]?\s?\d{5,10}")

    def patterns(self):
        return (ck._BSB_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        for match in ck._BSB_RE.finditer(text):
            raw = match.group()
            if not ck.is_bsb(raw):
                continue
            start, end = match.span()
            window_before = text[max(0, start - 40) : start]
            window_after = text[end : end + 20]
            has_label = bool(self._CONTEXT.search(window_before))
            has_account = bool(self._ACCOUNT_AFTER.match(window_after))
            if not (has_label or has_account):
                continue
            confidence = 1.0 if has_label else 0.75
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=confidence,
                value=raw,
                metadata=(("context", "label" if has_label else "account"),),
            )


class DriversLicenceDetector(RegexDetector):
    """State driver licence numbers.

    Each state uses its own format, so the detector requires a licence label
    in the immediate context before redacting; the shape is then validated per
    state where a public format is documented.
    """

    name = "drivers_licence"
    category = Category.DRIVERS_LICENCE

    _LABEL = re.compile(
        r"(?i)\b(?:driver'?s?\s*licen[cs]e|drivers\s*licence|licence\s*(?:no|number|#)|"
        r"dl|lic(?:ence)?)\b\.?\s*(?:no\.?|number|#)?\s*:?\s*"
    )
    _VALUE = re.compile(r"(?i)\b([A-Z0-9]{5,14})\b")
    # Documented public formats, used only to score confidence.
    _SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^\d{8}$"), "nsw"),
        (re.compile(r"^[A-Z]\d{6}[A-Z]$"), "qld"),
        (re.compile(r"^[A-Z]\d{6}$"), "wa"),
        (re.compile(r"^\d{9}$"), "sa"),
        (re.compile(r"^[A-Z]\d{7}$"), "vic"),
        (re.compile(r"^[A-Z]{2}\d{5}$"), "tas"),
        (re.compile(r"^\d{6}[A-Z]$"), "act"),
        (re.compile(r"^\d{7}$"), "nt"),
    )

    def patterns(self):
        return (self._LABEL,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        for label in self._LABEL.finditer(text):
            tail_start = label.end()
            value_match = self._VALUE.match(text, tail_start)
            if value_match is None:
                continue
            raw = value_match.group(1)
            # Reject obvious non-licence words so the label does not swallow
            # the rest of the sentence.
            if raw.lower() in {"and", "the", "for", "exp", "expiry", "class"}:
                continue
            state = next((s for pattern, s in self._SHAPES if pattern.match(raw)), None)
            yield Finding(
                span=Span(*value_match.span(1)),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=1.0 if state else 0.7,
                value=raw,
                metadata=(("state_format", state or "unknown"),),
            )


class IBANDetector(RegexDetector):
    """International bank account number with a mod-97 check."""

    name = "iban"
    category = Category.IBAN

    _CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

    def patterns(self):
        return (self._CANDIDATE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if ck.iban_check(match.group()) else 0.4
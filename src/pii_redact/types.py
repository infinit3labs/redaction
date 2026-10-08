"""Core types shared across the redaction pipeline."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class Category(str, enum.Enum):
    """Detection category for a PII finding."""

    EMAIL = "email"
    PHONE = "phone"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    MAC = "mac"
    URL_CREDENTIAL = "url_credential"
    TFN = "tfn"
    ABN = "abn"
    ACN = "acn"
    MEDICARE = "medicare"
    MEDICARE_ENROLMENT = "medicare_enrolment"
    PASSPORT = "passport"
    DRIVERS_LICENCE = "drivers_licence"
    CREDIT_CARD = "credit_card"
    BANK_BSB = "bsb"
    POSTCODE = "postcode"
    STREET_ADDRESS = "street_address"
    STATE_POSTCODE = "state_postcode"
    IBAN = "iban"
    #: A date whose day/month order could not be determined.
    DATE = "date"
    DATE_AU = "date_au"
    DATE_ISO = "date_iso"
    NAME = "name"
    ORGANISATION = "organisation"
    #: A place name. Produced by the NLP backends and the gazetteer, not by the
    #: deterministic rules, which only handle postcodes and street addresses.
    LOCATION = "location"
    AGE = "age"
    GENDER = "gender"

    # --- employment attribution --------------------------------------------
    # A named employer or internal body. Redacting these stops a respondent
    # being identifiable even when they never give a name.
    ORGANISATION_NAME = "organisation_name"
    #: A vocabulary hit that narrows the employer to a small set: an internal
    #: system, program, agreement or site. Matched from the lexicon.
    INTERNAL_TERM = "internal_term"
    #: An Australian award or enterprise agreement reference.
    AWARD = "award"

    # --- quasi-identifiers --------------------------------------------------
    # Individually harmless, jointly identifying. Detected so they can be
    # generalised or scored rather than blindly masked.
    JOB_TITLE = "job_title"
    TENURE = "tenure"
    TEAM_SIZE = "team_size"
    EMPLOYMENT_STATUS = "employment_status"
    HEALTH_DETAIL = "health_detail"
    #: A word sequence that identifies a person or a workplace without
    #: containing a name or a direct identifier. See
    #: :mod:`pii_redact.detectors.implication`.
    IMPLICATION = "implication"

    @property
    def is_quasi_identifier(self) -> bool:
        """True for attributes that only identify someone in combination."""
        return self in _QUASI_IDENTIFIERS

    @property
    def is_direct_identifier(self) -> bool:
        """True for attributes that identify someone on their own."""
        return self in _DIRECT_IDENTIFIERS

    def __str__(self) -> str:
        return self.value


#: Attributes that are individually innocuous but jointly identifying.
#: "Grade 4", "night shift", "team of five" reveals very little each; together
#: with an employer they identify one person.
_QUASI_IDENTIFIERS = frozenset(
    {
        Category.JOB_TITLE,
        Category.TENURE,
        Category.TEAM_SIZE,
        Category.EMPLOYMENT_STATUS,
        Category.HEALTH_DETAIL,
        Category.IMPLICATION,
        Category.AGE,
        Category.GENDER,
        Category.DATE_AU,
        Category.DATE_ISO,
        Category.LOCATION,
    }
)

#: Attributes that identify a person without needing anything else.
_DIRECT_IDENTIFIERS = frozenset(
    {
        Category.EMAIL,
        Category.PHONE,
        Category.TFN,
        Category.MEDICARE,
        Category.MEDICARE_ENROLMENT,
        Category.PASSPORT,
        Category.DRIVERS_LICENCE,
        Category.CREDIT_CARD,
        Category.IBAN,
        Category.STREET_ADDRESS,
        Category.NAME,
        Category.IPV4,
        Category.IPV6,
        Category.BANK_BSB,
    }
)


class Stage(str, enum.Enum):
    """Which pipeline stage produced a finding."""

    DETERMINISTIC = "deterministic"
    NLP = "nlp"

    def __str__(self) -> str:
        return self.value


class MaskStyle(str, enum.Enum):
    """How a matched span is rewritten."""

    #: Replace every character with ``mask_char``, preserving length.
    MASK = "mask"
    #: Delete the span entirely, closing the surrounding text together.
    REMOVE = "remove"
    #: Keep a few leading or trailing characters, mask the middle.
    PARTIAL = "partial"
    #: Replace with a plain category label such as ``[NAME]``.
    #:
    #: Preferred over HASH and TOKEN for survey data. A stable identifier that
    #: persists across exports is itself a re-identification vector: anyone
    #: holding two extracts can join them on it. A category label carries no
    #: linkage between values, only the fact that a value of that kind was
    #: present, which is usually what an analysis needs.
    LABEL = "label"
    #: Replace with a salted digest; stable across runs with the same salt.
    HASH = "hash"
    #: Replace with a category-scoped pseudonym such as ``email_ab12cd34``.
    TOKEN = "token"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range ``[start, end)`` in the source text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def extract(self, text: str) -> str:
        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class Finding:
    """A detected PII span plus provenance and confidence."""

    span: Span
    category: Category
    detector: str
    stage: Stage = Stage.DETERMINISTIC
    confidence: float = 1.0
    value: str | None = None
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def priority(self) -> int:
        """Sort key: deterministic findings outrank NLP ones, then confidence."""
        base = 0 if self.stage is Stage.DETERMINISTIC else -1
        return (base, round(self.confidence, 6))

    def meta(self) -> dict[str, Any]:
        return dict(self.metadata)


class RedactionError(ValueError):
    """Raised when input cannot be redacted."""


def total_overlap(spans: Iterable[Span]) -> int:
    return sum(s.length for s in spans)
"""Australian-first PII redaction, with optional NLP augmentation.

Deterministic detectors run first and own the result; an optional NLP stage
adds recall only where the deterministic pass found nothing.

For survey free text use :func:`survey_redactor`, which also wires in employer
attribution and quasi-identifier detection:

    >>> from pii_redact import SurveyRedactor, survey_redactor
    >>> survey = SurveyRedactor(survey_redactor(salt=b"..."))
    >>> record = survey.redact_record("I am a Grade 4 nurse at Acme Pty Ltd")
    >>> record.risk.band.value
    'critical'

For a plain document or log line:

    >>> from pii_redact import Redactor, Policy
    >>> Redactor(Policy()).redact_text("TFN 123 456 782")
    'TFN *** *** ***'
"""

from __future__ import annotations

from .checksums import (
    iban_check,
    is_abn,
    is_acn,
    is_bsb,
    is_credit_card,
    is_medicare,
    is_medicare_enrolment,
    is_passport,
    is_tfn,
    luhn_check,
)
from .engine import (
    DEFAULT_DETECTORS,
    Policy,
    RedactionResult,
    Redactor,
    redact,
    resolve_overlaps,
)
from .generalise import generalise
from .lexicon import Lexicon, LexiconMatch, Term, load_builtin_lexicon
from .masking import consistent_token, partial_mask, set_process_salt, stable_hash
from .normalise import FoldedText, normalise, normalise_aligned
from .records import RecordRedaction, SurveyRedactor
from .residual import (
    DisclosureAction,
    DisclosureDecision,
    DisclosurePolicy,
    ResidualAttribute,
    ResidualKind,
    ResidualProfile,
    decide,
    measure,
    protect,
    summarise,
)
from .risk import Audience, RiskAssessment, RiskBand, RiskSignal, assess
from .survey import survey_detectors, survey_policy, survey_redactor
from .types import Category, Finding, MaskStyle, RedactionError, Span, Stage

__version__ = "0.3.0"

__all__ = [
    "DEFAULT_DETECTORS",
    "Audience",
    "Category",
    "DisclosureAction",
    "DisclosureDecision",
    "DisclosurePolicy",
    "Finding",
    "FoldedText",
    "Lexicon",
    "LexiconMatch",
    "MaskStyle",
    "Policy",
    "RecordRedaction",
    "RedactionError",
    "RedactionResult",
    "Redactor",
    "ResidualAttribute",
    "ResidualKind",
    "ResidualProfile",
    "RiskAssessment",
    "RiskBand",
    "RiskSignal",
    "Span",
    "Stage",
    "SurveyRedactor",
    "Term",
    "__version__",
    "assess",
    "consistent_token",
    "decide",
    "generalise",
    "iban_check",
    "is_abn",
    "is_acn",
    "is_bsb",
    "is_credit_card",
    "is_medicare",
    "is_medicare_enrolment",
    "is_passport",
    "is_tfn",
    "load_builtin_lexicon",
    "luhn_check",
    "measure",
    "normalise",
    "normalise_aligned",
    "partial_mask",
    "protect",
    "redact",
    "resolve_overlaps",
    "set_process_salt",
    "stable_hash",
    "summarise",
    "survey_detectors",
    "survey_policy",
    "survey_redactor",
]
"""Row-level API: redact one response and return the evidence with it.

This is the unit the Spark integration calls. Keeping it a plain function over
plain types matters: it is testable without Spark, and it is the same code path
whether a record is redacted in a cluster, in a notebook or in a test.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .engine import RedactionResult, Redactor
from .residual import DisclosureDecision, DisclosurePolicy, protect
from .risk import Audience, RiskAssessment, RiskBand, assess
from .types import Category, Finding

#: Categories whose metadata carries a lexicon label, used for attribution
#: scoring. Kept as a set so a new label-bearing category only needs one edit.
_LABEL_CATEGORIES = frozenset({Category.INTERNAL_TERM, Category.AWARD})


@dataclass(frozen=True, slots=True)
class RecordRedaction:
    """One redacted response, with the evidence behind the decision."""

    record_id: str | None
    text: str
    original_text: str
    findings: tuple[Finding, ...]
    risk: RiskAssessment
    #: What residual disclosure control decided about the redacted text, or
    #: None when the deployment did not enable it.
    disclosure: DisclosureDecision | None = None

    @property
    def redacted_categories(self) -> frozenset[Category]:
        return frozenset(f.category for f in self.findings)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def withheld(self) -> bool:
        """True when residual control withheld the record rather than releasing it."""
        return self.disclosure is not None and self.disclosure.action.value == "suppress"

    def findings_json(self) -> str:
        """Compact findings as JSON, for an audit column.

        Offsets are against ``original_text``, so an audit row can be joined
        back to the raw Qualtrics extract to verify any redaction.
        """
        return json.dumps(
            [
                {
                    "category": f.category.value,
                    "start": f.span.start,
                    "end": f.span.end,
                    "detector": f.detector,
                    "stage": f.stage.value,
                    "confidence": round(f.confidence, 4),
                    "metadata": f.meta(),
                }
                for f in self.findings
            ],
            separators=(",", ":"),
            default=str,
        )

    def audit_json(self) -> str:
        """Findings plus the risk assessment, as one JSON object."""
        return json.dumps(
            {
                "record_id": self.record_id,
                "finding_count": self.finding_count,
                "categories": sorted(c.value for c in self.redacted_categories),
                "risk": self.risk.as_dict(),
                "disclosure": (
                    None if self.disclosure is None else self.disclosure.as_dict()
                ),
                "findings": json.loads(self.findings_json()),
            },
            separators=(",", ":"),
            default=str,
        )


class SurveyRedactor:
    """A redactor plus the record-level risk assessment around it."""

    def __init__(
        self,
        redactor: Redactor,
        *,
        audience: Audience = Audience.INTERNAL,
        review_threshold: RiskBand = RiskBand.HIGH,
        residual_policy: DisclosurePolicy | None = None,
    ) -> None:
        """A redactor, an audience, and optionally a disclosure policy.

        ``residual_policy`` enables the second pass in :mod:`pii_redact.residual`,
        which measures what the first pass left behind. It is off by default
        because it needs a population figure to mean anything, and a wrong figure
        is worse than no figure: supply the size of the workforce these responses
        came from, and the module will tell you when a record is identifying
        because of what *survived* redaction rather than because a rule fired.
        """
        self.redactor = redactor
        self.audience = audience
        self.review_threshold = review_threshold
        self.residual_policy = residual_policy

    def redact_record(
        self,
        text: str | None,
        *,
        record_id: str | None = None,
    ) -> RecordRedaction:
        """Redact one free-text response.

        ``None`` and empty input are returned unchanged with a zero risk
        assessment, so a sparse survey does not need special handling.
        """
        if text is None or not text.strip():
            return RecordRedaction(
                record_id=record_id,
                text=text or "",
                original_text=text or "",
                findings=(),
                risk=RiskAssessment(score=0.0, band=RiskBand.LOW, signals=(), word_count=0),
            )

        result: RedactionResult = self.redactor.redact(text)
        labels = _labels_for(result.findings)
        risk = assess(text, result.findings, audience=self.audience, lexicon_labels=labels)

        # Residual control runs on the *output*: it exists to catch what the
        # detectors did not claim, which by definition is only visible here.
        released = result.text
        disclosure: DisclosureDecision | None = None
        if self.residual_policy is not None:
            released, disclosure = protect(result.text, self.residual_policy)

        return RecordRedaction(
            record_id=record_id,
            text=released,
            original_text=text,
            findings=result.findings,
            risk=risk,
            disclosure=disclosure,
        )

    def needs_review(self, redaction: RecordRedaction) -> bool:
        """True when a record is at or above the configured review threshold."""
        order = list(RiskBand)
        return order.index(redaction.risk.band) >= order.index(self.review_threshold)

    def redact_many(
        self,
        texts: Sequence[str | None],
        record_ids: Sequence[str | None] | None = None,
    ) -> list[RecordRedaction]:
        """Redact a batch, in input order.

        Provided for local preview of a sample before running the Spark job.
        """
        ids = record_ids if record_ids is not None else [None] * len(texts)
        return [
            self.redact_record(text, record_id=record_id)
            for text, record_id in zip(texts, ids, strict=False)
        ]


def _labels_for(findings: Sequence[Finding]) -> list[str]:
    """Lexicon labels present in ``findings``, for employer attribution."""
    return [
        str(f.meta()["label"])
        for f in findings
        if f.category in _LABEL_CATEGORIES and "label" in f.meta()
    ]


__all__ = ["RecordRedaction", "SurveyRedactor"]
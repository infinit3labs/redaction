"""Re-identification risk scoring for a single free-text response.

The hard question with survey data is not "is this text free of PII". It is
"could someone who knows the organisation work out who wrote this". That is a
property of a *combination* of attributes, so it cannot be answered by
inspecting spans one at a time.

The model here is an explicit additive score over named signals. Every
contribution is returned with its reason, so a reviewer can see exactly why a
record was flagged and argue with a specific weight rather than with a number
that came out of a model.

Two facts drive most of the weight:

* **Employer attribution multiplies risk.** Inside an organisation, an
  employee's job title, grade and tenure uniquely identify them to a colleague.
  The same text shared externally is far less identifying. The score therefore
  takes an ``audience`` argument.
* **A narrative about a specific incident is highly identifying** even with no
  names at all, because incidents are memorable and few.

Scores are advisory. They exist to rank records for human review, not to
replace it.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .types import Category, Finding


class RiskBand(enum.Enum):
    """Triage band. Ordered least to most identifying."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value


class Audience(enum.Enum):
    """Who the redacted data is being shared with.

    The same text carries different risk depending on the reader, and in
    workplace data the difference is large.
    """

    #: Retained internally under access control, e.g. an incident response
    #: team who already know who works there.
    INTERNAL = "internal"
    #: Shared with a privacy officer, regulator or researcher who knows the
    #: organisation exists but not who said what.
    EXTERNAL_ANALYST = "external_analyst"
    #: Published or shared beyond the organisation, where the respondent's
    #: colleagues may also read it.
    PUBLIC = "public"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RiskSignal:
    """One named contribution to the score."""

    name: str
    weight: float
    detail: str
    category: Category | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": self.weight,
            "detail": self.detail,
            "category": self.category.value if self.category else None,
        }


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """A record's re-identification risk, with its reasoning."""

    score: float
    band: RiskBand
    signals: tuple[RiskSignal, ...] = ()
    word_count: int = 0

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.signals)

    def as_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary, safe to write to an audit table."""
        return {
            "score": round(self.score, 4),
            "band": self.band.value,
            "word_count": self.word_count,
            "signals": [s.as_dict() for s in self.signals],
        }

    def __str__(self) -> str:
        # Deliberately free of any source text.
        return f"{self.band.value} ({self.score:.2f}): {', '.join(self.signal_names) or 'no signals'}"


#: Band thresholds, highest first.
_BANDS: tuple[tuple[float, RiskBand], ...] = (
    (0.75, RiskBand.CRITICAL),
    (0.50, RiskBand.HIGH),
    (0.25, RiskBand.MODERATE),
    (0.0, RiskBand.LOW),
)


# --- Signal vocabulary -------------------------------------------------------
# Incident and interpersonal-harm narrative. Present in most bullying and
# psychosocial risk responses.
_NARRATIVE_RE = re.compile(
    r"(?i)\b(?:"
    r"bully|bullying|bullied|harass|harassment|harass(?:ed|ing)|"
    r"abuse|abusive|aggress(?:ion|ive)|threat(?:en)?|intimidat\w+|"
    r"humiliat\w+|degrad\w+|victimi[sz]e|retaliat\w+|"
    r"gossip|back ?stab\w+|excluded|ostraci[sz]ed|"
    r"manager|m supervisor|team lead|"
    r"complain(?:t|ed|ts)?|grievance|report(?:ed)?\s+(?:it|this|them)"
    r")\b"
)
# Second-person or first-person incident framing: "they said", "he put me in".
_SCENARIO_RE = re.compile(
    r"(?i)\b(?:"
    r"i\s+(?:told|asked|reported|raised|complained)|"
    r"(?:they|he|she|my manager|my supervisor)\s+"
    r"(?:said|told|asked|put|kept|made|always|never)\b|"
    r"every\s+(?:week|day|shift)|"
    r"each\s+(?:week|day|shift)|"
    r"every\s+(?:single|day|week)\s+(?:shift|week|day)"
    r")\b"
)
# Explicit uniqueness claims. These are near-certain identifiers within a team.
_UNIQUE_RE = re.compile(
    r"(?i)\b(?:the\s+only|only\s+(?:one|woman|man|casual|temp)|"
    r"no\s+one\s+else|nobody\s+else|"
    r"i\s+am\s+the\s+only|first\s+woman|only\s+(?:woman|man)\s+in)"
)
# Voluntarily disclosed protected characteristics: in a bullying survey these
# are often the subject of the complaint, which makes them identifying.
_PROTECTED_RE = re.compile(
    r"(?i)\b(?:"
    r"pregnan\w+|breastfeed\w+|parental\s+leave|"
    r"disab\w+|wheelchair|"
    r"indigenous|aboriginal|torres\s+strait|"
    r"lgb\w+|lgbt\w+|queer|gay|lesbian|bisexual|trans(?:gender)?|"
    r"muslim|jewish|hindu|christian|"
    r"caucasian|white\s+australian|"
    r"overtime|on\s+call|second\s+job"
    r")\b"
)
# The respondent is describing an identifiable event rather than an opinion.
_SPECIFIC_EVENT_RE = re.compile(
    r"(?i)\b(?:incident|on\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?|"
    r"last\s+(?:week|month|year|night|shift)|"
    r"after\s+the\s+(?:meeting|restructure|reorg|change)|"
    r"when\s+i\s+(?:arrived|started)|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})"
)
# An explicit confidentiality request. The respondent is telling you the
# response identifies them, which is stronger evidence than any combination of
# attributes you could infer -- and no amount of masking addresses it, so it
# belongs at the top of the triage queue.
_CONFIDENTIALITY_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:do\s+not|don'?t|never)\s+(?:want\s+)?(?:my\s+|our\s+)?(?:employer|manager|"
    r"boss|supervisor|company|colleagues?|coworkers?|co-workers|HR|anyone)\s+"
    r"(?:to\s+)?(?:find\s+out|know|see|be\s+told)|"
    r"(?:please\s+)?(?:keep|make)\s+this\s+(?:anonymous|confidential)|"
    r"(?:i\s+)?(?:am|'m|was)\s+(?:worried|concerned|nervous|scared)\s+about\s+"
    r"(?:being\s+)?(?:identified|recognised|recognized|found\s+out|tracked)|"
    r"this\s+(?:is\s+)?(?:not\s+)?anonymous|"
    r"i\s+don'?t\s+want\s+my\s+name"
    r")"
)

# Job title plus a rare combination. Handled by the caller via findings.
_NON_TEXT_SIGNALS = frozenset(
    {
        Category.TFN,
        Category.MEDICARE,
        Category.PASSPORT,
        Category.DRIVERS_LICENCE,
        Category.EMAIL,
        Category.PHONE,
        Category.CREDIT_CARD,
        Category.STREET_ADDRESS,
        Category.BANK_BSB,
        Category.IPV4,
    }
)

#: Weights for text-based signals. Named so they can be tuned with evidence
#: rather than by feel.
WEIGHTS = {
    "narrative": 0.20,
    "scenario": 0.15,
    "unique": 0.30,
    "protected": 0.15,
    "specific_event": 0.10,
    "employer_attribution": 0.25,
    "named_organisation": 0.20,
    "internal_lexicon": 0.12,
    "job_title": 0.12,
    "grade": 0.15,
    "tenure": 0.10,
    "team_size_small": 0.12,
    "employment_status": 0.08,
    "health_detail": 0.12,
    "date_of_birth": 0.15,
    "age": 0.06,
    "location_specific": 0.10,
    "short_response": 0.08,
    "direct_identifier": 0.45,
    "confidentiality_request": 0.35,
}

#: Multiplier applied to quasi-identifier weight by audience. The same title is
#: far more identifying to a colleague than to an external analyst.
_AUDIENCE_QUASI_FACTOR = {
    Audience.INTERNAL: 1.0,
    Audience.EXTERNAL_ANALYST: 0.8,
    Audience.PUBLIC: 0.6,
}

#: Multiplier applied to employer-attribution weight by audience. Attribution
#: is what makes a record risky *to the organisation*, so it matters most
#: internally and for external analysts who know the sector.
_AUDIENCE_ATTRIBUTION_FACTOR = {
    Audience.INTERNAL: 1.0,
    Audience.EXTERNAL_ANALYST: 1.0,
    Audience.PUBLIC: 0.7,
}

#: Responses shorter than this carry risk out of proportion to their content:
#: a short, specific answer is more likely to be about one identifiable event.
SHORT_RESPONSE_WORDS = 12

#: Ceiling on the short-response amplification. Without a cap, a three-word
#: answer multiplies its weight fourfold and lands in CRITICAL on its own,
#: which drowns out every other signal in the triage queue.
SHORT_RESPONSE_MAX_FACTOR = 2.0

#: A team this size or smaller makes a job title nearly unique.
SMALL_TEAM = 5


def assess(
    text: str,
    findings: Sequence[Finding],
    *,
    audience: Audience = Audience.INTERNAL,
    lexicon_labels: Iterable[str] = (),
) -> RiskAssessment:
    """Score one response.

    ``findings`` are the detections for ``text`` (from
    :meth:`pii_redact.Redactor.scan`). ``lexicon_labels`` are the labels of any
    lexicon hits, so employer attribution can be scored even when those spans
    are not separately reported.
    """
    words = text.split()
    word_count = len(words)
    signals: list[RiskSignal] = []
    quasi_factor = _AUDIENCE_QUASI_FACTOR[audience]
    attribution_factor = _AUDIENCE_ATTRIBUTION_FACTOR[audience]

    def add(name: str, detail: str, category: Category | None = None, factor: float = 1.0) -> None:
        weight = WEIGHTS[name] * factor
        if weight <= 0:
            return
        signals.append(RiskSignal(name=name, weight=weight, detail=detail, category=category))

    categories = {f.category for f in findings}
    categories_by_finding = {f.category: f for f in findings}

    # -- direct identifiers --------------------------------------------------
    direct = sorted(c for c in categories if c.is_direct_identifier)
    for category in direct:
        finding = categories_by_finding[category]
        add(
            "direct_identifier",
            f"{category.value} detected by {finding.detector}",
            category,
        )

    # -- text signals ---------------------------------------------------------
    if _NARRATIVE_RE.search(text):
        add("narrative", "describes interpersonal harm or a complaint")
    if _SCENARIO_RE.search(text):
        add("scenario", "describes a specific incident with named actors")
    if _UNIQUE_RE.search(text):
        add("unique", "claims to be unique within their group")
    if _PROTECTED_RE.search(text):
        add("protected", "mentions a protected or distinguishing characteristic")
    if _SPECIFIC_EVENT_RE.search(text):
        add("specific_event", "references a dated or named event")
    if _CONFIDENTIALITY_RE.search(text):
        add(
            "confidentiality_request",
            "asks that this not be attributed, or fears being identified",
        )
    if word_count and word_count <= SHORT_RESPONSE_WORDS:
        add(
            "short_response",
            f"only {word_count} words, so a single identifying fact dominates",
            factor=min(SHORT_RESPONSE_MAX_FACTOR, SHORT_RESPONSE_WORDS / max(word_count, 1)),
        )

    # -- employer attribution -------------------------------------------------
    labels = list(lexicon_labels)
    if labels:
        add(
            "internal_lexicon",
            f"{len(labels)} internal or award term(s): {', '.join(sorted(set(labels))[:3])}",
            Category.INTERNAL_TERM,
            # Several distinct internal terms are much stronger evidence of
            # attribution than one, but the effect saturates.
            factor=attribution_factor * min(1.0 + 0.15 * (len(labels) - 1), 1.6),
        )
    if Category.ORGANISATION_NAME in categories:
        add(
            "named_organisation",
            "names the employer or an internal body",
            Category.ORGANISATION_NAME,
            factor=attribution_factor,
        )
    if Category.AWARD in categories:
        add(
            "employer_attribution",
            "references an award or agreement",
            Category.AWARD,
            factor=attribution_factor,
        )
    if Category.INTERNAL_TERM in categories:
        add(
            "employer_attribution",
            "references an internal system or process",
            Category.INTERNAL_TERM,
            factor=attribution_factor,
        )

    # -- quasi-identifiers ----------------------------------------------------
    if Category.JOB_TITLE in categories:
        finding = categories_by_finding[Category.JOB_TITLE]
        strong = finding.confidence >= 0.8
        add(
            "job_title",
            f"states a job title or grade ({finding.meta().get('cue', '')})".strip(),
            Category.JOB_TITLE,
            factor=quasi_factor * (1.3 if strong else 1.0),
        )
    if Category.TENURE in categories:
        add(
            "tenure",
            f"states length of service ({categories_by_finding[Category.TENURE].meta().get('cue', '')})".strip(),
            Category.TENURE,
            factor=quasi_factor,
        )
    if Category.TEAM_SIZE in categories:
        finding = categories_by_finding[Category.TEAM_SIZE]
        detail = finding.meta().get("cue", "")
        small = bool(detail) and _looks_small(detail)
        add(
            "team_size_small",
            f"describes a small or unique group ({detail})".strip(),
            Category.TEAM_SIZE,
            factor=quasi_factor * (1.4 if small else 1.0),
        )
    if Category.EMPLOYMENT_STATUS in categories:
        add(
            "employment_status",
            "states casual, temporary or part-time status",
            Category.EMPLOYMENT_STATUS,
            factor=quasi_factor,
        )
    if Category.HEALTH_DETAIL in categories:
        add(
            "health_detail",
            "discloses a health condition",
            Category.HEALTH_DETAIL,
            factor=quasi_factor,
        )
    if Category.AGE in categories:
        add("age", "states an age", Category.AGE, factor=quasi_factor)
    if Category.DATE_AU in categories or Category.DATE_ISO in categories:
        add(
            "date_of_birth",
            "states a calendar date",
            Category.DATE_AU if Category.DATE_AU in categories else Category.DATE_ISO,
            factor=quasi_factor,
        )
    if Category.STREET_ADDRESS in categories or Category.LOCATION in categories:
        add(
            "location_specific",
            "states a workplace location",
            Category.STREET_ADDRESS if Category.STREET_ADDRESS in categories else Category.LOCATION,
            factor=quasi_factor,
        )

    # -- combination effects ---------------------------------------------------
    quasi_count = sum(
            1 for signal in signals if signal.category and signal.category.is_quasi_identifier
        )
    if quasi_count >= 3:
        # The whole point of this module: three innocuous attributes are not
        # innocuous together.
        bonus = min(0.30, 0.10 * (quasi_count - 2)) * quasi_factor
        signals.append(
            RiskSignal(
                name="quasi_combination",
                weight=bonus,
                detail=f"{quasi_count} quasi-identifiers combine to narrow the candidate pool",
            )
        )

    score = min(1.0, sum(s.weight for s in signals))
    return RiskAssessment(
        score=score,
        band=_band(score),
        signals=tuple(sorted(signals, key=lambda s: -s.weight)),
        word_count=word_count,
    )


_SMALL_WORDS = {
    "1": 1, "one": 1, "2": 2, "two": 2, "3": 3, "three": 3, "4": 4, "four": 4,
    "5": 5, "five": 5, "6": 6, "six": 6, "7": 7, "seven": 7, "8": 8, "eight": 8,
    "9": 9, "nine": 9, "10": 10, "ten": 10, "a few": 3, "several": 5,
}


def _looks_small(detail: str) -> bool:
    """True when a team-size cue denotes a group of five or fewer."""
    cleaned = detail.strip().lower().rstrip("+ ")
    return _SMALL_WORDS.get(cleaned, 99) <= SMALL_TEAM


def _band(score: float) -> RiskBand:
    for threshold, band in _BANDS:
        if score >= threshold:
            return band
    return RiskBand.LOW


def filter_by_band(
    assessments: Iterable[RiskAssessment],
    minimum: RiskBand,
) -> list[RiskAssessment]:
    """Select assessments at or above ``minimum``, for review triage."""
    order = list(RiskBand)
    cutoff = order.index(minimum)
    return [a for a in assessments if order.index(a.band) >= cutoff]


__all__ = [
    "SHORT_RESPONSE_WORDS",
    "SMALL_TEAM",
    "WEIGHTS",
    "Audience",
    "RiskAssessment",
    "RiskBand",
    "RiskSignal",
    "assess",
    "filter_by_band",
]
"""Quasi-identifier detectors.

These are the attributes that identify nobody on their own and everybody in
combination. In a WHS or bullying survey the combination is the problem:
"Grade 4", "night shift", "team of five", "18 years of service" and one
unusual injury is a person, even with no name anywhere in the text.

They are detected rather than masked, because each carries analytic value that
the survey presumably needs. The right response is usually to generalise
(``"18 years"`` to a service band) or to score the record for review, not to
delete. See :mod:`pii_redact.generalise` and :mod:`pii_redact.risk`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..generalise import is_banded
from ..normalise import normalise_aligned
from ..types import Category, Finding, Span, Stage
from ..vocabulary import APS_LEVEL

# --- Job title ---------------------------------------------------------------
# A curated list of common Australian occupations. Deliberately generic: the
# goal is to recognise "I am a <title>" framing and a known occupation noun,
# not to build a job taxonomy. Add to it as your workforce requires.
#
# Only specific occupations. Abstract nouns like "supervisor", "operator" and
# "analyst" are excluded because they match almost any sentence and destroy
# the signal this detector is supposed to carry.
_AU_OCCUPATIONS = (
    r"nurse|midwife|doctor|physician|surgeon|dentist|pharmacist|"
    r"teacher|lecturer|principal|"
    r"police officer|detective|constable|firefighter|paramedic|ambulance officer|"
    r"electrician|plumber|carpenter|welder|builder|labourer|"
    r"bus driver|truck driver|forklift driver|taxi driver|delivery driver|"
    r"cleaner|housekeeper|janitor|custodian|"
    r"chef|cook|baker|barista|"
    r"care worker|carer|support worker|"
    r"secretary|receptionist|accountant|auditor|"
    r"cashier|storeperson|"
    r"trainee|apprentice|intern|"
    r"librarian|curator|"
    r"security officer|"
    r"registered nurse|enrolled nurse"
)

_JOB_TITLE_LABEL_RE = re.compile(
    rf"(?i)\b(?:my\s+(?:job\s*title|role|position|occupation)|"
    rf"job\s*title|position\s*title|designation|"
    rf"i\s+(?:am|work|worked)\s+as\s+an?)\s*:?\s*"
    rf"(?P<value>(?:(?:[a-z]+\s+){{0,2}})(?:{_AU_OCCUPATIONS}))\b"
)
# Bare occupation, no framing. Low confidence by design: "the nurse said" is a
# signal, but a weak one, and the policy's confidence floor should decide
# whether it is worth masking.
_JOB_TITLE_BARE_RE = re.compile(
    rf"\b(?P<value>(?:registered|enrolled|senior|junior|trained)?\s*(?:{_AU_OCCUPATIONS}))\b",
    re.IGNORECASE,
)

# --- Grade / classification ---------------------------------------------------
# Pay grades and classifications are strong within-employer identifiers.
_GRADE_RE = re.compile(
    r"(?i)\b(?:grade|band|class|category|step|pay\s*point)\s*[:#]?\s*"
    r"(?P<value>[a-z]{1,2}\s?\d{1,2}|\d{1,2}[a-z]?)\b"
)

# --- Tenure -------------------------------------------------------------------
# The whole run is captured so it can be generalised to a service band rather
# than deleted.
#
# An employment context is mandatory, either before or after the number.
# Without it, "6 years" of anything reads as tenure, and "45 years old" would
# be reported as 45 years of service. Both directions of that error corrupt the
# age and tenure distributions the survey exists to measure.
# Interpolated alternations are always wrapped in a non-capturing group.
# Interpolating a bare "a|b" applies the alternation to the whole surrounding
# expression, so the group stops being a group and the capture it contains
# silently stops capturing.
_TENURE_AFTER = (
    r"(?:"
    r"\s+of\s+service|"
    r"\s+(?:with|at)\s+(?:the|my|our|this)\s+\w+|"
    r"\s+in\s+(?:the|my|our|this)\s+\w+|"
    r"\s+there|"
    r"\s+at\s+(?:the|my|our)\s+(?:company|organisation|employer|workplace)"
    r")"
)
# "worked here 18 years", "been with them 6 years": the cue precedes the number.
_TENURE_BEFORE = (
    r"(?:"
    r"(?:i'?ve\s+)?(?:been|worked|working)\s+"
    r"(?:here|there|for\s+them|with\s+(?:them|the\s+company))\s+"
    r"(?:now\s+|already\s+)?"
    r")"
)
_TENURE_RE = re.compile(
    # One flag position for the whole pattern; re does not allow a global flag
    # in the middle of an alternation.
    rf"(?i)\b(?:{_TENURE_BEFORE})?(?P<value>\d{{1,2}}(?:\.\d)?\s*\+?\s*years?{_TENURE_AFTER})"
    rf"|\b{_TENURE_BEFORE}(?P<value2>\d{{1,2}}(?:\.\d)?\s*\+?\s*years?)"
)
_TENURE_SINCE_RE = re.compile(
    r"(?i)\b(?:since|from|started\s+there\s+in|been\s+(?:here|with\s+\w+)\s+since|"
    r"i\s+(?:joined|started)\s+(?:in|here\s+in))\s+(?P<value>(?:19|20)\d{2})\b"
)

# --- Team size ------------------------------------------------------------------
# The noun must be a real unit of people. "shift with 18 years of service" is
# not a team of eighteen, so the number is rejected when a time unit follows.
_TEAM_SIZE_NOUN = r"team|department|dept|unit|ward|crew|roster|school|store|classroom|section"
# Respondents write small numbers as words more often than as digits, so both
# forms are accepted.
_NUMBER = r"\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|" \
          r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|" \
          r"a\s+few|several"
_TEAM_SIZE_RE = re.compile(
    rf"(?i)\b(?:{_TEAM_SIZE_NOUN})\b\s*"
    r"(?:(?:of|with|has|had|was|is|comprising|consisting\s+of)\s*)?"
    r"(?:just|only|about|around|approximately|roughly)?\s*"
    rf"(?P<value>{_NUMBER})\s*(?:\+\s*)?"
    r"(?:\s*(?:people|staff|workers|employees|members|persons|of\s+us|strong))?"
    r"(?!\s*(?:year|month|week|hour|day|minute))"
    r"(?![\d:/\-])"
)
# Uniqueness claims ("the only woman on the night shift") used to be reported
# here as a team size. That was wrong twice over: the category is not a team
# size, and the generaliser then rewrote "the only one" into "1 person", which
# produced a number that other detectors matched on. They are handled by
# pii_redact.detectors.implication, which redacts the whole clause and labels
# it.

# --- Employment status -------------------------------------------------------------
_STATUS_VALUE = (
    r"casual|part[\s-]?time|full[\s-]?time|temporary|temp|contract(?:or)?|"
    r"agency|fixed[\s-]?term|relief|shift\s+worker|rotating\s+shift|night\s+shift|"
    r"subcontractor|apprentice|trainee|intern|volunteer"
)
_EMPLOYMENT_STATUS_RE = re.compile(
    # "a"/"an"/"currently"/"still" can stack or be absent: "I am currently a
    # fixed-term temp" and "I'm part-time" are both how respondents write it.
    rf"(?i)\b(?:i\s+am|i'?m|as)\s+(?:(?:a|an|currently|still|now)\s+)*"
    rf"(?P<value>{_STATUS_VALUE})\b"
)
# List phrasing. "I am a Grade 4 registered nurse, casual, 18 years of service"
# states employment status in a comma list with no framing cue at all, which is
# how most free-text answers are written.
_EMPLOYMENT_STATUS_LIST_RE = re.compile(
    rf"(?i)(?:,\s*|^)(?P<value>{_STATUS_VALUE})\s*(?:,|$|\.)",
)

# --- Health detail ------------------------------------------------------------------
# Health detail is frequently the *subject* of a bullying or psychosocial risk
# complaint, so deleting it destroys the finding the survey exists to surface.
# Detected for scoring and generalisation, never masked by default.
_HEALTH_DETAIL_RE = re.compile(
    r"(?i)\b(?:i\s+(?:am|'m|have|suffer\s+from)|my\s+(?:condition|illness|injury|disease|"
    r"disorder|symptom)s?|diagnosed\s+with|struggling\s+with)\s+"
    # Consume an optional copula so "my condition is chronic fatigue" captures
    # "chronic fatigue" rather than stopping on the word "is".
    r"(?:(?:is|was|are|were)\s+)?"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-']*(?:\s+[A-Za-z0-9][A-Za-z0-9\-']*){0,4})"
)
_MENTAL_HEALTH_RE = re.compile(
    r"(?i)\b(?:anxiety|depression|ptsd|psychological\s+distress|"
    r"mental\s+health|panic\s+attacks?|burnout|psychosis|"
    r"psychological\s+safety|mental\s+illness)\b"
)

#: Conditions that genuinely end in -ed or -ing. Everything else that does is
#: a past-tense verb, which is how "I have raised it" became a diagnosis.
_CONDITION_ED = frozenset(
    {
        "depressed", "undiagnosed", "undiagnosed?", "bedridden", "longterm",
        "overworked", "overtired", "understaffed", "overstretched",
    }
)

#: Common verbs and function words that follow "I have" or "I am" without
#: naming a condition.
#:
#: The irregular past tenses are here for the same reason as the "-ed" check
#: above: "I have rung the service centre four times" matched the "I have" cue
#: and reported the verb as a diagnosis. They are listed rather than
#: pattern-matched, because the regular rule cannot see them.
_NOT_HEALTH_VERBS = frozenset(
    {
        "put", "got", "gotten", "taken", "done", "seen", "noticed", "heard",
        "read", "written", "said", "told", "asked", "found", "used", "left",
        "been", "going", "being", "doing", "working", "trying", "hoping",
        "already", "also", "just", "only", "still", "even", "never", "always",
        "twice", "thrice", "since", "before", "after", "again", "twice,",
        "raised", "worked", "let", "set", "kept", "gave", "made",
        "rung", "brought", "held", "spent", "sent", "paid",
    }
)

#: Nouns that name a household, a possession, or a piece of paperwork. "I have
#: family at home", "I have my own transport" and "I have screenshots" all match
#: the "I have" cue and were reported as health disclosures. This is a denylist
#: and cannot be complete: an unexpected noun after the cue is still reported.
#: That direction is deliberate -- the cost of a false positive is a noisy
#: finding a reviewer discards, and the cost of a false negative is a health
#: disclosure that no longer appears anywhere in the redacted record.
_NOT_HEALTH_NOUNS = frozenset(
    {
        "family", "families", "kids", "twins", "mortgage", "transport", "passport",
        "licence", "license", "visa", "loan", "degree", "certificate", "keys",
        "phone", "laptop", "car", "house", "home", "job", "holiday", "vacation",
        "screenshots", "screenshot", "receipts", "receipt", "paperwork",
        "references", "reference", "questions", "concerns", "notes", "records",
        "documents", "forms", "photos", "copies", "invoices", "invoice",
        "bookings", "password", "reminders", "appointments", "files",
        "statements", "passwords", "spare", "spares",
    }
)

#: Occupations. "I am registered nurse" matches the "I am" cue, but the words
#: after it are a job, not a condition.
_OCCUPATION_WORDS = frozenset(
    {
        "nurse", "midwife", "teacher", "doctor", "driver", "cleaner",
        "electrician", "plumber", "barista", "chef", "cook", "warden",
        "carer", "care", "worker", "labourer", "technician", "officer",
        "clerk", "accountant", "librarian", "pharmacist",
    }
)

#: Function and employment words that must never appear in a captured health
#: value. Without this, "I am the only woman in my unit" is read as a health
#: condition because "I am" is the cue.
_NOT_HEALTH = frozenset(
    {
        "the", "a", "an", "my", "in", "of", "at", "to", "for", "on", "and", "or",
        "only", "woman", "man", "guy", "lady", "girl", "person", "casual", "temp",
        "part", "time", "full", "contract", "contractor", "apprentice", "trainee",
        "intern", "volunteer", "team", "unit", "department", "shift", "grade",
        "new", "old", "sure", "not", "so", "very", "really", "just", "still",
        "always", "never", "here", "there", "back", "doing", "working", "been",
        "one", "two", "three", "no", "yes", "about", "after", "before", "since",
        "lot", "bit", "kind", "sort", "same", "other", "another", "each",
        "is", "was", "are", "were", "be", "being", "am",
        # Time units. "I am 45 years old" matches the "I am" cue, but an age is
        # not a condition; without these the capture reads "45 years".
        "years", "year", "yrs", "yr", "months", "month", "weeks", "week",
        "days", "day", "hours", "hour", "aged",
        "every", "all", "most", "some", "any", "many", "few",
        "last", "next", "past", "recent", "recently", "lastly", "finally",
        "who", "what", "when", "where", "why", "how", "which", "with", "from",
        "this", "that", "these", "those", "have", "has", "had",
    }
)

#: Words that legitimately continue a condition name.
_HEALTH_ALLOWED_CONNECTORS = frozenset({"and", "or", "of", "in", "to", "with"})

#: A health value longer than this is almost certainly prose, not a condition.
_MAX_HEALTH_WORDS = 4


def _value_group(regex: re.Pattern[str], match: re.Match[str]) -> str | int | None:
    """The group holding the matched value.

    A pattern may have several alternatives ("6 years of service" versus
    "worked here 6 years"), each with its own named group, so the first group
    that actually participated is the value.
    """
    named = [name for name in regex.groupindex if name.startswith("value")]
    for name in named:
        if match.group(name) is not None:
            return name
    return 0 if not named else None


class _RegexCategoryDetector:
    """Small base: compile ``(regex, category, confidence)`` triples and scan.

    Subclasses may override :meth:`clean_value` to trim or reject a capture.
    That matters for the looser patterns, where a capture can run past the end
    of the thing being described.
    """

    name = "quasi"
    stage = Stage.DETERMINISTIC

    def patterns(self) -> tuple[tuple[re.Pattern[str], Category, float], ...]:
        raise NotImplementedError

    def clean_value(self, raw: str) -> str | None:
        """Return the value to report, or ``None`` to reject the match."""
        return raw

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        # Length-preserving normalisation: smart quotes and dashes are folded
        # in place, so every span below is a valid offset into ``text``.
        aligned = normalise_aligned(text)
        for regex, category, confidence in self.patterns():
            for match in regex.finditer(aligned):
                group = _value_group(regex, match)
                if group is None:
                    continue
                captured = match.group(group)
                raw = captured.strip()
                cleaned = self.clean_value(raw)
                if cleaned is None:
                    continue
                # Stripping leading whitespace moves the value start, so the
                # offset must move with it or the span truncates the value.
                offset = match.start(group) + (len(captured) - len(captured.lstrip()))
                if cleaned != raw:
                    index = aligned.find(cleaned, offset)
                    if index < 0:
                        continue
                    offset = index
                start, end = offset, offset + len(cleaned)
                yield Finding(
                    span=Span(start, end),
                    category=category,
                    detector=self.name,
                    stage=self.stage,
                    confidence=confidence,
                    value=text[start:end],
                    metadata=(("cue", cleaned.lower()),),
                )


#: Employment classifications and officer levels that sit immediately after the
#: "I am" cue. "I am APS 6 and that makes me one of very few" matched the cue
#: and was reported as a health disclosure, which put a ``[HEALTH DETAIL]``
#: label in front of a value that identifies the respondent just as well. The
#: level is reported as an implication, by
#: :mod:`pii_redact.detectors.implication`.
_CLASSIFICATION_ONLY_RE = re.compile(
    rf"(?i)^(?:{APS_LEVEL}|ses|so[0-3]|grade\s*\d{{1,2}}|band\s*[a-z]?\d{{1,2}}|"
    r"step\s*\d|class\s*\d|category\s*\d)$"
)


class JobTitleDetector(_RegexCategoryDetector):
    """Occupation, and pay grade or classification."""

    name = "job_title"

    def patterns(self):
        return (
            (_JOB_TITLE_LABEL_RE, Category.JOB_TITLE, 0.95),
            (_GRADE_RE, Category.JOB_TITLE, 0.9),
            (_JOB_TITLE_BARE_RE, Category.JOB_TITLE, 0.55),
        )


class TenureDetector(_RegexCategoryDetector):
    """Length of service."""

    name = "tenure"

    def patterns(self):
        return (
            (_TENURE_SINCE_RE, Category.TENURE, 0.75),
            (_TENURE_RE, Category.TENURE, 0.85),
        )


class TeamSizeDetector(_RegexCategoryDetector):
    """How small the respondent's group is, and uniqueness claims."""

    name = "team_size"

    def patterns(self):
        return ((_TEAM_SIZE_RE, Category.TEAM_SIZE, 0.85),)


class EmploymentStatusDetector(_RegexCategoryDetector):
    """Casual, temporary, part-time and similar."""

    name = "employment_status"

    def patterns(self):
        return (
            (_EMPLOYMENT_STATUS_RE, Category.EMPLOYMENT_STATUS, 0.85),
            # Lower confidence: a bare list item has no framing cue, so "casual"
            # could be describing something other than employment.
            (_EMPLOYMENT_STATUS_LIST_RE, Category.EMPLOYMENT_STATUS, 0.6),
        )


class HealthDetailDetector(_RegexCategoryDetector):
    """Health conditions mentioned about the respondent.

    Reported so the record can be scored and generalised. This is deliberately
    not masked by default: in a psychosocial risk survey, "I developed anxiety
    after the restructure" is the finding, not the identifier.
    """

    name = "health_detail"

    def patterns(self):
        return (
            (_MENTAL_HEALTH_RE, Category.HEALTH_DETAIL, 0.8),
            (_HEALTH_DETAIL_RE, Category.HEALTH_DETAIL, 0.7),
        )

    def clean_value(self, raw: str) -> str | None:
        """Trim the capture to the longest plausible condition name.

        "I am the only woman in my unit" matches the cue "I am", but the words
        after it are not a condition. The capture is therefore cut at the first
        word that cannot belong to a condition, and rejected if nothing usable
        is left.
        """
        words = raw.split()
        if not words:
            return None
        # A classification or officer level is an identifier, not a condition.
        if _CLASSIFICATION_ONLY_RE.match(raw):
            return None
        # A value beginning with an -ing gerund is prose, not a condition:
        # "I am looking elsewhere" matches the "I am" cue, but "looking
        # elsewhere" is what someone is doing, not what they are suffering from.
        if words[0].lower().endswith("ing"):
            return None
        # The same holds for a past-tense verb. "I have raised it twice now" and
        # "I have worked here for years" both match the "I have" cue, and the
        # word after it is an action, not a diagnosis. Conditions that do end in
        # -ed or -ing are the exceptions, so they are listed rather than
        # pattern-matched around.
        if words[0].lower().endswith("ed") and words[0].lower() not in _CONDITION_ED:
            return None
        kept: list[str] = []
        for index, word in enumerate(words):
            lowered = word.lower().strip(".,;:!?")
            if not lowered:
                break
            if (
                lowered in _NOT_HEALTH
                or lowered in _OCCUPATION_WORDS
                or lowered in _NOT_HEALTH_VERBS
                or lowered in _NOT_HEALTH_NOUNS
            ):
                break
            if index and lowered in _HEALTH_ALLOWED_CONNECTORS and kept:
                # "anxiety and depression": keep the connector and the word
                # after it, but never end the value on a connector.
                kept.append(word)
                continue
            kept.append(word)
            if len(kept) >= _MAX_HEALTH_WORDS:
                break
        # A trailing connector means the phrase was cut mid-way.
        while kept and kept[-1].lower() in _HEALTH_ALLOWED_CONNECTORS:
            kept.pop()
        if not kept:
            return None
        # "I am 34" matches the "I am" cue, but a bare number is an age, not a
        # condition. Reject it rather than reporting nonsense.
        if all(word.isdigit() for word in kept):
            return None
        # A band emitted by the generaliser ("25-34", "10+ years") reads as a
        # condition after a redaction pass. Rejecting bands keeps redaction
        # idempotent: without this, "I am 25-34" becomes "I am health_detail_x"
        # on the second pass.
        if is_banded(raw):
            return None
        return " ".join(kept)


#: The quasi-identifier detectors, in a stable order. The engine resolves
#: overlaps, so this order only affects tie-breaking.
QUASI_DETECTORS: tuple[type[_RegexCategoryDetector], ...] = (
    JobTitleDetector,
    TenureDetector,
    TeamSizeDetector,
    EmploymentStatusDetector,
    HealthDetailDetector,
)


def default_quasi_detectors() -> tuple[_RegexCategoryDetector, ...]:
    return tuple(cls() for cls in QUASI_DETECTORS)


__all__ = [
    "EmploymentStatusDetector",
    "HealthDetailDetector",
    "JobTitleDetector",
    "TeamSizeDetector",
    "TenureDetector",
    "default_quasi_detectors",
]
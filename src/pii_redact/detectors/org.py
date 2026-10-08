"""Organisation, internal-term and award detectors.

This is the module that addresses the "identifiable as an employer" axis
separately from "identifiable as a person". Two sources feed it:

* **Named entities**: an employer written out, e.g. "at Acme Pty Ltd",
  "employed by Reticule Health". These are redacted, because the name alone
  identifies the workplace even when no individual is named.
* **Lexicon hits**: internal systems, programs and agreements. A respondent who
  writes "the intranet ignored my report" has named their employer as surely as
  if they had written its name, without using a single proper noun.

Span offsets are always against the caller's original string. Matching runs on
:func:`normalise_aligned`, which substitutes punctuation in place without
changing length, so a match position is a valid original offset.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..lexicon import Lexicon, LexiconMatch, load_builtin_lexicon
from ..normalise import normalise_aligned, token_key
from ..types import Category, Finding, Span, Stage

# Entity suffixes. Kept specific: a bare capitalised word is far too common in
# survey free text to treat as a name, and this module must not redact
# "Safety", "Health" or "Support" when they appear mid-sentence.
_ENTITY_SUFFIX = (
    r"Pty(?:\s+Ltd)?|Ltd|Limited|Inc|Corp(?:oration)?|Company|Co|Group|Holdings|"
    r"Solutions|Technologies|Partners|Associates|"
    r"Foundation|Trust|Society|Association|Union|Council|"
    r"Academy|College|University|School|"
    r"Hospital|Medical\s+Centre|Clinic|Church|Charity|Institute|"
    # Sector words. Australian employers are very often named for what they do
    # ("Reticule Health", "Meridian Care"), and without these the proper noun
    # rule only catches them when employment framing happens to be present.
    r"Health|Healthcare|Care|Caring|Services|Resources|Industries|"
    r"Organisation|Organization"
)

# Employment framing. A proper noun after one of these is an employer far more
# often than anything else.
_EMPLOYMENT_CONTEXT = (
    r"work(?:ed|ing)?\s+(?:at|for)|employ(?:ed|ment)?\s+(?:by|at)|"
    r"join(?:ed|ing)|joined|at\s+my|workplace|employer|"
    r"my\s+company|our\s+company|contract(?:ed|or)?\s+(?:to|by)|"
    r"this\s+is\s+about|it\s+is\s+about|specifically\s+about|"
    r"employed\s+with|works?\s+with"
)

# The verbs above are also capitalised at the start of a sentence, so they must
# be stripped from the captured name: "Joined Northline Services" names
# Northline Services, not "Joined Northline Services".
_TRIGGER_VERBS = frozenset(
    {"joined", "join", "joining", "worked", "work", "working", "employed", "employment", "contracted"}
)

_ORG_NAME_RE = re.compile(
    # Capitalised sequence ending in an entity suffix: "Acme Pty Ltd",
    # "Reticule Health". Two to five tokens keeps this off ordinary sentence
    # starts, where capitalisation carries no information.
    rf"\b(?P<name>(?:[A-Z][\w&'’.\-]*\s+){{1,4}}(?:{_ENTITY_SUFFIX}))\b"
)
_EMPLOYMENT_ORG_RE = re.compile(
    rf"(?i:{_EMPLOYMENT_CONTEXT})\s+(?:the\s+)?"
    r"(?P<name>[A-Z][\w&'’.\-]*(?:\s+[A-Z][\w&'’.\-]*){0,3})"
)
_AU_ACN_RE = re.compile(
    r"\b(?:ACN|ABN)\s*:?\s*\d{1,4}(?:[ -]?\d{1,4}){1,4}\b", re.IGNORECASE
)

#: Full names that look organisational but describe a process, a role or a
#: body the respondent works through rather than their employer. Matched on the
#: folded key, so spelling and spacing do not matter.
_GENERIC_ORG_NAMES = frozenset(
    {
        "safetycommittee", "healthandsafetycommittee", "hsccommittee",
        "safetycommitteeteam", "wesafetycommittee",
        "humanresources", "hrteam", "hrdepartment", "peopleandculture",
        "thecompany", "ourcompany", "mycompany", "theorganisation",
        "ourorganisation", "myorganisation", "myemployer", "ouremployer",
        "thesite", "mysite", "oursite", "headoffice", "ouroffice", "theoffice",
        "mymanager", "mysupervisor", "myteam", "ourteam", "theteam",
        "seniormanagement", "uppermmanagement", "middlemanagement",
        "themanagementteam", "managementteam", "leadershipteam",
        "everyone", "everybody", "nobody", "noone", "noleone",
        "workplace", "theworkplace", "myworkplace", "ourworkplace",
        "incident", "theincident", "grievance", "thegrievance", "complaint",
        "award", "theaward", "union", "theunion", "safety", "health",
        "trainingteam", "recruitmentteam", "payrollteam", "helpdesk",
        "people", "staff", "employees", "workers", "customers", "client", "clients",
        "thepolice", "thecouncil", "thegovernment",
    }
)

#: Words that signal a generic phrase on their own. A name made *entirely* of
#: these is rejected; a name containing one is not, because "Reticule Health"
#: is specific even though "Health" is not.
_GENERIC_WORDS = frozenset(
    {
        "the", "our", "my", "a", "an",
        "safety", "health", "support", "management", "team", "staff",
        "workers", "employee", "employees", "workplace", "site", "office",
        "head", "senior", "upper", "middle", "leadership", "human",
        "resources", "incident", "grievance", "complaint", "award", "union",
        "company", "organisation", "organization", "group", "everyone",
        "nobody", "no", "one", "people", "committee", "department", "division",
        "training", "recruitment", "payroll", "help", "desk", "police",
        "council", "government", "system", "systems", "member",
        "customers", "client", "clients", "customer",
    }
)


#: Function words that cannot appear inside an organisation's name. Their
#: presence means the capture ran past the name and into the sentence around it.
#: "DO NOT BUY WORKPLACE COVER FROM THESE PEOPLE" matched the employment-context
#: rule on "workplace" and swallowed four capitalised tokens of a shouted
#: sentence; the sentence was lost and nothing was redacted that identifies
#: anybody.
_NAME_FUNCTION_WORDS = frozenset(
    {
        "from", "to", "for", "with", "and", "or", "of", "in", "at", "on",
        "by", "as", "this", "that", "these", "those", "their", "our", "your",
        "my", "but", "if", "so", "then", "than", "they", "them", "we", "us",
        "is", "are", "was", "were", "be", "been", "being", "am", "not",
    }
)


class OrganisationNameDetector:
    """Named employers, sites and internal bodies.

    Implements the :class:`~pii_redact.detectors.base.Detector` protocol
    without inheriting it, because it reconciles several regexes that overlap
    before emitting. Lexicon terms are *not* handled here; see
    :class:`LexiconTermDetector`, which the engine runs separately so the two
    sources cannot double-report the same span.
    """

    name = "organisation_name"
    stage = Stage.DETERMINISTIC

    def __init__(self, *, enable_named_entities: bool = True) -> None:
        self.enable_named_entities = enable_named_entities

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        aligned = normalise_aligned(text)
        claimed: list[tuple[int, int]] = []
        for regex, confidence, reason in (
            (_ORG_NAME_RE, 0.95, "entity suffix"),
            (_EMPLOYMENT_ORG_RE, 0.8, "employment context"),
        ):
            for match in regex.finditer(aligned):
                start = match.start("name")
                full = aligned[match.start("name") : match.end("name")]
                raw = _trim_name(full)
                if raw is None:
                    continue
                # _trim_name can drop a leading trigger verb, so both ends move.
                # Re-anchoring on the cleaned text is correct however much was
                # removed; computing the end from the start alone silently
                # shifted the span left and truncated the name.
                offset = full.find(raw)
                if offset < 0:
                    continue
                start += offset
                end = start + len(raw)
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                yield Finding(
                    span=Span(start, end),
                    category=Category.ORGANISATION_NAME,
                    detector=self.name,
                    stage=self.stage,
                    confidence=confidence,
                    value=text[start:end],
                    metadata=(("entity", raw), ("reason", reason)),
                )
        for match in _AU_ACN_RE.finditer(aligned):
            # Only an explicitly labelled ABN. A label plus nine digits is an
            # ACN, and the checksum detectors already handle both correctly --
            # claiming it here with the wrong category made a valid ACN lose to
            # an 11-character mislabel.
            if not re.match(r"(?i)abn", match.group()):
                continue
            yield Finding(
                span=Span(*match.span()),
                category=Category.ABN,
                detector=self.name,
                stage=self.stage,
                confidence=0.95,
                value=match.group(),
                metadata=(("reason", "employer identifier"),),
            )


class LexiconTermDetector:
    """Expose a :class:`Lexicon` as an engine detector."""

    name = "lexicon_term"
    stage = Stage.DETERMINISTIC

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon if lexicon is not None else Lexicon()

    def detect(self, text: str) -> Iterator[Finding]:
        for match in self.lexicon.match(text):
            yield _lexicon_finding(match, self.name)


class AwardCodeDetector:
    """Modern award identifiers such as ``MA000019``.

    A code like this identifies an employer to within a small industry set, so
    it is an attribution signal rather than ordinary text.
    """

    name = "award_code"
    stage = Stage.DETERMINISTIC

    _CODE_RE = re.compile(r"\bMA\d{6}\b", re.IGNORECASE)

    def detect(self, text: str) -> Iterator[Finding]:
        for match in self._CODE_RE.finditer(text):
            yield Finding(
                span=Span(*match.span()),
                category=Category.AWARD,
                detector=self.name,
                stage=self.stage,
                confidence=1.0,
                value=match.group(),
                metadata=(("code", match.group().upper()),),
            )


def _trim_name(raw: str) -> str | None:
    """Clean a captured name, or return ``None`` if it is not an organisation.

    Strips a leading trigger verb and rejects names that are entirely generic
    vocabulary, which is how "the Safety Committee" and "my Human Resources
    team" are kept out of the findings.
    """
    name = raw.strip(" \t.,;:")
    if not name:
        return None

    words = name.split()
    while len(words) > 1 and words[0].casefold() in _TRIGGER_VERBS:
        words = words[1:]
    name = " ".join(words).strip()
    if not name:
        return None

    folded = token_key(name)
    if folded in _GENERIC_ORG_NAMES:
        return None
    lowered = {w.casefold().strip(".,") for w in words}
    if lowered and lowered <= _GENERIC_WORDS:
        return None
    if lowered & _NAME_FUNCTION_WORDS:
        return None
    return name


def _lexicon_finding(match: LexiconMatch, detector: str) -> Finding:
    metadata: dict[str, object] = {"canonical": match.canonical, "label": match.label}
    if match.fuzzy:
        # A fuzzy hit is a guess about what the respondent meant. Report the
        # distance so a reviewer can judge it.
        metadata["fuzzy"] = True
        metadata["distance"] = match.distance
    return Finding(
        span=Span(*match.span),
        category=match.category,
        detector=detector,
        stage=Stage.DETERMINISTIC,
        confidence=0.85 if match.fuzzy else 1.0,
        value=match.matched_text,
        metadata=tuple(sorted(metadata.items())),
    )


def default_organisation_detectors(
    lexicon: Lexicon | None = None,
) -> tuple[OrganisationNameDetector, LexiconTermDetector, AwardCodeDetector]:
    """The detector trio, with the bundled lexicon when none is supplied."""
    resolved = lexicon if lexicon is not None else load_builtin_lexicon()
    return (
        OrganisationNameDetector(),
        LexiconTermDetector(resolved),
        AwardCodeDetector(),
    )


__all__ = [
    "AwardCodeDetector",
    "LexiconTermDetector",
    "OrganisationNameDetector",
    "default_organisation_detectors",
]
"""Implication rules: word sequences that identify a person or a workplace.

Every other detector in this package looks for a *thing* -- a phone number, a
name, a postcode. This one looks for a *sequence*, because in workplace free
text the identifying information is usually spread across a sentence and no
single span is a name.

    "I am the only woman on the night shift."
    "One of two apprentices on the yard team."
    "After the Kwinana depot closed I was the only one who did the handover."

None of those contains an address, a phone number or a date of birth, and a
reader inside the organisation can still name the person. Redacting only the
entities leaves the sentence fully identifying, which is why masking alone is
not enough here: the phrase itself has to go.

Two design choices worth stating:

**The whole clause is the span.** Removing "woman" from "the only woman on the
night shift" leaves "the only [NAME] on the night shift", which still
identifies. So the span covers the full matched clause.

**The default replacement is a label, not a deletion.** These clauses are often
the substance of the complaint -- "I am the only woman on the night shift" is
the finding a psychosocial risk survey exists to surface. Replacing it with
``[IMPLICATION]`` keeps the fact that a uniqueness claim was made, which an
analyst needs in order to weight severity, while removing the identifying
content. Deleting it would destroy the evidence too.

Rules are data. :data:`RULES` is a plain tuple, so a deployment can add patterns
for its own workforce without subclassing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..normalise import normalise_aligned
from ..types import Category, Finding, Span, Stage
from ..vocabulary import (
    APS_LEVEL,
    CLEARANCE_LEVEL,
    CLEARANCE_MARKER,
    COHORT_MARKER,
    CORPORATE_MARKER,
    DEPARTMENT_STRUCTURE,
    DEPARTMENT_STRUCTURE_QUALIFIED,
    GOVERNMENT_PROCESS,
    OCCUPATION_FAMILY,
    PUBLIC_SECTOR_ROLE,
    SENIOR_ROLE,
    SERVICE_FUNCTION,
    WORKPLACE_TYPE,
    WORKPLACE_TYPE_STRONG,
    alt,
)


def _typo(word: str) -> str:
    """A regex fragment matching ``word`` within one edit.

    Real respondents misspell closed vocabulary constantly -- "warrehouse",
    "storre", "stoke" -- and a rule that requires the exact spelling reports a
    named site as clean. One edit is the right budget: it covers the four typos
    people actually make (an extra letter, a missing letter, a substitution and
    a transposition) without opening the rule to anything further out.

    Every direction is generated, not just deletion. "warrehouse" is an
    insertion into "warehouse" and no deletion of the correct word produces it,
    so a deletion-only neighbourhood would have missed exactly the case that
    prompted this.

    Substitution and insertion are restricted to ``[a-z]``. Allowing digits or
    punctuation here would collide with the numeric and identifier rules, which
    consume the same text and have to keep winning.
    """
    alts = {word}
    for index in range(len(word)):
        alts.add(word[:index] + word[index + 1 :])             # dropped letter
        alts.add(word[:index] + "[a-z]" + word[index + 1 :])  # mistyped letter
        alts.add(word[:index] + "[a-z]" + word[index:])       # doubled letter
    for index in range(len(word) - 1):
        alts.add(word[:index] + word[index + 1] + word[index] + word[index + 2 :])
    return alt(*sorted(alts))


# --- Vocabulary the rules share ---------------------------------------------
_ROSTER = alt(
    "team", "unit", "ward", "crew", "shift", "roster", "rota", "department",
    "store", "depot", "yard", "class", "section", "branch", "site",
    "workshop", "floor", "line", "cell",
)
_SHIFT = alt(
    "shift", "roster", "rota", "day", "night", "weekend", "early", "late",
    "daytime", "overnight", "backfill",
)
_ROLE = alt(
    *(_typo(w) if len(w) >= 5 else w for w in (
        "nurse", "teacher", "carer", "cleaner", "driver", "technician", "officer",
        "cook", "chef", "barista", "clerk", "labourer", "electrician", "plumber",
        "warden", "midwife", "allocator", "caseworker", "auditor", "pharmacist",
    ))
)
_AGE_PHASE = alt("youngest", "oldest", "newest", "new", "senior", "junior", "only")
_SMALL_NUMBER = alt(
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    r"\d{1,2}",
)
_ARTICLE = alt("the", "my", "our", "his", "her", "its")
_UNIQUE_KIND = alt(
    "woman", "man", "person", "casual", "temp", "migrant", "indigenous",
    "apprentice", "parent", "student", "trainee", "graduate", "lone",
)
# A bare "i" is allowed: "I cover all three depots" has no copula.
_SELF = alt(r"i\s+am", r"i'?m", r"i\s+was", r"being\s+the",
            r"it\s+is\s+that\s+i\s+am", r"i\b")
_SITE_NOUN = alt(
    *(word if len(word) < 5 else _typo(word)
      for word in (
          "depot", "store", "site", "branch", "plant", "facility", "terminal",
          "warehouse", "hub", "campus", "centre", "center", "workshop", "yard",
          "floor", "building", "lab", "refinery", "kiln", "mine", "exchange",
      ))
)

#: Blocks the head modifier of a "the <modifier> <unit>" phrase when the
#: modifier names a service function rather than a place inside one employer.
#: See ``vocabulary.SERVICE_FUNCTION`` for why, and for what this costs.
_NOT_SERVICE_FUNCTION = rf"(?!(?:{SERVICE_FUNCTION})\b)"

#: Determiners, possessives and bare adjectives that capitalise a site *noun*
#: without naming a site. Without this guard "My site" matched: the capital
#: letter, one word, the optional middle words backtracking to zero, then
#: "site" as the site noun. The adjectives were added when the held-out probe
#: set showed "the Main branch" and "the Big store" redacting as named sites.
#: Directional words are deliberately absent: "North", "South" and "West" are
#: real Australian place names and a site genuinely called the North depot
#: would be missed.
_NOT_A_NAME = alt(
    "My", "The", "Our", "His", "Her", "Its", "This", "That", "A", "An",
    "Their", "Your", "Each", "Every", "New", "Old",
    "Main", "Big", "Small", "Large", "Central", "Upper", "Lower",
    "Inner", "Outer", "Front", "Rear",
)
#: Workplace events that genuinely narrow the employer and the timeframe.
#: "restructure", "reorg", "move" and "downgrade" were removed: they are common
#: across Australian employers and carry clinically useful causal context
#: ("after the restructure") without identifying anybody. The ones kept are rare
#: enough to be recognisable by a colleague.
_WORK_EVENT = alt(
    "closure", r"shut\s?down", _typo("merger"), rf"take-?{_typo('over')}",
    _typo("announcement"), _typo("inspection"), _typo("audit"),
    _typo("investigation"), _typo("reopening"), _typo("relocation"),
    "sale", _typo("commissioning"), _typo("handover"), _typo("demolition"),
    _typo("walkout"), _typo("lockout"),
)
#: Deliberately specific. "system", "process", "form" and "report" were tried
#: and removed: "my report was wrong" appears in almost every response and
#: identifies nothing. Named internal systems are handled by the lexicon, which
#: is the right place for a name that varies between employers.
_INTERNAL_THING = alt(
    "intranet", "portal", "dashboard", "database", "register", "workflow",
    "helpdesk", "timesheet", "rostering", "reporting tool", "incident system",
    "reporting system", "hr system", "claims system", "payroll system",
)
_PERSON_RELATION = alt(
    "manager", "supervisor", "boss", r"team\s*lead", "coordinator", "colleague",
    r"co-?worker", "mate", "partner", "friend", "brother", "sister", "ceo",
    "director", "gm", r"line\s*manager",
)

#: Words that must not sit between a determiner and the noun it modifies.
#:
#: They are the *verb* of the sentence, not an adjective describing a unit, and
#: letting the modifier run swallow one turned "my manager keeps rostering me"
#: into "Meanwhile [IMPLICATION] me on the graveyard shift": a whole clause
#: redacted to label a system that was never named. A person relation is
#: excluded for the same reason -- "my manager sent the team" is a sentence
#: about a person, not a unit called manager team.
_NOT_A_MODIFIER = (
    rf"(?!(?:{_PERSON_RELATION}|"
    r"keeps?|kept|sends?|sent|uses?|used|fills?|filled|opens?|opened|"
    r"updates?|updated|ignores?|ignored|reports?|reported|asks?|asked|"
    r"tells?|told|says?|said|needs?|needed|wants?|wanted|makes?|made|"
    r"runs?|ran|owns?|owned|manages?|managed)\b)"
)


@dataclass(frozen=True, slots=True)
class ImplicationRule:
    """One pattern that identifies an implicating sequence.

    ``pattern`` should expose a ``clause`` group when only part of the match is
    the identifying phrase; without one the whole match is redacted. A rule may
    also expose ``workplace`` for a named-site alternative.
    """

    name: str
    pattern: re.Pattern[str]
    #: Human-readable reason, surfaced in findings metadata for review.
    label: str
    confidence: float = 0.8
    #: "person" or "workplace": what the sequence gives away.
    implicates: str = "person"

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pattern": self.pattern.pattern,
            "label": self.label,
            "confidence": self.confidence,
            "implicates": self.implicates,
        }


def _rule(
    name: str,
    pattern: str,
    label: str,
    *,
    confidence: float = 0.8,
    implicates: str = "person",
) -> ImplicationRule:
    return ImplicationRule(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE),
        label=label,
        confidence=confidence,
        implicates=implicates,
    )


#: The shipped rule set, grouped by what the sequence gives away.
RULES: tuple[ImplicationRule, ...] = (
    # -- uniqueness: "only one like me" is close to self-identification --------
    _rule(
        "only_one_on_team",
        rf"(?P<clause>(?:the\s+)?only\s+(?:\w+\s+){{0,2}}?{_UNIQUE_KIND}"
        rf"(?:\s+(?:{_ROLE}))?\s+(?:in|on|of|at)\s+"
        rf"(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}{_ROSTER})",
        "claims to be the only person of their kind in a named group",
    ),
    _rule(
        "one_of_n",
        rf"(?P<clause>one\s+of\s+(?:just\s+|only\s+)?{_SMALL_NUMBER}\s+"
        rf"(?:\w+\s+){{0,2}}(?:on|in|of|at)\s+(?:{_ARTICLE}\s+)?{_ROSTER})",
        "states membership of a very small named group",
    ),
    _rule(
        "sole_holder",
        rf"(?P<clause>(?:i\s+am|i'?m|the)\s+(?:sole|only|lone)\s+"
        rf"(?:\w+\s+){{0,2}}?(?:{_ROSTER}|{_ROLE}))",
        "describes sole responsibility for a group or role",
    ),
    _rule(
        "no_one_else",
        r"(?P<clause>(?:no\s+one\s+else|not\s+anyone\s+else|nobody\s+else)"
    r"(?:\s+\w+){0,3})",
        "claims no one else shares this duty",
    ),
    _rule(
        "unique_attribute",
        # The copula is optional: "I am the youngest here" and "the youngest on
        # the team here" are both how respondents put it. The group noun after
        # the preposition is consumed when present, so the clause ends on a
        # noun: leaving " team" outside the span produced "I am [IMPLICATION]
        # team".
        rf"(?P<clause>(?:{_SELF}\s+)?the\s+{_AGE_PHASE}\s+(?:\w+\s+){{0,2}}"
        r"(?:here|(?:on|at|in)\s+the(?:\s+\w+)?))",
        "identifies by comparative age or seniority within a group",
    ),
    _rule(
        "only_one_does",
        r"(?P<clause>(?:the\s+)?only\s+one\s+(?:who\s+|that\s+|to\s+)?"
    r"(?:\w+\s+){0,3}?(?:does|do|did|handles|covers|operates|runs|starts))",
        "claims a unique duty",
    ),
    # -- a small named group plus a shift or roster -----------------------------
    _rule(
        "shift_group",
        # The lookbehind stops "5 people on nights" matching inside the
        # generalised band "2-5 people", which is how a second redaction pass
        # over already-redacted text ended up clobbering half a band.
        rf"(?P<clause>(?:the\s+)?(?<![\d-])(?:\d{{1,2}}|{_SMALL_NUMBER})\s+"
        r"(?:people|staff|workers|employees|members|of\s+us)\s+"
        rf"(?:on|across|covering)\s+(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}{_SHIFT})",
        "describes the size of a named shift group",
        confidence=0.75,
    ),
    _rule(
        "covers_sites",
        rf"(?P<clause>{_SELF}\s+(?:the\s+)?(?:sole|only|entire)?\s*\w*\s*"
        r"(?:cover|covering|covers)\s+(?:all\s+)?"
        rf"(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}?"
        r"(?:sites|depots|stores|branches|locations|facilities))",
        "describes sole coverage of several workplaces",
        implicates="workplace",
    ),
    _rule(
        "unique_training",
        r"(?P<clause>(?:the\s+)?only\s+one\s+(?:here|on\s+the\s+\w+|at\s+the\s+\w+)\s+"
    r"(?:who|that|to)\s+(?:\w+\s+){0,3}?(?:trained|certified|qualified|ticketed))",
        "states a unique qualification in a group",
    ),
    # -- workplace events that pin a site in time -------------------------------
    _rule(
        "post_event",
        rf"(?P<clause>(?:after|since|when|following|post)\s+(?:the|our)\s+"
        rf"(?:\w+\s+){{0,2}}{_WORK_EVENT})",
        "anchors the response to a workplace event",
        confidence=0.75,
        implicates="workplace",
    ),
    _rule(
        "named_site",
        # (?-i:[A-Z]) is required: the pattern is compiled with IGNORECASE, and
        # a plain [A-Z] there would match "all" and "my", turning every site
        # noun into a named site.
        rf"(?P<workplace>(?!(?i:{_NOT_A_NAME})\s)"
        rf"(?-i:[A-Z])[A-Za-z'’-]+(?:\s+[A-Za-z'’-]+){{0,2}}\s+{_SITE_NOUN}\b)",
        "names a workplace site",
        confidence=0.7,
        implicates="workplace",
    ),
    _rule(
        "site_change",
        r"(?P<clause>(?:when|after|since)\s+(?:they|we|the\s+company|it|the\s+\w+)\s+"
        r"(?:was|were|is|are|got|got\s+)?\s*"
        r"(?:closed|moved|sold|merged|renamed|demolished|reopened|shut\s?down))",
        "describes a change to a specific site",
        implicates="workplace",
    ),
    _rule(
        "named_site_lowercase",
        rf"(?P<workplace>(?:work\s+out\s+of|based\s+(?:in|at)|site\s+is|"
        rf"located\s+(?:at|in)|posted\s+to)\s+(?:the\s+|our\s+|my\s+)?"
        rf"[a-z][a-z'\u2019-]+(?:\s+[a-z][a-z'\u2019-]+){{0,2}}\s+{_SITE_NOUN}\b)",
        "names a workplace site without capitalisation",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "internal_thing",
        # The leading \b is load-bearing. Without it the determiner matches
        # *inside* a word: "Your portal" matched on "our portal" starting one
        # character in, which replaced the middle of the word and left the
        # reader with "Y[IMPLICATION]". The possessive forms people actually
        # use are all listed, for the same reason.
        rf"(?P<clause>\b(?:the|our|my|your|his|her|its|their)\b\s+"
        rf"(?:{_NOT_A_MODIFIER}\w+\s+){{0,2}}?{_INTERNAL_THING})",
        "describes an internal system or process",
        confidence=0.65,
        implicates="workplace",
    ),
    # -- second-hand naming: identifies a third party -----------------------------
    _rule(
        "names_third_party",
        # A role word alone identifies nobody: "my supervisor is supportive"
        # appears in almost every positive response. A role *plus a name*
        # identifies one person, so a capitalised token is required.
        rf"(?P<clause>(?:my|our|his|her|their)\s+(?:\w+\s+){{0,2}}"
        rf"{_PERSON_RELATION}\s+(?P<who>(?-i:[A-Z])[A-Za-z'’\-]+))",
        "names a colleague by role and name",
        confidence=0.9,
    ),
    _rule(
        "reported_by",
        r"(?P<clause>(?:told|said|reported|mentioned)\s+(?:it\s+)?(?:to|by)\s*"
        rf"(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}{_PERSON_RELATION}"
        r"(?:\s+(?P<who>(?-i:[A-Z])[A-Za-z'’\-]+))?)",
        "attributes a statement to a specific person",
        confidence=0.7,
    ),
    # -- structural uniqueness from a classification ------------------------------
    _rule(
        "classification",
        r"(?P<clause>(?:pay\s*)?(?:grade|band|step|category|classification)\s*[:#]?\s*"
        r"[a-z]?\d{1,2}\b(?:\s+(?:increment|point|salary|rate))?)",
        "states a pay grade or classification, unique within an employer",
        confidence=0.85,
    ),

    # -- Australian public sector: workplace type -----------------------------
    _rule(
        "department_structure",
        # Modifiers are allowed between the determiner and the structure word:
        # "the digital services branch", "the enforcement directorate". The
        # qualified nouns require one, so that "my team has always had my back"
        # is not treated as an identifier.
        #
        # Modifier words must be at least three characters. Otherwise a
        # character-spaced string such as "the a b c team" matches, which turns
        # obfuscation text into a false positive.
        #
        # Two modifiers at most. Three let the run swallow the verb in a
        # sentence like "what my manager sent the team", which redacted a whole
        # clause to label a unit that was never named.
        rf"(?P<clause>\b(?:my|our|the|his|her)\b\s+(?:"
        rf"(?:\w+\s+){{0,3}}?{DEPARTMENT_STRUCTURE}"
        rf"|{_NOT_SERVICE_FUNCTION}{_NOT_A_MODIFIER}(?:\w{{3,}}\s+){{1,2}}"
        rf"{DEPARTMENT_STRUCTURE_QUALIFIED})\b"
        rf"(?:\s+(?:of|for)\s+(?:the\s+)?\w+){{0,2}}?)",
        "identifies the respondent by their department or branch",
        confidence=0.7,
        implicates="workplace",
    ),
    _rule(
        "unit_by_preposition",
        # "I work in Services division", "I sit within the integrity unit".
        # A preposition supplies the framing, so no determiner is required.
        # Each branch has its own determiner. A single optional determiner in
        # front of both let the article "a" serve as the required qualifier, so
        # "in a team of nine" matched as an administrative unit.
        rf"(?P<clause>\b(?:in|at|within|across|into)\s+(?:"
        rf"(?:the\s+|a\s+|an\s+|my\s+|our\s+)?(?:\w+\s+){{0,3}}?{DEPARTMENT_STRUCTURE}"
        rf"|(?:the\s+|a\s+|an\s+|my\s+|our\s+){_NOT_SERVICE_FUNCTION}"
        rf"{_NOT_A_MODIFIER}(?:\w{{3,}}\s+){{1,2}}{DEPARTMENT_STRUCTURE_QUALIFIED})\b)",
        "identifies an administrative unit by preposition",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "government_process",
        rf"(?P<clause>\b(?:the|our|my|a|an)\b\s+(?:\w+\s+){{0,2}}?{GOVERNMENT_PROCESS})",
        "describes an administrative act that only happens in government",
        confidence=0.7,
        implicates="workplace",
    ),
    _rule(
        "workplace_type",
        # A possessive determiner for the general list, or a bare "the" only
        # for nouns that are unambiguously a workplace. "our store" is a
        # workplace; "declined at the store" is a transaction.
        rf"(?P<clause>(?:\b(?:our|my|their|its)\b\s+{WORKPLACE_TYPE}\b"
        rf"|\b(?:the|a|an)\b\s+{WORKPLACE_TYPE_STRONG}\b)"
        rf"(?:\s+(?:in|at|near)\s+(?:the\s+)?\w+){{0,2}}?)",
        "identifies the type of workplace without naming it",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "senior_role",
        rf"(?P<clause>\b(?:i\s+am|i'?m|as)\b\s+(?:the\s+|a\s+|an\s+)?{SENIOR_ROLE}\b"
        rf"(?:\s+of\s+(?:the\s+)?\w+){{0,2}}?)",
        "names a role that has very few holders in any organisation",
        confidence=0.8,
    ),
    # -- occupation type -------------------------------------------------------
    _rule(
        "occupation_family",
        rf"(?P<clause>\b(?:i\s+am|i'?m|as|works?\s+as\s+an?|my\s+(?:role|title|job|"
        rf"position)\s+is)\b\s+(?:a\s+|an\s+|the\s+)?{OCCUPATION_FAMILY}\b)",
        "names a specific occupation family",
        confidence=0.7,
    ),
    _rule(
        "occupation_with_structure",
        # A specific occupation plus an administrative unit is close to unique.
        rf"(?P<clause>(?:a|an|the)\s+{PUBLIC_SECTOR_ROLE}\s+(?:in|at|with)\s+"
        rf"(?:the\s+|my\s+)?(?:\w+\s+){{0,2}}{DEPARTMENT_STRUCTURE})",
        "an occupation inside a named unit identifies one person",
        confidence=0.85,
    ),
    _rule(
        "aps_level",
        rf"(?P<clause>(?:a|an|as\s+an?)\s+{APS_LEVEL}\b"
        rf"(?:\s+(?:in|at|of)\s+(?:the\s+|my\s+)?(?:\w+\s+){{0,2}}"
        rf"{DEPARTMENT_STRUCTURE})?)",
        "states an APS officer level, usually unique within a branch",
        confidence=0.85,
    ),
    _rule(
        "aps_level_alone",
        # The trailing \b is what keeps "APS 61" from matching as "APS 6", and
        # the optional punctuation is what lets a level inside a comma list
        # match at all: "I am a case officer, APS 6, in the complaints branch"
        # needs \s+ to fail on the comma before it can be caught at all.
        rf"(?P<clause>{APS_LEVEL}\b(?:[,.;:]\s*|\s+(?:officer|role|classification|position)?))",
        "states an APS officer level",
        confidence=0.7,
    ),
    _rule(
        "seniority_in_unit",
        # The connectors deliberately do not consume the preposition: the unit
        # tail needs it, and eating it here truncated the match to
        # "one of three of us in the grants".
        rf"(?P<clause>(?:one\s+of|there\s+are|only)\s+(?:just\s+|only\s+)?"
        rf"{_SMALL_NUMBER}\s+"
        rf"(?:of\s+us\s+|of\s+them\s+|people\s+|of\s+|left\s+|remaining\s+)*"
        rf"(?:\w+\s+){{0,2}}?(?:{APS_LEVEL}s?\b|(?:{OCCUPATION_FAMILY})s?\b"
        rf"|senior|junior|acting|permanent|provisional|others?|ones?|us)?"
        rf"(?:\s+(?:in|of|on)\s+(?:the\s+|my\s+)?(?:"
        rf"(?:\w+\s+){{0,3}}?{DEPARTMENT_STRUCTURE}"
        rf"|(?:\w+\s+){{0,2}}{DEPARTMENT_STRUCTURE_QUALIFIED}"
        rf"|(?:\w+\s+){{0,2}}{_ROSTER}))?)",
        "states seniority within a small named unit",
        confidence=0.85,
    ),
    # -- cohort, programme and clearance ---------------------------------------
    _rule(
        "cohort_marker",
        rf"(?P<clause>\b(?:i\s+am|i'?m|came\s+through|am)\b\s+"
        rf"(?:part\s+of\s+|through\s+|in\s+|on\s+)?{COHORT_MARKER}\b)",
        "places the respondent in a small named cohort",
        confidence=0.75,
    ),
    _rule(
        "clearance",
        rf"(?P<clause>\b(?:i\s+(?:am|'m|hold|have)|my|(?:to\s+)?get|hold)\b\s+"
        rf"(?:a\s+|an\s+|the\s+)?{CLEARANCE_MARKER}\s*(?:{CLEARANCE_LEVEL}\b)?)",
        "names a security clearance level",
        confidence=0.9,
    ),
    _rule(
        "corporate_marker",
        rf"(?P<clause>\b(?:our|my|the|a|an)\b\s+(?:\w+\s+){{0,2}}?{CORPORATE_MARKER})",
        "identifies a corporate setting",
        confidence=0.65,
        implicates="workplace",
    ),
)


class ImplicationDetector:
    """Finds sequences of words that identify a person or a workplace.

    Implements the :class:`~pii_redact.detectors.base.Detector` protocol.
    """

    name = "implication"
    stage = Stage.DETERMINISTIC

    def __init__(
        self,
        rules: Sequence[ImplicationRule] | None = None,
        *,
        enable_person: bool = True,
        enable_workplace: bool = True,
    ) -> None:
        self.rules = tuple(rules) if rules is not None else RULES
        self.enable_person = enable_person
        self.enable_workplace = enable_workplace

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        # Length preserving, so every span below indexes the original string.
        aligned = normalise_aligned(text)
        claimed: list[tuple[int, int]] = []
        # Longest clauses first: a short rule firing inside a longer one must not
        # win, or the redacted span stops being a whole clause.
        for rule in sorted(self.rules, key=lambda r: -r.confidence):
            if rule.implicates == "person" and not self.enable_person:
                continue
            if rule.implicates == "workplace" and not self.enable_workplace:
                continue
            for match in rule.pattern.finditer(aligned):
                start, end = self._span_for(match, aligned)
                if start is None or end is None:
                    continue
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                yield Finding(
                    span=Span(start, end),
                    category=Category.IMPLICATION,
                    detector=self.name,
                    stage=self.stage,
                    confidence=rule.confidence,
                    value=text[start:end],
                    metadata=(
                        ("rule", rule.name),
                        ("label", rule.label),
                        ("implicates", rule.implicates),
                    ),
                )

    @staticmethod
    def _span_for(match: re.Match[str], text: str) -> tuple[int | None, int | None]:
        """Prefer an explicit group, else use the whole match.

        Trailing whitespace is trimmed. A pattern whose tail is entirely
        optional -- "I hold a security clearance" -- otherwise swallows the
        space that belongs to the sentence, and replacing it deletes the word
        boundary.
        """
        for name in ("clause", "workplace"):
            if name in match.re.groupindex and match.group(name) is not None:
                start, end = match.span(name)
                break
        else:
            start, end = match.span()
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end


def default_implication_detector() -> ImplicationDetector:
    return ImplicationDetector()


__all__ = [
    "RULES",
    "ImplicationDetector",
    "ImplicationRule",
    "alt",
    "default_implication_detector",
]
"""Residual disclosure control: measure what is left, then act on the measurement.

Rules detect. They cannot *prove absence*. On a recurring ingest the question is
not "did every rule fire" but "what would a colleague still be able to work out
from this text" -- and the answer depends on the combination of what remains,
not on whether any single detector matched.

This module inverts the problem. Instead of trying to recognise every way a
workplace can be implicated, it asks three bounded questions about whatever text
survived redaction:

1. **What classes of information remain?** Person attributes, workplace
   attributes, temporal, numeric, organisational, narrative. The space of
   ways to implicate a workplace is unbounded; the space of disclosure classes
   is not.
2. **Is the combination unique?** Each attribute carries an estimated
   selectivity -- roughly how many people in an organisation share it, as a
   1-in-N figure within that organisation. Dividing the workforce size by the
   product gives the expected number of people the combination matches. Below
   a threshold, the record is identifying even though no rule fired.
3. **What do we do about it?** Pass, generalise the residual high-selectivity
   literals away, or withhold the record entirely.

The important consequence: a detector miss does not have to be detected by
another detector. A missed identifier survives into the output as an unusual
specific value, and this module sees it as an attribute with very high
selectivity. That is what makes the module robust on data it has never seen,
which is the actual requirement for a recurring ingest.

Every number here is an estimate with a stated basis, and every estimate is in
one place (:data:`SELECTIVITY`, :data:`DisclosurePolicy`) so it can be
calibrated against real review outcomes.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .masking import PLACEHOLDER_PATTERN
from .vocabulary import OCCUPATION_FAMILY


class ResidualKind(str, enum.Enum):
    """What class of information a surviving value discloses."""

    #: Age band, tenure, occupation, employment status, team size.
    PERSON = "person"
    #: Unit type, site type, shift pattern, grade, clearance, cohort.
    WORKPLACE = "workplace"
    #: A date, a year, a relative period.
    TEMPORAL = "temporal"
    #: A count, a duration, a long digit run.
    NUMERIC = "numeric"
    #: A surviving proper noun that may be an organisation or a person.
    ORGANISATION = "organisation"
    #: A described incident, grievance or health event.
    NARRATIVE = "narrative"

    def __str__(self) -> str:
        return self.value


#: Estimated 1-in-N share for one attribute, before any adjustment.
#:
#: These are deliberately pessimistic. A number that is too selective will
#: escalate records to review that would have been fine; a number that is too
#: permissive will let identifying records through. Erring toward escalation is
#: the safe direction, and review is cheap next to a disclosure.
SELECTIVITY: dict[ResidualKind, float] = {
    ResidualKind.PERSON: 8.0,
    ResidualKind.WORKPLACE: 12.0,
    ResidualKind.TEMPORAL: 40.0,
    ResidualKind.NUMERIC: 25.0,
    ResidualKind.ORGANISATION: 60.0,
    ResidualKind.NARRATIVE: 200.0,
}

# --- Recognising what survived ----------------------------------------------
# Applied to the *redacted* text. A run of digits that long appearing after
# redaction is, with very high probability, something the detectors missed.
_LONG_DIGITS = re.compile(r"(?<![\w-])\d{6,}(?![\w-])")
# Grouped digits: an identifier the checksum rejected survives as "123 456 783",
# which a single long-run pattern never sees.
_GROUPED_DIGITS = re.compile(r"(?<![\w-])\d{2,4}(?:[ -]\d{2,4}){1,4}(?![\w-])")
# A case or project reference: "GW-2026-4471", "PS-2291".
_REFERENCE = re.compile(r"(?<![\w-])[A-Z]{2,4}-?\d{3,}(?:-\d{2,})*(?![\w-])")
# A system named after something: "the Meridien platform", "our Corvus register".
# The lowercase noun is what makes this safe; a lone capitalised word is not.
_NAMED_SYSTEM = re.compile(
    r"(?<![.!?]\s)(?P<who>[A-Z][A-Za-z'\u2019-]{2,})\s+"
    r"(?:platform|system|portal|register|workflow|dashboard|tool|suite|"
    r"network|intranet|repository|spreadsheet|helpdesk|ticketing)\b"
)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_EXACT_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_BAND = re.compile(r"\b\d{2,3}\s*-\s*\d{2,3}\b|\b\d{2,3}\+|"
                   r"\bunder\s+\d{1,3}\b|\b\d{2,3}\s+years\b|\b\d{2,3}\+?\s+people\b")
_SMALL_COUNT = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,3})"
                          r"\s+(?:of\s+us\s+)?(?:incidents?|complaints?|reports?|"
                          r"people|staff|workers|members)\b", re.I)
# "Not after a sentence terminator", not "at the start of the string":
# capitalisation mid-sentence is the evidence, capitalisation at the start of a
# sentence is not.
_PROPER_NOUN = re.compile(r"(?<![.!?]\s)\b[A-Z][A-Za-z'’-]+"
                          r"(?:\s+[A-Z][A-Za-z'’-]+){1,3}\b")
_AGE = re.compile(r"\b\d{1,3}\s*(?:years?\s*old|yrs?)\b", re.I)
_TENURE = re.compile(r"\b\d{1,2}\+?\s*(?:years?|yrs?)\b", re.I)
_OCCUPATION = re.compile(rf"(?<![.!?]\s)\b(?:{OCCUPATION_FAMILY})\b", re.IGNORECASE)
_NARRATIVE = re.compile(
    r"(?i)\b(?:bully\w*|harass\w*|abuse\w*|aggress\w*|threat\w*|humiliat\w*|"
    r"grievance|complaint|incident|retaliat\w*|excluded|ostraci[sz]ed|"
    r"anxiety|depression|ptsd|psychological|burnout|unsafe|danger\w*|"
    r"injur\w*|discriminat\w*|bullied|"
    # Behaviour rather than vocabulary. A respondent describes what was done to
    # them, not what it is called, and this layer only ever sees the *output* of
    # redaction, where the vocabulary has often been replaced by a marker.
    r"singl(?:ed|ing)\s+(?:me\s+)?out|pick(?:ed|ing)\s+(?:me\s+)?on|"
    r"target(?:ed|ing|s)?|"
    r"shout(?:ed|ing)|humiliat\w*|embarrass\w*|undermin\w*|"
    r"intimidat\w*|mock(?:ed|ing)|barrage|excluded\s+me|left\s+out|"
    r"set\s+me\s+up|put\s+me\s+on|against\s+me|"
    r"did\s+not\s+listen|would\s+not\s+listen|nobody\s+(?:cared|acted)|"
    r"nothing\s+happened|no\s+action)\b"
)
#: A redaction marker this module itself emits. Measuring it as surviving
#: information is the same mistake as re-detecting it: "[HEALTH DETAIL]" is two
#: capitalised words and would otherwise read as an organisation.
_PLACEHOLDER = re.compile(PLACEHOLDER_PATTERN)

#: Function words that capitalise without naming anything. "My TFN" is a label,
#: not a proper noun.
_NOT_A_NAME = frozenset(
    {
        "My", "Our", "The", "This", "That", "These", "Those", "His", "Her",
        "Its", "Their", "Your", "A", "An", "I", "We", "They", "He", "She",
        "It", "No", "Not", "Yes", "But", "And", "So", "When", "After",
        "Before", "Since", "If", "Then", "There", "Here", "Now", "One",
        "Two", "Three", "Each", "Every", "New", "Old", "Both", "All",
        "Some", "Any", "Most", "Many", "Few", "More", "Less", "Just",
        "Still", "Very", "Really", "Quite", "Also", "Even", "Only",
        "TFN", "ABN", "ACN", "DOB", "BSB", "EAP", "PA", "ID", "HR", "APS",
    }
)


@dataclass(frozen=True, slots=True)
class ResidualAttribute:
    """One piece of information that survived redaction."""

    kind: ResidualKind
    value: str
    #: Estimated 1-in-N share. Larger means more selective, so more identifying.
    selectivity: float
    span: tuple[int, int] | None = None

    @property
    def is_high_selectivity(self) -> bool:
        """True when few enough people share this that it narrows the field."""
        return self.selectivity >= HIGH_SELECTIVITY

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value[:60],
            "selectivity": round(self.selectivity, 2),
            "span": list(self.span) if self.span else None,
        }


#: An attribute shared by fewer than this many people is treated as
#: identifying on its own.
HIGH_SELECTIVITY = 1000.0

#: Selectivity applied to an organisation-like proper noun that survived. Two
#: consecutive capitalised words that a detector did not claim are, for a
#: respondent who did not name their employer, worth treating as a name.
_SURVIVING_PROPER_NOUN_SELECTIVITY = 2000.0


@dataclass(frozen=True)
class ResidualProfile:
    """Everything a redacted record still discloses."""

    attributes: tuple[ResidualAttribute, ...] = ()
    text: str = ""

    @property
    def selectivity_product(self) -> float:
        """Product of the per-attribute selectivities, as a 1-in-N figure.

        This is not a cohort size and must not be compared with one: it only
        means something once divided by a population size, which is why the
        arithmetic lives on :class:`DisclosurePolicy`. An empty product is 1.0,
        meaning "nothing left to narrow it" -- every member of the population
        matches -- rather than a single person.
        """
        product = 1.0
        for attribute in self.attributes:
            product *= attribute.selectivity
        return product

    @property
    def classes(self) -> frozenset[ResidualKind]:
        return frozenset(a.kind for a in self.attributes)

    @property
    def high_selectivity(self) -> tuple[ResidualAttribute, ...]:
        return tuple(a for a in self.attributes if a.is_high_selectivity)

    @property
    def identifying(self) -> bool:
        """True when a surviving value is close to unique on its own.

        Distinct from the cohort test: this looks for a near-unique literal and
        says nothing about combinations.
        """
        return any(a.is_high_selectivity for a in self.attributes)

    @property
    def has_reducible(self) -> bool:
        """Whether :func:`generalise_residual` would actually remove something.

        It has to be asked of :attr:`high_selectivity` and not of every
        attribute: generalising replaces only the near-unique literals, so a
        record whose residual risk is one age band and one narrative has nothing
        to redact even though both attributes carry a span. Reporting
        GENERALISE there would claim an action that never happened.
        """
        return any(
            a.span is not None and a.kind is not ResidualKind.NARRATIVE
            for a in self.high_selectivity
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "selectivity_product": round(self.selectivity_product, 1),
            "identifying": self.identifying,
            "classes": sorted(k.value for k in self.classes),
            "attributes": [a.as_dict() for a in self.attributes],
        }


class DisclosureAction(str, enum.Enum):
    """What to do with a record whose residual risk is too high."""

    #: Leave it. The surviving combination is plausibly shared.
    PASS = "pass"
    #: Replace the residual high-selectivity literals with category labels.
    GENERALISE = "generalise"
    #: Withhold the narrative. For records whose residual risk cannot be
    #: reduced without destroying the finding.
    SUPPRESS = "suppress"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DisclosurePolicy:
    """The thresholds that decide escalation.

    ``population`` is the one number a deployment must supply: how many people
    work in the workplace. Every selectivity is a share *of that population*, so
    the same text is identifying in a 30-person depot and unremarkable in a
    4000-person agency, and only this figure tells the two apart. The default
    is the workforce the table in :data:`SELECTIVITY` was calibrated against,
    which makes the defaults meaningful but almost never right.
    """

    #: Estimated workforce size of the workplace a record came from.
    population: int = 1000
    #: A record matching fewer people than this is treated as identifying.
    min_cohort: float = 5.0
    #: Scale factor on the expected matches, to acknowledge that attributes in
    #: a workplace are not independent. Left at 1.0, which is an honest default
    #: rather than a fudge: the independence assumption has no single defensible
    #: sign. Age, tenure and grade are positively correlated, so their joint
    #: share is *smaller* than the product, but grievance narratives cluster
    #: with everything else about a person's circumstances, so theirs is larger.
    #: Multiplies the expected matches, so raising it loosens the test. Set it
    #: per deployment against observed review outcomes.
    attribute_independence: float = 1.0
    #: 1-in-N share, *within the population*, for a surviving incident
    #: narrative. The trap here is the response rate: a grievance carried by one
    #: in twenty respondents is carried by roughly one in seven of the workforce
    #: when a third of it replied, so the share within the population is several
    #: times larger than the share among responses. The default reflects a
    #: general workplace survey; lower it for a narrowly scoped one, where the
    #: narrative really is close to unique.
    narrative_selectivity: float = SELECTIVITY[ResidualKind.NARRATIVE]

    def expected_matches(self, profile: ResidualProfile) -> float:
        """Expected number of people in the workplace matching every attribute.

        Each attribute is a 1-in-N share, so multiplying them gives the
        combined 1-in-N and the population divided by it gives a headcount.
        The result is usually fractional: 0.4 means "this combination is
        expected to match about two people in five", which for a cohort
        threshold is a single person, because a fractional expectation is not a
        fractional person.

        Attributes in a workplace are correlated, so this overestimates the
        matching group; ``attribute_independence`` corrects for that, and the
        default of 1.0 leaves the estimate optimistic on purpose -- over-escalating
        a review queue is cheaper than publishing a singular record.
        """
        return self.population / profile.selectivity_product * self.attribute_independence


#: What a withheld record becomes. Deliberately not empty: the fact that a
#: sensitive response exists is itself something the analysis needs.
WITHHELD_TEXT = "[SENSITIVE RESPONSE WITHHELD FOR DISCLOSURE CONTROL]"


@dataclass(frozen=True)
class DisclosureDecision:
    """The outcome of residual risk assessment, with its reasoning."""

    action: DisclosureAction
    profile: ResidualProfile
    #: Expected matches that justified the action. Deliberately the figure from
    #: *before* any remediation: a review queue needs to know how close the
    #: record came to identifying, which is invisible once it has been cleaned.
    cohort: float
    reason: str
    #: Attributes that drove an escalation, for the review queue.
    drivers: tuple[str, ...] = ()
    #: Expected matches after the action was applied. None when nothing changed.
    post_cohort: float | None = None

    @property
    def escalated(self) -> bool:
        return self.action is not DisclosureAction.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "expected_matches": round(self.cohort, 4),
            "expected_matches_after": (
                None if self.post_cohort is None else round(self.post_cohort, 4)
            ),
            "reason": self.reason,
            "drivers": list(self.drivers),
            "residual": self.profile.as_dict(),
        }


def measure(text: str, narrative_selectivity: float | None = None) -> ResidualProfile:
    """Inventory what survives in already-redacted ``text``.

    Run this on the *output* of a redaction pass, not the input. Anything with
    high selectivity here is either legitimate analytical content or a detector
    miss, and the two are indistinguishable from the text alone -- which is why
    this is a measurement and not a detector.
    """
    if not text or not text.strip():
        return ResidualProfile()

    attributes: list[ResidualAttribute] = []
    claimed: list[tuple[int, int]] = []
    markers = [m.span() for m in _PLACEHOLDER.finditer(text)]

    def add(kind: ResidualKind, match: re.Match[str], selectivity: float,
            group: int | str = 0) -> None:
        span = match.span(group)
        # Anything inside a redaction marker is this module's own output, not
        # residual disclosure.
        if any(span[0] < end and start < span[1] for start, end in markers):
            return
        if any(span[0] < end and start < span[1] for start, end in claimed):
            return
        claimed.append(span)
        attributes.append(
            ResidualAttribute(kind, match.group(group), selectivity, span)
        )

    # Digit runs and grouped digits surviving redaction are almost certainly
    # missed identifiers, and by far the most selective thing a record can
    # contain. Grouped digits matter because a checksum failure leaves the value
    # looking like "123 456 783" rather than one unbroken run.
    for match in _LONG_DIGITS.finditer(text):
        add(ResidualKind.NUMERIC, match, 1e9)

    for match in _GROUPED_DIGITS.finditer(text):
        if sum(len(part) for part in match.group().split()) >= 7:
            add(ResidualKind.NUMERIC, match, 1e8)

    for match in _REFERENCE.finditer(text):
        add(ResidualKind.ORGANISATION, match, 5e6)

    for match in _NAMED_SYSTEM.finditer(text):
        add(ResidualKind.ORGANISATION, match, _SURVIVING_PROPER_NOUN_SELECTIVITY)

    for match in _EXACT_DATE.finditer(text):
        add(ResidualKind.TEMPORAL, match, SELECTIVITY[ResidualKind.TEMPORAL] * 0.1)

    for match in _AGE.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON] * 3)

    for match in _OCCUPATION.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON] * 4)

    for match in _TENURE.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON] * 2)

    for match in _BAND.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON])

    for match in _SMALL_COUNT.finditer(text):
        add(ResidualKind.NUMERIC, match, SELECTIVITY[ResidualKind.NUMERIC] * 0.5)

    for match in _YEAR.finditer(text):
        add(ResidualKind.TEMPORAL, match, SELECTIVITY[ResidualKind.TEMPORAL] * 0.5)

    # Proper nouns that survived: possibly a missed organisation or person.
    # Checked last so the specific patterns above claim their spans first.
    for match in _PROPER_NOUN.finditer(text):
        words = match.group(0).split()
        if not words or words[0] in _NOT_A_NAME:
            continue
        add(ResidualKind.ORGANISATION, match, _SURVIVING_PROPER_NOUN_SELECTIVITY)

    if _NARRATIVE.search(text):
        attributes.append(
            ResidualAttribute(
                ResidualKind.NARRATIVE,
                "incident narrative",
                narrative_selectivity or SELECTIVITY[ResidualKind.NARRATIVE],
            )
        )

    return ResidualProfile(tuple(attributes), text)


def decide(
    profile: ResidualProfile,
    policy: DisclosurePolicy | None = None,
) -> DisclosureDecision:
    """Decide whether a redacted record may be released as it stands."""
    policy = policy or DisclosurePolicy()
    expected = policy.expected_matches(profile)
    drivers = tuple(sorted({a.value for a in profile.high_selectivity}))

    if not profile.attributes:
        return DisclosureDecision(
            action=DisclosureAction.PASS,
            profile=profile,
            cohort=expected,
            reason="nothing survived redaction, so nothing is disclosed",
        )

    if expected < policy.min_cohort:
        # Nothing left can be generalised away when the whole of the risk sits
        # in an unspanned narrative, or when every literal that could be
        # replaced is already below the threshold. Those records describe a
        # single person by construction and no amount of redacting changes
        # that, so the honest action is to withhold and let a human decide.
        # Deciding this by asking whether generalisation would change the text
        # is more honest than a second magic threshold.
        if not profile.has_reducible:
            return DisclosureDecision(
                action=DisclosureAction.SUPPRESS,
                profile=profile,
                cohort=expected,
                reason=(
                    f"the combination matches about {expected:.2f} of a "
                    f"population of {policy.population} and cannot be "
                    f"generalised without discarding the response"
                ),
                drivers=drivers,
            )
        return DisclosureDecision(
            action=DisclosureAction.GENERALISE,
            profile=profile,
            cohort=expected,
            reason=(
                f"the combination matches about {expected:.2f} of a population "
                f"of {policy.population}, below the threshold of "
                f"{policy.min_cohort}"
            ),
            drivers=drivers,
        )

    return DisclosureDecision(
        action=DisclosureAction.PASS,
        profile=profile,
        cohort=expected,
        reason=(
            f"the combination matches about {expected:.1f} of a population of "
            f"{policy.population}, which is large enough to be unremarkable"
        ),
    )


def generalise_residual(text: str, profile: ResidualProfile) -> str:
    """Replace surviving high-selectivity literals with a category marker.

    This is the ``GENERALISE`` action made concrete: the residue that would
    identify someone is removed without discarding the record, which matters
    because the narrative is usually the reason the response was collected.
    """
    out = text
    for attribute in sorted(
        profile.high_selectivity,
        key=lambda a: a.span[0] if a.span else 0,
        reverse=True,
    ):
        if attribute.span is None or attribute.kind is ResidualKind.NARRATIVE:
            continue
        start, end = attribute.span
        out = out[:start] + f"[{attribute.kind.value.upper()} REDACTED]" + out[end:]
    return out


def protect(
    text: str,
    policy: DisclosurePolicy | None = None,
) -> tuple[str, DisclosureDecision]:
    """Measure and act on a redacted record. Returns ``(text, decision)``.

    ``text`` is returned unchanged unless the decision escalates, in which case
    the residual literals are generalised away or the record is withheld.
    """
    policy = policy or DisclosurePolicy()
    profile = measure(text, policy.narrative_selectivity)
    decision = decide(profile, policy)

    if decision.action is DisclosureAction.SUPPRESS:
        return WITHHELD_TEXT, decision
    if decision.action is DisclosureAction.GENERALISE:
        cleaned = generalise_residual(text, profile)
        # Re-measure with the same narrative weight, or the reassessment would
        # compare two different models.
        # One reassessment, and no loop: whether a record still needs
        # withholding is a judgement for a reviewer, not a heuristic. The
        # returned decision is the one that was *applied*, because a caller
        # needs to know the record was escalated, not that the result looks
        # clean afterwards.
        after = measure(cleaned, policy.narrative_selectivity)
        return cleaned, replace(decision, post_cohort=policy.expected_matches(after))
    return text, decision


def summarise(decisions: Iterable[DisclosureDecision]) -> dict[str, Any]:
    """Aggregate decisions across an ingest, for the run report."""
    decisions = list(decisions)
    total = len(decisions)
    counts = {action.value: 0 for action in DisclosureAction}
    for decision in decisions:
        counts[decision.action.value] += 1
    escalated = counts[DisclosureAction.GENERALISE.value] + counts[DisclosureAction.SUPPRESS.value]
    return {
        "records": total,
        "actions": counts,
        "escalation_rate": round(escalated / total, 4) if total else 0.0,
        "mean_cohort": round(
            sum(d.cohort for d in decisions) / total, 1
        ) if total else 0.0,
    }


__all__ = [
    "HIGH_SELECTIVITY",
    "SELECTIVITY",
    "WITHHELD_TEXT",
    "DisclosureAction",
    "DisclosureDecision",
    "DisclosurePolicy",
    "ResidualAttribute",
    "ResidualKind",
    "ResidualProfile",
    "decide",
    "generalise_residual",
    "measure",
    "protect",
    "summarise",
]
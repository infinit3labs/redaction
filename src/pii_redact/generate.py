"""Synthetic survey generator with ground truth.

Hand-written fixtures are easy to overfit to. This module composes realistic
WHS free text from vocabularies and templates, injects known secrets at
controlled rates, and records exactly what it injected. That turns "did it
catch the thing" into a measurable question over thousands of records instead of
an opinion about twenty.

Two things it is careful about:

**Ground truth is what was injected, not what a detector reports.** A record
knows it contained a phone number because the generator put one there, so
recall is measurable without circularity.

**Degraded variants are first class.** Roughly half the injected values are
misspelled, leet-spelled, spaced out or fullwidth, because that is what free
text actually looks like. A detector that only scores well on clean values is
not measuring anything useful.

Everything is seeded, so a given ``--seed`` always produces the same corpus.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .checksums import acn_check_digit, is_tfn, medicare_check_digit

# --- Public sector and corporate context -------------------------------------
# Agency *types* rather than named agencies: the generator must not know any
# real department, because the point is to check that the module catches a
# workplace type from its vocabulary rather than from a supplied list.
AGENCY_UNITS = [
    "Migration directorate", "Home Affairs branch", "Services division",
    "the digital services branch", "the integrity unit", "the policy centre",
    "the enforcement directorate", "the complaints branch",
    "the grants secretariat", "the regional office", "the state office",
]
APS_LEVELS = ["EL1", "EL2", "EL3", "EL4", "EL6", "SES", "APS 4", "APS 6"]
PUBLIC_ROLES = [
    "policy officer", "assistant director", "branch manager",
    "eligibility specialist", "case officer", "contact centre agent",
    "grants administrator", "contracts officer", "investigator",
    "compliance officer", "statistician", "economist", "archivist",
    "policy adviser", "program manager",
]
CORPORATE_ROLES = [
    "site supervisor", "depot manager", "shift supervisor", "store manager",
    "senior trader", "relationship manager", "fund manager", "mine supervisor",
]
# Only unambiguously governmental acts. "The annual report" was tried and
# removed: it appears in corpora and on the open internet, so it identifies no
# workplace, and matching it deletes a phrase from otherwise ordinary text.
GOVERNMENT_ACTS = [
    "a machinery of government review", "a parliamentary estimates submission",
    "a freedom of information request", "an outcome roadmap",
    "a gateway review", "a cabinet submission", "a ministerial submission",
]
WORKPLACE_TYPES = [
    "a residential aged care facility", "a correctional centre",
    "our distribution centre", "our depot", "a water authority",
    "a community health centre", "an open cut mine", "our freight terminal",
    "a disability accommodation service", "a waste facility",
]
COHORTS = [
    "the graduate cohort", "the 2021 intake", "the pilot cohort",
    "the reference site", "the review team",
]
CLEARANCES = ["a security clearance", "clearance at the secret level",
              "national security vetting", "an NVL clearance"]

# --- Vocabularies -------------------------------------------------------------
SITES = [
    "Kwinana depot", "Altona distribution centre", "Fremantle yard",
    "Geelong workshop", "Parramatta hub", "Box Hill store",
    "Newcastle terminal", "Townsville depot", "Ballarat site",
    "Cairns branch", "Sunshine warehouse", "Dandenong store",
]
SUBURBS = [
    "Melbourne", "Sydney", "Brisbane", "Perth", "Adelaide", "Hobart",
    "Darwin", "Surry Hills", "Newtown", "Fremantle", "Kwinana", "Altona",
]
ROLES = [
    "registered nurse", "teacher", "care worker", "cleaner", "truck driver",
    "electrician", "barista", "labourer", "warden", "caseworker",
]
TEAMS = ["night shift", "day shift", "ward team", "yard team", "store team", "roster"]
GRADES = ["Grade 3", "Grade 4", "Band 6", "Step 3", "Class 2", "Level 4"]
HEALTH = ["anxiety", "chronic fatigue", "depression", "migraines", "a shoulder injury"]
SURNAMES = [
    "Whitlam", "Nguyen", "Okafor", "Brennan", "Kowalski", "Ferreira",
    "Nakamura", "Abadi", "Sinclair", "Halvorsen", "Petrov", "Mbeki",
]
GIVEN = [
    "Fiona", "Jane", "Ahmed", "Priya", "Tomas", "Grace", "Daniel",
    "Marisol", "Kiran", "Aoife", "Hassan", "Lena",
]
ORGS = [
    "Acme Pty Ltd", "Reticule Health", "Northline Services", "Corvus Group",
    "Halden Holdings", "Pinnacle Corporation",
]
FIRST_NAMES_PHRASES = [
    "the only woman on the night shift",
    "the only one here who does the handover",
    "one of two apprentices on the yard team",
    "the youngest on the team here",
    "the sole nurse in the unit",
    "the only casual on the roster",
    "nobody else does the late round",
    "the only one who does the 6am start",
    "I cover all three depots on my own",
]
SECOND_NAMES_PHRASES = [
    "my manager Fiona",
    "my team lead Daniel",
    "told my supervisor Grace",
    "my supervisor Ahmed",
    "my manager Priya",
]
WORK_EVENTS_PHRASES = [
    "after the depot closure",
    "since the take-over",
    "after the merger announcement",
    "when they closed the Box Hill store",
    "after the safety inspection",
    "since the site was sold",
]

HARMFUL = [
    "constantly undermined in front of the team",
    "targeted me in meetings until I cried",
    "threatened me when I raised a safety issue",
    "excluded me from the roster after I complained",
    "humiliated me in front of the new starters",
    "retaliated against me the week I reported it",
    "shouted at me in the break room",
    "made comments about my age whenever I pushed back",
]
POSITIVE = [
    "my team has always had my back",
    "the training here is genuinely good",
    "my supervisor listens and acts on it",
    "I feel safe raising concerns here",
    "the flexible roster makes a real difference",
]
HEDGES = [
    "I would rather not say too much in writing.",
    "I am not the only one who feels this way.",
    "Please do not share this with my team.",
    "I do not want my employer to find out I said this.",
    "This is anonymous but the detail is real.",
]

#: The service half of a survey run by a provider. A record that only ever
#: complains teaches a redactor about complaints: it never sees the praise, the
#: brief answers, or the vocabulary of claims and referrals that makes up most of
#: a real response set, and it will happily redact any of it.
SERVICE_GOOD = [
    "the claim was paid without argument",
    "the referral was approved the same day",
    "whoever answers the phone was polite and knew my name",
    "telehealth saved me a two hour drive each way",
    "the physiotherapist was thorough and did not rush me",
    "everything has been paid on time for two years",
]
SERVICE_BAD = [
    "three months and still chasing a claim of four hundred dollars",
    "nobody could tell me where the file had gone",
    "the pre-approval took four months and was never needed",
    "I was put on hold for an hour and then cut off",
    "the refund has not landed and nobody owns the delay",
    "the decision does not match what the specialist wrote",
]
#: Psychosocial hazards that are not bullying. Bullying dominates published
#: bullying data and this vocabulary would leave the other nine Model Code
#: hazards untested: workload, role clarity, change, fatigue, remote work,
#: understaffing, poor support and unconsulted change.
HAZARDS = [
    "we are short three people and everyone is picking up the rest",
    "the workload has doubled since the restructure",
    "roles are unclear and everyone has three managers",
    "the roster changes are announced the night before",
    "I do two twelve hour shifts back to back",
    "working from home means nobody sees how bad it has got",
    "the consultation happened after the decision was made",
    "I have been off sick twice this quarter and neither was discussed",
]
#: Register. Free text arrives in every shape: shouted, terse, lowercase,
#: abbreviated. A corpus of tidy prose makes an over-redaction look fine.
REGISTERS = [
    "THIS IS UNACCEPTABLE AND I WANT SOMEONE TO OWN IT.",
    "no complaints",
    "same as last year",
    "fine i guess",
    "The whole thing has been a joke from the start.",
    "honestly, who is reading these?",
]

_OPENERS = [
    "To be honest,", "Long story short,", "I have thought about this for weeks.",
    "For what it is worth,", "I probably should not have said this, but",
]
_TAILS = [
    "It has been going on for months.", "I have raised it twice now.",
    "Nobody has done anything about it.", "I just needed to put it somewhere.",
    "", "I am done pretending it is fine.",
]


@dataclass(frozen=True, slots=True)
class Secret:
    """One thing the generator planted, and how to recognise it afterwards."""

    kind: str
    #: The exact text inserted, which must not survive redaction.
    literal: str
    #: Category the redactor is expected to report, if any.
    expect_category: str | None = None
    #: True when this item is a bare number that only a checksum can confirm.
    checksummed: bool = False
    #: How the generator altered the value: clean, misspelled, leet,
    #: fullwidth, spaced or lower. Reported alongside recall, because the
    #: sequence rules are exact patterns and "misspelled" is a ceiling rather
    #: than a defect.
    degradation: str = "clean"


@dataclass
class GeneratedRecord:
    """One synthetic response plus its ground truth."""

    response_id: str
    answers: dict[str, str]
    secrets: list[Secret] = field(default_factory=list)
    #: Phrases that must survive redaction, used to measure over-redaction.
    must_survive: list[str] = field(default_factory=list)
    template: str = ""

    def to_json(self) -> dict[str, object]:
        record: dict[str, object] = {"ResponseID": self.response_id}
        record.update(self.answers)
        return record


# --- Degradation: the point of the generator ---------------------------------
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}


def misspell(text: str, rng: random.Random) -> str:
    """One realistic typo: a doubled, dropped or transposed letter."""
    if len(text) < 5:
        return text
    index = rng.randrange(1, len(text) - 1)
    kind = rng.choice(("drop", "double", "swap", "substitute"))
    if kind == "drop":
        return text[:index] + text[index + 1 :]
    if kind == "double":
        return text[:index] + text[index] + text[index:]
    if kind == "swap" and index + 1 < len(text):
        return text[:index] + text[index + 1] + text[index] + text[index + 2 :]
    return text[:index] + rng.choice("abcdefghijklmnopqrstuvwxyz") + text[index + 1 :]


def to_leet(text: str) -> str:
    return "".join(_LEET.get(c, c) if c.isdigit() else c for c in text)


def to_fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "0" <= c <= "9" else c for c in text)


def space_out(text: str) -> str:
    return " ".join(text)


def degrade(text: str, rng: random.Random) -> tuple[str, str]:
    """Return ``(value, degradation)`` for ``text``.

    Two thirds of injected values are degraded, which is closer to real survey
    data than clean values would be.
    """
    roll = rng.random()
    if roll < 0.35:
        return misspell(text, rng), "misspelled"
    if roll < 0.45:
        return to_leet(text), "leet"
    if roll < 0.55:
        return to_fullwidth(text), "fullwidth"
    if roll < 0.60:
        return space_out(text), "spaced"
    if roll < 0.68:
        return text.lower(), "lower"
    return text, "clean"


# --- Value builders -----------------------------------------------------------
def valid_tfn(rng: random.Random) -> str:
    while True:
        candidate = rng.randrange(100_000_000, 999_999_999)
        if is_tfn(str(candidate)):
            digits = str(candidate)
            return f"{digits[:3]} {digits[3:6]} {digits[6:]}"


def valid_medicare(rng: random.Random) -> str:
    base = rng.choice((2, 3, 4, 5, 6, 9)) * 100_000_000 + rng.randrange(10_000_000, 99_999_999)
    digits = f"{base}{medicare_check_digit(str(base))}"
    return f"{digits[0]} {digits[1:5]} {digits[5:9]} {digits[9]}"


def valid_acn(rng: random.Random) -> str:
    base = f"{rng.randrange(10_000_000, 99_999_999):08d}"
    acn = base + str(acn_check_digit(base))
    return f"{acn[:3]} {acn[3:6]} {acn[6:]}"


def valid_phone(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return f"04{rng.randrange(10, 99)} {rng.randrange(100, 999)} {rng.randrange(100, 999)}"
    area = rng.choice(("02", "03", "07", "08"))
    return f"{area}9{rng.randrange(100, 999)} {rng.randrange(1000, 9999)}"


def valid_email(rng: random.Random) -> str:
    return (
        f"{rng.choice(GIVEN).lower()}.{rng.choice(SURNAMES).lower()}@"
        f"{rng.choice(('mail', 'corp', 'group', 'connect')).lower()}"
        f".{rng.choice(('com.au', 'com.au', 'net.au'))}"
    )


def valid_card(rng: random.Random) -> str:
    prefix = rng.choice(("4111", "555534", "378282"))
    return f"{prefix} {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}"


def valid_dob(rng: random.Random) -> str:
    return (
        f"{rng.randrange(1, 29):02d}/{rng.randrange(1, 13):02d}/"
        f"{rng.randrange(1955, 2006)}"
    )


def valid_bsb(rng: random.Random) -> str:
    return f"{rng.randrange(10, 97)}{rng.randrange(1000, 9999)}"


# --- Injection ----------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Injector:
    """One kind of secret and how to weave it into a sentence."""

    kind: str
    expect_category: str | None
    checksummed: bool = False
    sentences: tuple[str, ...] = ()
    builders: tuple = ()
    #: Only inject these where the template already talks about employment.
    employment_only: bool = False
    #: Leave the value byte-for-byte intact. Set for anything whose *syntax* is
    #: the evidence: mangling the letters of an email address or a domain
    #: produces a string that is not an email address any more, so recall would
    #: be measuring the generator rather than the detector. Degradation is
    #: exercised instead by the dedicated obfuscated injectors.
    exact: bool = False


INJECTORS: tuple[Injector, ...] = (
    Injector(
        "phone",
        "phone",
        exact=True,
        sentences=(
            "You can reach me on {value} if that helps.",
            "My mobile is {value}, though I cannot always answer.",
        ),
        builders=(valid_phone,),
    ),
    Injector(
        "email",
        "email",
        exact=True,
        sentences=(
            "Email me at {value} if you need more detail.",
            "I have put my address {value} in the other box.",
        ),
        builders=(valid_email,),
    ),
    Injector(
        "tfn",
        "tfn",
        checksummed=True,
        exact=True,
        sentences=("My TFN is {value} if you need to look me up.",),
        builders=(valid_tfn,),
    ),
    Injector(
        "medicare",
        "medicare",
        checksummed=True,
        exact=True,
        sentences=("My Medicare number is {value}.",),
        builders=(valid_medicare,),
    ),
    Injector(
        "acn",
        "acn",
        checksummed=True,
        exact=True,
        sentences=("The employer ACN is {value} if that is useful.",),
        builders=(valid_acn,),
    ),
    Injector(
        "credit_card",
        "credit_card",
        checksummed=True,
        exact=True,
        sentences=("Pay me on {value} if it is easier.",),
        builders=(valid_card,),
    ),
    Injector(
        "dob",
        "date_au",
        exact=True,
        sentences=("DOB {value}, just so the age gap is clear.",),
        builders=(valid_dob,),
    ),
    Injector(
        "bsb",
        "bsb",
        exact=True,
        employment_only=True,
        sentences=("BSB {value} if you need to run a report.",),
        builders=(valid_bsb,),
    ),
    Injector(
        "organisation",
        "organisation_name",
        exact=True,
        employment_only=True,
        sentences=("For context I work at {value}.", "This is about {value} specifically."),
        builders=(lambda rng: rng.choice(ORGS),),
    ),
    Injector(
        "site",
        "implication",
        employment_only=True,
        sentences=("I work out of {value} most weeks.", "My site is {value}."),
        builders=(lambda rng: rng.choice(SITES),),
    ),
    Injector(
        "implication_person",
        "implication",
        employment_only=True,
        sentences=(
            "Realistically I am {value}.",
            "To put it plainly, {value}.",
        ),
        builders=(lambda rng: rng.choice(FIRST_NAMES_PHRASES),),
    ),
    Injector(
        "implication_workplace",
        "implication",
        employment_only=True,
        sentences=("It all happened {value}.", "Things changed {value}."),
        builders=(lambda rng: rng.choice(WORK_EVENTS_PHRASES),),
    ),
    Injector(
        "third_party",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("It was {value} who signed it off.",),
        builders=(lambda rng: rng.choice(SECOND_NAMES_PHRASES),),
    ),
    Injector(
        "suburb",
        "location",
        employment_only=True,
        exact=True,
        sentences=("I am based in {value} these days.",),
        builders=(lambda rng: rng.choice(SUBURBS),),
    ),
    Injector(
        "aps_unit",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I work in {value}.", "My role sits in {value}."),
        builders=(lambda rng: rng.choice(AGENCY_UNITS),),
    ),
    Injector(
        "aps_level",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I am {value} and that makes me one of very few.",),
        builders=(lambda rng: rng.choice(APS_LEVELS),),
    ),
    Injector(
        "public_role",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I work as a {value}.", "My title is {value}."),
        builders=(lambda rng: rng.choice(PUBLIC_ROLES + CORPORATE_ROLES),),
    ),
    Injector(
        "seniority_in_unit",
        "implication",
        employment_only=True,
        exact=True,
        sentences=(
            "I am one of three of us in {value}.",
            "There are two of us left in {value}.",
        ),
        builders=(lambda rng: rng.choice(AGENCY_UNITS),),
    ),
    Injector(
        "government_act",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("Last year we handled {value}.", "It went through {value}."),
        builders=(lambda rng: rng.choice(GOVERNMENT_ACTS),),
    ),
    Injector(
        "workplace_type",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I have worked in {value} for years.", "I am based in {value}."),
        builders=(lambda rng: rng.choice(WORKPLACE_TYPES),),
    ),
    Injector(
        "cohort",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I am part of {value}.", "I came through {value}."),
        builders=(lambda rng: rng.choice(COHORTS),),
    ),
    Injector(
        "clearance",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I hold {value}.", "It took years to get {value}."),
        builders=(lambda rng: rng.choice(CLEARANCES),),
    ),
    Injector(
        "phone_obfuscated",
        "phone",
        sentences=("Reach me on {value}.", "Best number is {value}."),
        builders=(valid_phone,),
    ),
    Injector(
        "email_spaced",
        "email",
        sentences=("My address is {value}.",),
        builders=(valid_email,),
    ),
)


def _join(text: str, sentence: str) -> str:
    """Append a sentence with correct punctuation and capitalisation."""
    sentence = sentence.strip()
    if not sentence:
        return text
    if not sentence[0].isupper():
        sentence = sentence[0].upper() + sentence[1:]
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return f"{text} {sentence}"


def _degraded(value: str, injector: Injector, rng: random.Random) -> str:
    """Apply the degradation appropriate to ``injector``, and label it.

    Obfuscated variants are chosen by kind so the corpus contains fullwidth
    phone numbers and spaced email addresses deliberately, rather than by
    accident. Everything else is either left intact (syntax is the evidence) or
    lightly misspelled.
    """
    if injector.kind == "phone_obfuscated":
        return to_fullwidth(value), "fullwidth"
    if injector.kind == "email_spaced":
        return space_out(value), "spaced"
    if injector.exact or injector.checksummed:
        return value, "clean"
    # Agency and occupation phrasing is already degraded by the phrase pool, and
    # a typo in "directorate" measures nothing about the detector.
    return degrade(value, rng)


class SurveyGenerator:
    """Composes synthetic WHS responses with known ground truth."""

    def __init__(
        self,
        seed: int = 20260308,
        *,
        injection_rate: float = 0.85,
        employment_rate: float = 0.7,
    ) -> None:
        self.rng = random.Random(seed)
        self.injection_rate = injection_rate
        self.employment_rate = employment_rate
        self._counter = 0

    def record(self, index: int) -> GeneratedRecord:
        """One synthetic response."""
        rng = self.rng
        # Weighted towards harm, because that is the shape a psychosocial survey
        # produces most of -- but not exclusively. A generator that only ever
        # writes complaints cannot show that praise and administrative answers
        # survive redaction intact.
        template = rng.choice(
            ("harm", "positive", "harm", "harm", "health", "service", "hazard", "terse")
        )
        employment = rng.random() < self.employment_rate

        body = self._body(template, employment)
        secrets: list[Secret] = []
        must_survive: list[str] = []

        if rng.random() < self.injection_rate:
            for injector in self._choose_injectors(employment):
                built = injector.builders[rng.randrange(len(injector.builders))](rng)
                value, how = _degraded(built, injector, rng)
                sentence = injector.sentences[rng.randrange(len(injector.sentences))]
                body = _join(body, sentence.format(value=value))
                secrets.append(
                    Secret(
                        kind=injector.kind,
                        literal=value,
                        expect_category=injector.expect_category,
                        checksummed=injector.checksummed,
                        degradation=how,
                    )
                )

        must_survive.extend(self._stable_fragments(template, body))
        answers = self._split_answers(body)

        return GeneratedRecord(
            response_id=f"G{index:05d}",
            answers=answers,
            secrets=secrets,
            must_survive=must_survive,
            template=template,
        )

    def corpus(self, count: int) -> list[GeneratedRecord]:
        return [self.record(i) for i in range(count)]

    # -- internals ----------------------------------------------------------
    def _choose_injectors(self, employment: bool) -> list[Injector]:
        pool = [
            i
            for i in INJECTORS
            if employment or not i.employment_only
        ]
        rng = self.rng
        chosen = [i for i in pool if rng.random() < 0.3]
        return chosen[:3]

    def _body(self, template: str, employment: bool) -> str:
        rng = self.rng
        parts: list[str] = []
        if rng.random() < 0.5:
            parts.append(_OPENERS[rng.randrange(len(_OPENERS))])

        # Each clause is terminated so the composed text reads as prose.
        if template == "harm":
            parts.append(
                f"My manager {HARMFUL[rng.randrange(len(HARMFUL))]}."
            )
        elif template == "positive":
            parts.append(f"{POSITIVE[rng.randrange(len(POSITIVE))]}.")
            parts.append(f"{SERVICE_GOOD[rng.randrange(len(SERVICE_GOOD))]}.")
        elif template == "service":
            pool = SERVICE_GOOD if rng.random() < 0.4 else SERVICE_BAD
            parts.append(f"Cover wise, {pool[rng.randrange(len(pool))]}.")
        elif template == "hazard":
            parts.append(f"{HAZARDS[rng.randrange(len(HAZARDS))]}.")
            parts.append(f"{SERVICE_BAD[rng.randrange(len(SERVICE_BAD))]}.")
        elif template == "terse":
            parts.append(REGISTERS[rng.randrange(len(REGISTERS))])
        else:
            parts.append(
                "I have been struggling since this started and it has left me with "
                f"{HEALTH[rng.randrange(len(HEALTH))]}."
            )

        if rng.random() < 0.6:
            condition = rng.choice(HEDGES)
            parts.append(condition)
            # A confidentiality request only makes sense in employment context,
            # so make sure the record has some when that is missing.
            if not employment and (
                "do not want my employer" in condition
                or "Please do not share" in condition
            ):
                parts.append(
                    f"I work at {rng.choice(ORGS)} as a {rng.choice(ROLES)}."
                )

        if employment and rng.random() < 0.5:
            if rng.random() < 0.55:
                # Public-sector framing: level plus unit is near-unique.
                parts.append(
                    f"I am a {rng.choice(PUBLIC_ROLES)}, {rng.choice(APS_LEVELS)}, "
                    f"in {rng.choice(AGENCY_UNITS)}, with "
                    f"{rng.randrange(2, 30)} years of service."
                )
            else:
                parts.append(
                    f"I am a {rng.choice(ROLES + CORPORATE_ROLES)}, "
                    f"{rng.choice(GRADES)}, {rng.randrange(2, 30)} years of "
                    f"service, on the {rng.choice(TEAMS)} of a team of "
                    f"{rng.randrange(2, 15)}."
                )

        tail = _TAILS[rng.randrange(len(_TAILS))]
        if tail:
            parts.append(tail)
        return " ".join(parts)

    @staticmethod
    def _stable_fragments(template: str, body: str) -> list[str]:
        """Phrases a redactor must not touch, for measuring over-redaction.

        Chosen from the templates rather than the whole body so a secret that
        happens to sit inside them does not create a false failure.
        """
        fragments: list[str] = []
        for pool in (HARMFUL, POSITIVE, HAZARDS, SERVICE_GOOD, SERVICE_BAD, REGISTERS):
            for phrase in pool:
                if phrase in body:
                    fragments.append(phrase.split()[0] + " " + phrase.split()[1])
                    break
        return fragments

    def _split_answers(self, body: str) -> dict[str, str]:
        """Distribute one response across two free-text Qualtrics columns.

        The split lands on a sentence boundary. Splitting mid-sentence would
        occasionally cut a planted secret in half across Q1 and Q2, and no
        per-field redactor can catch a value that spans two fields -- the
        measurement would be recording an artefact of the generator rather than
        a property of the detector.
        """
        rng = self.rng
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        if len(sentences) < 2:
            return {"Q1": body}
        cut = rng.randrange(1, len(sentences))
        return {"Q1": " ".join(sentences[:cut]), "Q2": " ".join(sentences[cut:])}


def write_corpus(
    records: Sequence[GeneratedRecord],
    ndjson_path: Path,
    manifest_path: Path,
) -> None:
    """Write an NDJSON corpus and its ground-truth manifest."""
    ndjson_path.write_text(
        "\n".join(json.dumps(r.to_json(), ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "$comment": [
            "Ground truth from pii_redact.generate. Every entry in secrets was",
            "planted by the generator, so recall is measurable without reference",
            "to what a detector reported. must_survive is the inverse: text the",
            "redactor must leave alone, which is how over-redaction is caught."
        ],
        "count": len(records),
        "records": [
            {
                "id": r.response_id,
                "template": r.template,
                "secrets": [
                    {
                        "kind": s.kind,
                        "literal": s.literal,
                        "expect_category": s.expect_category,
                        "checksummed": s.checksummed,
                        "degradation": s.degradation,
                    }
                    for s in r.secrets
                ],
                "must_survive": r.must_survive,
            }
            for r in records
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "INJECTORS",
    "GeneratedRecord",
    "Secret",
    "SurveyGenerator",
    "degrade",
    "misspell",
    "space_out",
    "to_fullwidth",
    "to_leet",
    "valid_acn",
    "valid_bsb",
    "valid_card",
    "valid_dob",
    "valid_email",
    "valid_medicare",
    "valid_phone",
    "valid_tfn",
    "write_corpus",
]
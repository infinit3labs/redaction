"""NLP augmentation stage.

The deterministic detectors are the primary defence: they are auditable, fast
and have known false-positive rates. This stage adds recall, never precision
authority. Every finding it emits is tagged ``Stage.NLP`` and scores below all
deterministic findings in the engine, so an NLP span can only win where the
deterministic pass found nothing.

Two backends are supported:

``SpacyBackend``
    Uses a spaCy NER pipeline when a model is installed.

``GazetteerBackend``
    A dependency-free fallback built from context cues (honorifics, name
    labels) and an Australian gazetteer of place names. This runs anywhere,
    which keeps the module usable with an empty dependency set.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import ClassVar

from ..types import Category, Finding, Span, Stage

# Confidence assigned to heuristic hits. Deliberately below the engine's
# default NLP threshold-adjacent band so callers can tune per category.
GAZETTEER_CONFIDENCE = 0.6

# --- Australian place gazetteer ---------------------------------------------
# Capital-city and major-city names, used as a weak location signal. Kept
# deliberately small: a large list costs maintenance and adds noise.
_AU_CITIES: frozenset[str] = frozenset(
    {
        "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra",
        "Hobart", "Darwin", "Gold Coast", "Newcastle", "Wollongong", "Geelong",
        "Townsville", "Cairns", "Toowoomba", "Ballarat", "Bendigo", "Launceston",
        "Mackay", "Rockhampton", "Sunshine Coast", "Central Coast", "Hunter",
        "Illawarra", "Fraser Coast", "Wagga Wagga", "Alice Springs", "Broome",
    }
)

_AU_STATES: frozenset[str] = frozenset(
    {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}
)

_HONORIFICS = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Rev|A/Prof|Capt|Sgt|Justice|Lord|Lady)"

_NAME_LABEL = re.compile(
    rf"(?i)\b(?:name|patient|customer|client|employee|staff|member|contact|"
    rf"applicant|claimant|guardian|parent|spouse|next\s*of\s*kin|nok|"
    # Role labels. Respondents name their own supervisor in exactly this form,
    # and a supervisor's name is among the most identifying strings in the set.
    rf"manager|supervisor|team\s*lead|coordinator|colleague|co-?worker|"
    rf"my\s+boss|boss|colleague'?s?\s*name)"
    rf"\b\s*[:\-]\s*(?P<name>(?:{_HONORIFICS}\.?\s+)?[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){{1,3}})"
)

_HONORIFIC_NAME = re.compile(
    # _HONORIFICS is wrapped because it is an alternation. Interpolating it
    # bare puts the alternation inside the name group, so the group captures
    # only the matched honorific and the rest of the name falls outside the
    # reported span.
    rf"\b(?P<name>(?:{_HONORIFICS})\.?\s+[A-Z][a-z'\u2019-]+(?:\s+[A-Z][a-z'\u2019-]+){{1,3}})"
)

# The trigger phrase is case-insensitive; the place itself is not, because a
# capitalised word is the evidence. Scoped inline flags keep both true.
_LOCATION_PHRASE = re.compile(
    r"(?i:\b(?:based\s+in|located\s+in|living\s+in|resident\s+in|currently\s+in|from))\s+"
    r"(?P<place>[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){0,2})"
)
# "in the Melbourne store", "at the Parramatta site". A capitalised place name
# immediately before a site noun is a workplace location, which is one of the
# strongest employer-attribution signals available in free text.
# Every interpolated alternation must be wrapped in a non-capturing group.
# Interpolating a bare "a|b" into a larger pattern applies the alternation to
# the whole expression, so the single word "site" matches on its own.
_SITE_NOUN = (
    r"(?:"
    r"store|depot|site|branch|office|hub|campus|centre|center|warehouse|plant|"
    r"factory|terminal|yard|facility|hospital|clinic|school|college|university|"
    r"workshop|garage|building|headquarters"
    r")"
)
_LOCATION_SITE_RE = re.compile(
    rf"\b(?:in|at|on|from)\s+(?:the\s+|our\s+|my\s+)?"
    rf"(?P<place>[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?)\s+{_SITE_NOUN}\b"
)

_STATE_PHRASE = re.compile(rf"\b(?P<place>{'|'.join(sorted(_AU_STATES))})\b")

_ADDRESS_LABEL = re.compile(
    r"(?i)\b(?:address|residential\s*address|home\s*address|mailing\s*address|"
    r"street|addr\.?)\b\s*[:\-]\s*"
    r"(?P<value>[^,;\n]{6,80}?(?:,\s*[A-Z][a-z'\u2019-]+\s+\d{4})?)"
)

# An age needs an age marker. A bare "6 years" is a duration -- of service, of
# sickness, of anything -- so "N years" on its own is deliberately not an age.
# Two shapes qualify: "N years old" / "N years of age", and an explicit framing
# cue immediately before the number.
#
# The digit guards are load bearing for idempotence. Without them "25-34" (an
# age band this package emitted on a previous pass) is read as the age 25 or
# 34, generalised again, and turns into "25-25-34".
_AGE_RE = re.compile(
    r"(?i)"
    # "34 years old", "34 years of age"
    r"(?<![\d-])(?P<value>\d{1,3})\s?(?:-|\s)?\s*(?:years?|yrs?)\s*(?:old|of\s+age)(?![\d-])\b"
    # "I am 34", "I'm 34", "aged 34" -- but not the 25 of "25-34"
    r"|\b(?:i\s+am|i'?m|im|aged|age\s+of)\s+(?P<value2>\d{1,3})(?![\d-])(?!\s*-)"
)

_SEX_LABEL = re.compile(
    r"(?i)\b(?:sex|gender|dob|date\s*of\s*birth)\b\s*[:\-]\s*(?P<value>[A-Za-z/ ]{3,20})"
)

# Words that must never be treated as a person's name by the gazetteer.
_NAME_STOPWORDS = frozenset(
    {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December", "Australia", "Australian",
    }
)


class GazetteerBackend:
    """Dependency-free heuristic NER built on context cues and a gazetteer."""

    name = "gazetteer"

    def detect(self, text: str) -> Iterator[Finding]:
        yield from self._names(text)
        yield from self._addresses(text)
        yield from self._locations(text)
        yield from self._age_gender(text)

    # -- people --------------------------------------------------------------
    def _names(self, text: str) -> Iterator[Finding]:
        seen: set[Span] = set()
        for pattern, boost in ((_NAME_LABEL, 0.2), (_HONORIFIC_NAME, 0.15)):
            for match in pattern.finditer(text):
                raw = match.group("name")
                span = Span(*match.span("name"))
                if span in seen or not self._plausible_name(raw):
                    continue
                seen.add(span)
                yield self._finding(span, raw, Category.NAME, GAZETTEER_CONFIDENCE + boost)

    @staticmethod
    def _plausible_name(raw: str) -> bool:
        words = [w for w in raw.split() if w[:1].isupper()]
        if len(words) < 2:
            return False
        if raw.split()[0] in _NAME_STOPWORDS:
            return False
        # Reject titles/headings such as "Dear Customer".
        return not any(w in {"dear", "regards", "sincerely", "hi", "hello"} for w in words)

    # -- addresses -----------------------------------------------------------
    def _addresses(self, text: str) -> Iterator[Finding]:
        for match in _ADDRESS_LABEL.finditer(text):
            value = match.group("value").strip()
            if len(value) < 6:
                continue
            yield self._finding(
                Span(*match.span("value")), value, Category.STREET_ADDRESS, 0.7
            )

    # -- places --------------------------------------------------------------
    def _locations(self, text: str) -> Iterator[Finding]:
        for match in _LOCATION_PHRASE.finditer(text):
            place = match.group("place")
            confidence = 0.75 if any(w in place for w in _AU_CITIES) else GAZETTEER_CONFIDENCE
            yield self._finding(
                Span(*match.span("place")), place, Category.LOCATION, confidence
            )
        for match in _LOCATION_SITE_RE.finditer(text):
            place = match.group("place")
            confidence = 0.8 if any(w in place for w in _AU_CITIES) else 0.65
            yield self._finding(
                Span(*match.span("place")), place, Category.LOCATION, confidence
            )

    @staticmethod
    def _finding(span: Span, value: str, category: Category, confidence: float) -> Finding:
        return Finding(
            span=span,
            category=category,
            detector="gazetteer",
            stage=Stage.NLP,
            confidence=confidence,
            value=value,
        )

    def _age_gender(self, text: str) -> Iterator[Finding]:
        for match in _AGE_RE.finditer(text):
            group = "value" if match.group("value") else "value2"
            age = int(match.group(group))
            if age <= 120:
                yield self._finding(
                    Span(*match.span(group)), match.group(group), Category.AGE, 0.7
                )
        for match in _SEX_LABEL.finditer(text):
            value = match.group("value").strip()
            if value.lower() in {"male", "female", "m", "f", "non-binary", "other"}:
                yield self._finding(
                    Span(*match.span("value")), value, Category.GENDER, 0.7
                )


class SpacyBackend:
    """spaCy NER adapter.

    Imports are deferred so the package installs and runs without spaCy. If the
    model is missing, :meth:`available` returns False and the engine falls back
    to the gazetteer rather than failing.
    """

    name = "spacy"

    # spaCy label -> our category.
    _LABEL_MAP: ClassVar[dict[str, Category]] = {
        "PERSON": Category.NAME,
        "GPE": Category.LOCATION,
        "LOC": Category.LOCATION,
        "FAC": Category.STREET_ADDRESS,
        "ORG": Category.ORGANISATION,
    }

    def __init__(self, model: str = "en_core_web_lg", min_score: float = 0.35) -> None:
        self.model = model
        self.min_score = min_score
        self._nlp = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._nlp is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import spacy
        except ImportError as exc:
            self._load_error = f"spaCy not installed: {exc}"
            return False
        try:
            self._nlp = spacy.load(self.model, disable=["lemmatizer"])
        except Exception as exc:
            self._load_error = f"could not load spaCy model {self.model!r}: {exc}"
            return False
        return True

    def available(self) -> bool:
        return self._ensure_loaded()

    def detect(self, text: str) -> Iterator[Finding]:
        if not self._ensure_loaded() or self._nlp is None:
            return
        doc = self._nlp(text)
        for ent in doc.ents:
            category = self._LABEL_MAP.get(ent.label_)
            if category is None:
                continue
            score = getattr(ent, "_", None)
            confidence = float(getattr(score, "score", 1.0) or 1.0)
            if confidence < self.min_score:
                continue
            if ent.start_char < 0 or ent.end_char <= ent.start_char:
                continue
            yield Finding(
                span=Span(ent.start_char, ent.end_char),
                category=category,
                detector=f"spacy:{ent.label_}",
                stage=Stage.NLP,
                confidence=confidence,
                value=ent.text,
            )
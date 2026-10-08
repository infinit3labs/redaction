"""Deterministic detectors for contact details, network identifiers and locations.

Phone, postcode and address rules are Australian-first: mobile numbers require
a ``04`` prefix, landlines a leading ``0`` with a valid area code, 1300/1800
numbers are handled separately, and postcodes are validated against real
state allocations rather than a bare four-digit pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..normalise import normalise_aligned
from ..types import Category, Finding, Span, Stage
from .base import CompiledPatternDetector, RegexDetector

# --- Email -----------------------------------------------------------------
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}\b"
)

# Deliberate obfuscation: respondents, and people avoiding an employer, type
# addresses with a space between every character. Modelled as a single class of
# "one address character, optionally followed by one space", repeated. The
# literal "@" and "." stay unspaced because that is how people actually write
# it, and requiring them keeps the pattern from matching ordinary prose.
# Deliberate obfuscation: respondents, and people avoiding an employer, type
# addresses with a space between every character. The signature of that is that
# *every* character is followed by a space, which is what distinguishes it from
# ordinary prose and lets a greedy match stop at "j a n e" rather than starting
# from the first letter of the preceding word.
#
# Three character classes are needed because the parts of an address allow
# different characters. The local part may contain dots; a domain label may
# not; the top level domain is letters only. Using one class for all three lets
# the domain label swallow the TLD and the whole pattern stops matching.
_LOCAL_U = r"(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~.\-]\s)"
_LABEL_U = r"(?:[A-Za-z0-9\-]\s)"
_TLD_U = r"(?:[A-Za-z]\s)"
# Every character is followed by a space, so a separator is a dot, an optional
# space, and then the next character. _UNIT consumes the space that *follows*
# its own character, which is why the optional space belongs after the dot.
_EMAIL_SPACED_RE = re.compile(
    r"(?<![\w.])"                                # start at a word boundary
    rf"{_LOCAL_U}{{3,64}}"                       # local part, dots allowed
    r"@\s?"
    rf"{_LABEL_U}{{1,63}}"
    # Each loop label must be followed by another dot. Without the lookahead a
    # greedy loop eats the top level domain as one more label and the required
    # TLD has nothing left to match, so the whole pattern fails.
    rf"(?:\.\s?{_LABEL_U}{{1,63}}(?=\s?\.))*"
    # The final letter is matched on its own because the last character of an
    # address at the end of a sentence has no trailing space to consume.
    rf"\.\s?{_TLD_U}{{1,62}}[A-Za-z]",           # top level domain
    re.IGNORECASE,
)


class EmailDetector(RegexDetector):
    """Email addresses.

    Matched against NFKC-normalised text so a fullwidth or compatibility form
    pasted from a phone is still found. NFKC is length preserving for the
    characters involved here, and the detector re-anchors defensively rather
    than assuming it.
    """

    name = "email"
    stage = Stage.DETERMINISTIC

    def patterns(self):
        return (EMAIL_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return Category.EMAIL

    def detect(self, text: str) -> Iterator[Finding]:
        # Folds fullwidth characters, which are common in text pasted from a
        # phone. Length preserving, so spans stay valid original offsets.
        aligned = normalise_aligned(text)
        for match in EMAIL_RE.finditer(aligned):
            start, end = match.span()
            yield Finding(
                span=Span(start, end),
                category=Category.EMAIL,
                detector=self.name,
                stage=self.stage,
                confidence=1.0,
                value=text[start:end],
            )
        # Both patterns are scanned over the whole text, always. An earlier
        # version ran the spaced pattern only when no canonical address was
        # found anywhere, as a guard against double reporting. That guard was
        # wrong: a response containing both a normal address and a spaced-out
        # one is ordinary, and the spaced address was silently skipped. The two
        # patterns cannot match the same text, since one requires no spaces and
        # the other requires them.
        for match in _EMAIL_SPACED_RE.finditer(aligned):
            start, end = match.span()
            # The trailing "\s?" of the last TLD character can swallow a space
            # that belongs to the sentence.
            while end > start and aligned[end - 1].isspace():
                end -= 1
            yield Finding(
                span=Span(start, end),
                category=Category.EMAIL,
                detector=self.name + "_spaced",
                stage=self.stage,
                confidence=0.9,
                value=text[start:end],
                metadata=(("note", "characters separated by spaces"),),
            )


# --- Australian phone numbers ----------------------------------------------
# Candidate pattern only. Shape is deliberately permissive; _normalise
# enforces the real Australian numbering plan (2-digit area codes, 04 mobiles,
# 1300/1800 service numbers), which is what keeps false positives out.
# Leftmost-longest behaviour means "(03) 9123 4567" is captured whole rather
# than from the "9123" onwards.
_PHONE_CANDIDATE = re.compile(
    # A leading "+" is consumed by the match, so exclude it from the guard;
    # otherwise "+61 412 345 678" starts matching at "61" and leaves the "+".
    r"(?<![\w\d.+])"
    r"(?P<international>\+?61[\s.-]?)?"
    r"(?:"
    r"\(\d{2,4}\)"            # "(03)" or "(0412)"
    # 1-4 digits: covers the national "03"/"0412"/"1300" forms and the single
    # area-code digit that follows an international prefix, "+61 3 9123 4567".
    r"|\d{1,4}"
    r")"
    r"(?:[\s.-]?\d){5,10}"
    r"(?![\d])"
)

# The first digit of an Australian landline area code: 2 (NSW/ACT), 3 (VIC),
# 7 (QLD) or 8 (WA/SA/NT). The second digit of the area code varies by region,
# so validating on the first digit keeps the rule short and still rejects
# numbers from other countries that happen to start with 0.
_AU_LANDLINE_LEADING_DIGITS = frozenset("2378")




class PhoneDetector(RegexDetector):
    """Australian phone numbers, normalised and shape-checked.

    Accepts ``+61``/``61`` international forms and local ``0x`` forms. Mobile
    numbers (prefix ``04``) and ``1300``/``1800`` service numbers are reported
    with full confidence; landlines require a recognised area code.
    """

    name = "phone"
    category = Category.PHONE

    def patterns(self):
        return (_PHONE_CANDIDATE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        # Folds fullwidth digits, which are common in text pasted from a phone.
        # Length preserving, so every span below is a valid original offset.
        aligned = normalise_aligned(text)
        for match in _PHONE_CANDIDATE.finditer(aligned):
            start, end = match.span()
            # Trailing punctuation such as the "." at the end of a sentence.
            while end > start and aligned[end - 1] in ".,;:":
                end -= 1

            digits = re.sub(r"\D", "", aligned[start:end])
            normalised = self._normalise(digits)
            if normalised is None:
                continue
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=self._confidence(normalised),
                value=text[start:end],
                metadata=(("normalised", normalised),),
            )

    @staticmethod
    def _normalise(digits: str) -> str | None:
        """Return the canonical form, or ``None`` if the numbering plan rejects it.

        Australian national numbers:
          landline  ``0`` + area (2, 3, 7 or 8) + 8 subscriber digits = 10
          mobile    ``04`` + 9 digits = 11
          service   ``0`` + ``1300``/``1800`` + 6 digits = 11

        The ``+61`` prefix is detected from the digits themselves rather than
        from the regex group, so ``61 3 9123 4567`` and ``+61 412 345 678`` are
        both recognised.
        """
        if digits.startswith("61"):
            body = digits[2:]
            return f"+61{body}" if _is_au_nsn(body) else None

        # 1300/1800 are quoted with or without the trunk zero in practice.
        if re.fullmatch(r"(?:1300|1800)\d{6}", digits):
            return "0" + digits

        if not digits.startswith("0"):
            return None
        body = digits[1:]

        if body.startswith("4") and len(body) == 9:
            return "04" + body[1:]
        if len(body) == 9 and body[0] in _AU_LANDLINE_LEADING_DIGITS:
            return "0" + body
        return None

    @staticmethod
    def _confidence(normalised: str) -> float:
        # 04 mobiles and 1300/1800 service numbers are unambiguous.
        if normalised.startswith(("04", "01300", "01800")):
            return 1.0
        return 0.95


def _is_au_nsn(body: str) -> bool:
    """True for a nine-digit Australian national significant number."""
    if len(body) != 9:
        return False
    return body[0] == "4" or body[0] in _AU_LANDLINE_LEADING_DIGITS


# --- IP and MAC -------------------------------------------------------------
IPV4_RE = re.compile(
    # Guards exclude a word character or dot on either side, so "v1.2.3.4" and
    # "1.2.3.4.5" do not yield a partial match. The release context guard drops
    # "Release 2.4.0.1" and "version 2.4.0.1": in survey free text a dotted quad
    # is far more often a version string than a network address.
    r"(?<![\w.])"
    r"(?<!release\s)(?<!version\s)(?<!build\s)(?<!v)\b"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    r"(?![\w.])",
    re.IGNORECASE,
)
IPV6_RE = re.compile(
    r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,7}:\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}\b"
    r"|\b[A-Fa-f0-9]{1,4}:(?::[A-Fa-f0-9]{1,4}){1,6}\b"
    r"|::(?:[A-Fa-f0-9]{1,4}:){0,6}[A-Fa-f0-9]{1,4}\b"
)
MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")


class IPv4Detector(CompiledPatternDetector):
    name = "ipv4"
    regex = IPV4_RE
    category = Category.IPV4


class IPv6Detector(CompiledPatternDetector):
    name = "ipv6"
    regex = IPV6_RE
    category = Category.IPV6


class MACDetector(CompiledPatternDetector):
    name = "mac"
    regex = MAC_RE
    category = Category.MAC


# --- URL credentials --------------------------------------------------------
URL_CREDENTIAL_RE = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+(?::[^\s/@]*)?@|"
    r"\b(?:api[_-]?key|apikey|token|access[_-]?token|secret|password|passwd|pwd)"
    r"[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+=]{8,}",
    re.IGNORECASE,
)


class URLCredentialDetector(RegexDetector):
    """Credentials embedded in URLs and key=value secret pairs."""

    name = "url_credential"
    category = Category.URL_CREDENTIAL

    def patterns(self):
        return (URL_CREDENTIAL_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category


# --- Dates ------------------------------------------------------------------
_DAYS = (
    "mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday"
    "|friday|saturday|sunday"
)
_MONTHS = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    "|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# dd/mm/yyyy and yyyy/mm/dd. Both orders are seen in AU data sets, so both are
# matched and the ambiguity is reported in metadata for downstream sorting.
DATE_AU_RE = re.compile(
    r"\b(?:(?P<d1>\d{1,2})[/-](?P<m1>\d{1,2})[/-](?P<y1>\d{2,4})"
    r"|(?P<y2>\d{4})[/-](?P<m2>\d{1,2})[/-](?P<d2>\d{1,2}))\b"
)
_MONTH_NUMBER = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

DATE_NAMED_RE = re.compile(
    rf"\b(?:(?P<d1>\d{{1,2}})\s+(?P<m1>{_MONTHS})\s+(?P<y1>\d{{4}})"
    rf"|(?P<d2>\d{{1,2}})\s+(?P<m2>{_MONTHS})"
    rf"|(?P<m3>{_MONTHS})\s+(?P<d3>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y3>\d{{4}})"
    rf"|(?:{_DAYS})\s+(?P<d4>\d{{1,2}})\s+(?:of\s+)?(?P<m4>{_MONTHS})\s+(?P<y4>\d{{4}}))\b",
    # Month and day names are matched case-insensitively; the captured group
    # keeps the source casing, so _MONTH_NUMBER lookups are case-folded.
    re.IGNORECASE,
)

_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


class DateDetector(RegexDetector):
    """Calendar dates in Australian (day-first) and ISO orders.

    Only real dates pass: ``31/02/2024`` and month ``13`` are rejected, which
    removes a large class of false positives on version strings and ratio
    figures such as ``1/2/3``.
    """

    name = "date"
    stage = Stage.DETERMINISTIC

    def patterns(self):
        return (DATE_AU_RE, DATE_NAMED_RE)

    def category_for(self, match: re.Match[str]) -> Category:
        parsed = self._parse(match)
        if parsed is None:
            return Category.DATE_AU
        return Category.DATE_ISO if parsed[3] == "iso" else Category.DATE_AU

    def detect(self, text: str) -> Iterator[Finding]:
        # Length preserving, so fullwidth digits and smart punctuation are
        # handled while every span stays a valid offset into the original.
        aligned = normalise_aligned(text)
        for pattern in (DATE_AU_RE, DATE_NAMED_RE):
            for match in pattern.finditer(aligned):
                parsed = self._parse(match)
                if parsed is None:
                    continue
                day, month, year, order = parsed
                start, end = match.span()
                iso = f"{year:04d}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"
                yield Finding(
                    span=Span(start, end),
                    category=Category.DATE_ISO if order == "iso" else Category.DATE_AU,
                    detector=self.name,
                    stage=self.stage,
                    # A date with no year is weaker evidence on its own.
                    confidence=1.0 if year else 0.8,
                    value=text[start:end],
                    metadata=(("iso", iso), ("order", order)),
                )

    @staticmethod
    def _parse(match: re.Match[str]) -> tuple[int, int, int | None, str] | None:
        g = {k: (v.lower() if isinstance(v, str) else v) for k, v in match.groupdict().items()}

        if g.get("d1") and g.get("y1") and g.get("m1") not in _MONTH_NUMBER:
            # "12/03/1985"
            day, month, year = int(g["d1"]), int(g["m1"]), int(g["y1"])
            order = "day_first"
        elif g.get("d1") and g.get("m1") in _MONTH_NUMBER and g.get("y1"):
            # "12 March 2024"
            day, month, year = int(g["d1"]), _MONTH_NUMBER[g["m1"]], int(g["y1"])
            order = "day_first"
        elif g.get("y2") and g.get("m2") and g.get("d2"):
            # "2024-03-12"
            day, month, year = int(g["d2"]), int(g["m2"]), int(g["y2"])
            order = "iso"
        elif g.get("d3") and g.get("m3") and g.get("y3"):
            # "March 12, 2024"
            day, month, year = int(g["d3"]), _MONTH_NUMBER[g["m3"]], int(g["y3"])
            order = "month_first"
        elif g.get("d4") and g.get("m4") and g.get("y4"):
            # "Tuesday 12 March 2024"
            day, month, year = int(g["d4"]), _MONTH_NUMBER[g["m4"]], int(g["y4"])
            order = "day_first"
        elif g.get("d2") and g.get("m2") in _MONTH_NUMBER:
            # Day and month with no year, e.g. "12 March".
            day, month, year = int(g["d2"]), _MONTH_NUMBER[g["m2"]], None
            order = "day_first"
        else:
            return None

        if not 1 <= month <= 12:
            return None
        if not 1 <= day <= _DAYS_IN_MONTH[month - 1]:
            return None
        if year is not None:
            if year < 100:
                year += 2000 if year <= 69 else 1900
            if not 1900 <= year <= 2099:
                return None
        return day, month, year, order


# --- Postcodes and addresses ------------------------------------------------
# Real postcode allocations per state, as (lo, hi) ranges. Validating against
# these instead of \d{4} removes matches on years, quantities and amounts.
_STATE_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "NSW": ((1000, 1999), (2000, 2599), (2620, 2899), (2900, 2999)),
    "ACT": ((2600, 2618), (2620, 2631), (2900, 2920)),
    "VIC": ((3000, 3999),),
    "QLD": ((4000, 4499), (4500, 4911), (4920, 4929), (4940, 4950)),
    "SA": ((5000, 5999),),
    "WA": ((6000, 6799), (6800, 6999)),
    "NT": ((800, 899),),
    "TAS": ((7000, 7999),),
    # Other states/territories, for completeness.
    "QLD_OTHER": ((4920, 4999),),
}


def valid_au_postcode(code: str) -> bool:
    """True when ``code`` falls inside a postcode allocation for any state."""
    if not re.fullmatch(r"\d{4}", code):
        return False
    value = int(code)
    if value == 0:
        return False
    return any(
        lo <= value <= hi
        for ranges in _STATE_RANGES.values()
        for lo, hi in ranges
    )


STATE_CODES = ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT")

# "NSW 2000" / "Sydney NSW 2000" / "2000 NSW"
_STATE_POSTCODE_RE = re.compile(
    rf"\b(?P<state>{'|'.join(STATE_CODES)})\s+(?P<code>\d{{4}})\b"
    rf"|\b(?P<code2>\d{{4}})\s+(?P<state2>{'|'.join(STATE_CODES)})\b"
)

POSTCODE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:post\s*code|postcode|zip|suburb|suburb/town|city)\b"
)

_STREET_SUFFIX = (
    "street|st|road|rd|avenue|ave|parade|pde|highway|hwy|place|pl|square|sq|court|ct"
    "|crescent|cres|close|cl|drive|dr|lane|ln|terrace|tce|boulevard|bvd|way|walk|circle|cir"
    "|esplanade|esp|ter|view|vd|rise|walkway|wy|loop|grove|gr|bay|cove|cv|mews|parkway|pkwy"
    "|triangle|plaza|quay|qy|alley|aly|promenade|prom|bend|glade|haven|vdge"
)
_UNIT_RE = r"(?:unit|u|flat|apt|apartment|suite|ste)\s*\d+[a-z]?(?:/\d+[a-z]?)?"
_STREET_RE = re.compile(
    rf"\b\d{{1,5}}\s+"
    rf"(?:(?:{_UNIT_RE})\s+)?"
    rf"(?:[A-Z][A-Za-z'\u2019\-]+\s+){{1,4}}"
    rf"(?:{_STREET_SUFFIX})\b",
    re.IGNORECASE,
)
_PO_BOX_RE = re.compile(r"\bPO\s+Box\s+\d+\b", re.IGNORECASE)


class StatePostcodeDetector(RegexDetector):
    """A postcode that is qualified by a state abbreviation is unambiguous."""

    name = "state_postcode"
    category = Category.POSTCODE

    def patterns(self):
        return (_STATE_POSTCODE_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        for match in _STATE_POSTCODE_RE.finditer(text):
            code = match.group("code") or match.group("code2")
            state = match.group("state") or match.group("state2")
            if not code or not valid_au_postcode(code):
                continue
            # Redact the whole "STATE 2000" pair: the pair is the location.
            yield Finding(
                span=Span(*match.span()),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=1.0,
                value=match.group(),
                metadata=(("postcode", code), ("state", state)),
            )


class PostcodeDetector(RegexDetector):
    """Bare four-digit postcode, accepted only with postal context nearby."""

    name = "postcode"
    category = Category.POSTCODE

    def patterns(self):
        return (re.compile(r"\b\d{4}\b"),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        # Skip anything already claimed by a state-qualified pair.
        claimed = [f.span for f in StatePostcodeDetector().detect(text)]
        for match in re.finditer(r"\b\d{4}\b", text):
            start, end = match.span()
            if any(s.start < end and start < s.end for s in claimed):
                continue
            code = match.group()
            if not valid_au_postcode(code):
                continue
            window = text[max(0, start - 30) : end + 30]
            if not POSTCODE_CONTEXT_RE.search(window):
                continue
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=0.75,
                value=code,
                metadata=(("context", "label"),),
            )


class StreetAddressDetector(RegexDetector):
    """Street addresses, PO boxes and unit/street pairs."""

    name = "street_address"
    category = Category.STREET_ADDRESS

    def patterns(self):
        return (_STREET_RE, _PO_BOX_RE)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if _PO_BOX_RE.fullmatch(match.group()) else 0.85
"""Redaction policy and the engine that applies it.

Execution order is the design centre of this module:

1. **Deterministic detectors** run first. They are exact, auditable and
   produce high-confidence spans.
2. Deterministic findings are resolved for overlap: a longer span wins over a
   shorter one it contains, and ties break on confidence then declaration
   order, so results are deterministic.
3. **NLP findings** are collected, then filtered against the spans already
   claimed by step 2. An NLP span that touches a deterministic span is
   discarded, never merged or overridden.
4. The surviving spans are applied right-to-left so earlier offsets stay valid.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from . import masking
from .detectors import au_ids, generic, nlp
from .detectors.base import Detector
from .generalise import generalise
from .masking import build_masker
from .types import (
    Category,
    Finding,
    MaskStyle,
    RedactionError,
    Span,
    Stage,
)

#: Style map shape: category (or the string "default") to a mask style.
StyleMap = dict[Category | str, MaskStyle | str]


@dataclass(frozen=True, slots=True)
class Policy:
    """What to detect, how to mask it, and how strict to be.

    Defaults are conservative: high-confidence categories only, a full mask
    replacement, and NLP disabled because it is the slower, less predictable
    stage.
    """

    #: Categories to redact. ``None`` means every category the engine knows.
    categories: frozenset[Category] | None = None
    #: Categories to widen rather than mask. Orthogonal to ``styles``: direct
    #: identifiers can be masked while quasi-identifiers are generalised, which
    #: is usually what a survey analysis needs.
    generalise_categories: frozenset[Category] = frozenset()
    #: Replacement style, per category or as a default for all.
    styles: StyleMap = field(default_factory=dict)
    #: Enable the NLP augmentation stage.
    nlp_enabled: bool = False
    #: NLP backend name: "gazetteer" or "spacy".
    nlp_backend: str = "gazetteer"
    #: spaCy model to load when ``nlp_backend="spacy"``.
    spacy_model: str = "en_core_web_lg"
    #: Drop NLP findings scoring below this.
    nlp_min_confidence: float = 0.55
    #: Drop *all* findings below this confidence, deterministic included.
    min_confidence: float = 0.0
    #: Minimum evidence before a bare four-digit number counts as a postcode.
    require_context: bool = True
    #: HMAC salt for hashed/token styles. Keep it secret and stable.
    salt: bytes | None = None
    #: Replacement used for unclassified or unmatched categories.
    default_style: MaskStyle = MaskStyle.MASK
    #: Replacement character for mask styles.
    mask_char: str = "*"
    #: Characters kept by ``MaskStyle.PARTIAL``, at each end.
    partial_keep_prefix: int = 0
    partial_keep_suffix: int = 4
    #: When true, raise if a value would be masked but the span is empty.
    strict: bool = False

    def style_for(self, category: Category) -> MaskStyle:
        """Replacement style for ``category``, falling back to the default."""
        style = self.styles.get(category, self.styles.get("default"))
        if style is None:
            return self.default_style
        return style if isinstance(style, MaskStyle) else MaskStyle(str(style))

    def enabled(self, category: Category) -> bool:
        return self.categories is None or category in self.categories


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of a redaction pass."""

    text: str
    findings: tuple[Finding, ...]

    @cached_property
    def counts(self) -> Counter[str]:
        """Finding counts per category."""
        return Counter(f.category.value for f in self.findings)

    @cached_property
    def stage_counts(self) -> Counter[str]:
        return Counter(f.stage.value for f in self.findings)

    @property
    def redacted_categories(self) -> frozenset[Category]:
        return frozenset(f.category for f in self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def summary(self) -> str:
        """A one-line description, safe to log."""
        parts = ", ".join(f"{name}={count}" for name, count in sorted(self.counts.items()))
        deterministic = self.stage_counts.get(Stage.DETERMINISTIC.value, 0)
        nlp = self.stage_counts.get(Stage.NLP.value, 0)
        return f"{len(self.findings)} findings ({deterministic} deterministic, {nlp} nlp): {parts}"


# Order matters for tie-breaking: earlier detectors win equal-confidence ties.
DEFAULT_DETECTORS: tuple[type[Detector], ...] = (
    # Strong identifiers first: a validated ABN beats a bare nine-digit span.
    au_ids.TFNDetector,
    au_ids.ABNDetector,
    au_ids.ACNDetector,
    au_ids.MedicareDetector,
    au_ids.MedicareEnrolmentDetector,
    au_ids.CreditCardDetector,
    au_ids.IBANDetector,
    au_ids.BSBDetector,
    au_ids.DriversLicenceDetector,
    au_ids.PassportDetector,
    # Then contact and network details.
    generic.URLCredentialDetector,
    generic.EmailDetector,
    generic.PhoneDetector,
    generic.IPv4Detector,
    generic.IPv6Detector,
    generic.MACDetector,
    # Then locations and dates, which are noisier.
    generic.StatePostcodeDetector,
    generic.PostcodeDetector,
    generic.StreetAddressDetector,
    generic.DateDetector,
)


class Redactor:
    """Applies a :class:`Policy` to text.

    The detector instances are built once at construction because the majority
    of them compile regexes; reusing a ``Redactor`` across many documents is
    the intended usage.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        detectors: Sequence[Detector] | None = None,
    ) -> None:
        self.policy = policy or Policy()
        self._detectors: tuple[Detector, ...] = (
            tuple(detectors) if detectors is not None
            else tuple(cls() for cls in DEFAULT_DETECTORS)  # type: ignore[call-arg]
        )
        self._order = {d.name: index for index, d in enumerate(self._detectors)}
        self._nlp_backend = self._build_nlp_backend()

    def _build_nlp_backend(self) -> nlp.GazetteerBackend | nlp.SpacyBackend | None:
        if not self.policy.nlp_enabled:
            return None
        if self.policy.nlp_backend == "spacy":
            backend = nlp.SpacyBackend(self.policy.spacy_model)
            if backend.available():
                return backend
            # Fall back rather than fail: a missing model should degrade
            # coverage, not raise in production.
            return nlp.GazetteerBackend()
        return nlp.GazetteerBackend()

    def with_policy(self, **changes: Any) -> Redactor:
        """A new redactor with the same detectors and an updated policy."""
        return Redactor(policy=replace(self.policy, **changes), detectors=self._detectors)

    # -- public API ---------------------------------------------------------
    def scan(self, text: str) -> list[Finding]:
        """Detect PII without altering ``text``."""
        return list(self._resolve(text))

    def redact(self, text: str) -> RedactionResult:
        """Redact ``text`` and return the new string plus all findings."""
        if not isinstance(text, str):
            raise RedactionError(f"expected str, got {type(text).__name__}")
        findings = list(self._resolve(text))
        return RedactionResult(text=self._apply(text, findings), findings=tuple(findings))

    def redact_text(self, text: str) -> str:
        """Convenience wrapper returning only the redacted string."""
        return self.redact(text).text

    # -- internals ----------------------------------------------------------
    def _resolve(self, text: str) -> Iterable[Finding]:
        already = _placeholder_regions(text)
        deterministic = self._run_deterministic(text)
        claimed = [f.span for f in deterministic]
        nlp_findings = self._run_nlp(text)
        surviving = self._filter_nlp(nlp_findings, claimed)
        if already:
            # Skip anything found inside a token this module produced earlier.
            # Without this, a second pass matches the "award_" inside
            # "award_97b87895" and the token grows a new hash every time.
            deterministic = [f for f in deterministic if not _inside_any(f.span, already)]
            surviving = [f for f in surviving if not _inside_any(f.span, already)]
        return [*deterministic, *surviving]

    def _run_deterministic(self, text: str) -> list[Finding]:
        collected: list[Finding] = []
        for detector in self._detectors:
            for finding in detector.detect(text):
                if not self.policy.enabled(finding.category):
                    continue
                if finding.confidence < self.policy.min_confidence:
                    continue
                collected.append(finding)
        return resolve_overlaps(collected, self._order)

    def _run_nlp(self, text: str) -> list[Finding]:
        backend = self._nlp_backend
        if backend is None:
            return []
        collected: list[Finding] = []
        for finding in backend.detect(text):
            if not self.policy.enabled(finding.category):
                continue
            if finding.confidence < self.policy.nlp_min_confidence:
                continue
            if finding.confidence < self.policy.min_confidence:
                continue
            collected.append(finding)
        return collected

    @staticmethod
    def _filter_nlp(findings: list[Finding], claimed: list[Span]) -> list[Finding]:
        """Drop any NLP span touching a span already claimed deterministically."""
        if not claimed:
            return findings
        surviving = [
            f for f in findings if not any(f.span.overlaps(s) for s in claimed)
        ]
        return resolve_overlaps(surviving, {})

    def _apply(self, text: str, findings: Sequence[Finding]) -> str:
        # Right to left keeps earlier offsets valid as we rewrite.
        ordered = sorted(findings, key=lambda f: f.span.start, reverse=True)
        out = text
        for finding in ordered:
            raw = finding.value if finding.value is not None else finding.span.extract(text)
            replacement = self._replacement(finding, raw)
            if self.policy.strict and not replacement and finding.span.length:
                raise RedactionError(
                    f"strict mode produced an empty replacement for {finding.category.value}"
                )
            out = out[: finding.span.start] + replacement + out[finding.span.end :]
        return out

    def _replacement(self, finding: Finding, raw: str) -> str:
        """Generalise when configured for the category, otherwise mask."""
        if finding.category in self.policy.generalise_categories:
            return generalise(finding.category, raw)
        style = self.policy.style_for(finding.category)
        masker = build_masker(
            style,
            finding.category.value,
            self.policy.salt,
            self.policy.mask_char,
            self.policy.partial_keep_prefix,
            self.policy.partial_keep_suffix,
        )
        return masker(raw)


#: Matches a redaction marker already present in ``text``, in either the label
#: or the pseudonym form. Any detection lying wholly inside one of these is
#: treated as already redacted.
#:
#: No ``\b`` around the label alternative: "[" is not a word character, so a
#: word boundary before it only matches when the preceding character is a word
#: character. With one, "[AWARD]" was re-detected as the award term and grew a
#: second pair of brackets on every pass.
_PLACEHOLDER_RE = re.compile(
    rf"{masking.LABEL_PATTERN}|\b{masking.TOKEN_PATTERN}\b"
)


def _placeholder_regions(text: str) -> list[Span]:
    """Spans of ``text`` that are already-redacted markers."""
    if not text or ("[" not in text and "_" not in text):
        return []
    return [Span(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]


def _inside_any(span: Span, regions: list[Span]) -> bool:
    return any(span.start >= r.start and span.end <= r.end for r in regions)


def resolve_overlaps(findings: Iterable[Finding], order: dict[str, int]) -> list[Finding]:
    """Drop overlapping findings, keeping the most specific one.

    Ranking, strongest first: a span that covers more characters beats a span
    contained in it, then deterministic beats NLP, then higher confidence, then
    earlier detector in ``order``. Sorting on ``(-length, stage, -confidence,
    detector index)`` makes the outcome independent of input iteration order.
    """
    ranked = sorted(
        findings,
        key=lambda f: (
            -f.span.length,
            0 if f.stage is Stage.DETERMINISTIC else 1,
            -f.confidence,
            order.get(f.detector, len(order)),
            f.span.start,
        ),
    )
    kept: list[Finding] = []
    for candidate in ranked:
        if any(candidate.span.overlaps(k.span) for k in kept):
            continue
        kept.append(candidate)
    kept.sort(key=lambda f: f.span.start)
    return kept


def redact(text: str, policy: Policy | None = None) -> str:
    """Module-level convenience: build a redactor and return the masked text.

    Convenient for one-offs; construct a :class:`Redactor` and reuse it when
    processing many documents.
    """
    return Redactor(policy).redact_text(text)
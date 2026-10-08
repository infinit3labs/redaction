"""Detector protocol and helpers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..types import Category, Finding, Span, Stage


@runtime_checkable
class Detector(Protocol):
    """Finds candidate spans in text.

    Detectors report *candidates*; the engine resolves overlaps, so a
    detector may safely report a broad span that a more specific detector
    later narrows.
    """

    name: str

    def detect(self, text: str) -> Iterator[Finding]:
        ...


class RegexDetector:
    """Base class for detectors built from one or more regexes.

    Subclasses provide ``name`` and ``detect``; this class only supplies the
    scan loop, group handling and confidence plumbing so each concrete
    detector stays a few lines long.
    """

    name = "regex"
    stage = Stage.DETERMINISTIC
    confidence = 1.0

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        raise NotImplementedError

    def category_for(self, match: re.Match[str]) -> Category:
        raise NotImplementedError

    def confidence_for(self, match: re.Match[str]) -> float:
        return self.confidence

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {}

    def detect(self, text: str) -> Iterator[Finding]:
        for pattern in self.patterns():
            for match in pattern.finditer(text):
                # A named or explicit group narrows the redacted span; otherwise
                # the whole match is the span.
                if "span" in pattern.groupindex:
                    start, end = match.span("span")
                else:
                    start, end = match.span()
                if start < 0:
                    continue
                yield Finding(
                    span=Span(start, end),
                    category=self.category_for(match),
                    detector=self.name,
                    stage=self.stage,
                    confidence=self.confidence_for(match),
                    value=text[start:end],
                    metadata=tuple(self.metadata_for(match).items()),
                )


class SinglePatternDetector(RegexDetector):
    """A detector backed by exactly one regex and one category."""

    pattern: str
    category: Category
    flags: int = 0

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (re.compile(self.pattern, self.flags),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category


class CompiledPatternDetector(RegexDetector):
    """A detector backed by one already-compiled regex and one category.

    Prefer this over :class:`SinglePatternDetector` wherever a module-level
    regex exists. Carrying the pattern as a string means recompiling it, and
    recompiling drops every flag that was set on the original object, so a
    detector can silently behave differently from the regex it is named
    after. That is not a hypothetical: ``IPv4Detector`` lost its
    ``re.IGNORECASE`` that way and stopped recognising "Release 2.4.0.1" as a
    version string.
    """

    regex: re.Pattern[str]
    category: Category

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self.regex,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category
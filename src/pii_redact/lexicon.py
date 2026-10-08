"""Term lexicon with fuzzy matching, for organisational attribution.

The deterministic rules cannot know that "the Kestrel rollout" or "our LMS"
identify a specific employer. That requires a vocabulary. This module provides
one, loadable at runtime, and tolerates the misspellings and spacing variants
that free text produces.

Two matching strategies, cheapest first:

1. **Phrase match** over folded tokens, longest term first, so "Project
   Kestrel" wins over a bare "Kestrel".
2. **Token match** against a SymSpell-style delete index, which catches a
   respondent typing "Kestral" or "Kestrell" for "Kestrel".

Matching is deliberately conservative. A term that can be mistaken for an
ordinary English word ("Care", "Grade", "Shift") is only matched exactly, and
a term shorter than :data:`MIN_FUZZY_LENGTH` never fuzzy matches, because edit
distance is meaningless at that size and every short string is near every
other short string.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .normalise import token_key, tokenise
from .types import Category

#: Below this length, fuzzy matching is disabled: one edit in a five-character
#: word matches far too many unrelated words.
MIN_FUZZY_LENGTH = 6

#: Deletion distance. 1 handles "kestral"; 2 handles "kestrl" and
#: "tiffanyy" without exploding the index.
DEFAULT_MAX_DISTANCE = 1

#: A term matching more than this fraction of a comment is treated as "the
#: whole comment is about our internal system", which is a different signal
#: from an incidental mention.
DENSE_MENTION_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class Term:
    """One lexicon entry.

    ``category`` selects the finding type and, with ``HASH``/``TOKEN`` masking,
    the pseudonym prefix. ``aliases`` covers the abbreviations and expansions
    people actually type ("lms", "learning management system").
    """

    term: str
    category: Category
    aliases: tuple[str, ...] = ()
    #: Free-text label used in risk explanations, e.g. "internal system".
    label: str = "internal term"
    #: Match this term only when it is written correctly.
    #:
    #: For a term that is also an ordinary English word, a one-edit tolerance
    #: buys nothing: "intranet" misspelled is still an intranet worth catching,
    #: but "risk assessment" misspelled is just a typo in a sentence about
    #: anything. Delete-index matching also lets a short term capture unrelated
    #: tokens that happen to be one deletion away, which is a false positive
    #: with no upside. Set this on everyday vocabulary and on short forms.
    exact_only: bool = False

    @property
    def key(self) -> str:
        return token_key(self.term)

    def all_forms(self) -> tuple[str, ...]:
        return (self.term, *self.aliases)


@dataclass(frozen=True, slots=True)
class LexiconMatch:
    """A lexicon hit, resolved back to original-string offsets."""

    span: tuple[int, int]
    matched_text: str
    canonical: str
    category: Category
    label: str
    fuzzy: bool

    @property
    def distance(self) -> int:
        return 1 if self.fuzzy else 0


@dataclass(frozen=True)
class Lexicon:
    """An immutable, pre-indexed term set. Build once, reuse across records.

    Constructing the delete index is the expensive part, so a ``Lexicon``
    should be built on the driver and shipped to workers, or cached per
    process. It is a frozen dataclass, but its indexes are built lazily and
    cached on the instance, so it must not be mutated after first use.
    """

    terms: tuple[Term, ...] = ()
    #: Tolerated edit distance for single-token terms.
    max_distance: int = DEFAULT_MAX_DISTANCE

    _phrase_index: dict[str, Term] = field(default=None, compare=False, repr=False)
    _delete_index: dict[str, Term] = field(default=None, compare=False, repr=False)
    _max_phrase_tokens: int = field(default=1, compare=False, repr=False)
    _built: bool = field(default=False, compare=False, repr=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_terms(cls, terms: Iterable[Term], *, max_distance: int = DEFAULT_MAX_DISTANCE) -> Lexicon:
        return cls(terms=tuple(terms), max_distance=max_distance)

    @classmethod
    def from_json(cls, path: str | Path, *, max_distance: int = DEFAULT_MAX_DISTANCE) -> Lexicon:
        """Load a lexicon from a JSON file.

        Expected shape::

            {
              "max_distance": 1,
              "terms": [
                {"term": "Project Kestrel",
                 "category": "internal_term",
                 "aliases": ["kestrel", "proj kestrel"],
                 "label": "internal project"}
              ]
            }
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_spec(data, max_distance=max_distance)

    @classmethod
    def from_spec(cls, spec: Mapping[str, object], *, max_distance: int | None = None) -> Lexicon:
        """Build from an already-parsed mapping. See :meth:`from_json`."""
        distance = spec.get("max_distance", max_distance)
        raw_terms = spec.get("terms", ())
        terms = [
            Term(
                term=str(entry["term"]),
                category=Category(str(entry.get("category", Category.INTERNAL_TERM.value))),
                aliases=tuple(str(a) for a in entry.get("aliases", ())),  # type: ignore[union-attr]
                label=str(entry.get("label", "internal term")),
                exact_only=bool(entry.get("exact_only", False)),
            )
            for entry in raw_terms  # type: ignore[union-attr]
        ]
        return cls(
            terms=tuple(terms),
            max_distance=int(distance) if distance is not None else DEFAULT_MAX_DISTANCE,  # type: ignore[arg-type]
        )

    def merged(self, other: Lexicon) -> Lexicon:
        """A new lexicon containing both term sets. Neither is mutated."""
        return Lexicon(
            terms=self.terms + other.terms,
            max_distance=max(self.max_distance, other.max_distance),
        )

    # -- indexing (lazy, once per instance) ---------------------------------
    def _build(self) -> None:
        if self._built:
            return
        phrase: dict[str, Term] = {}
        delete: dict[str, Term] = {}
        max_tokens = 1
        for term in self.terms:
            for form in term.all_forms():
                key = token_key(form)
                if not key:
                    continue
                # First writer wins: a longer, more specific term registered
                # earlier should not be shadowed by a later short alias.
                if key not in phrase:
                    phrase[key] = term
                max_tokens = max(max_tokens, len(form.split()))
                # exact_only terms stay out of the delete index entirely, so
                # they are reachable only by an exact token or exact phrase.
                if _is_single_token(form) and not term.exact_only:
                    for variant in _deletions(key, self.max_distance):
                        delete.setdefault(variant, term)
        object.__setattr__(self, "_phrase_index", phrase)
        object.__setattr__(self, "_delete_index", delete)
        object.__setattr__(self, "_max_phrase_tokens", max_tokens)
        object.__setattr__(self, "_built", True)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    # -- matching -----------------------------------------------------------
    def match(self, text: str) -> list[LexiconMatch]:
        """Every lexicon hit in ``text``, in ascending offset order."""
        self._build()
        if not text.strip():
            return []

        tokens = tokenise(text)
        if not tokens:
            return []

        hits: list[LexiconMatch] = []
        claimed: list[tuple[int, int]] = []

        # Pass 1: multi-word phrases, longest first, so "Project Kestrel" is not
        # preempted by a single-token "Kestrel".
        if self._max_phrase_tokens > 1:
            for width in range(self._max_phrase_tokens, 1, -1):
                for start_index in range(len(tokens) - width + 1):
                    window = tokens[start_index : start_index + width]
                    key = "".join(token_key(tok) for tok, _, _ in window)
                    term = self._phrase_index.get(key)
                    if term is None:
                        continue
                    span = (window[0][1], window[-1][2])
                    if _any_overlap(span, claimed):
                        continue
                    claimed.append(span)
                    hits.append(
                        LexiconMatch(
                            span=span,
                            matched_text=text[span[0] : span[1]],
                            canonical=term.term,
                            category=term.category,
                            label=term.label,
                            fuzzy=False,
                        )
                    )

        # Pass 2: single tokens, exact then fuzzy.
        for token, start, end in tokens:
            if _any_overlap((start, end), claimed):
                continue
            key = token_key(token)
            if not key:
                continue
            term = self._phrase_index.get(key)
            if term is not None:
                claimed.append((start, end))
                hits.append(
                    LexiconMatch(
                        span=(start, end),
                        matched_text=token,
                        canonical=term.term,
                        category=term.category,
                        label=term.label,
                        fuzzy=False,
                    )
                )
                continue
            fuzzy = self._fuzzy_term(key)
            if fuzzy is not None and len(key) >= MIN_FUZZY_LENGTH:
                claimed.append((start, end))
                hits.append(
                    LexiconMatch(
                        span=(start, end),
                        matched_text=token,
                        canonical=fuzzy.term,
                        category=fuzzy.category,
                        label=fuzzy.label,
                        fuzzy=True,
                    )
                )

        hits.sort(key=lambda h: h.span[0])
        return hits

    def _fuzzy_term(self, key: str) -> Term | None:
        """Look up a single-token key allowing up to ``max_distance`` deletions."""
        if len(key) < MIN_FUZZY_LENGTH:
            return None
        best: tuple[int, int, Term] | None = None
        for distance in range(0, self.max_distance + 1):
            for variant in _deletions(key, distance):
                term = self._delete_index.get(variant)
                if term is None:
                    continue
                candidate = (distance, -len(term.key), term)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if best is not None:
                return best[2]
        return None

    # -- reporting ----------------------------------------------------------
    def describe(self) -> list[dict[str, str]]:
        """A JSON-serialisable summary, for shipping a lexicon to workers."""
        return [
            {
                "term": t.term,
                "category": t.category.value,
                "label": t.label,
                "exact_only": t.exact_only,
                "aliases": list(t.aliases),
            }
            for t in self.terms
        ]


def _is_single_token(form: str) -> bool:
    """True when a form was written as a single whitespace-delimited word.

    Must be asked of the raw form, never of a folded key. ``token_key`` strips
    spaces, so once a key exists this question cannot be answered from it: the
    key for ``"toolbox talk"`` is ``"toolboxtalk"``, indistinguishable from a
    single eleven-letter word. Asking it anyway made the guard vacuously true
    and put every phrase in the delete index, which both let phrases fuzzy-match
    and inflated the index about twentyfold.
    """
    return bool(form) and " " not in form


def _deletions(key: str, distance: int) -> list[str]:
    """Every string reachable by deleting up to ``distance`` characters.

    SymSpell's trick: indexing deletions rather than computing distances at
    lookup time turns fuzzy matching into a dict hit. Set membership does the
    distance filtering for free.
    """
    if distance <= 0:
        return [key]
    seen = {key}
    frontier = [key]
    for _ in range(distance):
        next_frontier: list[str] = []
        for variant in frontier:
            for index in range(len(variant)):
                candidate = variant[:index] + variant[index + 1 :]
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(seen)


def _any_overlap(span: tuple[int, int], claimed: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in claimed)


def load_builtin_lexicon() -> Lexicon:
    """The bundled seed lexicon.

    This is deliberately generic: Australian WHS and employment vocabulary plus
    the well-known national award identifiers. It will not know your internal
    project names. Its purpose is to catch the vocabulary that narrows an
    employer to a small set on its own, and to give ``Lexicon`` a
    known-good starting point to merge a bespoke list into.
    """
    from importlib.resources import files

    resource = files("pii_redact").joinpath("resources/lexicon.json")
    return Lexicon.from_json(resource)


__all__ = [
    "DEFAULT_MAX_DISTANCE",
    "MIN_FUZZY_LENGTH",
    "Lexicon",
    "LexiconMatch",
    "Term",
    "load_builtin_lexicon",
]
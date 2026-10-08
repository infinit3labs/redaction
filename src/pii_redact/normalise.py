"""Offset-preserving text normalisation for free text.

Survey free text is messy: respondents paste from phones, use curly quotes,
type inconsistent spacing, and misspell names. Matching the raw string
directly makes both recall and offsets unreliable.

Two levels of normalisation are provided, and the distinction matters:

``normalise``
    Conservative. Case folding, Unicode normalisation, unified dashes and
    quotes, whitespace collapsing. Preserves enough structure that token
    boundaries survive. Used for ordinary regex detectors.

``FoldedText``
    Aggressive. Keeps only alphanumerics and folds leetspeak, so ``Bui11ing``,
    ``Bullying`` and ``b u i l l y n g`` collapse together. Used only for
    lexicon lookup, where a misspell must still match.

Both preserve a map back to the original string, so every span reported by a
detector is expressed in original-string offsets and can be applied safely.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Punctuation that respondents substitute for spaces, or that smart-input
# keyboards inject. Treating these as separators is what makes
# "O'Brien" == "O Brien" == "O-Brien".
_SEPARATORS = re.compile(r"[\s _/\\|]+")

# Characters a person might type instead of the letter they mean. Applied only
# in the aggressive fold, never to text that will be reported back to the user.
_LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "l",
        "+": "t",
    }
)


@dataclass(frozen=True, slots=True)
class FoldedText:
    """An aggressively normalised view of ``source`` with a back-map.

    ``text[i]`` corresponds to ``source[source_index[i]]``. Several folded
    characters can map to one source character (Unicode expansion), so the map
    is one-to-many in that direction only.
    """

    source: str
    text: str
    source_index: tuple[int, ...]

    @classmethod
    def build(cls, source: str, *, leet: bool = True) -> FoldedText:
        folded_chars: list[str] = []
        indices: list[int] = []
        for source_pos, char in enumerate(source):
            # Decompose so accented characters fold to their base letter:
            # "José" -> "jose", "Nguyễn" -> "nguyen".
            decomposed = unicodedata.normalize("NFKD", char)
            base = "".join(
                c for c in decomposed if not unicodedata.combining(c) and not unicodedata.category(c).startswith("M")
            )
            base = base.casefold()
            if leet:
                base = base.translate(_LEET)
            for out_char in base:
                if out_char.isalnum():
                    folded_chars.append(out_char)
                    indices.append(source_pos)
        return cls(source=source, text="".join(folded_chars), source_index=tuple(indices))

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Translate a span in folded space to original-string offsets."""
        if start >= end or not self.source_index:
            return (start, end)
        lo = min(self.source_index[start:end])
        hi = max(self.source_index[start:end]) + 1
        return (lo, hi)

    def contains(self, needle: str) -> bool:
        return needle in self.text

    def __bool__(self) -> bool:
        return bool(self.text)


def normalise(text: str) -> str:
    """Conservative normalisation for regex detectors.

    Case folds, applies Unicode NFKC, unifies smart punctuation, and collapses
    whitespace runs to a single space. Word and digit structure is preserved, so
    a detector matching ``\\d{3}`` still behaves predictably.
    """
    if not text:
        return ""
    # NFKC first: it folds fullwidth digits to ASCII, which is exactly what we
    # want for digit patterns, and composes most compatibility characters.
    text = unicodedata.normalize("NFKC", text)
    text = _EXPANSIONS_RE.sub(lambda m: _EXPANSIONS[m.group()], text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP.get(m.group(), m.group()), text)
    text = _SEPARATORS.sub(" ", text)
    return text.strip()


# Built after the function so the module reads top-down, but referenced inside
# it at call time.
# Smart punctuation a respondent's keyboard inserts, mapped to its ASCII
# equivalent. Applied after NFKC so fullwidth and composed forms already match.
#
# Every entry here maps one character to one character, which is what makes
# normalise_aligned safe for offset-preserving matching.
_PUNCT_RE = re.compile("[‘’‛ʼ′＇`´]|[“”‟″＂]|[–—―−]|[   ⁠]")
_PUNCT_MAP = {
    **{c: "'" for c in "‘’‛ʼ′＇`´"},
    **{c: '"' for c in "“”‟″＂"},
    **{c: "-" for c in "–—―−"},
    **{c: " " for c in "   ⁠"},
}

#: Expansions that change length, so they apply to :func:`normalise` only.
_EXPANSIONS = {"…": "..."}
_EXPANSIONS_RE = re.compile("|".join(map(re.escape, _EXPANSIONS)))


def normalise_aligned(text: str) -> str:
    """Length-preserving normalisation, including fullwidth folding.

    Strictly one output character per input character, so offsets into the
    result are also offsets into the original: whitespace runs are not
    collapsed and nothing is inserted or deleted, guaranteeing
    ``len(result) == len(text)``.

    That guarantee is what lets a detector match on the result and report a
    span against the caller's string. It also forces two choices:

    * NFKC is tried first, because it is the only form that folds fullwidth
      digits and letters to ASCII ("０" -> "0"). Respondents paste those from
      phones and CJK keyboards, and a digit-only detector misses every one.
    * Any character whose normalisation is not exactly one character wide is
      left alone. The ellipsis becomes "..." and the "fi" ligature becomes
      "fi"; both would shift every following offset, so neither is applied.
      Surrogate pairs and emoji are one Python character already and pass
      through unchanged.

    Case is deliberately preserved: callers use this when a pattern depends on
    capitalisation (proper-noun detection). Pass ``re.IGNORECASE`` to the
    pattern where case must not matter.
    """
    if not text:
        return ""
    out: list[str] = []
    for char in text:
        if char in _PUNCT_MAP:
            out.append(_PUNCT_MAP[char])
            continue
        composed = unicodedata.normalize("NFKC", char)
        if len(composed) == 1:
            out.append(composed)
            continue
        # NFKC widened the character, so it is not safe to apply. Try dropping
        # combining marks instead, which collapses "é" to "e" at one to one.
        decomposed = unicodedata.normalize("NFKD", char)
        base = "".join(
            c for c in decomposed if not unicodedata.combining(c) and not unicodedata.category(c).startswith("M")
        )
        out.append(base if len(base) == 1 else char)
    return "".join(out)



def tokenise(text: str) -> list[str]:
    """Split already-normalised text into tokens, keeping their offsets.

    Returns ``(token, start, end)`` triples against ``text``. Used by the
    lexicon matcher, which needs token positions to map a fuzzy match back to
    a span.
    """
    out: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’&.\-]*", text):
        token = match.group()
        start, end = match.span()
        # Trim trailing punctuation that the pattern allowed in, e.g. the dot
        # in "Acme." or the apostrophe in "O'Brien'" at a word boundary.
        while end > start and token[-1] in ".-&'’":
            token = token[:-1]
            end -= 1
        if token:
            out.append((token, start, end))
    return out


def token_key(token: str) -> str:
    """Comparison key for a token: letters and digits only, case folded.

    ``"O'Brien"``, ``"o brien"``, ``"OBrien"`` and ``"o’brien"`` all yield
    ``"obrien"``.
    """
    return "".join(c for c in token.casefold() if c.isalnum())




__all__ = [
    "FoldedText",
    "normalise",
    "normalise_aligned",
    "token_key",
    "tokenise",
]
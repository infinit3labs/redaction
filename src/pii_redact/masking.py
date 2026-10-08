"""Masking strategies.

Each strategy turns a matched value into a replacement. The default is a
full-length character mask, which preserves the shape of the text without
revealing the value. Partial, hash and token styles are available for cases
where referential integrity matters more than minimality.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Callable

from .types import MaskStyle

_DEFAULT_MASK_CHAR = "*"


def full_mask(value: str, char: str = _DEFAULT_MASK_CHAR) -> str:
    """Replace every character, preserving length and whitespace structure."""
    return "".join(char if not c.isspace() else c for c in value)



def partial_mask(
    value: str,
    char: str = _DEFAULT_MASK_CHAR,
    keep_prefix: int = 0,
    keep_suffix: int = 4,
) -> str:
    """Keep a few leading or trailing characters, mask the middle.

    Used for identifiers where a human needs the last few digits to reconcile
    a record without seeing the whole value.
    """
    if len(value) <= keep_prefix + keep_suffix:
        return full_mask(value, char)
    hidden = len(value) - keep_prefix - keep_suffix
    return value[:keep_prefix] + char * hidden + (value[-keep_suffix:] if keep_suffix else "")


def stable_hash(value: str, salt: bytes | None = None, length: int = 12) -> str:
    """A deterministic, salted digest of ``value``.

    Keyed with HMAC-SHA256 when a ``salt`` is configured so an attacker
    cannot confirm a guessed value against the output. Same input and salt
    always produce the same token, which keeps joins across redacted data
    working. The salt must be treated as a secret and rotated deliberately:
    losing it makes all historical hashes unmatchable.
    """
    if salt:
        digest = hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"[{digest[:length]}]"


def consistent_token(category: str, value: str, salt: bytes | None = None) -> str:
    """A category-scoped pseudonym such as ``email_ab12cd34``.

    Derived from HMAC-SHA256 so identical values map to identical tokens
    without revealing the original.

    The token is not opaque to the detectors, and that matters. A second
    redaction pass over already-redacted text would match the ``award_`` inside
    ``award_97b87895`` against the award lexicon term, and emit
    ``award_ab12cd34_97b87895``. Repeated passes grow the token without bound.
    :mod:`pii_redact.engine` therefore treats a token-shaped region as already
    redacted and skips detections inside it, which is what makes redaction
    idempotent.
    """
    salt = salt or _process_salt()
    digest = hmac.new(salt, f"{category}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"{category}_{digest[:8]}"


#: The shape of a token produced above, and the label form. Used by the engine
#: to recognise text it has already redacted. Deliberately specific: an
#: accidental match in survey prose would silently suppress a real detection.
TOKEN_PATTERN = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*_[0-9a-f]{8}"
LABEL_PATTERN = r"\[[A-Z][A-Z ]*\]"

#: Either marker. The engine skips detections lying wholly inside one of these.
PLACEHOLDER_PATTERN = rf"(?:{LABEL_PATTERN}|{TOKEN_PATTERN})"


def is_placeholder(text: str) -> bool:
    """True when ``text`` is exactly one redaction placeholder."""
    return re.fullmatch(PLACEHOLDER_PATTERN, text) is not None



_PROCESS_SALT: bytes | None = None


def set_process_salt(salt: bytes | None) -> None:
    """Set the module-level salt used by :func:`consistent_token`."""
    global _PROCESS_SALT
    _PROCESS_SALT = salt


def _process_salt() -> bytes | None:
    """The configured salt, or an ephemeral one created on first use.

    An ephemeral salt keeps unkeyed use from producing a value that is
    reversible by brute force against the output alone. The cost is that
    tokens do not survive a restart unless a salt is set explicitly.
    """
    global _PROCESS_SALT
    if _PROCESS_SALT is None:
        _PROCESS_SALT = secrets.token_bytes(32)
    return _PROCESS_SALT


Masker = Callable[[str], str]


def category_label(category: str) -> str:
    """A plain category marker such as ``[NAME]``.

    Deliberately carries no information about the value: every name becomes the
    same ``[NAME]``, so two exports of the same data cannot be joined on it.
    ``[ ... ]`` rather than a bare word so the marker is unambiguous in prose and
    survives a second redaction pass, which is what keeps redaction idempotent.
    """
    return f"[{category.upper().replace('_', ' ')}]"


def build_masker(
    style: MaskStyle,
    category: str,
    salt: bytes | None = None,
    mask_char: str = _DEFAULT_MASK_CHAR,
    keep_prefix: int = 0,
    keep_suffix: int = 4,
) -> Masker:
    """Return a callable mapping a raw value to its replacement.

    ``keep_prefix`` and ``keep_suffix`` apply only to ``MaskStyle.PARTIAL``.
    """
    if style is MaskStyle.REMOVE:
        return lambda value: ""
    if style is MaskStyle.LABEL:
        return lambda value: category_label(category)
    if style is MaskStyle.HASH:
        return lambda value: stable_hash(value, salt)
    if style is MaskStyle.TOKEN:
        return lambda value: consistent_token(category, value, salt)
    if style is MaskStyle.PARTIAL:
        return lambda value: partial_mask(value, mask_char, keep_prefix, keep_suffix)
    return lambda value: full_mask(value, mask_char)
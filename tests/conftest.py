"""Shared fixtures."""

from __future__ import annotations

import pytest

from pii_redact import Policy, Redactor


@pytest.fixture
def redactor() -> Redactor:
    """A redactor with the default deterministic policy."""
    return Redactor(Policy())


@pytest.fixture
def nlp_redactor() -> Redactor:
    """A redactor with the NLP augmentation stage enabled."""
    return Redactor(Policy(nlp_enabled=True))
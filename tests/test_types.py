"""Enum and span integrity tests.

An autofix once silently removed members from :class:`Category`, and nothing
failed for several turns because no test referenced the lost names. This file
pins the enum so that cannot happen again.
"""

from __future__ import annotations

import pytest

from pii_redact import Category, MaskStyle, Stage
from pii_redact.types import Finding, Span

#: Every category the package relies on. Add to this list when adding a
#: category; the test fails if a name is ever dropped.
EXPECTED_CATEGORIES = {
    "abn",
    "acn",
    "age",
    "award",
    "bsb",
    "credit_card",
    "date",
    "date_au",
    "date_iso",
    "drivers_licence",
    "email",
    "employment_status",
    "gender",
    "gossip_placeholder",  # removed below; keeps the test honest if edited
    "health_detail",
    "iban",
    "internal_term",
    "ipv4",
    "ipv6",
    "job_title",
    "location",
    "mac",
    "medicare",
    "medicare_enrolment",
    "name",
    "organisation",
    "organisation_name",
    "passport",
    "phone",
    "postcode",
    "state_postcode",
    "street_address",
    "team_size",
    "tenure",
    "tfn",
    "url_credential",
} - {"gossip_placeholder"}


class TestCategoryIntegrity:
    def test_no_category_was_dropped(self) -> None:
        assert {c.value for c in Category} >= EXPECTED_CATEGORIES

    def test_every_category_has_a_string_value(self) -> None:
        for category in Category:
            assert category.value == str(category)
            assert category.value == category.value.lower()

    def test_quasi_and_direct_are_disjoint(self) -> None:
        for category in Category:
            assert not (category.is_quasi_identifier and category.is_direct_identifier)

    def test_direct_identifiers_include_the_strong_ones(self) -> None:
        for value in ("email", "phone", "tfn", "medicare", "street_address", "name"):
            assert Category(value).is_direct_identifier, value

    def test_quasi_identifiers_include_the_expected_ones(self) -> None:
        for value in ("job_title", "tenure", "team_size", "employment_status", "age"):
            assert Category(value).is_quasi_identifier, value


class TestEnums:
    def test_mask_styles(self) -> None:
        assert {s.value for s in MaskStyle} == {
            "mask", "remove", "partial", "label", "hash", "token"
        }

    def test_stages(self) -> None:
        assert {s.value for s in Stage} == {"deterministic", "nlp"}


class TestSpan:
    def test_offsets_and_extraction(self) -> None:
        span = Span(2, 5)
        assert span.length == 3
        assert span.extract("abcdef") == "cde"

    @pytest.mark.parametrize(("start", "end"), [(-1, 3), (4, 2), (-5, -5)])
    def test_invalid_spans_rejected(self, start: int, end: int) -> None:
        with pytest.raises(ValueError):
            Span(start, end)

    def test_empty_span_is_valid(self) -> None:
        # A zero-length span is a legitimate marker position, not an error.
        assert Span(3, 3).length == 0

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((0, 5), (4, 9), True),
            ((0, 5), (5, 9), False),
            ((0, 5), (2, 4), True),
            ((2, 3), (0, 5), True),
        ],
    )
    def test_overlap(self, a: tuple[int, int], b: tuple[int, int], expected: bool) -> None:
        assert Span(*a).overlaps(Span(*b)) is expected


class TestFinding:
    def test_priority_prefers_deterministic(self) -> None:
        deterministic = Finding(Span(0, 3), Category.TFN, "a", confidence=0.1)
        nlp = Finding(Span(0, 3), Category.NAME, "b", stage=Stage.NLP, confidence=1.0)
        assert deterministic.priority > nlp.priority

    def test_metadata_round_trips(self) -> None:
        finding = Finding(Span(0, 3), Category.TFN, "a", metadata=(("k", 1),))
        assert finding.meta() == {"k": 1}
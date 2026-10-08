"""NLP stage tests, independent of the engine."""

from __future__ import annotations

import pytest

from pii_redact.detectors.nlp import GazetteerBackend, SpacyBackend
from pii_redact.types import Category, Stage


@pytest.fixture
def gazetteer() -> GazetteerBackend:
    return GazetteerBackend()


def texts(gazetteer: GazetteerBackend, text: str) -> list[str]:
    return [f.span.extract(text) for f in gazetteer.detect(text)]


class TestGazetteerNames:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Patient: Jane Citizen", "Jane Citizen"),
            ("Contact: Michael O'Brien", "Michael O'Brien"),
            ("Applicant: Anna-Louise Smith", "Anna-Louise Smith"),
        ],
    )
    def test_labelled_names(
        self, gazetteer: GazetteerBackend, text: str, expected: str
    ) -> None:
        assert expected in texts(gazetteer, text)

    def test_honorific_names(self, gazetteer: GazetteerBackend) -> None:
        assert "Dr Fiona Whitlam" in texts(gazetteer, "Reviewed by Dr Fiona Whitlam on site")

    def test_labels_without_capitalised_names_are_ignored(
        self, gazetteer: GazetteerBackend
    ) -> None:
        # A single token is not enough evidence for a person.
        assert texts(gazetteer, "Patient: unknown") == []

    @pytest.mark.parametrize(
        "text",
        [
            "Dear Customer, your order shipped.",
            "Monday is the first day",
            "Patient: Australia Wide",
        ],
    )
    def test_rejects_non_names(self, gazetteer: GazetteerBackend, text: str) -> None:
        assert texts(gazetteer, text) == []

    def test_confidence_above_default_threshold(self, gazetteer: GazetteerBackend) -> None:
        findings = list(gazetteer.detect("Patient: Jane Citizen"))
        assert findings and all(f.confidence >= 0.55 for f in findings)


class TestGazetteerPlaces:
    def test_context_phrase(self, gazetteer: GazetteerBackend) -> None:
        assert "Melbourne" in texts(gazetteer, "Currently in Melbourne for the audit")

    def test_known_city_scores_higher(self, gazetteer: GazetteerBackend) -> None:
        known = next(
            f for f in gazetteer.detect("based in Melbourne") if f.category is Category.LOCATION
        )
        unknown = next(
            f for f in gazetteer.detect("based in Ashbyvale") if f.category is Category.LOCATION
        )
        assert known.confidence > unknown.confidence


class TestGazetteerDemographics:
    def test_age(self, gazetteer: GazetteerBackend) -> None:
        findings = [f for f in gazetteer.detect("Patient is 34 years old")]
        assert any(f.category is Category.AGE for f in findings)

    def test_implausible_age_ignored(self, gazetteer: GazetteerBackend) -> None:
        assert not any(
            f.category is Category.AGE for f in gazetteer.detect("task took 400 hours")
        )

    def test_labelled_sex(self, gazetteer: GazetteerBackend) -> None:
        assert "female" in texts(gazetteer, "Sex: female")


class TestNLPStageTagging:
    def test_every_gazetteer_finding_is_nlp(self, gazetteer: GazetteerBackend) -> None:
        text = "Patient: Jane Citizen, based in Melbourne, 34 years old"
        findings = list(gazetteer.detect(text))
        assert findings
        assert all(f.stage is Stage.NLP for f in findings)


class TestSpacyBackend:
    def test_missing_model_degrades_quietly(self) -> None:
        backend = SpacyBackend("definitely-not-installed-model")
        assert backend.available() is False
        # Must not raise; it simply yields nothing.
        assert list(backend.detect("Patient: Jane Citizen")) == []

    def test_falls_back_when_model_missing(self) -> None:
        from pii_redact import Policy, Redactor

        redactor = Redactor(Policy(nlp_enabled=True, nlp_backend="spacy"))
        result = redactor.redact("Patient: Jane Citizen")
        assert any(f.detector == "gazetteer" for f in result)
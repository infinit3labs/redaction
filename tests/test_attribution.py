"""Organisation attribution and quasi-identifier detector tests."""

from __future__ import annotations

import pytest

from pii_redact.detectors.org import (
    AwardCodeDetector,
    LexiconTermDetector,
    OrganisationNameDetector,
    default_organisation_detectors,
)
from pii_redact.detectors.quasi import (
    EmploymentStatusDetector,
    HealthDetailDetector,
    JobTitleDetector,
    TeamSizeDetector,
    TenureDetector,
    default_quasi_detectors,
)
from pii_redact.lexicon import Lexicon, Term, load_builtin_lexicon
from pii_redact.types import Category


def hits(detector, text: str) -> list[tuple[str, Category]]:
    return [(f.span.extract(text), f.category) for f in detector.detect(text)]


def values(detector, text: str) -> list[str]:
    return [f.span.extract(text) for f in detector.detect(text)]


class TestNamedOrganisations:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I work at Acme Pty Ltd", "Acme Pty Ltd"),
            ("we work at Acme Corporation", "Acme Corporation"),
            ("I was employed by Reticule Health from 2015", "Reticule Health"),
            ("Joined Northline Services last year", "Northline Services"),
            ("my employer is a large hospital trust", None),
        ],
    )
    def test_named_employers(self, text: str, expected: str | None) -> None:
        found = values(OrganisationNameDetector(), text)
        if expected is None:
            assert not found
        else:
            assert expected in found

    def test_trigger_verb_is_stripped_from_the_name(self) -> None:
        # "Joined" is capitalised because it starts the sentence; it is not
        # part of the organisation's name.
        found = values(OrganisationNameDetector(), "Joined Northline Services last year")
        assert found == ["Northline Services"]

    @pytest.mark.parametrize(
        "text",
        [
            "My Safety Committee did nothing",
            "Human Resources ignored my report",
            "Senior Management blocked the claim",
            "The Company has a policy",
            "Everyone at Head Office agreed",
            "my manager was useless",
            "I emailed my supervisor",
        ],
    )
    def test_generic_phrases_are_not_organisations(self, text: str) -> None:
        assert values(OrganisationNameDetector(), text) == []

    def test_no_capitalisation_at_sentence_start(self) -> None:
        # Capitalisation alone carries no information here.
        assert values(OrganisationNameDetector(), "Safety is everyone's concern") == []

    @pytest.mark.parametrize(
        "text",
        [
            # "workplace" is employment framing, so the run of capitalised
            # words behind it looked like a name. The capture held function
            # words, which is the tell that it ran into the sentence.
            "DO NOT BUY WORKPLACE COVER FROM THESE PEOPLE.",
            "WORKPLACE COVER IS THE WORST I HAVE USED.",
        ],
    )
    def test_shouted_sentences_are_not_employers(self, text: str) -> None:
        assert values(OrganisationNameDetector(), text) == []

    def test_labelled_abn_is_caught(self) -> None:
        found = hits(OrganisationNameDetector(), "ABN: 64 004 085 616")
        assert any(category is Category.ABN for _, category in found)

    def test_labelled_acn_is_not_mislabelled_as_an_abn(self) -> None:
        # A label plus nine digits is an ACN. The checksum detectors already
        # handle both correctly, so claiming it here with the wrong category
        # made a valid ACN lose to an eleven-character mislabel.
        found = hits(OrganisationNameDetector(), "ACN 004 085 616 is my employer registration")
        assert not any(category is Category.ABN for _, category in found)

    def test_labelled_acn_still_caught_by_the_checksum_detector(self) -> None:
        from pii_redact.detectors import au_ids

        found = hits(au_ids.ACNDetector(), "ACN 004 085 616 is my employer registration")
        assert any(category is Category.ACN for _, category in found)

    def test_empty_input(self) -> None:
        assert list(OrganisationNameDetector().detect("")) == []
        assert list(OrganisationNameDetector().detect("   ")) == []

    def test_offsets_match_the_original_text(self) -> None:
        text = "I work at Acme Pty Ltd — they ignored me"
        for finding in OrganisationNameDetector().detect(text):
            assert finding.span.extract(text) == finding.value


class TestLexiconTerms:
    @pytest.fixture
    def detector(self) -> LexiconTermDetector:
        return LexiconTermDetector(load_builtin_lexicon())

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("our EBA is unfair", "enterprise agreement"),
            ("the intranet ignored my report", "intranet"),
            ("I called the EAP and they never called back", "Employee Assistance Program"),
        ],
    )
    def test_matches(self, detector: LexiconTermDetector, text: str, expected: str) -> None:
        assert any(f.meta()["canonical"] == expected for f in detector.detect(text))

    def test_misspelling_still_matches(self, detector: LexiconTermDetector) -> None:
        findings = list(detector.detect("the timsheet sytem rejected my claim"))
        assert any(f.meta()["canonical"] == "timesheet" for f in findings)

    def test_fuzzy_hits_carry_the_distance(self, detector: LexiconTermDetector) -> None:
        finding = next(
            f for f in detector.detect("the timsheet was wrong") if f.meta().get("fuzzy")
        )
        assert finding.meta()["distance"] == 1
        assert finding.confidence < 1.0

    def test_offsets_match_original(self, detector: LexiconTermDetector) -> None:
        text = "  the intranet ignored my report"
        for finding in detector.detect(text):
            assert text[finding.span.start : finding.span.end] == finding.value


class TestAwardCodes:
    def test_detects_modern_award_code(self) -> None:
        assert values(AwardCodeDetector(), "I am covered by MA000019") == ["MA000019"]

    def test_case_insensitive(self) -> None:
        assert values(AwardCodeDetector(), "award ma000019") == ["ma000019"]

    def test_rejects_other_codes(self) -> None:
        assert values(AwardCodeDetector(), "invoice ORD-2024-8812") == []


class TestOrganisationDefaults:
    def test_triple_has_no_duplicate_spans(self) -> None:
        detectors = default_organisation_detectors(load_builtin_lexicon())
        text = "I work at Acme Pty Ltd and the intranet ignored me"
        seen: list[tuple[int, int]] = []
        for detector in detectors:
            for finding in detector.detect(text):
                span = (finding.span.start, finding.span.end)
                assert span not in seen, (detector.name, span)
                seen.append(span)


class TestJobTitle:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I am a Grade 4 registered nurse", "4"),
            ("my job title is teacher", "teacher"),
            ("I work as a forklift driver", "forklift driver"),
        ],
    )
    def test_labelled(self, text: str, expected: str) -> None:
        assert expected in values(JobTitleDetector(), text)

    def test_bare_occupation_is_low_confidence(self) -> None:
        finding = next(f for f in JobTitleDetector().detect("she is a doctor") if f.category is Category.JOB_TITLE)
        assert finding.confidence < 0.8

    def test_abstract_roles_are_excluded(self) -> None:
        # "supervisor", "operator" and "analyst" match almost any sentence and
        # would destroy the signal.
        assert values(JobTitleDetector(), "the operator said so") == []

    def test_offsets_are_exact(self) -> None:
        text = "I am a Grade 4 registered nurse on nights"
        for finding in JobTitleDetector().detect(text):
            assert text[finding.span.start : finding.span.end] == finding.value


class TestTenure:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I have 18 years of service", "18 years of service"),
            ("I joined in 2019", "2019"),
            ("been here since 2015", "2015"),
        ],
    )
    def test_detects(self, text: str, expected: str) -> None:
        assert expected in values(TenureDetector(), text)

    def test_rejects_unrelated_numbers(self) -> None:
        assert values(TenureDetector(), "there were 12 items in stock") == []


class TestTeamSize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("our team of three", "three"),
            ("a team of 12 people", "12"),
            ("our ward of nine staff", "nine"),
        ],
    )
    def test_detects(self, text: str, expected: str) -> None:
        assert expected in values(TeamSizeDetector(), text)

    def test_rejects_a_duration_masquerading_as_a_count(self) -> None:
        # "shift with 18 years of service" is a duration, not a team of 18.
        assert "18" not in values(TeamSizeDetector(), "night shift with 18 years of service")

    def test_uniqueness_claims_are_not_a_team_size(self) -> None:
        """A uniqueness claim is an implication, not a count.

        Reporting it as a team size let the generaliser rewrite "the only one"
        into "1 person", which then matched other detectors on the next pass.
        """
        assert values(TeamSizeDetector(), "I am the only woman in my unit") == []
        from pii_redact.detectors.implication import ImplicationDetector

        assert "the only woman in my unit" in [
            f.span.extract("I am the only woman in my unit")
            for f in ImplicationDetector().detect("I am the only woman in my unit")
        ]


class TestEmploymentStatus:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I am a casual", "casual"),
            ("I'm part-time", "part-time"),
            ("I am currently a temp", "temp"),
            ("I am a fixed-term contractor", "fixed-term"),
        ],
    )
    def test_detects(self, text: str, expected: str) -> None:
        assert expected in values(EmploymentStatusDetector(), text)


class TestHealthDetail:
    def test_named_condition(self) -> None:
        assert "anxiety" in values(HealthDetailDetector(), "I was diagnosed with anxiety")

    def test_mental_health_vocabulary(self) -> None:
        assert "depression" in values(HealthDetailDetector(), "I developed depression")

    def test_trims_to_the_condition(self) -> None:
        assert values(HealthDetailDetector(), "My condition is chronic fatigue") == [
            "chronic fatigue"
        ]

    def test_rejects_prose_after_the_cue(self) -> None:
        # "I am" is the cue, but what follows is not a condition.
        assert values(HealthDetailDetector(), "I am the only woman in my unit") == []

    def test_rejects_a_bare_number(self) -> None:
        # "I am 34" matches the cue; a number is an age, not a condition.
        assert values(HealthDetailDetector(), "I am 34") == []

    @pytest.mark.parametrize(
        "text",
        [
            # Households, possessions and irregular verbs all match the "I have"
            # cue. None of them is a diagnosis, and reporting one puts a
            # health finding in the triage queue that a reviewer has to undo.
            "I have family at home.",
            "I have my own transport and two kids.",
            "I have rung the service centre four times.",
            "I have brought this up before.",
        ],
    )
    def test_rejects_non_conditions_behind_the_cue(self, text: str) -> None:
        assert values(HealthDetailDetector(), text) == []

    def test_offsets_are_exact(self) -> None:
        text = "I was diagnosed with anxiety after the restructure"
        for finding in HealthDetailDetector().detect(text):
            assert text[finding.span.start : finding.span.end] == finding.value


class TestQuasiContract:
    @pytest.mark.parametrize(
        "detector",
        [
            JobTitleDetector(),
            TenureDetector(),
            TeamSizeDetector(),
            EmploymentStatusDetector(),
            HealthDetailDetector(),
        ],
    )
    def test_findings_are_well_formed(self, detector) -> None:
        text = "I am a casual Grade 4 nurse with 18 years of service in a team of three"
        for finding in detector.detect(text):
            assert 0 <= finding.span.start < finding.span.end <= len(text)
            assert finding.value == text[finding.span.start : finding.span.end], detector.name

    def test_empty_input(self) -> None:
        for detector in default_quasi_detectors():
            assert list(detector.detect("")) == []
            assert list(detector.detect("   ")) == []

    def test_smart_punctuation_preserves_offsets(self) -> None:
        # The detector matches on normalised text; the span must still be a
        # valid offset into the original.
        text = "I’m a casual — 18 years of service"
        for detector in default_quasi_detectors():
            for finding in detector.detect(text):
                assert text[finding.span.start : finding.span.end] == finding.value


class TestQuasiTuning:
    def test_occupations_are_extensible(self) -> None:
        # The occupation list is a module constant, so a deployment can extend
        # it without subclassing.
        from pii_redact.detectors import quasi

        assert "paramedic" in quasi._AU_OCCUPATIONS
        assert isinstance(quasi._AU_OCCUPATIONS, str)


class TestEmptyLexicon:
    def test_no_terms_means_no_findings(self) -> None:
        assert list(LexiconTermDetector(Lexicon()).detect("the intranet")) == []

    def test_term_requires_a_category(self) -> None:
        term = Term("Widget Program", Category.INTERNAL_TERM)
        assert term.category is Category.INTERNAL_TERM
        assert term.all_forms() == ("Widget Program",)
"""Implication rules: sequences that identify a person or a workplace."""

from __future__ import annotations

import re

import pytest

from pii_redact.detectors import implication
from pii_redact.detectors.implication import (
    RULES,
    ImplicationDetector,
    ImplicationRule,
    alt,
)
from pii_redact.types import Category


@pytest.fixture
def detector() -> ImplicationDetector:
    return ImplicationDetector()


def spans(detector: ImplicationDetector, text: str) -> list[str]:
    return [f.span.extract(text) for f in detector.detect(text)]


class TestAltHelper:
    def test_alt_produces_a_group(self) -> None:
        assert alt("a", "b") == "(?:a|b)"

    @pytest.mark.parametrize(
        "name",
        [
            "_ROSTER", "_SHIFT", "_ROLE", "_AGE_PHASE", "_SMALL_NUMBER",
            "_ARTICLE", "_UNIQUE_KIND", "_SELF", "_SITE_NOUN", "_WORK_EVENT",
            "_INTERNAL_THING", "_PERSON_RELATION",
        ],
    )
    def test_every_shared_vocabulary_is_grouped(self, name: str) -> None:
        """Interpolating a bare ``a|b`` applies the alternation to the whole pattern.

        That turns the surrounding capture group into a set of top-level
        branches and lets a single word match on its own. It has silently broken
        this package three times, so the invariant is asserted rather than
        trusted.
        """
        value = getattr(implication, name)
        assert value.startswith("(?:") and value.endswith(")"), f"{name} is not grouped"

    def test_no_rule_fires_on_a_single_vocabulary_word(self) -> None:
        """The failure mode of an ungrouped alternation, asserted directly."""
        d = ImplicationDetector()
        for word in ("shift", "site", "roster", "depot", "the", "award", "team"):
            hits = [f for f in d.detect(word) if f.span.extract(word) == word]
            assert not hits, f"bare vocabulary word {word!r} matched on its own"


class TestImplicationDetector:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I am the only woman on the night shift.", "the only woman on the night shift"),
            ("One of two apprentices on the yard team.",
             "One of two apprentices on the yard team"),
            ("To be clear, the only casual on the roster here.", "the only casual on the roster"),
            ("I am the youngest on the team here.", "I am the youngest on the team"),
            ("I cover all three depots on my own.", "I cover all three depots"),
            ("My manager Fiona ignored my report.", "My manager Fiona"),
        ],
    )
    def test_person_rules(
        self, detector: ImplicationDetector, text: str, expected: str
    ) -> None:
        assert expected in spans(detector, text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Things changed after the depot closure.", "after the depot closure"),
            ("It all happened since the take-over.", "since the take-over"),
            ("Since the site was sold nothing was the same.", "Since the site was sold"),
            ("I work out of Altona distribution centre most weeks.", None),
        ],
    )
    def test_workplace_rules(
        self, detector: ImplicationDetector, text: str, expected: str | None
    ) -> None:
        hits = spans(detector, text)
        if expected is None:
            # The capitalised form is claimed instead; either is acceptable.
            assert hits
        else:
            assert expected in hits

    @pytest.mark.parametrize(
        "text",
        [
            "The training was good and my supervisor is supportive.",
            "Nothing much to report this time.",
            "I would rather not say too much in writing.",
            "my team has always had my back",
            "I feel safe raising concerns here.",
        ],
    )
    def test_benign_text_is_silent(self, detector: ImplicationDetector, text: str) -> None:
        """Positive and neutral responses must not be mangled."""
        assert spans(detector, text) == []

    def test_role_word_alone_does_not_identify(self, detector: ImplicationDetector) -> None:
        # "my supervisor is supportive" appears in almost every positive
        # response. A role word alone must not be treated as naming a person.
        assert spans(detector, "my supervisor is supportive") == []

    def test_whole_clause_is_the_span(self, detector: ImplicationDetector) -> None:
        """Redacting only the noun leaves the sentence still identifying."""
        text = "I am the only woman on the night shift."
        for finding in detector.detect(text):
            assert len(finding.span.extract(text)) > len("woman")

    def test_offsets_are_exact(self, detector: ImplicationDetector) -> None:
        text = "Honestly, I am the only woman on the night shift here."
        for finding in detector.detect(text):
            assert finding.span.extract(text) == finding.value

    def test_smart_punctuation_preserves_offsets(self, detector: ImplicationDetector) -> None:
        text = "I’m the only woman on the night shift — always."
        for finding in detector.detect(text):
            assert finding.span.extract(text) == finding.value

    def test_findings_carry_the_rule(self, detector: ImplicationDetector) -> None:
        finding = next(iter(detector.detect("I am the only woman on the night shift.")))
        assert finding.meta()["rule"] == "only_one_on_team"
        assert finding.meta()["implicates"] == "person"
        assert finding.category is Category.IMPLICATION

    def test_empty_input(self, detector: ImplicationDetector) -> None:
        assert list(detector.detect("")) == []
        assert list(detector.detect("   ")) == []

    def test_determiner_does_not_match_inside_a_word(
        self, detector: ImplicationDetector
    ) -> None:
        """A determiner inside a word is not a determiner.

        "Your portal" once matched on "our portal" one character in. The span
        then started mid-word, so redaction replaced the middle of "Your" and
        left the reader with "Y[IMPLICATION]".
        """
        text = "Your portal says the upload failed."
        assert spans(detector, text) == ["Your portal"]

    def test_service_function_units_are_not_units(
        self, detector: ImplicationDetector
    ) -> None:
        """A survey to a provider describes the provider's own service units.

        Every insurer has a claims team and a member services centre, so those
        phrases narrow nothing, and redacting them destroys the only sentence
        that says how the respondent was treated.
        """
        for text in [
            "The claims team has not answered.",
            "Nobody in the member services centre could help.",
            "I was on hold with the customer service team for an hour.",
        ]:
            assert spans(detector, text) == [], text

    def test_named_units_still_fire(self, detector: ImplicationDetector) -> None:
        """The guard is on the head modifier, not on unit words generally."""
        assert spans(detector, "My role sits in the integrity unit.") == [
            "the integrity unit"
        ]
        assert spans(detector, "I work in the digital services branch.") == [
            "the digital services branch"
        ]

    def test_comparative_age_clause_ends_on_a_noun(
        self, detector: ImplicationDetector
    ) -> None:
        """The group noun belongs inside the clause, not after it."""
        assert spans(detector, "I am 27 and the youngest on the team.") == [
            "the youngest on the team"
        ]


class TestToggles:
    def test_person_rules_can_be_disabled(self) -> None:
        text = "I am the only woman on the night shift."
        assert spans(ImplicationDetector(enable_person=False), text) == []

    def test_workplace_rules_can_be_disabled(self) -> None:
        text = "Things changed after the depot closure."
        assert spans(ImplicationDetector(enable_workplace=False), text) == []


class TestCustomRules:
    def test_a_deployment_can_add_its_own(self) -> None:
        rule = ImplicationRule(
            name="tier_two_only",
            pattern=re.compile(r"the only tier two staff member", re.IGNORECASE),
            label="a tier unique to this employer",
        )
        detector = ImplicationDetector(rules=[rule])
        assert spans(detector, "I am the only tier two staff member here.") == [
            "the only tier two staff member"
        ]

    def test_describe_is_serialisable(self) -> None:
        import json

        payload = json.dumps([r.describe() for r in RULES])
        assert "only_one_on_team" in payload

    def test_rule_names_are_unique(self) -> None:
        names = [r.name for r in RULES]
        assert len(names) == len(set(names))

    def test_every_rule_compiles_and_has_a_label(self) -> None:
        for rule in RULES:
            assert rule.pattern.pattern
            assert rule.label
            assert 0 < rule.confidence <= 1
            assert rule.implicates in {"person", "workplace"}


class TestIdempotenceInteraction:
    def test_generalised_bands_are_not_re_read(self) -> None:
        """A generalised band must not look like a new shift-group fact.

        ``team of 2-5 people on nights`` is the generaliser's output. Without a
        guard the "5" inside it matches a shift-group rule and a second pass
        replaces half the band.
        """
        text = "in a team of 2-5 people on nights"
        assert spans(ImplicationDetector(), text) == []

    def test_label_placeholders_are_not_re_read(self) -> None:
        """[AWARD] must not be re-detected as the award term."""
        from pii_redact import SurveyRedactor, survey_redactor

        survey = SurveyRedactor(survey_redactor(salt=b"t"))
        text = "We are covered by the Clerical Award MA000019."
        once = survey.redact_record(text).text
        twice = survey.redact_record(once).text
        assert once == twice
        assert "[[" not in once
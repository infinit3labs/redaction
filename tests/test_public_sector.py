"""Australian public-sector and corporate vocabulary, and the rules using it.

Two things are asserted here that are easy to lose:

* every shared vocabulary is a *grouped* alternation, because an ungrouped one
  silently changes what a rule matches. That bug appeared three times in this
  package;
* public-sector rules redact without damaging benign text. The vocabulary is
  large and mostly made of ordinary words, so precision is the risk.
"""

from __future__ import annotations

import pytest

from pii_redact import vocabulary
from pii_redact.detectors.implication import RULES, ImplicationDetector


@pytest.fixture
def detector() -> ImplicationDetector:
    return ImplicationDetector()


def spans(detector: ImplicationDetector, text: str) -> list[str]:
    return [f.span.extract(text) for f in detector.detect(text)]


def rules_fired(detector: ImplicationDetector, text: str) -> list[str]:
    return [f.meta()["rule"] for f in detector.detect(text)]


class TestGroupedAlternations:
    @pytest.mark.parametrize(
        "name",
        [
            "DEPARTMENT_STRUCTURE",
            "DEPARTMENT_STRUCTURE_QUALIFIED",
            "GOVERNMENT_PROCESS",
            "APS_LEVEL",
            "SENIOR_ROLE",
            "OCCUPATION_FAMILY",
            "PUBLIC_SECTOR_ROLE",
            "WORKPLACE_TYPE",
            "WORKPLACE_TYPE_STRONG",
            "CORPORATE_MARKER",
            "COHORT_MARKER",
            "CLEARANCE_MARKER",
            "CLEARANCE_LEVEL",
            "NON_CONDITION_VERBS",
        ],
    )
    def test_vocabulary_is_a_group(self, name: str) -> None:
        value = getattr(vocabulary, name)
        assert value.startswith("(?:") and value.endswith(")"), f"{name} is not grouped"

    def test_alt_helper_wraps(self) -> None:
        assert vocabulary.alt("a", "b") == "(?:a|b)"

    def test_no_rule_fires_on_a_bare_vocabulary_word(self, detector) -> None:
        """The observable symptom of an ungrouped alternation."""
        for word in ("shift", "site", "team", "office", "office", "branch",
                     "grade", "manager", "the", "EL2", "unit"):
            hits = [f for f in detector.detect(word) if f.span.extract(word) == word]
            assert not hits, f"bare word {word!r} matched on its own"

    def test_structure_vocabulary_is_bare_nouns_only(self) -> None:
        """Rules consume the determiner outside the group.

        An entry like "the centre" can never match for that reason, and its
        presence is misleading during review.
        """
        for entry in vocabulary.DEPARTMENT_STRUCTURE.strip("(?:)").split("|"):
            assert not entry.lower().startswith("the "), entry


class TestPublicSectorRules:
    @pytest.mark.parametrize(
        ("text", "rule"),
        [
            ("I work in the digital services branch.", "department_structure"),
            ("My directorate reviewed the annual report.", "department_structure"),
            ("My role sits in the integrity unit.", "department_structure"),
            ("I work in Services division.", "unit_by_preposition"),
            ("Last year we handled a machinery of government review.", "government_process"),
            ("Our agency ran a freedom of information request.", "government_process"),
            ("I am an assistant director in the Home Affairs branch.",
             "occupation_with_structure"),
            ("I am an eligibility specialist in the Services division.",
             "occupation_with_structure"),
            ("My role is policy adviser.", "occupation_family"),
            ("I am one of three EL2s in the Migration directorate.",
             "seniority_in_unit"),
            ("There are two of us left in the integrity unit.", "seniority_in_unit"),
            ("I am part of the pilot cohort.", "cohort_marker"),
            ("I came through the 2021 intake.", "cohort_marker"),
            ("I hold a security clearance.", "clearance"),
            ("I hold clearance at the secret level.", "clearance"),
            ("I am the branch manager.", "senior_role"),
        ],
    )
    def test_rule_fires(
        self, detector: ImplicationDetector, text: str, rule: str
    ) -> None:
        assert rule in rules_fired(detector, text), (text, rules_fired(detector, text))

    def test_clauses_are_redacted_whole(self, detector: ImplicationDetector) -> None:
        """The whole recognisable clause, not just the identifying noun.

        Removing "EL2" from "one of three EL2s in the Migration directorate"
        leaves a sentence that still identifies the person.
        """
        text = "I am one of three EL2s in the Migration directorate."
        hit = max(detector.detect(text), key=lambda f: f.span.length)
        assert hit.span.extract(text) == "one of three EL2s in the Migration directorate"

    def test_workplace_type_without_a_name(self, detector: ImplicationDetector) -> None:
        for text in (
            "I work in a residential aged care facility.",
            "I work at our depot most weeks.",
            "The correctional centre ignored my report.",
        ):
            assert spans(detector, text), text


class TestPrecisionInPublicSectorText:
    """The vocabulary is mostly ordinary words. Precision is the risk."""

    @pytest.mark.parametrize(
        "text",
        [
            "My team has always had my back.",
            "The training here is genuinely good.",
            "my supervisor listens and acts on it",
            "I feel safe raising concerns here.",
            "I work in a team of nine.",
            "My office is cold.",
            "I read the annual report at home last week.",
            "I met him at 5 pm about the report and the a b c team",
            "I read the annual report at home last week.",
            "The weather has been lovely and I went for a walk.",
            "We worked the 6am start on weekends.",
            "I am a team player and I like the office.",
        ],
    )
    def test_benign_text_is_silent(self, detector: ImplicationDetector, text: str) -> None:
        assert spans(detector, text) == [], (text, spans(detector, text))

    def test_role_word_alone_is_not_a_person(self, detector: ImplicationDetector) -> None:
        assert spans(detector, "my director is supportive") == []

    def test_transaction_is_not_a_workplace(self, detector: ImplicationDetector) -> None:
        # "at the store" here is a purchase, not a workplace.
        assert spans(detector, "My card was declined at the store.") == []

    def test_noun_inside_a_longer_word_is_not_a_site(self, detector) -> None:
        # "lab" must not match inside "laboratory" or "labourer".
        for text in (
            "I work at Halden Holdings as a labourer.",
            "I spent the afternoon in the laboratory.",
        ):
            assert spans(detector, text) == [], (text, spans(detector, text))


class TestRuleHygiene:
    def test_rule_names_unique(self) -> None:
        names = [r.name for r in RULES]
        assert len(names) == len(set(names))

    def test_every_rule_compiles_and_is_labelled(self) -> None:
        for rule in RULES:
            assert rule.pattern.pattern
            assert rule.label
            assert 0 < rule.confidence <= 1
            assert rule.implicates in {"person", "workplace"}

    def test_public_sector_rules_exist(self) -> None:
        expected = {
            "department_structure", "unit_by_preposition", "government_process",
            "occupation_family", "occupation_with_structure", "aps_level",
            "seniority_in_unit", "cohort_marker", "clearance", "senior_role",
            "workplace_type",
        }
        assert expected <= {r.name for r in RULES}

    def test_offsets_are_exact(self, detector: ImplicationDetector) -> None:
        text = "I’m one of three EL2s in the Migration directorate — always."
        for finding in detector.detect(text):
            assert finding.span.extract(text) == finding.value

    def test_trailing_whitespace_is_not_captured(self, detector) -> None:
        # An all-optional tail would otherwise swallow the word boundary.
        text = "I hold a security clearance and I regret it."
        for finding in detector.detect(text):
            assert not finding.span.extract(text).endswith(" ")
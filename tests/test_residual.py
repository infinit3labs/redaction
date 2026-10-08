"""Tests for residual disclosure control.

The thresholds in this module are estimates, and pinning exact numbers here
would make the tests a change-detector rather than a specification. What is
asserted instead is behaviour that has to hold for the estimates to be usable:

* a record with nothing left in it discloses nothing;
* the module never reads its own redaction markers as residual disclosure;
* the *same* text is identifying in a small workplace and unremarkable in a
  large one, which is the entire reason ``population`` exists;
* a format the detectors do not know about is still caught, because a value
  that survives as an unusual specific thing is measurable without recognising
  what it is.

Anything asserted about a specific threshold is marked as calibration and
should change together with the numbers it depends on.
"""

from __future__ import annotations

import pytest

from pii_redact.residual import (
    DisclosureAction,
    DisclosurePolicy,
    ResidualKind,
    decide,
    generalise_residual,
    measure,
    protect,
    summarise,
)

AGENCY = DisclosurePolicy(population=4000)
DEPOT = DisclosurePolicy(population=30)


# --------------------------------------------------------------------------
# Nothing left is nothing disclosed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Great workplace, plenty of training. I feel safe raising concerns here.",
        "The training was good and my supervisor is supportive.",
        "Flexible hours and a decent kitchen. Nothing to complain about.",
    ],
)
def test_benign_text_releases_unchanged(text: str) -> None:
    final, decision = protect(text, AGENCY)
    assert final == text
    assert decision.action is DisclosureAction.PASS
    assert not decision.escalated


def test_empty_profile_is_not_treated_as_one_person() -> None:
    """An empty product is 1.0, meaning 'everyone matches', not 'one person'."""
    profile = measure("")
    assert profile.attributes == ()
    assert decide(profile, AGENCY).action is DisclosureAction.PASS


# --------------------------------------------------------------------------
# The module must not read its own output
# --------------------------------------------------------------------------


def test_placeholder_markers_are_not_residual_disclosure() -> None:
    """[HEALTH DETAIL] is two capitalised words and would read as a name."""
    text = "I developed [HEALTH DETAIL] after the restructure."
    profile = measure(text)
    assert not [
        a for a in profile.attributes if "HEALTH" in a.value
    ]
    assert decide(profile, AGENCY).action is DisclosureAction.PASS


def test_every_category_label_is_invisible_to_the_residual_layer() -> None:
    """Built from the real labels, not from a guess at their shape."""
    from pii_redact.masking import category_label

    for category in (
        "name", "location", "organisation_name", "phone", "date", "numeric",
        "health_detail", "implication",
    ):
        marker = category_label(category)
        # Neutral carrier text: the phrase has to be free of anything the
        # residual layer legitimately measures, or this test proves nothing.
        text = f"The report mentions {marker} in the summary table."
        assert measure(text).attributes == (), f"{marker} was measured as disclosure"


@pytest.mark.parametrize(
    "marker",
    ["**** *** ***", "tok_4f3a2b1c", "[MASKED]"],
)
def test_token_style_placeholders_are_not_measured(marker: str) -> None:
    """Pseudonymous output must not read back as residual disclosure either."""
    text = f"The report mentions {marker} in the summary table."
    assert measure(text).attributes == ()


def test_residual_layer_is_idempotent_over_its_own_output() -> None:
    """Protecting an already-protected record must not escalate it again."""
    _, first = protect("My employee number is 0041827.", AGENCY)
    assert first.escalated
    cleaned = "My employee number is [NUMERIC REDACTED]."
    second_text, second = protect(cleaned, AGENCY)
    assert second.action is DisclosureAction.PASS
    assert second_text == cleaned


# --------------------------------------------------------------------------
# Population is the whole point
# --------------------------------------------------------------------------


def test_same_text_escalates_in_a_small_workplace_and_not_a_large_one() -> None:
    """The property that matters is that a crossing population exists."""
    text = "I am 25-34 and I have worked here 11 years. They singled me out again."
    assert protect(text, DEPOT)[1].escalated

    crossings = [
        pop
        for pop in (10, 30, 100, 1_000, 10_000, 100_000, 1_000_000)
        if not protect(text, DisclosurePolicy(population=pop))[1].escalated
    ]
    assert crossings, "no workforce size makes this text safe to release"
    assert min(crossings) > DEPOT.population


def test_population_changes_the_decision_monotonically() -> None:
    """Escalation is not allowed to flip back to passing as the population grows."""
    text = "I am 25-34. I have worked here 11 years. They singled me out again."
    actions = [
        protect(text, DisclosurePolicy(population=pop))[1].action
        for pop in (30, 100, 1000, 10_000)
    ]
    order = {DisclosureAction.SUPPRESS: 0, DisclosureAction.GENERALISE: 1, DisclosureAction.PASS: 2}
    ranks = [order[a] for a in actions]
    assert ranks == sorted(ranks), f"non-monotonic escalation: {actions}"


def test_expected_matches_scales_with_population() -> None:
    """A bigger workplace means more people share the description."""
    profile = measure("I am 25-34.")
    small = DisclosurePolicy(population=100).expected_matches(profile)
    large = DisclosurePolicy(population=1000).expected_matches(profile)
    assert small == pytest.approx(large / 10)


def test_attribute_independence_loosens_the_test() -> None:
    profile = measure("I am 25-34. I have worked here 11 years.")
    strict = DisclosurePolicy(population=4000).expected_matches(profile)
    loose = DisclosurePolicy(population=4000, attribute_independence=50).expected_matches(
        profile
    )
    assert loose > strict


# --------------------------------------------------------------------------
# Catching what the detectors missed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("My employee number is 0041827.", ResidualKind.NUMERIC),
        ("My TFN is 123 456 783, I think it has a typo.", ResidualKind.NUMERIC),
        ("Our internal case reference is GW-2026-4471.", ResidualKind.ORGANISATION),
        ("The Meridien platform rejected my claim.", ResidualKind.ORGANISATION),
        ("They singled me out again in the meeting.", ResidualKind.NARRATIVE),
    ],
)
def test_unseen_formats_are_still_measured(text: str, expected_kind: ResidualKind) -> None:
    """Formats no detector knows about must still register as disclosure."""
    profile = measure(text)
    assert expected_kind in profile.classes, (
        f"{expected_kind} not measured in {text!r}: "
        f"got {sorted(k.value for k in profile.classes)}"
    )


def test_a_missed_identifier_is_escalated_in_both_workplace_sizes() -> None:
    for policy in (AGENCY, DEPOT):
        _, decision = protect("My employee number is 0041827.", policy)
        assert decision.escalated
        assert decision.action is DisclosureAction.GENERALISE


def test_generalise_removes_the_missed_value() -> None:
    final, _ = protect("My employee number is 0041827.", AGENCY)
    assert "0041827" not in final
    assert final == "My employee number is [NUMERIC REDACTED]."


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


def test_unreducible_risk_is_withheld_rather_than_generalised() -> None:
    """Reporting GENERALISE when nothing was replaced would be a false claim."""
    _, decision = protect(
        "I am 25-34. I have worked here 11 years. They singled me out again.", DEPOT
    )
    assert decision.action is DisclosureAction.SUPPRESS


def test_generalise_leaves_the_record_usable() -> None:
    """The narrative is why the response exists and must survive redaction."""
    final, _ = protect("My employee number is 0041827 and I was singled out.", AGENCY)
    assert "singled out" in final


def test_reporting_keeps_the_pre_remediation_figure() -> None:
    """A review queue needs to know how close the record came to identifying."""
    _, decision = protect("My employee number is 0041827.", AGENCY)
    assert decision.cohort < AGENCY.min_cohort
    assert decision.post_cohort is not None
    assert decision.post_cohort > decision.cohort


def test_generalise_is_idempotent_as_an_operation() -> None:
    text = "My employee number is 0041827."
    profile = measure(text)
    once = generalise_residual(text, profile)
    twice = generalise_residual(once, measure(once))
    assert once == twice


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_summarises_an_ingest() -> None:
    decisions = [
        protect("Great workplace, plenty of training.", AGENCY)[1],
        protect("My employee number is 0041827.", AGENCY)[1],
        protect("I am 25-34. They singled me out.", DEPOT)[1],
    ]
    summary = summarise(decisions)
    assert summary["records"] == 3
    assert sum(summary["actions"].values()) == 3
    assert 0.0 < summary["escalation_rate"] < 1.0


def test_decision_is_serialisable_with_both_figures() -> None:
    payload = protect("My employee number is 0041827.", AGENCY)[1].as_dict()
    assert payload["action"] == "generalise"
    assert "expected_matches" in payload
    assert "expected_matches_after" in payload
    assert payload["residual"]["classes"]


# --------------------------------------------------------------------------
# Calibration notes, not behaviour
# --------------------------------------------------------------------------


def test_defaults_are_a_stated_estimate_not_a_fact() -> None:
    """If these numbers are ever 'fixed', this test is the thing to delete."""
    policy = DisclosurePolicy()
    assert policy.population == 1000
    assert policy.min_cohort == 5.0
    assert policy.attribute_independence == 1.0
    assert policy.narrative_selectivity > 1.0


def test_labels_carry_no_value_information() -> None:
    """A marker depends only on its category, so two exports cannot be joined on it."""
    from pii_redact.masking import category_label

    assert category_label("name") == category_label("name")
    assert category_label("organisation_name") == "[ORGANISATION NAME]"
    assert measure(f"value is {category_label('numeric')} here").attributes == ()
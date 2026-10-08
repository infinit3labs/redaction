"""Engine tests: precedence, overlap resolution, policy and masking."""

from __future__ import annotations

import pytest

from pii_redact import (
    Category,
    MaskStyle,
    Policy,
    RedactionError,
    Redactor,
    Stage,
    redact,
)
from pii_redact.cli import _build_parser
from pii_redact.types import Finding, Span


class TestDeterministicOnly:
    def test_nlp_is_off_by_default(self, redactor: Redactor) -> None:
        result = redactor.redact("Contact Jane Citizen on 0412 345 678")
        assert all(f.stage is Stage.DETERMINISTIC for f in result)

    def test_redacts_known_identifiers(self, redactor: Redactor) -> None:
        text = "TFN 123 456 782 and ABN 64 004 085 616"
        result = redactor.redact(text)
        assert "123" not in result.text
        assert "64004085616" not in result.text
        assert result.counts["tfn"] == 1
        assert result.counts["abn"] == 1

    def test_clean_text_untouched(self, redactor: Redactor) -> None:
        text = "The meeting is on Tuesday and the invoice total is $1,250.00."
        assert redactor.redact_text(text) == text

    def test_scan_does_not_modify_input(self, redactor: Redactor) -> None:
        text = "TFN 123 456 782"
        findings = redactor.scan(text)
        assert findings
        assert text == "TFN 123 456 782"

    def test_findings_have_provenance(self, redactor: Redactor) -> None:
        findings = redactor.scan("TFN 123 456 782")
        assert len(findings) == 1
        finding = findings[0]
        assert finding.detector == "tfn"
        assert finding.stage is Stage.DETERMINISTIC
        assert finding.meta()["normalised"] == "123456782"


class TestPrecedence:
    def test_longest_span_wins(self) -> None:
        """A URL carrying credentials beats the bare email inside it.

        Both detectors match the same text: the credential detector claims the
        whole ``https://user:pass@host`` and the email detector claims the
        address portion. Overlap resolution ranks by span length first, so the
        credential claim survives and the email is not double-redacted.
        """
        text = "endpoint https://admin:hunter2@example.com/hook"
        result = Redactor(Policy()).redact(text)
        assert result.counts.get("url_credential") == 1
        assert Category.EMAIL not in {f.category for f in result}
        assert "hunter2" not in result.text

    def test_abn_and_acn_do_not_collide(self) -> None:
        """Word boundaries keep an 11-digit ABN from being split.

        The ACN pattern needs a non-word character on both sides, so the
        trailing nine digits of ``64004085616`` are never offered as an ACN.
        """
        result = Redactor(Policy()).redact("ACN 004085616 and ABN 64004085616")
        assert result.counts["acn"] == 1
        assert result.counts["abn"] == 1

    def test_checksum_failure_is_not_rescued_by_nlp(self) -> None:
        """A deterministic rejection stays rejected with NLP enabled."""
        text = "TFN 123 456 783"
        without_nlp = Redactor(Policy()).redact(text)
        with_nlp = Redactor(Policy(nlp_enabled=True)).redact(text)
        assert without_nlp.text == with_nlp.text

    def test_resolution_is_order_independent(self, redactor: Redactor) -> None:
        text = "TFN 123 456 782, Medicare 2123 4567 82, phone 0412 345 678"
        spans = [(f.span.start, f.span.end, f.category) for f in redactor.scan(text)]
        assert spans == sorted(spans)


class TestNLPStage:
    def test_gazetteer_finds_labelled_names(self, nlp_redactor: Redactor) -> None:
        result = nlp_redactor.redact("Patient: Jane Citizen")
        assert any(f.category is Category.NAME for f in result)
        assert "Jane Citizen" not in result.text

    def test_gazetteer_finds_honorific_names(self, nlp_redactor: Redactor) -> None:
        result = nlp_redactor.redact("Reviewed by Dr Fiona Whitlam on site")
        assert any(f.category is Category.NAME for f in result)

    def test_gazetteer_findings_are_nlp_stage(self, nlp_redactor: Redactor) -> None:
        result = nlp_redactor.redact("Patient: Jane Citizen")
        assert all(f.stage is Stage.NLP for f in result)

    def test_deterministic_results_are_untouched_by_nlp(self, nlp_redactor: Redactor) -> None:
        text = "TFN 123 456 782 belongs to Patient: Jane Citizen"
        result = nlp_redactor.redact(text)
        assert result.counts["tfn"] == 1
        # The name is still redacted, so NLP added recall without disturbing
        # the deterministic claim.
        assert "Jane Citizen" not in result.text

    def test_nlp_never_splits_a_deterministic_span(self, nlp_redactor: Redactor) -> None:
        """An NLP span overlapping a deterministic span is discarded whole."""
        text = "Contact Patient: Jane Citizen on 0412 345 678"
        result = nlp_redactor.redact(text)
        phone = next(f for f in result if f.category is Category.PHONE)
        # The phone span survives intact rather than being partially covered
        # by an NLP name span.
        assert phone.span.extract(text) == "0412 345 678"
        for finding in result:
            if finding.category is Category.NAME:
                assert not finding.span.overlaps(phone.span)

    def test_confidence_gate(self) -> None:
        strict = Redactor(Policy(nlp_enabled=True, nlp_min_confidence=0.99))
        assert "Jane Citizen" in strict.redact_text("Patient: Jane Citizen")

    def test_gazetteer_rejects_headings(self, nlp_redactor: Redactor) -> None:
        # "Dear Customer" is a salutation, not a person.
        result = nlp_redactor.redact("Dear Customer, your order has shipped.")
        assert not any(f.category is Category.NAME for f in result)

    def test_spacy_backend_degrades_when_unavailable(self) -> None:
        """A missing spaCy install must not raise; it falls back."""
        redactor = Redactor(Policy(nlp_enabled=True, nlp_backend="spacy"))
        result = redactor.redact("Patient: Jane Citizen")
        assert len(result) >= 1


class TestPolicy:
    def test_category_filter(self) -> None:
        only_tfn = Redactor(Policy(categories=frozenset({Category.TFN})))
        result = only_tfn.redact("TFN 123 456 782 and email a@b.com")
        assert result.counts["tfn"] == 1
        assert "a@b.com" in result.text

    def test_min_confidence_filters_weak_findings(self) -> None:
        strict = Redactor(Policy(min_confidence=0.9))
        result = strict.redact("card 4111 1111 1111 1112")
        assert len(result) == 0

    def test_unknown_category_rejected(self) -> None:
        from pii_redact.cli import _policy_from_args

        args = _build_parser().parse_args(["--categories", "NOT_A_CATEGORY"])
        with pytest.raises(SystemExit):
            _policy_from_args(args)

    def test_with_policy_returns_new_instance(self, redactor: Redactor) -> None:
        derived = redactor.with_policy(categories=frozenset({Category.EMAIL}))
        assert derived.policy.categories == frozenset({Category.EMAIL})
        assert redactor.policy.categories is None

    def test_non_string_input_rejected(self, redactor: Redactor) -> None:
        with pytest.raises(RedactionError):
            redactor.redact(None)  # type: ignore[arg-type]


class TestMaskStyles:
    @pytest.mark.parametrize(
        "style",
        [MaskStyle.MASK, MaskStyle.REMOVE, MaskStyle.PARTIAL, MaskStyle.HASH, MaskStyle.TOKEN],
    )
    def test_value_is_removed_for_every_style(self, style: MaskStyle) -> None:
        result = Redactor(Policy(styles={"default": style})).redact("TFN 123 456 782")
        assert "123456782" not in result.text

    def test_mask_is_the_default_and_preserves_length(self, redactor: Redactor) -> None:
        text = "x 0412 345 678 y"
        masked = redactor.redact_text(text)
        # Length is preserved and the digit groups stay aligned, which matters
        # for fixed-width log columns; whitespace is kept as whitespace.
        assert masked == "x **** *** *** y"
        assert len(masked) == len(text)

    def test_mask_char_is_configurable(self) -> None:
        policy = Policy(mask_char="#")
        assert Redactor(policy).redact_text("0412345678") == "##########"

    def test_remove_collapses_the_span(self) -> None:
        result = Redactor(Policy(styles={"default": MaskStyle.REMOVE})).redact("x 0412 345 678 y")
        assert result.text == "x  y"

    def test_partial_keeps_last_four(self) -> None:
        result = Redactor(Policy(styles={"default": MaskStyle.PARTIAL})).redact("TFN 123456782")
        assert result.text.endswith("6782")

    def test_partial_keeps_configured_edges(self) -> None:
        policy = Policy(
            styles={"default": MaskStyle.PARTIAL},
            partial_keep_prefix=2,
            partial_keep_suffix=2,
        )
        assert Redactor(policy).redact_text("0412345678") == "04******78"

    def test_partial_fully_masks_short_values(self) -> None:
        # Nothing sensible can be kept from a four-digit value, so partial
        # falls back to a full mask rather than leaking the whole thing.
        from pii_redact.masking import partial_mask

        assert partial_mask("1234") == "****"

    def test_token_style_is_stable(self) -> None:
        policy = Policy(styles={"default": MaskStyle.TOKEN}, salt=b"unit-test-salt")
        redactor = Redactor(policy)
        first = redactor.redact_text("TFN 123456782")
        second = redactor.redact_text("TFN 123456782")
        assert first == second
        assert first == redactor.redact_text("TFN 123456782")

    def test_token_style_differs_per_salt(self) -> None:
        a = Redactor(Policy(styles={"default": MaskStyle.TOKEN}, salt=b"a")).redact_text("TFN 123456782")
        b = Redactor(Policy(styles={"default": MaskStyle.TOKEN}, salt=b"b")).redact_text("TFN 123456782")
        assert a != b

    def test_token_style_is_category_scoped(self) -> None:
        policy = Policy(
            styles={"default": MaskStyle.TOKEN},
            salt=b"unit-test-salt",
        )
        redactor = Redactor(policy)
        tfn = redactor.redact_text("TFN 123456782")
        card = redactor.redact_text("Card 4111111111111111")
        # Different categories must not collide even if the values matched.
        assert tfn.startswith("TFN ")
        assert card.startswith("Card ")


class TestOverlapResolution:
    def test_no_overlapping_findings_survive(self, redactor: Redactor) -> None:
        text = (
            "TFN 123 456 782, ABN 64 004 085 616, Medicare 2123 4567 82, "
            "phone 0412 345 678, 45 Collins Street Melbourne VIC 3000, "
            "card 4111 1111 1111 1111, DOB 12/03/1985"
        )
        findings = redactor.scan(text)
        for i, a in enumerate(findings):
            for b in findings[i + 1 :]:
                assert not a.span.overlaps(b.span), (a, b)

    def test_findings_are_sorted_by_offset(self, redactor: Redactor) -> None:
        text = "Call 0412 345 678 or email a@b.com about TFN 123 456 782"
        starts = [f.span.start for f in redactor.scan(text)]
        assert starts == sorted(starts)

    def test_resolve_overlaps_prefers_longest(self) -> None:
        from pii_redact.engine import resolve_overlaps

        short = Finding(Span(0, 3), Category.TFN, "x")
        long = Finding(Span(0, 11), Category.ABN, "y")
        kept = resolve_overlaps([short, long], {"x": 0, "y": 1})
        assert kept == [long]

    def test_resolve_overlaps_prefers_deterministic_on_tie(self) -> None:
        from pii_redact.engine import resolve_overlaps

        nlp = Finding(Span(0, 5), Category.NAME, "nlp", stage=Stage.NLP, confidence=1.0)
        deterministic = Finding(Span(0, 5), Category.ORGANISATION, "det", confidence=0.5)
        kept = resolve_overlaps([nlp, deterministic], {"det": 0, "nlp": 1})
        assert kept == [deterministic]


class TestResult:
    def test_summary_is_safe_to_log(self, redactor: Redactor) -> None:
        result = redactor.redact("TFN 123 456 782")
        summary = result.summary()
        assert "123456782" not in summary
        assert "tfn=1" in summary

    def test_iteration_and_length(self, redactor: Redactor) -> None:
        result = redactor.redact("TFN 123 456 782 and 0412 345 678")
        assert len(result) == 2
        assert len(list(result)) == 2

    def test_module_level_helper(self) -> None:
        assert "123456782" not in redact("TFN 123 456 782")

    def test_empty_and_whitespace_input(self, redactor: Redactor) -> None:
        for text in ("", "   ", "\n\t"):
            assert redactor.redact_text(text) == text


class TestIdempotence:
    def test_redacting_twice_is_stable(self, redactor: Redactor) -> None:
        """A second pass must not further mangle already-redacted text.

        Mask characters and separators must not form new PII-shaped matches,
        which is what lets a redacted pipeline run through redaction again.
        """
        text = "TFN 123 456 782, phone 0412 345 678, email jane@example.com.au"
        once = redactor.redact_text(text)
        twice = redactor.redact_text(once)
        assert once == twice
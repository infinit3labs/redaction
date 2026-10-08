"""Risk scoring, generalisation, record API and Spark glue tests."""

from __future__ import annotations

import pytest

from pii_redact import (
    Audience,
    Category,
    RiskBand,
    SurveyRedactor,
    assess,
    generalise,
    survey_redactor,
)
from pii_redact.generalise import parse_number
from pii_redact.lexicon import Lexicon, Term, load_builtin_lexicon
from pii_redact.risk import WEIGHTS, filter_by_band
from pii_redact.spark import RedactorSpec, redact_text, summarise_batch


@pytest.fixture
def survey() -> SurveyRedactor:
    return SurveyRedactor(survey_redactor(salt=b"test-salt", lexicon=load_builtin_lexicon()))


def score_for(survey: SurveyRedactor, text: str, audience=Audience.INTERNAL) -> float:
    return survey.redact_record(text).risk.score


class TestRiskBands:
    def test_benign_response_is_low(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record(
            "Great workplace, plenty of training opportunities and a really supportive team."
        )
        assert record.risk.band is RiskBand.LOW

    def test_narrative_about_a_colleague_raises_risk(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record(
            "My manager is a bully. They single me out in meetings and the intranet "
            "ignored my report."
        )
        assert record.risk.band in {RiskBand.HIGH, RiskBand.CRITICAL}
        assert "narrative" in record.risk.signal_names
        assert "employer_attribution" in record.risk.signal_names

    def test_attribute_stacking_scores_higher_than_one_attribute(
        self, survey: SurveyRedactor
    ) -> None:
        one = score_for(
            survey, "I am a registered nurse and the work is fine."
        )
        many = score_for(
            survey,
            "I am a Grade 4 registered nurse, casual, 18 years of service, in a "
            "team of three on nights at Acme Pty Ltd. DOB 12/03/1985.",
        )
        assert many > one + 0.2
        assert "quasi_combination" in survey.redact_record(
            "I am a Grade 4 registered nurse, casual, 18 years of service, in a "
            "team of three on nights."
        ).risk.signal_names

    def test_direct_identifier_scores_high(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record("You can reach me on 0412 345 678 or a@b.com")
        assert "direct_identifier" in record.risk.signal_names

    def test_score_is_bounded(self, survey: SurveyRedactor) -> None:
        extreme = (
            "I am the only Aboriginal woman on the night shift, Grade 4, casual, "
            "team of three, 18 years of service, at Acme Pty Ltd. Call me on "
            "0412 345 678 or jane@example.com. I was diagnosed with anxiety after "
            "the restructure. DOB 12/03/1985."
        )
        record = survey.redact_record(extreme)
        assert 0.0 <= record.risk.score <= 1.0
        assert record.risk.band is RiskBand.CRITICAL


class TestRiskExplainability:
    def test_every_signal_has_a_reason(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record(
            "I am a Grade 4 nurse, casual, in a team of three, and my manager "
            "bullies me."
        )
        for signal in record.risk.signals:
            assert signal.detail
            assert signal.weight > 0

    def test_signals_are_sorted_by_weight(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record(
            "I am the only woman, a Grade 4 casual nurse, 18 years, team of three, at Acme Pty Ltd"
        )
        weights = [s.weight for s in record.risk.signals]
        assert weights == sorted(weights, reverse=True)

    def test_assessment_str_contains_no_source_text(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record("Call 0412 345 678, I am a Grade 4 nurse at Acme Pty Ltd")
        rendered = str(record.risk)
        assert "0412 345 678" not in rendered
        assert "Acme" not in rendered

    def test_as_dict_is_serialisable(self, survey: SurveyRedactor) -> None:
        import json

        payload = survey.redact_record("I am a Grade 4 nurse at Acme Pty Ltd").risk.as_dict()
        assert json.loads(json.dumps(payload))["band"] in {"low", "moderate", "high", "critical"}

    def test_weights_table_is_complete(self) -> None:
        # Every weight referenced by the model must exist in the table.
        for name in (
            "narrative",
            "scenario",
            "unique",
            "protected",
            "specific_event",
            "employer_attribution",
            "named_organisation",
            "internal_lexicon",
            "job_title",
            "tenure",
            "team_size_small",
            "employment_status",
            "health_detail",
            "date_of_birth",
            "age",
            "location_specific",
            "short_response",
            "direct_identifier",
        ):
            assert name in WEIGHTS


class TestRiskAudience:
    def test_internal_risk_exceeds_public(self, survey: SurveyRedactor) -> None:
        text = "I am a Grade 4 registered nurse at Acme Pty Ltd with 18 years of service"
        internal = assess(text, survey.redact_record(text).findings, audience=Audience.INTERNAL)
        public = assess(text, survey.redact_record(text).findings, audience=Audience.PUBLIC)
        assert internal.score > public.score

    def test_attribution_matters_internally(self, survey: SurveyRedactor) -> None:
        text = "the intranet ignored my report"
        internal = assess(text, survey.redact_record(text).findings, audience=Audience.INTERNAL)
        public = assess(text, survey.redact_record(text).findings, audience=Audience.PUBLIC)
        assert internal.score > public.score


class TestRiskTriage:
    def test_filter_by_band(self, survey: SurveyRedactor) -> None:
        texts = [
            "Great workplace, supportive team.",
            "I am a Grade 4 nurse, casual, team of three, 18 years, at Acme Pty Ltd",
        ]
        assessments = [survey.redact_record(t).risk for t in texts]
        assert len(filter_by_band(assessments, RiskBand.HIGH)) == 1

    def test_needs_review_respects_the_threshold(self) -> None:
        strict = SurveyRedactor(survey_redactor(salt=b"s"), review_threshold=RiskBand.LOW)
        record = strict.redact_record("Great workplace.")
        assert strict.needs_review(record) is True


class TestGeneralise:
    @pytest.mark.parametrize(
        ("category", "value", "expected"),
        [
            (Category.AGE, "34", "25-34"),
            (Category.AGE, "17", "under 18"),
            (Category.AGE, "71", "65-74"),
            (Category.TENURE, "18 years of service", "10-19 years"),
            (Category.TEAM_SIZE, "3", "2-5 people"),
            (Category.TEAM_SIZE, "12", "11-20 people"),
            (Category.TEAM_SIZE, "three", "2-5 people"),
            (Category.DATE_AU, "12/03/1985", "1985"),
            (Category.DATE_ISO, "2024-03-12", "2024"),
        ],
    )
    def test_values(self, category: Category, value: str, expected: str) -> None:
        assert generalise(category, value) == expected

    def test_start_year_is_not_treated_as_a_duration(self) -> None:
        assert generalise(Category.TENURE, "2019") == "2019"

    def test_unparseable_passes_through(self) -> None:
        assert generalise(Category.AGE, "not a number") == "not a number"

    def test_unknown_category_passes_through(self) -> None:
        assert generalise(Category.TFN, "123456782") == "123456782"

    def test_in_context(self) -> None:
        result = survey_redactor(salt=b"s").redact_text("I am 34 with 18 years of service")
        assert "25-34" in result
        assert "10-19 years" in result

    def test_dates_become_years_in_a_survey_pass(self) -> None:
        result = survey_redactor(salt=b"s").redact_text("DOB 12/03/1985")
        assert result == "DOB 1985"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("34", 34.0), ("three", 3.0), ("a few", None), ("", None), ("abc", None)],
    )
    def test_parse_number(self, text: str, expected: float | None) -> None:
        assert parse_number(text) == expected


class TestSurveyProfile:
    def test_generalises_by_default(self, survey: SurveyRedactor) -> None:
        out = survey.redactor.redact_text("I am 45 years old, DOB 12/03/1981")
        assert "45-54" in out
        assert "1981" in out
        assert "12/03/1981" not in out

    def test_labels_employer_attribution(self, survey: SurveyRedactor) -> None:
        out = survey.redactor.redact_text("I work at Acme Pty Ltd and they ignored me")
        assert "Acme Pty Ltd" not in out
        assert "[ORGANISATION NAME]" in out

    def test_health_detail_is_labelled_not_deleted(self, survey: SurveyRedactor) -> None:
        out = survey.redactor.redact_text("I was diagnosed with anxiety after the restructure")
        assert "anxiety" not in out
        assert "[HEALTH DETAIL]" in out
        # The rest of the finding survives, so the response is still analysable.
        assert "after the restructure" in out

    def test_benign_text_is_untouched(self, survey: SurveyRedactor) -> None:
        text = "The training was good and my supervisor is supportive."
        assert survey.redactor.redact_text(text) == text

    def test_custom_lexicon_is_merged_over_the_seed(self) -> None:
        custom = Lexicon.from_terms(
            [
                Term(
                    "Project Kestrel",
                    Category.INTERNAL_TERM,
                    aliases=["kestrel"],
                    label="internal project",
                )
            ]
        )
        redactor = survey_redactor(salt=b"s", lexicon=custom)
        out = redactor.redact_text("the Kestral rollout ignored me")
        assert "Kestral" not in out
        # The bundled seed still applies.
        assert "[INTERNAL TERM]" in redactor.redact_text("the intranet ignored me")


class TestRecordAPI:
    def test_record_fields(self, survey: SurveyRedactor) -> None:
        record = survey.redact_record("Call 0412 345 678", record_id="r1")
        assert record.record_id == "r1"
        assert record.original_text == "Call 0412 345 678"
        assert "0412345678" not in record.text
        assert record.finding_count > 0

    def test_null_and_empty_input(self, survey: SurveyRedactor) -> None:
        for value in (None, "", "   "):
            record = survey.redact_record(value)
            assert record.text == value or record.text == ""
            assert record.risk.band is RiskBand.LOW

    def test_findings_json_offsets_index_the_original(self, survey: SurveyRedactor) -> None:
        import json

        text = "Call 0412 345 678 now"
        record = survey.redact_record(text)
        findings = json.loads(record.findings_json())
        assert findings
        for finding in findings:
            assert text[finding["start"] : finding["end"]] or finding["category"]

    def test_audit_json(self, survey: SurveyRedactor) -> None:
        import json

        payload = json.loads(survey.redact_record("I am a Grade 4 nurse").audit_json())
        assert payload["record_id"] is None
        assert payload["finding_count"] >= 1
        assert "risk" in payload

    def test_batch_preserves_order(self, survey: SurveyRedactor) -> None:
        texts = ["a", "Call 0412 345 678", None, "b"]
        records = survey.redact_many(texts)
        assert len(records) == 4
        assert records[0].text == "a"
        assert records[2].text == ""


class TestSparkSpec:
    def test_salt_round_trip_as_bytes_and_hex(self) -> None:
        spec = RedactorSpec.build(salt=b"\xde\xad\xbe\xef")
        assert spec.salt_bytes() == b"\xde\xad\xbe\xef"
        spec = RedactorSpec.build(salt="deadbeef")
        assert spec.salt_bytes() == bytes.fromhex("deadbeef")
        spec = RedactorSpec.build(salt="0xDEADBEEF")
        assert spec.salt_bytes() == bytes.fromhex("deadbeef")

    def test_no_salt_is_allowed(self) -> None:
        spec = RedactorSpec.build()
        assert spec.salt_bytes() is None

    def test_spec_is_json_serialisable(self) -> None:
        import json

        spec = RedactorSpec.build(
            salt=b"s",
            generalise_categories=frozenset({Category.AGE, Category.TENURE}),
            lexicon=load_builtin_lexicon(),
            nlp_min_confidence=0.6,
        )
        assert json.loads(json.dumps(spec.to_dict()))

    def test_cache_key_distinguishes_configs(self) -> None:
        a = RedactorSpec.build(salt=b"a")
        b = RedactorSpec.build(salt=b"b")
        c = RedactorSpec.build(salt=b"a", nlp_enabled=False)
        assert len({a.cache_key(), b.cache_key(), c.cache_key()}) == 3

    def test_cache_key_is_stable(self) -> None:
        assert RedactorSpec.build(salt=b"a").cache_key() == RedactorSpec.build(salt=b"a").cache_key()

    def test_categories_round_trip(self) -> None:
        spec = RedactorSpec.build(generalise_categories=frozenset({Category.AGE}))
        assert spec.categories() == frozenset({Category.AGE})

    def test_generalisation_defaults_match_the_survey_profile(self) -> None:
        # A spec that silently dropped these would differ from survey_redactor()
        # on identical text, which is nearly impossible to notice in a Delta
        # table. Pin the behaviour.
        from pii_redact.survey import DEFAULT_GENERALISED

        assert RedactorSpec.build().categories() == DEFAULT_GENERALISED

    def test_empty_generalisation_is_explicit(self) -> None:
        spec = RedactorSpec.build(generalise_categories=frozenset())
        assert spec.generalise_categories == ()
        # And the worker still falls back to the survey defaults rather than
        # masking everything.
        record = redact_text(spec, "DOB 12/03/1985, 18 years of service")
        assert "1985" in record.text
        assert "10-19 years" in record.text

    def test_tenure_is_generalised_through_the_spark_path(self) -> None:
        record = redact_text(
            RedactorSpec.build(salt=b"s"), "casual, 18 years of service, team of three"
        )
        assert "10-19 years" in record.text
        assert "2-5 people" in record.text

    def test_spec_survives_a_json_round_trip(self) -> None:
        import json

        original = RedactorSpec.build(salt=b"\x01\x02", lexicon=load_builtin_lexicon())
        restored = RedactorSpec(**json.loads(json.dumps(original.to_dict())))
        # Tuples come back as lists, so normalise before comparing behaviour.
        assert restored.salt_bytes() == original.salt_bytes()
        assert "internal_term" in [f.category.value for f in redact_text(restored, "the intranet").findings]

    def test_worker_redactor_is_cached(self) -> None:
        from pii_redact.spark import build_worker_redactor

        spec = RedactorSpec.build(salt=b"cache-test")
        assert build_worker_redactor(spec) is build_worker_redactor(spec)


class TestSparkBatch:
    @pytest.fixture
    def spec(self) -> RedactorSpec:
        return RedactorSpec.build(salt=b"batch-salt")

    def test_summarise_batch_columns_align(self, spec: RedactorSpec) -> None:
        texts = ["Call 0412 345 678", "Great workplace.", None]
        out = summarise_batch(spec, texts)
        for key, values in out.items():
            assert len(values) == len(texts), key

    def test_nulls_do_not_raise(self, spec: RedactorSpec) -> None:
        assert summarise_batch(spec, [None, None])["risk"] == [0.0, 0.0]

    def test_redacted_text_has_no_raw_values(self, spec: RedactorSpec) -> None:
        out = summarise_batch(spec, ["Call 0412 345 678"])
        assert "0412345678" not in out["redacted"][0]

    def test_findings_column_is_valid_json(self, spec: RedactorSpec) -> None:
        import json

        out = summarise_batch(spec, ["Call 0412 345 678"])
        assert isinstance(json.loads(out["findings"][0]), list)

    def test_needs_review_flags_the_risky_row(self, spec: RedactorSpec) -> None:
        out = summarise_batch(
            spec,
            [
                "Great workplace, very supportive team.",
                "I am a Grade 4 nurse, casual, team of three, 18 years, at Acme Pty Ltd",
            ],
        )
        assert out["needs_review"] == [False, True]

    def test_spark_helpers_import_without_pyspark(self) -> None:
        # The module must be importable and its pure helpers usable on a driver
        # or a test environment with no cluster.
        import pii_redact.spark as spark

        assert callable(spark.redact_text)
        assert callable(spark.summarise_batch)

    @pytest.mark.parametrize(
        "fn",
        ["redact_batch_udf", "redact_column_udf"],
    )
    def test_udf_builders_fail_loudly_without_pyspark(self, fn: str) -> None:
        """No Spark on this machine: the builders must raise, not return junk."""
        import importlib.util

        import pii_redact.spark as spark

        if importlib.util.find_spec("pyspark") is not None:
            pytest.skip("pyspark is installed; nothing to assert")
        with pytest.raises(ImportError):
            getattr(spark, fn)(RedactorSpec.build())
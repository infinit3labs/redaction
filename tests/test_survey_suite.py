"""The sample survey suite, as a regression gate.

``data/survey_responses.ndjson`` plus ``data/survey_manifest.json`` are the
observation fixtures. The expectations in the manifest are authored from the
specification of what a WHS redactor ought to do, so this file asserts against
them rather than against whatever the module currently does.

Risk bands are treated separately from detection. They are calibration
placeholders, not ground truth, so they are pinned only against the same
recorded values and a change there is a prompt to re-derive them rather than a
failure.

To re-observe the suite by hand::

    python examples/run_survey_suite.py --show-clean
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pii_redact import Category, Lexicon, RiskBand, Term
from pii_redact.spark import RedactorSpec, summarise_batch

ROOT = Path(__file__).resolve().parent.parent
NDJSON = ROOT / "data" / "survey_responses.ndjson"
MANIFEST = ROOT / "data" / "survey_manifest.json"

#: Mirrors examples/run_survey_suite.py. The internal project name stands in for
#: the term list a deployment supplies; "Kestral" in the fixture is misspelled
#: so the fuzzy path is exercised, not just the exact one.
DEMO_LEXICON = Lexicon.from_terms(
    [
        Term(
            "Project Kestrel",
            Category.INTERNAL_TERM,
            aliases=["kestrel"],
            label="internal project",
        ),
        Term("Beacon", Category.INTERNAL_TERM, label="internal program"),
        Term("Kwinana depot", Category.INTERNAL_TERM, label="workplace site"),
    ]
)

#: Deterministic, and not a secret. The suite must produce identical output on
#: every run for the assertions below to mean anything.
DEMO_SALT = "0" * 32

#: Qualtrics metadata, not free text.
METADATA_COLUMNS = {"ResponseID", "StartDate", "EndDate", "IPAddress"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records() -> list[dict]:
    out = []
    for number, line in enumerate(NDJSON.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - fixture integrity
            raise AssertionError(f"{NDJSON}:{number}: invalid JSON: {exc}") from exc
    return out


@pytest.fixture(scope="module")
def spec() -> RedactorSpec:
    return RedactorSpec.build(salt=DEMO_SALT, lexicon=DEMO_LEXICON)


@pytest.fixture(scope="module")
def redacted(spec: RedactorSpec, records: list[dict]) -> dict[str, dict[str, dict]]:
    """Redact every record once: ``{response_id: {column: {text, findings, band}}}``."""
    columns = sorted({c for r in records for c in r if c not in METADATA_COLUMNS})
    batch = {c: summarise_batch(spec, [r.get(c) for r in records]) for c in columns}
    index = {r["ResponseID"]: i for i, r in enumerate(records)}
    out: dict[str, dict[str, dict]] = {}
    for response_id, position in index.items():
        entry: dict[str, dict] = {}
        for column in columns:
            findings = json.loads(batch[column]["findings"][position])
            entry[column] = {
                "text": batch[column]["redacted"][position],
                "categories": {f["category"] for f in findings},
                "band": batch[column]["risk_band"][position],
                "risk": batch[column]["risk"][position],
            }
        out[response_id] = entry
    return out


def joined(redacted: dict, response_id: str) -> str:
    return "\n".join(column["text"] for column in redacted[response_id].values())


def categories_of(redacted: dict, response_id: str) -> set[str]:
    found: set[str] = set()
    for column in redacted[response_id].values():
        found |= column["categories"]
    return found


class TestFixtureIntegrity:
    def test_every_record_has_a_manifest_entry(self, manifest: dict, records: list[dict]) -> None:
        declared = {e["id"] for e in manifest["records"]}
        actual = {r["ResponseID"] for r in records}
        assert declared == actual

    def test_response_ids_are_unique(self, records: list[dict]) -> None:
        ids = [r["ResponseID"] for r in records]
        assert len(ids) == len(set(ids))

    def test_the_fixture_actually_covers_the_hard_cases(self, records: list[dict]) -> None:
        """A suite that stops testing anything is worse than no suite."""
        blob = json.dumps(records, ensure_ascii=False).lower()
        for marker, why in [
            ("bullylng", "misspelled bullying"),
            ("timsheet", "misspelled internal system"),
            ("kestral", "misspelled internal project"),
            ("servcie", "misspelled tenure"),
            ("０", "fullwidth digits"),
            ("j a n e", "spaced out email"),
            ("only woman", "uniqueness claim"),
            ("anxiety", "health detail"),
            ("bully", "incident narrative"),
            ("nguyễn", "a non-ASCII name"),
            ("công ty", "a second language response"),
            ("eap", "counselling referred by the provider"),
            ("pre-existing", "a quoted decision letter"),
            ("pre-approval", "an authorisation complaint"),
            ("telehealth", "an access barrier"),
            ("roster", "a psychosocial hazard named as one"),
        ]:
            assert marker in blob, f"fixture no longer contains {why}"

    @pytest.mark.parametrize(
        "needle",
        ["CLM-2026-8812", "$1,250.00", "v1.2.3.4", "2.4.0.1", "5300000100"],
    )
    def test_hard_negatives_are_still_here(self, records: list[dict], needle: str) -> None:
        """Claim references, amounts, version strings and near-miss ABNs.

        These are the values a redaction module gets wrong when it is greedy,
        so removing one from the fixture quietly deletes the test.
        """
        assert any(needle in r.get("Q1", "") or needle in r.get("Q2", "") for r in records)

    def test_the_survey_is_answered_as_a_service_survey(
        self, records: list[dict]
    ) -> None:
        """The fixture has to be the survey that was actually sent.

        A corpus of complaints only proves the module survives them. Q1 asks
        about the service received, so almost every record has to answer it,
        and the psychosocial answers sit in their own column rather than
        trailing behind the first thing the respondent typed.
        """
        answered = [
            r for r in records if r.get("Q1", "").strip() and r["ResponseID"] != "S013"
        ]
        assert len(answered) >= len(records) - 1
        psychosocial = [r for r in records if r.get("Q3", "").strip()]
        assert len(psychosocial) >= len(records) // 3, "too few psychosocial answers"
        assert len({r["ResponseID"] for r in psychosocial}) == len(psychosocial)

    def test_metadata_is_export_shaped(self, records: list[dict]) -> None:
        """Start and end timestamps must order, and the response must be shorter.

        A fixture with negative durations or end dates before start dates cannot
        be used to test anything that looks at response time.
        """
        for record in records:
            start = record["StartDate"]
            end = record["EndDate"]
            assert start < end, record["ResponseID"]

    def test_non_ascii_survives_a_pass(self, redacted: dict, records: list[dict]) -> None:
        """Offsets are byte offsets, and survey text is not all ASCII.

        A name with diacritics and a whole answer in a second language both
        have to come out the other side unchanged, or every downstream offset
        into the column is wrong.
        """
        record = next(r for r in records if r["ResponseID"] == "S043")
        non_ascii = {
            column: value
            for column, value in record.items()
            if isinstance(value, str) and not value.isascii()
        }
        assert len(non_ascii) == 2, non_ascii
        for column, value in non_ascii.items():
            assert redacted["S043"][column]["text"] == value


class TestRedactionExpectations:
    def test_nothing_that_must_be_redacted_survives(
        self, manifest: dict, redacted: dict
    ) -> None:
        failures: list[str] = []
        for entry in manifest["records"]:
            text = joined(redacted, entry["id"])
            for needle in entry.get("must_not_appear", []):
                if needle in text:
                    failures.append(f"{entry['id']}: {needle!r} still present")
        assert not failures, "not redacted:\n  " + "\n  ".join(failures)

    def test_nothing_that_must_survive_was_over_redacted(
        self, manifest: dict, redacted: dict
    ) -> None:
        failures: list[str] = []
        for entry in manifest["records"]:
            text = joined(redacted, entry["id"])
            for needle in entry.get("must_appear", []):
                if needle not in text:
                    failures.append(f"{entry['id']}: {needle!r} was removed")
        assert not failures, "over-redacted:\n  " + "\n  ".join(failures)

    def test_every_expected_category_is_detected(self, manifest: dict, redacted: dict) -> None:
        failures: list[str] = []
        for entry in manifest["records"]:
            found = categories_of(redacted, entry["id"])
            for category in entry.get("categories", []):
                if category not in found:
                    failures.append(f"{entry['id']}: {category} not detected")
        assert not failures, "missing detections:\n  " + "\n  ".join(failures)


class TestNegativeControls:
    """Records with nothing to find must produce nothing."""

    @pytest.mark.parametrize(
        "response_id",
        ["S001", "S002", "S003", "S004", "S013", "S024", "S027", "S031", "S032", "S045"],
    )
    def test_no_false_positives(
        self, manifest: dict, redacted: dict, response_id: str
    ) -> None:
        entry = next(e for e in manifest["records"] if e["id"] == response_id)
        assert entry["categories"] == [], f"{response_id} was meant to be a negative control"
        # "short_response" is a length signal, not a detection, so exclude it.
        findings = categories_of(redacted, response_id)
        assert findings == set(), f"{response_id}: unexpected detections {sorted(findings)}"

    def test_every_declared_control_is_really_clean(
        self, manifest: dict, redacted: dict
    ) -> None:
        """A record with no expected categories must produce no categories."""
        failures = []
        for entry in manifest["records"]:
            if entry["categories"]:
                continue
            found = categories_of(redacted, entry["id"])
            if found:
                failures.append(f"{entry['id']}: {sorted(found)}")
        assert not failures, "unexpected detections:\n  " + "\n  ".join(failures)

    def test_provider_side_units_are_not_employers(self, redacted: dict) -> None:
        """S027 describes the provider's own service function four times.

        Redacting these labels the surveyor's own call centre as if it were the
        respondent's employer, and takes the only sentence that records how the
        respondent was actually treated with it.
        """
        text = joined(redacted, "S027")
        assert "the claims team" in text
        assert "the service centre" in text
        assert "I have rung" in text

    def test_benign_text_is_byte_identical(self, spec: RedactorSpec) -> None:
        benign = "The training was good and my supervisor is supportive."
        assert summarise_batch(spec, [benign])["redacted"][0] == benign


class TestMisspellingAndObfuscation:
    @pytest.mark.parametrize(
        ("response_id", "needle", "why"),
        [
            ("S014", "intrant", "misspelled internal system, caught by fuzzy match"),
            ("S015", "timsheet", "misspelled internal system"),
            ("S015", "Kestral", "misspelled internal project from the supplied lexicon"),
            ("S017", "breifing", "misspelled seeded internal term"),
        ],
    )
    def test_misspellings_are_caught(
        self, redacted: dict, response_id: str, needle: str, why: str
    ) -> None:
        assert needle not in joined(redacted, response_id), why

    def test_prose_around_a_misspelling_survives(self, redacted: dict) -> None:
        # Catching the identifier must not take the sentence with it.
        assert "nobody told us why" in joined(redacted, "S015")

    def test_fullwidth_digits_are_caught(self, redacted: dict) -> None:
        assert "０" not in joined(redacted, "S016")

    def test_spaced_out_email_is_caught(self, redacted: dict) -> None:
        assert "j a n e" not in joined(redacted, "S016")

    def test_spaced_email_does_not_fire_on_prose(self, spec: RedactorSpec) -> None:
        out = summarise_batch(spec, ["I met him at 5 pm about the report and the a b c team"])
        assert out["redacted"][0] == "I met him at 5 pm about the report and the a b c team"


class TestOffsetIntegrity:
    def test_findings_index_the_original_text(self, spec: RedactorSpec, records: list[dict]) -> None:
        for record in records:
            for column, value in record.items():
                if column in METADATA_COLUMNS or not isinstance(value, str) or not value:
                    continue
                result = summarise_batch(spec, [value])
                for finding in json.loads(result["findings"][0]):
                    start, end = finding["start"], finding["end"]
                    assert 0 <= start < end <= len(value), (record["ResponseID"], column)
                    assert value[start:end].strip() != "", (record["ResponseID"], column)


class TestRiskCalibration:
    """Bands are placeholders, so this asserts stability, not correctness.

    A change here means the weights moved. Re-derive the manifest bands from a
    fresh run and confirm the shift is an improvement before accepting it.
    """

    def test_bands_match_the_recorded_calibration(self, manifest: dict, redacted: dict) -> None:
        failures = []
        for entry in manifest["records"]:
            expected = RiskBand(entry["risk_band"])
            observed = {
                RiskBand(column["band"]) for column in redacted[entry["id"]].values()
            }
            if max(observed, key=lambda b: list(RiskBand).index(b)) is not expected:
                failures.append(f"{entry['id']}: expected {expected.value}, saw {sorted(b.value for b in observed)}")
        assert not failures, "band drift:\n  " + "\n  ".join(failures)

    def test_confidentiality_request_raises_risk(self, redacted: dict) -> None:
        # S025 asks that this not be attributed to them. That must outweigh a
        # short answer with two quasi-identifiers.
        assert max(c["risk"] for c in redacted["S025"].values()) >= 0.4

    def test_clean_records_score_low(self, redacted: dict) -> None:
        for response_id in ("S001", "S013"):
            assert max(c["risk"] for c in redacted[response_id].values()) < 0.25

    def test_records_with_direct_identifiers_score_high(self, redacted: dict) -> None:
        for response_id in ("S012", "S023"):
            assert max(c["risk"] for c in redacted[response_id].values()) >= 0.5

    def test_band_is_not_severity(self, redacted: dict) -> None:
        """A severe response can be unidentifying, and that is not a bug.

        The band answers "could a colleague work out who wrote this". S029 is a
        harassment disclosure with a confidentiality request and still scores
        moderate, because nothing in it identifies anybody on its own. Severity
        needs a different signal, and mixing the two would push the triage queue
        in the wrong direction.
        """
        severe_but_unidentifying = {"S029": "moderate", "S032": "low", "S041": "moderate"}
        for response_id, band in severe_but_unidentifying.items():
            observed = {
                RiskBand(column["band"]) for column in redacted[response_id].values()
            }
            worst = max(observed, key=lambda b: list(RiskBand).index(b))
            assert worst is RiskBand(band), f"{response_id} bands drifted: {observed}"


class TestKnownGaps:
    """Gaps the fixture holds open on purpose, so a fix is a visible change.

    Each of these is a real exposure in a survey run by a provider, and none of
    them is caught by the rules today. When one is fixed, the failure that
    appears here is the signal to remove the record's note from the manifest.
    """

    def test_member_and_claim_numbers_are_not_caught(self, redacted: dict) -> None:
        # A member number plus a claim reference identifies a person to anyone
        # holding the provider's own export. Residual disclosure control is the
        # layer meant to catch it, and it is off unless a population is supplied.
        text = joined(redacted, "S027")
        assert "Member number 8841-22607" in text

    def test_a_third_party_named_without_a_label_is_not_caught(self, redacted: dict) -> None:
        # The provider's caseworker, named in full, in Vietnamese. A role word
        # plus a capitalised token needs a possessive or a label; "my caseworker
        # was <Name>" is neither.
        assert "Nguyễn Thị Hương" in joined(redacted, "S043")

    def test_an_unlabelled_surname_is_not_caught(self, redacted: dict) -> None:
        assert "Okafor" in joined(redacted, "S045")

    def test_bare_suburbs_are_not_caught(self, redacted: dict) -> None:
        # There is no gazetteer. "Surry Hills" survives inside the postal line,
        # while "the Box Hill site" is caught by the workplace-site rule because
        # it is described as a site. A site or suburb list has to come through
        # the lexicon, which is why the runner takes a --lexicon argument at all.
        assert "Surry Hills" in joined(redacted, "S020")
        assert "Box Hill" not in joined(redacted, "S020")

    def test_a_month_day_date_is_reported_but_not_rewritten(self, redacted: dict) -> None:
        # "14 August" is detected and reported, then passed through unchanged by
        # the generaliser: without a year there is nothing to narrow to, and
        # dropping the day would destroy the only timeline the response gives.
        assert "14 August" in joined(redacted, "S041")


class TestIdempotence:
    def test_redacting_twice_changes_nothing(self, spec: RedactorSpec, redacted: dict) -> None:
        """A redacted stream must survive a second pass unchanged.

        This matters because the Spark path writes the redacted text to Delta and
        a later job may read it back and redact again.
        """
        for response_id, columns in redacted.items():
            texts = [column["text"] for column in columns.values()]
            again = summarise_batch(spec, texts)["redacted"]
            assert again == texts, f"{response_id} is not stable under re-redaction"
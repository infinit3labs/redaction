#!/usr/bin/env python3
"""Run the redaction suite over the sample survey data and report.

This is the observation tool. It loads ``data/survey_responses.ndjson``,
redacts every free text field, and compares the result against
``data/survey_manifest.json`` -- expectations written from the specification of
what ought to be caught, not from what the module happens to do.

Three kinds of report, in priority order:

* **MISS**  something the manifest says must be redacted, and is not
* **LEAK**  an expectation the manifest did not anticipate was still redacted
* **BAND**  a risk band that differs from the expected one

Exit status is 1 when anything is missed or leaked, so this doubles as a gate.

    python examples/run_survey_suite.py
    python examples/run_survey_suite.py --show-clean
    python examples/run_survey_suite.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pii_redact import Audience, Category, Lexicon, RiskBand, Term  # noqa: E402
from pii_redact.spark import RedactorSpec, build_worker_redactor  # noqa: E402

DATA = ROOT / "data"
NDJSON = DATA / "survey_responses.ndjson"
MANIFEST = DATA / "survey_manifest.json"

#: Demo salt. Deterministic so repeated runs produce identical output; not a
#: secret and never to be used for real data.
DEMO_SALT = "00000000000000000000000000000000"

#: A stand-in for the internal term list a deployment would supply. Without one
#: of these the bundled seed is all there is, and it cannot know a project name.
#: "Kestral" in record S015 is deliberately misspelled so this exercises the
#: fuzzy path rather than the exact one.
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

#: Qualtrics metadata and structured instrument items are not free text and are
#: not redacted. ``Instrument`` records which survey the response came from and
#: ``Consent`` is the discrete follow-up permission item; the free-text
#: conditions on follow-up contact are Q5.
METADATA_COLUMNS = {
    "ResponseID",
    "StartDate",
    "EndDate",
    "IPAddress",
    "Instrument",
    "Consent",
}
FREE_TEXT_PREFIX = "Q"


@dataclass
class RecordReport:
    """One record's outcome against the manifest."""

    record_id: str
    note: str
    expected_band: str
    actual_band: str
    redacted: dict[str, str] = field(default_factory=dict)
    observed_categories: set[str] = field(default_factory=set)
    expected_categories: set[str] = field(default_factory=set)
    missed: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    unexpected_categories: list[str] = field(default_factory=list)
    missing_categories: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missed and not self.leaked and not self.missing_categories

    @property
    def band_ok(self) -> bool:
        return self.expected_band == self.actual_band


def load_records(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
    return records


def check_record(report: RecordReport, fields: dict[str, str]) -> RecordReport:
    """Compare redacted text against the manifest for one record."""
    joined = "\n".join(fields.values())

    for needle in report.expected_terms.get("must_not_appear", []):
        if needle in joined:
            report.missed.append(needle)

    for needle in report.expected_terms.get("must_appear", []):
        if needle not in joined:
            # Only an over-redaction if the text is now missing it entirely.
            # A substitution that preserves meaning is acceptable, so report as
            # a leak only when no field retains the phrase.
            report.leaked.append(needle)

    for category in report.expected_categories:
        if category not in report.observed_categories:
            report.missing_categories.append(category)

    report.unexpected_categories = sorted(report.observed_categories - report.expected_categories)
    return report


def build_reports(manifest: dict, records: list[dict], spec: RedactorSpec) -> list[RecordReport]:
    survey = build_worker_redactor(spec)
    by_id = {e["id"]: e for e in manifest["records"]}
    reports: list[RecordReport] = []

    for record in records:
        record_id = record.get("ResponseID", "<no id>")
        entry = by_id.get(record_id)
        if entry is None:
            reports.append(
                RecordReport(
                    record_id=record_id,
                    note="NO MANIFEST ENTRY",
                    expected_band="?",
                    actual_band="?",
                    missed=["<record absent from the manifest>"],
                )
            )
            continue

        report = RecordReport(
            record_id=record_id,
            note=entry.get("note", ""),
            expected_band=entry.get("risk_band", "low"),
            actual_band="low",
            expected_categories=set(entry.get("categories", [])),
        )
        report.expected_terms = entry  # type: ignore[attr-defined]

        worst_score = 0.0
        for column, value in record.items():
            if column in METADATA_COLUMNS or not column.startswith(FREE_TEXT_PREFIX):
                continue
            result = survey.redact_record(value)
            report.redacted[column] = result.text
            report.findings.extend(
                {
                    "column": column,
                    "category": f.category.value,
                    "detector": f.detector,
                    "confidence": round(f.confidence, 3),
                    "original": value[f.span.start : f.span.end],
                }
                for f in result.findings
            )
            report.observed_categories.update(f.category.value for f in result.findings)
            worst_score = max(worst_score, result.risk.score)

        report.actual_band = _band_for(worst_score).value
        reports.append(check_record(report, report.redacted))

    return reports


def _band_for(score: float) -> RiskBand:
    from pii_redact.risk import _band

    return _band(score)


def print_report(reports: list[RecordReport], show_clean: bool) -> None:
    misses = [r for r in reports if r.missed]
    leaks = [r for r in reports if r.leaked]
    missing_categories = [r for r in reports if r.missing_categories]
    bands = [r for r in reports if not r.band_ok]

    for report in reports:
        clean = report.ok and report.band_ok
        if clean and not show_clean:
            continue
        marker = "PASS" if report.ok else "FAIL"
        band_marker = " " if report.band_ok else "~"
        print(f"[{marker}{band_marker}] {report.record_id}  {report.note}")
        if not report.band_ok:
            print(f"        risk band: expected {report.expected_band}, got {report.actual_band}")
        for item in report.missed:
            print(f"        MISS   still present: {item!r}")
        for item in report.leaked:
            print(f"        LEAK   wrongly removed: {item!r}")
        for item in report.missing_categories:
            print(f"        MISS   category not detected: {item}")
        if report.unexpected_categories and not report.ok:
            print(f"        extra  categories detected: {', '.join(report.unexpected_categories)}")
        for column, text in report.redacted.items():
            print(f"        {column}: {text}")
        print()

    print("=" * 78)
    total = len(reports)
    passed = sum(1 for r in reports if r.ok)
    print(
        f"{passed}/{total} records fully met expectations   "
        f"({len(misses)} with misses, {len(leaks)} with over-redaction, "
        f"{len(missing_categories)} missing categories, {len(bands)} band mismatches)"
    )

    if misses or leaks or missing_categories:
        print()
        print("Outstanding gaps, by class:")
        for report in reports:
            if report.missed or report.leaked or report.missing_categories:
                for item in report.missed:
                    print(f"  {report.record_id}  not redacted: {item!r}")
                for item in report.leaked:
                    print(f"  {report.record_id}  over-redacted: {item!r}")
                for item in report.missing_categories:
                    print(f"  {report.record_id}  category absent: {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-clean", action="store_true", help="print passing records too")
    parser.add_argument("--json", action="store_true", help="emit a machine readable report")
    parser.add_argument("--salt", default=DEMO_SALT, help="demo salt, hex encoded")
    parser.add_argument(
        "--no-lexicon",
        action="store_true",
        help="use only the bundled seed lexicon, with no internal terms",
    )
    parser.add_argument(
        "--audience",
        choices=[a.value for a in Audience],
        default=Audience.INTERNAL.value,
    )
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = load_records(NDJSON)
    lexicon = None if args.no_lexicon else DEMO_LEXICON
    spec = RedactorSpec.build(salt=args.salt, audience=Audience(args.audience), lexicon=lexicon)
    reports = build_reports(manifest, records, spec)

    if args.json:
        print(
            json.dumps(
                {
                    "records": [
                        {
                            "id": r.record_id,
                            "ok": r.ok,
                            "expected_band": r.expected_band,
                            "actual_band": r.actual_band,
                            "missed": r.missed,
                            "leaked": r.leaked,
                            "missing_categories": r.missing_categories,
                            "unexpected_categories": r.unexpected_categories,
                            "redacted": r.redacted,
                            "findings": r.findings,
                        }
                        for r in reports
                    ]
                },
                indent=2,
            )
        )
    else:
        print_report(reports, args.show_clean)

    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
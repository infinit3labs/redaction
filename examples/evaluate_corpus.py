"""Measure recall against generated ground truth.

Runs a generated corpus through the redactor and reports, per secret kind, how
many planted values survived. This is the number that matters for a redaction
module: not how many findings it produced, but how many known secrets escaped.

    python examples/evaluate_corpus.py --count 400
    python examples/evaluate_corpus.py --count 400 --show-misses

A secret counts as caught when its literal text is absent from the redacted
output. That is deliberately stricter than "a finding overlapped it": the
standard is what a reader of the output can still see.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pii_redact import Category, Lexicon, Term  # noqa: E402
from pii_redact.generate import GeneratedRecord, SurveyGenerator  # noqa: E402
from pii_redact.spark import RedactorSpec, build_worker_redactor  # noqa: E402

#: Deterministic demo salt. Tokens are no longer hashed by default, so this
#: only affects records configured with a hash style.
DEMO_SALT = "0" * 32

DEMO_LEXICON = Lexicon.from_terms(
    [
        Term("Project Kestrel", Category.INTERNAL_TERM, aliases=["kestrel"],
             label="internal project"),
        Term("Beacon", Category.INTERNAL_TERM, label="internal program"),
    ]
)

#: Qualtrics metadata is not free text and is not evaluated.
METADATA_COLUMNS = {"ResponseID", "StartDate", "EndDate", "IPAddress"}


@dataclass
class Outcome:
    """Aggregate result of one evaluation run."""

    planted: Counter[str] = field(default_factory=Counter)
    caught: Counter[str] = field(default_factory=Counter)
    planted_by_form: Counter[str] = field(default_factory=Counter)
    caught_by_form: Counter[str] = field(default_factory=Counter)
    survived: list[tuple[str, str, str]] = field(default_factory=list)
    survived_stable: Counter[str] = field(default_factory=Counter)
    removed_stable: Counter[str] = field(default_factory=Counter)
    record_count: int = 0

    def rate(self, kind: str) -> float:
        total = self.planted[kind]
        return 1.0 if total == 0 else self.caught[kind] / total

    def form_rate(self, form: str) -> float:
        total = self.planted_by_form[form]
        return 1.0 if total == 0 else self.caught_by_form[form] / total

    @property
    def overall(self) -> float:
        total = sum(self.planted.values())
        return 1.0 if total == 0 else sum(self.caught.values()) / total

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.record_count,
            "planted": dict(self.planted),
            "caught": dict(self.caught),
            "recall_by_kind": {k: round(self.rate(k), 4) for k in sorted(self.planted)},
            "overall_recall": round(self.overall, 4),
            "recall_by_form": {
                k: round(self.form_rate(k), 4) for k in sorted(self.planted_by_form)
            },
            "over_redaction_by_kind": {
                k: v for k, v in sorted(self.removed_stable.items())
            },
        }


def evaluate(
    records: list[GeneratedRecord],
    spec: RedactorSpec,
    *,
    show_misses: int = 0,
) -> Outcome:
    survey = build_worker_redactor(spec)
    outcome = Outcome(record_count=len(records))

    for record in records:
        columns = [c for c in record.answers if c not in METADATA_COLUMNS]
        texts = [record.answers[c] for c in columns]
        batch = _summarise(survey, spec, texts)
        redacted = "\n".join(batch)
        original = "\n".join(texts)

        for secret in record.secrets:
            outcome.planted[secret.kind] += 1
            outcome.planted_by_form[secret.degradation] += 1
            if secret.literal not in redacted:
                outcome.caught[secret.kind] += 1
                outcome.caught_by_form[secret.degradation] += 1
            elif len(outcome.survived) < show_misses:
                outcome.survived.append((record.response_id, secret.kind, secret.literal))

        # Over-redaction: prose the generator did not plant a secret in.
        for fragment in record.must_survive:
            if fragment in original and fragment not in redacted:
                outcome.removed_stable[fragment.split()[0]] += 1
            elif fragment in original:
                outcome.survived_stable[fragment.split()[0]] += 1

    return outcome


def _summarise(survey, spec: RedactorSpec, texts: list[str]) -> list[str]:
    return [survey.redact_record(t).text for t in texts]


def print_report(outcome: Outcome, show_misses: int) -> None:
    print(f"{outcome.record_count} records, "
          f"{sum(outcome.planted.values())} planted secrets\n")
    print(f"{'kind':22} {'planted':>8} {'caught':>7} {'recall':>8}")
    print("-" * 48)
    for kind in sorted(outcome.planted):
        print(f"{kind:22} {outcome.planted[kind]:>8} {outcome.caught[kind]:>7} "
              f"{outcome.rate(kind) * 100:>7.1f}%")
    print("-" * 48)
    print(f"{'OVERALL':22} {sum(outcome.planted.values()):>8} "
          f"{sum(outcome.caught.values()):>7} {outcome.overall * 100:>7.1f}%")

    print("\nRecall by how the value was written:")
    for form in sorted(outcome.planted_by_form):
        print(f"  {form:14} {outcome.planted_by_form[form]:>5} planted  "
              f"{outcome.form_rate(form) * 100:>6.1f}% caught")

    if outcome.removed_stable:
        print("\nOver-redaction (prose the generator planted no secret in):")
        for fragment, count in sorted(outcome.removed_stable.items()):
            print(f"  {fragment:24} {count}")

    if show_misses and outcome.survived:
        print(f"\nSurviving secrets (first {show_misses}):")
        for response_id, kind, literal in outcome.survived:
            print(f"  {response_id}  {kind:20} {literal!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=400, help="records to generate")
    parser.add_argument("--seed", type=int, default=20260308)
    parser.add_argument("--injection-rate", type=float, default=0.85)
    parser.add_argument("--show-misses", type=int, default=15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    generator = SurveyGenerator(seed=args.seed, injection_rate=args.injection_rate)
    records = generator.corpus(args.count)
    spec = RedactorSpec.build(salt=DEMO_SALT, lexicon=DEMO_LEXICON)
    outcome = evaluate(records, spec, show_misses=args.show_misses)

    if args.json:
        print(json.dumps(outcome.as_dict(), indent=2))
    else:
        print_report(outcome, args.show_misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
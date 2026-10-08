"""Command-line interface: ``pii-redact [FILE]`` or stdin to stdout."""

from __future__ import annotations

import argparse
import enum
import json
import sys
from pathlib import Path

from .engine import Policy, Redactor
from .types import Category, Finding, MaskStyle


class OutputFormat(enum.Enum):
    TEXT = "text"
    JSON = "json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-redact",
        description="Redact Australian PII from text using deterministic rules "
        "and optional NLP augmentation.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="files to redact; omit to read stdin",
    )
    parser.add_argument(
        "--style",
        choices=[s.value for s in MaskStyle],
        default=MaskStyle.MASK.value,
        help="replacement style (default: mask)",
    )
    parser.add_argument(
        "--keep-suffix",
        type=int,
        default=4,
        help="characters preserved at the end under --style partial (default: 4)",
    )
    parser.add_argument(
        "--keep-prefix",
        type=int,
        default=0,
        help="characters preserved at the start under --style partial (default: 0)",
    )
    parser.add_argument(
        "--mask-char",
        default="*",
        help="replacement character for --style mask and partial (default: *)",
    )
    parser.add_argument(
        "--categories",
        help="comma-separated categories to redact; default is all",
    )
    parser.add_argument(
        "--nlp",
        action="store_true",
        help="enable the NLP augmentation stage",
    )
    parser.add_argument(
        "--nlp-backend",
        choices=["gazetteer", "spacy"],
        default="gazetteer",
        help="NLP backend to use with --nlp (default: gazetteer)",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="spaCy model to load with --nlp-backend spacy",
    )
    parser.add_argument(
        "--format",
        choices=[f.value for f in OutputFormat],
        default=OutputFormat.TEXT.value,
        help="output format (default: text)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="report findings with offsets and metadata instead of redacting",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a one-line summary to stderr after processing",
    )
    return parser


def _policy_from_args(args: argparse.Namespace) -> Policy:
    categories: frozenset[Category] | None = None
    if args.categories:
        wanted = {c.strip().upper() for c in args.categories.split(",") if c.strip()}
        known = {c.value.upper() for c in Category}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown categories: {', '.join(sorted(unknown))}")
        categories = frozenset(c for c in Category if c.value.upper() in wanted)
    return Policy(
        categories=categories,
        styles={"default": MaskStyle(args.style)},
        nlp_enabled=args.nlp,
        nlp_backend=args.nlp_backend,
        spacy_model=args.spacy_model,
        mask_char=args.mask_char,
        partial_keep_prefix=args.keep_prefix,
        partial_keep_suffix=args.keep_suffix,
    )


def _finding_dict(f: Finding) -> dict[str, object]:
    return {
        "category": f.category.value,
        "start": f.span.start,
        "end": f.span.end,
        "detector": f.detector,
        "stage": f.stage.value,
        "confidence": round(f.confidence, 4),
        "metadata": f.meta(),
    }


def _render_findings(findings: list[Finding]) -> str:
    return json.dumps([_finding_dict(f) for f in findings], indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    redactor = Redactor(_policy_from_args(args))

    paths = args.files or [None]
    out = sys.stdout
    findings_total = 0

    for path in paths:
        if path is None:
            text = sys.stdin.read()
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"pii-redact: {exc}", file=sys.stderr)
                return 2

        result = redactor.redact(text)
        findings_total += len(result)

        if args.scan:
            out.write(_render_findings(list(result)) + "\n")
        elif args.format == OutputFormat.JSON.value:
            out.write(
                json.dumps(
                    {
                        "text": result.text,
                        "findings": [_finding_dict(f) for f in result],
                    },
                    default=str,
                )
                + "\n"
            )
        else:
            out.write(result.text)

        if args.summary:
            print(result.summary(), file=sys.stderr)

    # Exit 1 under --scan when nothing was found, so shell pipelines can gate
    # on the presence of PII. Text and JSON output always exit 0.
    if args.scan and findings_total == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
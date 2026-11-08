#!/usr/bin/env python3
"""Scan text for compositional sensitivity using the Comcare research rules.

Dependency-free. Loads sensitive_context_lexicon.json and
sensitive_context_rules.json from this directory, then reports every rule
fired, the trigger terms that fired it, the action to take, and the sentence.

Usage:
    python context_scan.py FILE [FILE ...]        # text or NDJSON files
    python context_scan.py --json FILE            # machine-readable findings
    python context_scan.py --stats FILE           # per-rule counts, incl. guard cancellations
    python context_scan.py --no-guards FILE       # measure guard effect
    python context_scan.py --validate             # check groups, actions, regexes

NDJSON input: every string value in each JSON object is scanned; findings
carry the record index and key. This is how the survey fixture is scanned.

The scanner is deliberately conservative in the same direction as
pii_redact: a guard cancels a match, and a rule needs a co-trigger from every
outer all_of element. It is a research instrument for measuring rule quality
over the held-out probes, not a replacement for the engine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEXICON = json.loads((HERE / "sensitive_context_lexicon.json").read_text(encoding="utf-8"))
RULES = json.loads((HERE / "sensitive_context_rules.json").read_text(encoding="utf-8"))
GROUPS: dict[str, list[str]] = LEXICON["groups"]

# Token-distance ceilings approximating the declared proximity.
PROXIMITY_TOKENS = {"clause": 15, "sentence": 80, "paragraph": 400}
CAPITALISED_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
#: Capitalised tokens that are not names. Months, weekdays, sentence openers,
#: institutions and acronym-expansions would otherwise make every sentence
#: with a date look like a third-party naming.
_NOT_NAMES = frozenset(
    w.lower()
    for w in (
        "January February March April May June July August September October November December "
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday "
        "The This That These Those There Their They Them Then Than When Where Which While Who Why How "
        "My Our Your His Her Its We I He She You It "
        "After Before Since Until Because However Although Despite During Following "
        "Comcare WorkSafe SafeWork WHS SRC APS SES HR GP EAP ART AAT NSW VIC QLD WA SA TAS ACT NT "
        "Australia Australian Commonwealth Code Act Regulations Section Part Division Clause "
        "January February March April May June July August September October November December"
    ).split()
)
_WORD_TOKEN = re.compile(r"[A-Za-z0-9']+")

_phrase_cache: dict[str, re.Pattern[str]] = {}


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Compile one group phrase: literal, whitespace-tolerant, word-bounded."""
    if phrase not in _phrase_cache:
        body = re.escape(phrase.strip())
        body = body.replace(r"\ ", r"\s+")
        _phrase_cache[phrase] = re.compile(rf"(?<![\w]){body}(?![\w])", re.IGNORECASE)
    return _phrase_cache[phrase]


def iter_tokens(text: str):
    """(token, start, end) with absolute offsets."""
    for m in _WORD_TOKEN.finditer(text):
        yield m.group(0), m.start(), m.end()


class Sentence:
    """One sentence with its tokens, for proximity and guard checks."""

    def __init__(self, text: str, start: int):
        self.text = text
        self.start = start
        self.tokens = list(iter_tokens(text))

    def phrase_spans(self, group: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if group == "@capitalised_name":
            for m in CAPITALISED_NAME.finditer(self.text):
                before = self.text[: m.start()].rstrip()
                if m.start() == 0 or before.endswith((".", "!", "?", "\n")) or not before:
                    continue  # sentence-initial capital
                if m.group(0).lower() in _NOT_NAMES:
                    continue
                spans.append((m.start(), m.end()))
            return spans
        for phrase in GROUPS.get(group, []):
            for m in phrase_pattern(phrase).finditer(self.text):
                spans.append((m.start(), m.end()))
        return spans

    def guard_hit(self, guard_groups: list[str]) -> tuple[str, str] | None:
        for group in guard_groups:
            for phrase in GROUPS.get(group, []):
                if group == "@capitalised_name":
                    continue
                m = phrase_pattern(phrase).search(self.text)
                if m:
                    return group, self.text[m.start():m.end()]
        return None


def _token_gap(tokens, spans) -> int:
    """Tokens between the first and last chosen span."""
    lo = min(span[0] for span in spans)
    hi = max(span[1] for span in spans)
    return sum(1 for _, s, _ in tokens if lo <= s < hi)


def _assign(candidates, tokens, ceiling):
    """Pick one distinct span per outer element, minimising spread."""
    best = None
    caps = [c[:15] for c in candidates]

    def recurse(i, chosen):
        nonlocal best
        if i == len(caps):
            gap = _token_gap(tokens, chosen)
            if gap <= ceiling and (best is None or gap < best[0]):
                best = (gap, list(chosen))
            return
        for span in caps[i]:
            if any(span[0] < c[1] and c[0] < span[1] for c in chosen):
                continue
            chosen.append(span)
            recurse(i + 1, chosen)
            chosen.pop()

    recurse(0, [])
    return best[1] if best else None


def scan_text(text: str, use_guards: bool = True):
    findings, cancelled = [], 0
    sentences = [
        Sentence(text[m.start():m.end()], m.start())
        for m in re.finditer(r"[^.!?\n]+[.!?]*", text)
        if m.group().strip()
    ]
    # A paragraph rule may cross a full stop, but never crosses a line break:
    # in survey data each line/field is one answer, and merging answers would
    # invent combinations the respondent never wrote.
    paragraphs = [
        Sentence(text[m.start():m.end()], m.start())
        for m in re.finditer(r"[^\n]+", text)
        if m.group().strip()
    ]

    for rule in RULES["rules"]:
        ceiling = PROXIMITY_TOKENS.get(rule.get("proximity", "sentence"), 80)
        ceiling = rule.get("max_words_between", ceiling)
        units = paragraphs if rule.get("proximity") == "paragraph" else sentences
        seen_anchors: set[int] = set()
        for unit in units:
            per_element = []
            for alternatives in rule["all_of"]:
                candidates = []
                for group in alternatives:
                    for span in unit.phrase_spans(group):
                        candidates.append((span[0], span[1], group))
                if not candidates:
                    break
                per_element.append(sorted(candidates))
            if len(per_element) != len(rule["all_of"]):
                continue
            chosen = _assign(per_element, unit.tokens, ceiling)
            if chosen is None:
                continue
            anchor = chosen[0][0]
            anchor_key = unit.start + anchor
            if anchor_key in seen_anchors:
                continue
            seen_anchors.add(anchor_key)
            if use_guards:
                anchor_abs = unit.start + anchor
                anchor_sentence = next(
                    (s for s in sentences if s.start <= anchor_abs < s.start + len(s.text)),
                    unit,
                )
                guard = anchor_sentence.guard_hit(rule.get("guards", []))
                if guard:
                    cancelled += 1
                    continue
            matched_terms = sorted({g for _, _, g in chosen})
            findings.append(
                {
                    "rule": rule["id"],
                    "name": rule["name"],
                    "class": rule["class"],
                    "action": rule["action"],
                    "risk_weight": rule.get("risk_weight"),
                    "terms": matched_terms,
                    "sentence": unit.text.strip(),
                    "span": [unit.start, unit.start + len(unit.text)],
                }
            )
    return findings, cancelled


def scan_value(value, findings, prefix=""):
    if isinstance(value, str):
        for f in scan_text(value)[0]:
            f = dict(f)
            f["path"] = prefix
            findings.append(f)
    elif isinstance(value, dict):
        for k, v in value.items():
            scan_value(v, findings, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            scan_value(v, findings, f"{prefix}[{i}]")


def load_inputs(paths):
    for p in paths:
        p = Path(p)
        raw = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix == ".ndjson":
            for i, line in enumerate(raw.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield p.name, i, raw
                    continue
                for key, value in obj.items():
                    if isinstance(value, str):
                        yield p.name, i, value
        elif p.suffix == ".json":
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield p.name, 0, raw
                continue
            findings = []
            scan_value(obj, findings, p.stem)
            for f in findings:
                yield p.name, f.get("path", ""), f["sentence"]
        else:
            yield p.name, 0, raw


def validate() -> int:
    problems = 0
    known = set(GROUPS)
    for rule in RULES["rules"]:
        for alternatives in rule["all_of"]:
            for group in alternatives:
                if group != "@capitalised_name" and group not in known:
                    print(f"  MISSING GROUP {group} in {rule['id']}")
                    problems += 1
        for group in rule.get("guards", []):
            if group not in known:
                print(f"  MISSING GUARD GROUP {group} in {rule['id']}")
                problems += 1
        if "regex" in rule:
            try:
                re.compile(rule["regex"])
            except re.error as e:
                print(f"  BAD REGEX {rule['id']}: {e}")
                problems += 1
    print(f"validate: {len(RULES['rules'])} rules, {len(GROUPS)} groups, {problems} problem(s)")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-guards", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        return validate()
    if not args.files:
        print("nothing to scan; pass files or --validate", file=sys.stderr)
        return 2

    all_findings = []
    per_rule: dict[str, int] = {}
    cancelled_total = 0
    for name, ref, text in load_inputs(args.files):
        findings, cancelled = scan_text(text, use_guards=not args.no_guards)
        cancelled_total += cancelled
        for f in findings:
            f["source"] = (name, ref)
            all_findings.append(f)
            per_rule[f["rule"]] = per_rule.get(f["rule"], 0) + 1

    if args.stats:
        print(f"rules fired: {len(per_rule)}/{len(RULES['rules'])}   guard cancellations: {cancelled_total}")
        for rule in RULES["rules"]:
            print(f"  {rule['id']:6s} {per_rule.get(rule['id'], 0):4d}  {rule['name']}")
        return 0
    if args.json:
        print(json.dumps(all_findings, indent=2))
        return 0
    for f in all_findings:
        print(f"{f['rule']} [{','.join(f['action'])}] {f['name']} :: {', '.join(f['terms'])}")
        print(f"    {f['sentence'][:220]}")
    print(f"\n{len(all_findings)} finding(s); {cancelled_total} guard cancellation(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

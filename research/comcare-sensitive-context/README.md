# Comcare sensitive-context research

Research output from the Comcare / Australian WHS regulator scan, built to
feed the `pii_redact` detection model with **compositional** sensitivity: the
words, concepts, abbreviations and sentence structures that pass a normal PII
scan but are sensitive when combined.

Start with [`REPORT.md`](REPORT.md). It contains the vocabulary tables,
acronyms, sentence structures, the twenty-one context rules, the precision
guards and the integration map into `src/pii_redact/`.

| File | Contents |
| --- | --- |
| `REPORT.md` | The research report and reading order |
| `sensitive_context_lexicon.json` | Terms by sensitivity class; trigger phrase groups; guard groups |
| `sensitive_context_rules.json` | 21 context rules: triggers, proximity, action, rationale, evidence, ready-to-compile regex |
| `abbreviations.json` | Acronyms with expansions, class, corpus count, verification status |
| `corpus_stats.json` | Corpus size and per-term attestation counts |
| `context_scan.py` | Dependency-free scanner over the rules, used for measurement |
| `sensitivity_probes.txt` | One passing probe per rule plus guard cases |

## Running the scanner

```console
$ python3 context_scan.py --validate                 # check groups, actions, regexes
$ python3 context_scan.py sensitivity_probes.txt     # one line per finding
$ python3 context_scan.py --stats sensitivity_probes.txt
$ python3 context_scan.py --json data/survey_responses.ndjson > hits.json
$ python3 context_scan.py --no-guards FILE           # measure guard contribution
```

Measured on the repo fixtures (2026-10-17): the 58-record survey fixture
produces 5 findings, all on records the manifest already marks sensitive, and
zero on the 26 declared-clean negative controls; the 50 held-out probes
produce 13 findings across 7 probes, on the harm, condition, trauma,
aggression and uniqueness themes. The rules are literal here; the engine's
normalisation adds typo and leetspeak tolerance.

## Integration order

1. Add `Category.SENSITIVE_CONTEXT` and a `detectors/context.py` that loads
   `sensitive_context_rules.json` (see REPORT.md §10).
2. Add the new weights and signals to `risk.py`.
3. Add the covert-naming implication rules (`no_names_frame`, `certain_person`,
   `shared_knowledge`, `initials_only`, `role_plus_event`).
4. Add the scheme-specific terms to `resources/lexicon.json` only where they
   narrow the employer or jurisdiction: `Pracsys`, `CAIS`, `FCDR`,
   `reasonable administrative action`, `sensitive claim`, `NV1`.
5. Add the §5 sentence shapes and the Claims Manual example pattern to
   `generate.py` and to `data/heldout_probes.ndjson`, then measure.

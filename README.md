# pii-redact

PII redaction for Australian data, and re-identification risk scoring for
workplace health and safety survey free text.

Deterministic rules run first and own the result. An optional NLP stage adds
recall only where the rules found nothing. For survey data, employer attribution
and quasi-identifiers are first-class, because a bullying response can identify
a person without naming one.

No required dependencies. `spacy` is an optional extra. `pyspark` is only
needed on the cluster, and `pii_redact.spark` imports without it.

## The pipeline

1. **Deterministic detectors** — regex and checksum. Fast, auditable, with
   measurable false-positive rates.
2. **Overlap resolution** — longest span wins, then deterministic over NLP, then
   confidence, then detector order. Independent of iteration order.
3. **NLP** — fills gaps only. An NLP span touching a claimed span is discarded
   whole: never merged, never truncated, never allowed to split a validated TFN.
4. **Rewrite right to left**, so earlier offsets stay valid.
5. **Residual disclosure control** (opt-in) — measures what stages 1–4 left
   behind and escalates on it. Off by default; see
   [Residual disclosure control](#residual-disclosure-control).

## Two different risks

Most PII tooling treats "who is this" as the only question. Workplace data has a
second axis, and it is often the bigger one.

**Identifying a person.** TFN, Medicare, name, phone, address. Masked.

**Identifying the employer.** A respondent who writes "the intranet ignored my
report" has named their workplace as surely as if they had written its name,
without using a single proper noun. A named employer is redacted; internal
vocabulary is matched from a lexicon and both redacted *and* scored.

Everything below the employer is a **quasi-identifier**: a job title, a pay
grade, 18 years of service, night shift, a team of three. Individually
harmless. Together, inside one organisation, they are a person.

That combination is why masking is the wrong default for survey data. See
[Generalisation](#generalisation).

## Quick start

Documents and log lines:

```python
from pii_redact import Redactor, Policy

Redactor(Policy()).redact_text("TFN 123 456 782")   # 'TFN *** *** ***'
```

Survey free text:

```python
from pii_redact import SurveyRedactor, survey_redactor

survey = SurveyRedactor(survey_redactor(salt=b"..."))

record = survey.redact_record("I am a Grade 4 nurse at Acme Pty Ltd with 18 years of service")
record.text        # generalised and tokenised
record.risk        # 'critical (0.81): quasi_combination, named_organisation, ...'
record.risk.band   # <RiskBand.CRITICAL>
```

Databricks, before the Delta write — see
[`examples/databricks_redaction.md`](examples/databricks_redaction.md).

## Australian identifiers

Every checksummed identifier is two-stage: a narrow regex finds candidates, then
the checksum confirms each one. Without that gate a TFN detector is just "nine
digits", which matches order numbers and timestamps.

| Category | Rule |
| --- | --- |
| `tfn` | 9 digits, ATO mod-11 weights `[1,4,3,7,5,8,6,9,10]` |
| `abn` | 11 digits, ABR mod-89 weights, each digit reduced by one |
| `acn` | 9 digits, ASIC weights `[8..1]`, check digit is the mod-10 complement |
| `medicare` | 10 digits, weighted check digit `1,3,7,9,1,3,7,9,1` |
| `medicare_enrolment` | Shape only (no checksum exists), confidence reduced |
| `bsb` | 6 digits, unallocated prefixes rejected, **label required** |
| `passport` | One letter then seven digits, `E`/`O`/`I` rejected |
| `drivers_licence` | Label required, scored against documented state formats |
| `credit_card` | Recognised IIN plus Luhn, including eftpos, Co- and UnionPay |

The ABN's two leading digits are the only pair that satisfies the mod-89
checksum for a given ACN, so that check is a genuine integrity test.

## Contact, network and location

| Category | Notes |
| --- | --- |
| `email` | Standard address pattern |
| `phone` | AU numbering plan: `04` mobiles, `02/03/07/08` landlines, `1300`/`1800`, `+61` |
| `ipv4`, `ipv6`, `mac` | Guards exclude `v1.2.3.4` and `1.2.3.4.5` |
| `postcode` | Real state allocations; a bare 4-digit number needs a postal label |
| `implication` | A word sequence that identifies a person or workplace (see below) |
| `street_address` | Street suffixes, unit/street pairs, PO boxes |
| `url_credential` | Credentials in URLs and `api_key = ...` pairs |
| `date_au`, `date_iso` | Day-first and ISO, with real calendar validation |
| `organisation_name` | Named employers, including AU entity suffixes |
| `internal_term`, `award` | Lexicon hits; award codes like `MA000019` |

**Phone numbers are shape-checked, not pattern-matched.** The candidate regex
is loose and the numbering plan does the work. `12.03.1985`, `1.2.3.4`,
`2024-03-12` and `1250.00` all produce candidates and all get rejected.

**Dates are day-first.** `03/02/2024` is 3 February. That is the Australian
convention and it differs from US data, so the parsed ISO value is reported in
finding metadata for anything downstream that sorts or compares dates.

## Free text and misspellings

Survey responses are messy, so matching runs on two normalisation levels.
Neither changes what you get back: every span is reported in original-string
offsets.

* `normalise` — case, Unicode, smart punctuation, whitespace. Used by the
  ordinary regex detectors.
* `normalise_aligned` — the same substitutions but strictly one character in,
  one out, so offsets stay valid. Used wherever a pattern depends on
  capitalisation (proper nouns) or the caller needs original offsets.

`FoldedText` goes further, keeping only alphanumerics and folding leetspeak, so
`Bui11ing`, `Bullying` and `b u i l l y n g` collapse together.

The lexicon uses a SymSpell-style delete index, so one edit in a long token is a
dict lookup rather than a scan:

```python
from pii_redact import Lexicon, Term, Category

lexicon = Lexicon.from_terms([
    Term("Project Kestrel", Category.INTERNAL_TERM,
         aliases=["kestrel"], label="internal project"),
])
lexicon.match("the Kestral rollout")   # -> canonical "Project Kestrel", fuzzy
```

Fuzzy matching is disabled below six characters, and the first alias registered
wins the key, so list the distinctive token of a multi-word term as an alias.

**How the fuzzy step actually works.** Each single-token form is indexed
together with every string one character away from it, and a query is
looked up by generating its own deletions and hitting the same table. Set
membership does the distance filtering, so there is no per-token distance
computation at all. A hit means the query and the term share a string
reachable by deleting at most one character from each, which in practice
catches a deletion, an insertion, a substitution and an adjacent
transposition:

| Written | Reads as | Operation |
| --- | --- | --- |
| `intranet` | `intranet` | exact |
| `intanet` | `intranet` | dropped a character |
| `intrranet` | `intranet` | added one |
| `intranxt` | `intranet` | substituted one |
| `inratnet` | `intranet` | swapped two |

Three limits are worth stating plainly. Only **single-token** forms are
indexed, so a two-word term can never fuzzy-match: `toolbox talk` is found
when written correctly and missed when written as `toolbos talk`, and the
fix is to register the distinctive token (`toolbox`) as its own alias.
Ties resolve to the shortest distance and then the longest term, so a more
specific term wins. And `exact_only` opts a term out of this mechanism
entirely.

`load_builtin_lexicon()` ships a generic seed: Australian WHS and employment
vocabulary, state compensation schemes, and national award references. A
lexicon you pass is merged *over* it, so adding your internal names never loses
the baseline.

### What the seed lexicon is for, and what it leaves out

A lexicon term earns a place only if naming it tells a colleague something they
could not otherwise guess. That test rules out most safety vocabulary, because
"risk assessment" and "near miss" describe every Australian workplace — an
employee survey is full of them, so including them only manufactures false
attribution. They belong to the implication rules, which key on patterns.

What survives the test is terminology that pins a *jurisdiction* or an
*employer class*, which is where the attribution signal actually is:

| Entry | What it narrows |
| --- | --- |
| `Occupational Health and Safety Act` | Victoria — the only state still titled OHS rather than WHS, so even the acronym is decisive |
| `Work Health and Safety Act 2020` | Western Australia |
| `Work Health and Safety (National Uniform) Act` | Northern Territory |
| `icare`, `SIRA`, `State Insurance Corporation` | NSW, and for public sector employers a specific claims arrangement |
| `Employees Help Desk` | Queensland's general insurer for small employers |
| `Gallagher Bassett`, `self-insurer` | Of a few dozen employers each |
| `Dust Diseases Scheme` | NSW, and a specific occupational disease cohort |
| `SafeWork NSW`, `WorkSafe Victoria`, `ReturnToWorkSA` | The regulator itself, hence the state |
| `MA000098`, `MA000090`, `MA000141` | Aged care, construction, security |

Two mechanisms keep the additions from costing precision. `exact_only` keeps a
term out of the delete index, so an everyday word matches only when spelled
correctly — `fatigue` yes, `fatique` no, because a misspelling of a common word
is just a typo in a sentence about anything. And no term carries a bare alias:
`award` and `appraisal` were matching every response in which the respondent
merely mentioned winning one or being appraised, which is most of them.

Safe Work Australia is labelled the national policy body, because that is what
it is — it sets model law and explicitly does not regulate or enforce.

## Generalisation

Masking a date removes the age signal the survey exists to measure. For most
quasi-identifiers the right answer is to widen the value:

```
"12/03/1985"         -> "1985"
"I am 45 years old"  -> "45-54"
"18 years of service"-> "10-19 years"
"team of three"      -> "2-5 people"
```

`survey_redactor` generalises ages, dates, tenure and team size by default. This
is orthogonal to `MaskStyle`: direct identifiers are masked while
quasi-identifiers are generalised in the same pass.

Health detail is **tokenised**, not masked. "I developed anxiety after the
restructure" is the finding; masking it destroys the analysis while barely
reducing risk. A token still lets responses be grouped by condition without
naming it.

## Risk scoring

The hard question is not "is this free of PII". It is "could someone who knows
the organisation work out who wrote this". That is a property of a combination,
so it cannot be answered span by span.

```python
from pii_redact import assess, Audience

risk = assess(text, findings, audience=Audience.INTERNAL)
risk.band      # low | moderate | high | critical
risk.signals   # named contributions, sorted by weight, each with a reason
```

Signals include the narrative itself (a bullying account is highly identifying
on its own), uniqueness claims, protected characteristics, specific dated
events, employer attribution, and quasi-identifier counts. Three innocuous
attributes produce an explicit `quasi_combination` bonus, because that
combination is the actual risk.

`Audience` matters more here than in most PII tooling. The same text is far more
identifying to a colleague than to an external analyst, and
`audience=Audience.INTERNAL` reflects that by weighting quasi-identifiers and
attribution highest.

Weights live in a plain dict, `risk.WEIGHTS`. They are starting points, not
measurements. Tune them from your own review outcomes. Scores are advisory: they
rank records for human review, they do not replace it.

## Mask styles

| Style | Output for `4111 1111 1111 1111` | Use when |
| --- | --- | --- |
| `MASK` | `**** **** **** ****` | Default. Preserves length and whitespace |
| `REMOVE` | `` | Collapses the span |
| `PARTIAL` | `************1111` | A human needs to reconcile a record |
| `HASH` | `[3f9a2c1e8b4d]` | Referential integrity, no disclosure |
| `LABEL` | `[CREDIT CARD]` | **Default for survey data** |
| `TOKEN` | `credit_card_3f9a2c1e` | Category-scoped pseudonym |

`LABEL` is the default in the survey profile. It carries no information about the
value, so two exports of the same data cannot be joined on it — which is the
point, because a stable identifier that persists across extracts is itself a
re-identification vector. Every name becomes `[NAME]`; an analyst can still see
that a name was present and count by category, which is usually the analysis
that matters.

`HASH` and `TOKEN` are keyed with HMAC-SHA256, so an attacker cannot confirm a
guessed value against the output, and identical inputs map to identical tokens so
grouping still works. Choose them when you need to join across extracts.

`PARTIAL` is deliberately *not* the default for job titles: keeping the last four
characters of "registered nurse" produces `***********urse`, which leaks nothing
useful and reads badly.

**The salt is a secret and must be stable.** Persist it in Databricks secret
scope alongside the redacted data. Losing it makes every historical token
unmatchable; leaking it makes tokens reversible for anyone who can guess a
candidate. Without an explicit salt the module generates an ephemeral one, so
tokens do not survive a process restart — which silently breaks any
cross-run grouping you attempt on the Spark path.

## Spark

`RedactorSpec` is a small frozen config; executors build and cache their own
redactor from it. Compiled regexes and the lexicon delete index never cross the
wire, and a task that retries pays the build cost once per JVM, not once per
task.

```python
from pii_redact.spark import RedactorSpec, redact_frame

spec = RedactorSpec.build(salt=salt, audience=Audience.INTERNAL)
clean = redact_frame(raw, ["Q7", "Q8", "Q9"], spec)
```

Each column gains `_redacted`, `_findings`, `_risk`, `_risk_band` and
`_needs_review`. Use the pandas UDF (`redact_frame` / `redact_dataframe`) rather
than the scalar `redact_column_udf`; Arrow serialisation is amortised across a
batch, and that is the difference between minutes and hours.

Offsets in `_findings` are against the decoded column value, not the raw NDJSON
line. Qualtrics escapes text aggressively, so raw-line offsets would not line up
with what a reviewer sees.

## Implicating sequences

Most redaction tooling looks for a *thing*: a number, a name, an address. In
workplace free text the identifying information is often a *sequence*, with no
single nameable span:

> "I am the only woman on the night shift."
> "One of two apprentices on the yard team."
> "After the Kwinana depot closed I was the only one who did the handover."

Masking the entities leaves all three fully identifying. `detectors/implication.py`
redacts the whole clause and labels it `[IMPLICATION]`, because the *fact* that a
uniqueness claim was made is itself what an analyst needs in order to weight
severity.

The rule set is data (`RULES`), so a deployment can add patterns for its own
workforce.

### Public sector and corporate

For Australian government agencies, associated organisations and large
corporations, the risk is rarely the employer name. It is that the respondent
describes *what kind of workplace* and *what kind of job*, and the population
matching that description inside the organisation is a handful of people.

> "I am one of three EL2s in the Migration directorate."
> "An eligibility specialist in the Services division."
> "There are two of us left in the integrity unit."
> "I hold a security clearance at the secret level."

None of those names a person or an employer. All four identify one person to
anyone inside the agency. `vocabulary.py` carries the shared lists and the rules
consume them:

| Rule family | What it catches |
| --- | --- |
| `department_structure`, `unit_by_preposition` | "the digital services branch", "in Services division" |
| `government_process` | "a machinery of government review", "a FOI request" |
| `occupation_family`, `occupation_with_structure` | "an eligibility specialist in the Services division" |
| `aps_level`, `seniority_in_unit` | "one of three EL2s in the Migration directorate" |
| `senior_role` | "the branch manager", "the secretary" |
| `cohort_marker` | "the 2021 intake", "the graduate cohort" |
| `clearance` | a security clearance and its level |
| `workplace_type` | "a residential aged care facility", "our depot" |
| `corporate_marker` | "our values and behaviours", "the graduate program" |

Two vocabulary decisions are worth knowing about, because both cost recall to
buy precision:

* **Strong versus qualified structure nouns.** `department`, `directorate` and
  `secretariat` identify on their own. `unit`, `office`, `centre` and `team` only
  do so behind a qualifier — "the integrity unit" is an identifier, "my team has
  always had my back" is not, and it appears in nearly every positive response.
* **Weak versus strong workplace types.** "our depot" needs no more than a
  possessive, but "the store" does not: in "declined at the store" it is a
  transaction. Strong nouns like "depot", "refinery" and "correctional centre"
  work with a bare article.

Similarly, "the annual report" was tried as a government marker and removed. It
appears in corpora and on the open internet, so it identifies no workplace, and
matching it deletes a phrase from otherwise ordinary text.

`WORKPLACE_TYPE` and `OCCUPATION_FAMILY` deliberately exclude abstract nouns --
"supervisor", "operator", "specialist", "officer". They match almost any
sentence and would flatten the signal the rules are meant to carry.

Two lessons are encoded in that module rather than left as comments to be
ignored:

* Every shared vocabulary is built through `alt()`, which wraps it in a group.
  Interpolating a bare `"a|b"` into a larger pattern applies the alternation to
  the *whole* expression, so the surrounding capture group stops being a group.
  That bug appeared three times in this package and each time it was silent.
  `tests/test_implication.py` asserts every constant is grouped, and asserts
  that no rule fires on a bare vocabulary word.
* A rule word that appears only inside a placeholder must not be re-detected, or
  the placeholder grows on every pass. See idempotence below.

## Generated corpus and measured recall

Hand-written fixtures are easy to overfit to. `pii_redact.generate` composes
realistic responses from templates and vocabularies, plants known secrets at
controlled rates, and records exactly what it planted — so recall is measurable
without reference to what a detector reported.

Its templates cover the same ground as the hand-written fixture: the service
half of the survey, psychosocial hazards that are not bullying, and three
registers (shouted, terse, plain). That is not decoration. Adding the service
and hazard templates re-seeded the corpus and exposed two detection defects that
the bullying-only corpus had hidden, one of them an APS officer level being
reported as a health disclosure.

```console
$ python examples/evaluate_corpus.py --count 400
$ python examples/evaluate_corpus.py --count 400 --show-misses --json
```

A secret counts as caught when its literal text is *absent* from the redacted
output, which is stricter than "a finding overlapped it": the standard is what
a reader of the output can still see.

Current figures over 500 generated records, with public-sector and corporate
vocabulary in the mix:

```
kind                    planted  caught   recall
acn                         111     111   100.0%
aps_level                    11      11   100.0%
aps_unit                     10      10   100.0%
bsb                          57      57   100.0%
cohort                        5       5   100.0%
credit_card                 104     104   100.0%
dob                          90      90   100.0%
email                       139     139   100.0%
email_spaced                 17      17   100.0%
government_act                3       3   100.0%
medicare                    130     130   100.0%
organisation                 39      39   100.0%
phone                       137     137   100.0%
phone_obfuscated             30      30   100.0%
public_role                   9       9   100.0%
seniority_in_unit             9       9   100.0%
suburb                       16      16   100.0%
tfn                         133     133   100.0%
third_party                  25      25   100.0%
workplace_type                2       2   100.0%
implication_person           46      43    93.5%
implication_workplace        31      26    83.9%
site                         27      21    77.8%
OVERALL                    1182    1168    98.8%
```

Split by how the value was written, which is the more informative cut:

| form | planted | recall |
| --- | --- | --- |
| clean | 1068 | **100.0%** |
| fullwidth | 40 | **100.0%** |
| leet | 11 | **100.0%** |
| lower | 9 | **100.0%** |
| spaced | 19 | 89.5% |
| misspelled | 35 | 65.7% |

Read that honestly: **every well-formed value is caught, and the two shortfalls
are a ceiling rather than a defect.** The sequence rules and the site rules are
exact patterns, so a misspelled idiom ("since the atke-over") or a
character-spaced phrase ("t h e   o n l y") cannot match them. Closing that gap
means fuzzy matching over multi-word idioms, which trades a large
false-positive surface for the margin — a bad trade when a false positive means
a support manager's name is deleted from a bullying complaint.

Entity-level values do better: the lexicon's SymSpell index absorbs one edit, so
"timsheet" and "Kestral" match.

## Sample survey data and the evaluation suite

`data/survey_responses.ndjson` is 58 Qualtrics-shaped records from a workplace
health cover provider: `Q1` asks how the service they received was, `Q2` what to
improve, `Q3` comments on psychosocial issues at work, `Q4` anything else and
`Q5` follow-up consent. Almost every record answers the service question as well
as the psychosocial one, because a survey that only measured the workplace would
not be the survey that was sent, and a redactor tested only against complaints
learns nothing about praise.

### Two instruments, one export

The export merges two surveys, and every record says which one it came from in an
`Instrument` column. The distinction is enforced, not documented:

| `Instrument` | What it is | What it collects by design |
| --- | --- | --- |
| `service-pulse` | The routine member survey, `Q1`–`Q5`. `Q3` invites a psychosocial comment, so hazard prose appears here. | Nothing financial |
| `reimbursement` | The claims, authorisation and payment form path | TFN, Medicare, date of birth, ABN, ACN, BSB, bank account |

That split is what makes `S012` defensible. **No psychosocial survey collects a
TFN**, so a TFN, a Medicare number or a date of birth cannot have arrived
through the hazard half of the survey; they came through the reimbursement half.
A single-instrument fixture cannot account for them at all, which is the most
available realism error in this corpus.

`TestInstrumentStratum` fails if a reimbursement-category finding lands on a
`service-pulse` record — and it fails on the *fixture*, not the detector, because
the detection itself would have passed. Claim references are deliberately outside
that invariant: a respondent pastes a claim number into a hazard box, and `S044`
does exactly that.

Consent is modelled the same way, because the obvious simplification is wrong in
a way that shows. `Consent` is a discrete item — `yes`, `no`, `yes-not-employer`,
`yes-anonymous` — and `Q5` holds only the free-text *conditions* on follow-up
contact. A single free-text consent box is empty 44 times out of 45, which is not
a consent item, and `yes but not my employer` grants something quite different
from silence.

### The hazard mix is weighted, and not toward the nastiest hazards

Safe Work Australia's *Managing Psychosocial Hazards at Work* Code of Practice
names 17 hazards. This corpus does not distribute across them evenly, because the
national data says not to. The [2026 Australian Worker Exposure Survey](https://data.safeworkaustralia.gov.au/)
reports the largest single disagreement on influence — 21.8% disagree they have a
say in changes that affect them — while 94.4% agree they have clarity of duties
and 84.0% agree they receive support to work safely.

So change management, role clarity, recognition and job insecurity are
represented rather than left as a bullying monoculture, and so are
**protectives**: `S046`, `S047`, `S048` and `S058` are a worker describing
conditions that are working. A psychosocial corpus built only from complaints
misrepresents the population it claims to sample, and worse, teaches a redactor
that hazard prose is the only prose worth being careful with.

Worth noting how the protective records are worded. `"I am clear on what my
responsibilities are"` is the literal validated item, and it does not appear
here: the health cue is `I am` plus a denylist, and it reads `clear` as a health
adjective and reports `I am [HEALTH DETAIL] on what my responsibilities are`.
The protectives are phrased as free text instead, which is what a respondent
actually types into a box.

The response set deliberately spans the shapes a real export contains:

| | |
| --- | --- |
| Register | Terse one-liners, all-caps rants, all-lower-case replies, second-language English, a Vietnamese answer, an abandoned response with no text at all |
| Psychosocial hazard | Bullying, harassment, workload and understaffing, fatigue and shift work, unconsulted restructure, roster changes, return to work, a colleague dismissing a report, change announced the day it happened, three competing priorities, absent recognition, casual hours cut each quarter, indirect exposure to traumatic material, night work in an empty building, tracked time and keystroke metrics, a workplace at forty degrees, a customer who put a fist through the counter |
| Protective | Role clarity, practical support, a team that covers for each other, short handed but coping with a supervisor who asks what you need |
| Service | Claim delays, pre-approval, gap payment, referral approval, EAP counselling, telehealth, exclusion decisions, refunds that have not arrived |
| Identifier | Phone, email, TFN, Medicare, DOB, cards, ABN, ACN, BSB, street and postal addresses, a named employer, a named treating clinic, an award code, an internal system |
| Hard negatives | Claim references, dollar amounts, version strings, a four-digit year, an 11-digit run one digit off a valid ABN |

Twenty-six of the 58 records declare no categories at all and are asserted to
produce no detections. Thirteen of those are the new psychosocial records, which
means the suite now proves that hazard and protective prose survives redaction
byte for byte — a claim the old corpus, being almost entirely complaints, could
not make.

The corpus is fixture data, so it uses documentation IP ranges and invented
organisations throughout.

`data/survey_manifest.json` holds the expectations. They are authored from the
specification of what the module ought to catch, **not** from observed output:

* `must_not_appear` — substrings that must be gone from the redacted text
* `must_appear` — substrings that must survive, proving no over-redaction
* `categories` — finding categories expected anywhere in the record

Observe it:

```console
$ python examples/run_survey_suite.py --show-clean
$ python examples/run_survey_suite.py --json
$ python examples/run_survey_suite.py --no-lexicon    # seed vocabulary only
```

It exits nonzero on any miss, so it doubles as a gate. Three failure classes are
reported separately:

* **MISS** — something the manifest says must be redacted, and is not
* **LEAK** — an over-redaction the manifest did not anticipate
* **BAND** — a risk band that differs from the recorded calibration

`risk_band` in the manifest is a **calibration placeholder** captured from a run,
not ground truth. Bands depend on `risk.WEIGHTS`, which are starting points
rather than measurements. Tune them against your own human review outcomes, then
re-derive the manifest. That is why band drift is reported separately from
detection failure: a band change is a prompt to re-derive, not a bug.

**A band is not a severity score.** `S029` (harassment with a confidentiality
request), `S032` (an all-caps service rant) and `S041` (an exclusion decision)
are severe and score low or moderate, because nothing in them identifies
anybody on its own. Route severity through a separate signal or the triage
queue inverts.

### Known gaps the fixture holds open

`tests/test_survey_suite.py::TestKnownGaps` pins four exposures that are real in
a provider survey and are **not** caught today. Each one fails the test when it
is fixed, which is the signal to update the record's note in the manifest:

* Member and claim numbers (`S027`). The provider's own references identify a
  person to anyone holding the provider's export. `residual` is the layer meant
  to catch them and is off unless a population figure is supplied.
* A third party named without a label (`S043`). "My caseworker was `<Name>`" is
  neither a possessive nor an honorific, so the role-plus-name rule misses it.
* A bare surname (`S045`) — the documented free-text name gap.
* Bare suburbs (`S020`). "Surry Hills" survives; "the Box Hill site" does not,
  because it is described as a workplace site. The fix is a site list through the
  lexicon, never a regex.

The same fixtures run in `tests/test_survey_suite.py`, alongside idempotence and
offset-integrity checks. The suite found several real defects during
development, including non-idempotent pseudonyms that grew a new hash on every
pass, and three regex precedence bugs from interpolating an ungrouped
alternation into a larger pattern.

### Writing realistic mock data is what finds these bugs

The expansion of this fixture from 25 records to 45 surfaced nine defects that the
smaller, tidier corpus had hidden. Every one of them was a *false* detection:
ordinary survey prose being redacted, or a value reported under the wrong
category. None of them would have shown up in a corpus of tidy sentences.

| Text | What happened |
| --- | --- |
| "Your portal says the upload failed" | Matched on `our` one character in, replacing the middle of the word: `Y[IMPLICATION]` |
| "Nobody in the claims team has not answered" | The provider's own service function labelled as the respondent's employer |
| "what my manager sent the team" | A verb swallowed into the unit name, redacting a whole clause |
| "my manager keeps rostering me" | Same, via the internal-system rule |
| "I have family at home", "I have rung the service centre" | Household nouns and irregular verbs reported as health disclosures |
| "DO NOT BUY WORKPLACE COVER FROM THESE PEOPLE" | Four capitalised words of a shouted sentence taken for an employer |
| "I am the youngest on the team" | Clause stopped at "on the", leaving a stray noun after the label |

Re-seeding the synthetic corpus after adding its service and hazard templates
found two more: "I am APS 6" was claimed by the health cue and labelled
`[HEALTH DETAIL]`, and an officer level inside a comma list ("a case officer,
APS 6, in the complaints branch") matched no rule at all, because the level rule
required whitespace after the number.

Extending the fixture from 45 to 58 records against the Code of Practice hazard
list found three more, all on the protective side and all left open rather than
dodged in the fixture text:

| Text | What happened |
| --- | --- |
| "I am clear on what my responsibilities are" | The validated People at Work role-clarity item, reported as `I am [HEALTH DETAIL] on what my responsibilities are` |
| "I have support to work safely" | The validated support item, same route: `I have [HEALTH DETAIL] to work safely` |
| "I am the only one on site after 10pm" | `I am [IMPLICATION] after 10pm`, with the uniqueness clause cut at "the only one" and the trailing preposition orphaned into the sentence |

The first two are the same defect the README already documents — the health cue
is `I am`/`I have` plus a denylist — landing on the two most common protective
items in the national data. Adding `clear` and `support` to the denylist is the
obvious fix and is not applied here: the denylist is a blunt instrument, and the
evidence that it needs replacing rather than extending is that it is currently
catching validated survey wording. That is a detector change with its own
regression surface, so it belongs in its own pass with its own justification,
not smuggled in behind a data fixture.

The lesson worth keeping: a mock corpus is only as good as the register it is
written in. Tidy prose hides false positives, because tidy prose is not what
arrives in the export.

## Residual disclosure control

Rules detect. They cannot prove absence. `pii_redact.residual` measures what
the first pass *left behind* and escalates on that, which is the only defence
against a format nobody wrote a rule for:

```python
from pii_redact import SurveyRedactor, survey_redactor
from pii_redact.residual import DisclosurePolicy

# Off unless you supply a population figure, because the arithmetic needs one.
survey = SurveyRedactor(
    survey_redactor(salt=b"..."),
    residual_policy=DisclosurePolicy(population=4000),
)

record = survey.redact_record("My employee number is 0041827.")
record.disclosure.action   # DisclosureAction.GENERALISE
record.text                # 'My employee number is [NUMERIC REDACTED].'
```

It runs on the *output*, and it inventories six bounded classes of what
survived — person, workplace, temporal, numeric, organisation, narrative —
rather than trying to recognise unbounded ways of implicating a workplace. Each
attribute carries a 1-in-N selectivity, and:

```
expected_matches = population / (selectivity_1 * selectivity_2 * ...)
```

Below `min_cohort`, the record is identifying even though no rule fired. The
three actions are `PASS`, `GENERALISE` (replace the residual literals, keep the
narrative) and `SUPPRESS` (withhold, for risk that cannot be redacted away).

What it catches that the detectors do not:

| Surviving value | Why it is measurable |
| --- | --- |
| `0041827` | A digit run nothing claimed |
| `123 456 783` | Grouped digits — a TFN rejected by its checksum |
| `GW-2026-4471` | A case or project reference |
| `Meridien platform` | A capitalised word plus a system noun |
| `singled me out` | Behaviour, not the vocabulary a detector would key on |

Two properties are tested rather than assumed. The module never reads its own
output as disclosure — `[HEALTH DETAIL]` is two capitalised words and would
otherwise register as an organisation name, which is the same mistake as
re-detecting a pseudonym. And the same text escalates in a 30-person depot and
passes in a 4000-person agency, because that is the actual question.

### The numbers are estimates, and they need your data

The selectivity table is a starting point, not a result. Two of its entries are
known to be weak:

- **`SELECTIVITY[NARRATIVE]` is the least reliable number here.** One regex
  family spans "anxiety" and "singled me out by a named incident", whose real
  prevalence differs by orders of magnitude. The default of 200 reflects a
  general workplace survey; `narrative_selectivity` lowers it for a narrowly
  scoped one. Watch the response rate too: a grievance carried by one in twenty
  *respondents* is carried by roughly one in seven of the *workforce* when a
  third of it replied.
- **`attribute_independence` corrects an assumption with no defensible sign.**
  It defaults to 1.0, meaning the raw independence estimate. Age, tenure and
  grade correlate positively, which would make the joint group smaller than the
  product, but grievance narratives cluster with everything else about a person,
  which makes it larger. Raising it loosens the test.

So calibrate against observed review outcomes before trusting the escalation
rate. `summarise()` gives you the per-ingest counts to do that with, and every
decision carries the pre-remediation figure (`cohort`) alongside the result
(`post_cohort`), because a review queue needs to know how close a record came to
identifying — which is invisible once it has been cleaned.

## Known limits

Stated plainly, because a redaction module that overstates its coverage is worse
than none.

- **No checksum on several categories.** BSBs, driver licences, Medicare
  enrolment references, postcodes and dates can only be shape-matched. That is
  why BSBs and driver licences require a nearby label.
- **Free-text names need the NLP stage.** "Jane Citizen" with no label or
  honorific is not detected by default. The dependency-free gazetteer catches
  `Patient: Jane Citizen` and `Dr Fiona Whitlam`; a spaCy model catches more.
- **The seed lexicon has one known false positive.** Bare `briefing` matches
  "a briefing about the new website". It is kept because dropping it also drops
  typo tolerance for `breifing`, which is a tested capability, and the
  alternative is losing that to soften one ordinary-English phrase. Pinned by a
  test so the trade-off stays visible.
- **Employer inference is heuristic without your lexicon.** The bundled seed
  catches internal systems, awards and WHS processes. It cannot know your
  project names, and nothing else can. A lexicon is the only reliable way.
- **Suburbs inside a comma-separated address survive.** `45 Collins Street` is
  redacted and `Melbourne VIC 3000` is redacted, but the line is not treated as
  one span.
- **Dotted quads are genuinely ambiguous.** `1.2.3.4` is a valid IP and also a
  plausible version. The guards remove the common non-address cases, not the
  ambiguity.
- **Redaction is lossy by design.** `HASH` and `TOKEN` preserve linkage, not
  reversibility. Nothing here decrypts.
- **Suburb names are not covered.** `45 Collins Street` and `VIC 3000` are
  redacted; "Surry Hills" survives. There is no gazetteer, and adding one is a
  data-maintenance burden. The right fix is a site list through the lexicon,
  which also gets you your own depots and buildings.
- **Misspellings beyond one edit are not caught, and phrases never fuzzy-match.**
  The delete index catches one insertion, deletion, substitution or adjacent
  transposition, so "timsheet" and "Kestral" match. "toolbos" would match
  "toolbox" on that basis, but it does not, because "toolbox" is only ever
  registered inside the phrase "toolbox talk" and phrases are not indexed for
  fuzzy matching at all. Register the distinctive token as its own alias.
- **Sequence rules are exact patterns.** An implication phrase with a typo in it
  ("since the atke-over") will survive. This is a deliberate ceiling: see the
  measured recall table above.
- **Residual control measures, it does not prove safety.** `residual.py` shows
  that a record is *probably* safe by the arithmetic above. It cannot tell you a
  record is safe, and a record it releases has still not been through a human
  being who knows the workplace. Its thresholds are stated estimates pending
  calibration against your review outcomes.
- **Redaction must be idempotent, and it is tested to be.** A second pass over
  redacted text changes nothing. Getting there required two real fixes — a
  pseudonym that re-matched its own category term and grew a new hash every
  pass, and a generalised band like `25-34` being re-read as the age 25. Both
  are covered by tests over generated data, because neither is visible in a
  hand-written fixture.
- **The health cue is "I have" or "I am", and the guard behind it is a denylist.**
  "I have family at home" and "I have rung the service centre" were both
  reported as health disclosures until they were added to the list. The list
  cannot be complete: an unexpected noun after the cue is still reported. That
  direction is deliberate, because the cost of a false positive is a noisy
  finding a reviewer discards, and the cost of a false negative is a health
  disclosure that no longer appears anywhere in the redacted record.
- **"No complaints" scores as a complaint narrative.** The narrative signal keys
  on the word, not the polarity, so the most common way a positive answer ends
  raises the record's risk. In the sample corpus `S042` avoids the phrasing and
  the manifest says why.
- **A month-day date is reported and then left alone.** "14 August" is detected as
  `date_au`, but with no year to narrow to the generaliser passes it through
  unchanged. That is the deliberate choice over dropping a day from the only
  timeline the response gives.
- **A provider's own identifiers are not covered.** Member numbers, claim
  references and policy numbers are the most identifying values in an insurance
  survey and none of them is in the rules. See
  `TestKnownGaps` for the four fixtures that hold this open.

## Development

```console
$ pip install -e '.[dev]'
$ pytest
$ ruff check .
```

## Licence

MIT
# Sensitive content that passes a normal scan

Research against Comcare and associated Australian WHS sites, for the
`pii_redact` detection model. Retrieved 2026-10-17.

The goal is narrow and stated plainly: find the words, concepts, abbreviations
and sentence structures whose sensitivity is **compositional**. A normal scan
(a regex for a number, a name, a diagnosis, a slur) misses them because no
single span is sensitive. The combination is.

> "The employee tells their general practitioner that they feel bullied by
> their supervisor. They also report panicking before the catch-ups and anxious
> distress. The employer conducts an investigation... the actions performed by
> the supervisor constituted reasonable administrative action (informal
> performance management)."
> — Comcare Claims Manual, liability example 4

Every element alone is ordinary workplace English. Together it is a health
disclosure, an allegation against an identifiable person, a performance
management action and the exact fact pattern that decides a compensation claim.
[A normal scan finds none of it.](#2-the-five-mechanisms)

## Contents

1. [Corpus and method](#1-corpus-and-method)
2. [The five mechanisms](#2-the-five-mechanisms)
3. [Vocabulary by sensitivity class](#3-vocabulary-by-sensitivity-class)
4. [Abbreviations and acronyms](#4-abbreviations-and-acronyms)
5. [Sentence structures](#5-sentence-structures)
6. [Context rules](#6-context-rules)
7. [Covert naming and evasion](#7-covert-naming-and-evasion)
8. [Precision guards](#8-precision-guards)
9. [Handling map](#9-handling-map)
10. [Integration into pii_redact](#10-integration-into-pii_redact)
11. [Provenance](#11-provenance)

Machine-readable companions, consumed by `context_scan.py`:

| File | Contents |
| --- | --- |
| `sensitive_context_lexicon.json` | Terms by class, aliases, handling action, combinator word lists |
| `sensitive_context_rules.json` | The context rules: triggers, proximity, action, rationale, regex |
| `abbreviations.json` | Acronyms with expansions, class, corpus count, confidence |
| `corpus_stats.json` | Document/character/sentence counts and per-term counts |
| `context_scan.py` | Dependency-free scanner over the rules, for measurement |

## 1. Corpus and method

124 documents, 2.82M characters, 15,554 sentences, fetched in one pass from the
sites below (counts in `corpus_stats.json`; URLs in [Provenance](#11-provenance)).

| Source group | What it contributes |
| --- | --- |
| Comcare scheme guidance (SRC Act, WHS Act, rehabilitation, ART) | Claim, liability, review and return-to-work vocabulary; the RAA fact patterns |
| Comcare Claims Manual (four parts, incl. glossary) | The controlled vocabulary: liability, permanent impairment, sensitive claims, NWE |
| Comcare psychosocial hazard pages (17 topics) | Hazard definitions and effects language |
| Comcare incident notification guide and notifications pages | Notifiable incident, serious injury or illness, dangerous incident |
| Safe Work Australia (model Code, glossary, topic pages) | The 12-hazard core list, worker-language examples, WHS abbreviations |
| State and territory regulators (NSW, VIC, QLD, WA, SA, TAS, ACT, NT) | Variant vocabulary ("harmful behaviours", "psychological health", QLD's 14 hazards) |
| OAIC | The statutory meaning of sensitive information and health information |
| WHS Act 2011 (Cth) text | s35–38 notification, s36 definition, duties, enforcement |

**What this corpus is not.** It is regulator and scheme prose, not survey
responses. Frequencies below measure what the sites say, not how workers write.
They are used here for two things only: to prove a term is in current
Australian usage, and to select which terms deserve a rule. The held-out
probes in `data/heldout_probes.ndjson` remain the correct place to measure
recall against worker register, and this research should feed that file rather
than substitute for it.

## 2. The five mechanisms

### 2.1 Lexical ambiguity — the same string, two sensitivities

Most missed sensitive content is not rare vocabulary. It is common vocabulary
in its personal sense. A scan keyed on the term fires on the topical sense or
not at all, and either way misses.

| String | Topical / benign use (in the corpus) | Personal / sensitive use |
| --- | --- | --- |
| `claim` (2,609 hits) | "making a claim", "claims team" | "my claim", "the claim was denied" |
| `capacity`/`incapacity` (804) | "functional capacity", "team capacity" | "I have no capacity to work", "incapacity payments" |
| `condition` | "conditions of employment" | "my condition", "a pre-existing condition" |
| `performance` | "scheme performance", "high performing" | "performance management", "my performance review" |
| `notice` | "public notice", "notification" | "notice of injury" |
| `determination` (337) | "determination of liability" | "the determination about my case" |
| `adjustment` | "adjustment percentage" | "adjustment disorder", "reasonable adjustments" |
| `action` | "corrective action" | "reasonable administrative action", "adverse action" |
| `support` | "poor support" (a hazard) | "I have support", "support worker" |
| `incident` | "incident notification" | "the incident happened to me on..." |

A detector that keys on the bare term burns precision on the topical sense,
which is why `implication.py` and `quasi.py` already maintain denylists
(`clear`, `support`, `torn`, `confused`). The same trap applies here, at scale.
The rule files below therefore **never trigger on a bare term**: every rule
requires a co-trigger (possessive, cause, frequency, person, or claim frame).

### 2.2 Composition — the quasi-identifier set, extended for WHS

The existing `IMPLICATION` rules cover uniqueness and unit membership. The WHS
domain adds combinators with unusually high selectivity that a generic
quasi-identifier set does not know. Two or more from this list, in one answer,
with a hazard narrative, is near-certain identification inside the employer:

- shift pattern: `rotating roster`, `night shift`, `FIFO`, `4 on 4 off`, `backfill`
- work group: `DWG`, `work group`, `HSR for`, `our team of N`
- level/classification: `APS 6`, `EL1`, `SES Band 1`, `Grade 4`, `Band 3`
- cohort: `graduate intake`, `2021 intake`, `the restructure cohort`
- workplace site: `our depot`, `the [Name] office`, `the [suburb] campus`
- security: `NV1`, `NV2`, `PV`, `with clearance`, "sensitive claim" (NV1 claims)
- role + unit: "an eligibility specialist in the Services division"
- exposure history: `first responder`, `contact centre`, `case manager`,
  `custodial officer`, `emergency services`, `frontline`
- time anchors: "since March", "after the restructure", "for 18 months"

### 2.3 Relational uniqueness — the sentence identifies, not the span

"One of two physios on the ward who does the Tuesday handover" has no PII. The
sequence is the identifier, exactly the case `implication.py` was built for.
WHS-specific frames worth adding, all attested:

- "the only physio / the only female supervisor / the only APS 6 in the unit"
- "I was the last one left after the restructure"
- "there are two of us doing that caseload"
- "everyone on that shift knows"
- "the only one trained in [ticketed skill]" (base already covers
  `unique_training`; add first-responder and clinical qualifications)

### 2.4 Causal claims — hazard + outcome + employment link

s5B of the SRC Act requires employment to have contributed to a disease **to a
significant degree** ("substantially more than material", *Comcare v Power*).
A worker sentence that attributes a condition to work is therefore both health
information and evidence in a legal test. Corpus-attested sentence frames:

- "It has affected my mental health because of the workload."
- "The stress from the restructure caused my anxiety."
- "My depression got worse after the performance meeting."
- "I developed PTSD after dealing with the fatality."
- "Work contributed to / aggravated / flared up my condition."

The trigger is not `anxiety` alone (topic pages and Code text are full of it).
It is `condition` + (`causal marker` or `employment link`) + first person.

### 2.5 Meta-sensitivity — harmful regardless of identity

Some content is sensitive even with every identifier removed. It must not be
re-shown to a wide audience, logged verbatim, or included in free-text exports:

- suicidal ideation, self-harm, "intentionally self-inflicted" content
- sexual assault and harassment detail; abuse quotes
- family and domestic violence (FDV) disclosures, including immediate danger
- trauma material: fatalities, deceased persons, child abuse material,
  "vicarious exposure", "cumulative trauma"
- slurs, threats and profanity quoted from a named person
- serious and wilful misconduct allegations, drug/alcohol allegations

The last group is double-edged: the same text that evidences misconduct
exposure can destroy a worker's reputation if leaked. Sensitivity here is about
**restricted handling**, not just masking.

## 3. Vocabulary by sensitivity class

Counts are corpus occurrences (case-insensitive, all documents), as a currency
of attestation — not prevalence. Handling actions are defined in
[§9](#9-handling-map).

### 3.1 Harm behaviour allegations (`harm_behaviour`)

The most likely class to be missed, because the trigger words are also policy
titles. Attested definition (Comcare bullying page): "repeated, unreasonable
behaviour directed towards a worker or group of workers, that creates a risk
to health and safety."

| Term / phrase | Count | Note |
| --- | --- | --- |
| bullying, bullied | 207 | "repeated", "directed at", frequency markers co-occur |
| harassment | 254 | need not be repeated; "based on personal characteristics" |
| sexual harassment | 97 | "unwelcome", "suggestive jokes", "repeated invitations" |
| violence and aggression | 68 | "abused, threatened or assaulted", "service-related violence" |
| conflict / interpersonal conflict | 132 | "poor workplace relationships" |
| exclusion, ostracism, "excluded from the group chat" | 67 | covert, easy to miss |
| gossip, rumours | 8 | covert |
| intimidation, belittling, humiliation, unjustified criticism | 29 | Comcare examples list |
| discrimination | 95 | "based on a protected attribute" |
| victimisation / retaliation / reprisal | 8 | HSR context: s58–61 WHS Act |
| occupational / workplace violence | 18 | external vs internal; witnessing counts |
| misconduct (alleged) | 24 | pairs with investigation/Code of Conduct |
| abuse (verbal/physical) | in `maltreatment` family | "abusive and demeaning behaviour" (model Code) |

Co-triggers that turn a policy mention into a disclosure: `my manager`,
`my supervisor`, `he/she/they + verb`, `towards me`, `in front of`, `every
shift/week`, `for months`, `since March`, `repeatedly`, `again and again`.

### 3.2 Psychological health and injury (`psych_health`)

The model Code names the conditions: "anxiety, depression, post-traumatic
stress disorder (PTSD) and sleep disorders". Comcare adds the employment
concepts.

| Term / phrase | Count | Note |
| --- | --- | --- |
| psychological injury / injuries | 99 | the claim-class term |
| mental health (condition/injury) | 248 | also the stigma guard: "mental health" alone is topic talk |
| psychological health and safety | 152 | duty framing |
| psychosocial hazard(s) / risk(s) | 825 / 201 | **do not flag alone** — every page says it |
| anxiety / anxious | 34 | symptom and diagnosis |
| depression / depressive symptoms | 36 | |
| PTSD / post-traumatic stress disorder | 47 | first-responder fast track |
| adjustment disorder | 7 | the most common primary psychological claim, per the Claims Manual examples |
| burnout | 22 | model Code worker language: "burnt-out" |
| sleep disorders, insomnia, "unable to sleep" | 4 + "disrupted sleep" | model Code |
| stress response, psychological distress | in Code | "stress itself is not an injury" (SWA) |
| vicarious trauma / exposure | 8 | indirect exposure, cumulative trauma |
| pre-existing / underlying condition | many | health info of high claim relevance |
| flare-up, aggravation, acceleration, recurrence | glossary | s5A/5B terms; each is a health statement |

Treatment and capacity vocabulary: `diagnosed with`, `treating GP/psychiatrist/
psychologist`, `medication`, `antidepressant`, `medical certificate`,
`certificate of capacity`, `work restrictions`, `unfit for work`, `fit for
duty`, `functional capacity`, `capacity to work`, `time off work`, `stress
leave`, `Workcover/Comcare leave` (spelled variously), `EAP counselling`,
`clinical review`, `independent medical examination (IME)`,
`Fitness for Continued Duty Report (FCDR)`.

### 3.3 Compensation claim status (`claim_status`)

From the Claims Manual glossary; 2,609 `claim` hits overall.

| Term / phrase | Count | Where it matters |
| --- | --- | --- |
| claim, claimant | 2,609 | "my claim", "the claim was rejected" |
| liability, "liability denied/accepted" | 448 | the decision, and its reasons |
| determination | 337 | "the determination about my case" |
| reconsideration | 244 | internal review: "asked for a reconsideration" |
| ART / Administrative Review Tribunal | 224 | external review; ART replaced the AAT |
| AAT | 17 | historical matters |
| permanent impairment (PI) / WPI | 272 / 33 | PI is both a condition and a payment category |
| NWE / normal weekly earnings | 361 | income replacement figure — financially sensitive |
| NWH / normal weekly hours | 24 | |
| incapacity payments | in `capacity` count | amounts and dates |
| rehabilitation program, s36 referral | | statutory processes |
| overpayment, debt recovery | manual | financial + legal |
| sensitive claims (NV1 claims) | manual | claims requiring NV1-cleared case managers |
| reviewable decision | glossary | s62; rights and deadlines |

The most identifying single value in this class is the **claim or case
reference**, already listed in the repo's `TestKnownGaps` (`CLM-2026-8812`,
`S-027`). Nothing in the corpus changes that finding; the manual confirms the
reference exists in the client-facing correspondence.

### 3.4 Administrative action (`admin_action`)

The intersection class. s5A(2) SRC Act lists "a reasonable appraisal of the
employee's performance", "a reasonable counselling action (whether formal or
informal)", "a reasonable suspension action", and "a reasonable disciplinary
action (whether formal or informal)" — plus anything done in connection with
them. When one of these appears within a paragraph of a psychological
condition, the text is the RAA fact pattern. Attested vocabulary:

`performance appraisal`, `performance review`, `performance management`,
`informal performance management`, `informal performance improvement targets`,
`counselling (formal or informal)`, `PIP`, `underperformance`, `show cause`,
`Code of Conduct investigation`, `misconduct`, `disciplinary action`,
`formal warning`, `suspension`, `termination`, `redundancy`, `redeployment`,
`rehabilitation and performance`, `management of unsatisfactory performance`.

Counts are low in the corpus (16, 24, 11) precisely because these are
consequential, not topical. That is the reason a general scan misses them and
the reason the combination rule earns its weight.

### 3.5 Rehabilitation and return to work (`rehab_rtw`)

| Term / phrase | Count | Note |
| --- | --- | --- |
| rehabilitation | 1,054 | includes `rehabilitation authority`, s36/37/38 SRC Act |
| return to work / RTW | 423 | |
| suitable duties / suitable employment | 136 | often the sensitive fact: what the worker can and cannot do |
| reasonable adjustment(s) | 48 | Disability Discrimination Act overlap |
| graduated return to work / GRTW | 41 | reduced hours and duties |
| work trial, work hardening, redeployment | glossary | |
| workplace rehabilitation provider (WRP), RCM | 34 IME + | named provider + named worker |
| case conference | page | includes treating practitioner, employer, insurer |
| confidentiality of information about the employee | manual | the scheme's own handling rule |

The distinguish-the-person signal is `restrictions` + a named worker or a
first person: "the employee must avoid stressful tasks until reviewed".

### 3.6 WHS duty and enforcement (`whs_enforcement`)

Important for two reasons: it marks potential liability exposure in free text
("I reported it to the regulator"), and it names HSRs and witnesses whose
identity s58–61 protects from victimisation.

| Term | Count | Note |
| --- | --- | --- |
| PCBU, duty holder | 374 | |
| HSR / health and safety representative | 294 / 45 | s58: prohibited discriminatory conduct |
| DWG / designated work group | sitemap + model law | small-group identifier |
| notifiable incident | 75 | s35 |
| serious injury or illness | 43 | s36 |
| dangerous incident | 25 | s35(c) |
| inspector, entry powers | 130 | |
| improvement notice / prohibition notice | 73 / 67 | enforcement instruments |
| WHS undertaking / enforceable undertaking | 76 | includes admissions |
| prosecution, Category 1 offence, gross negligence | 83 / 13 | enforcement exposure |
| civil penalty | 4 | |
| dispute resolution, PIN | WHS Act Div 5 | HSR dispute pathway |
| right of entry / WHS entry permit | WHS Act Part 7 | union officials and advocates |

### 3.7 Trauma content (`trauma_content`)

Model Code: "Witnessing, investigating or being directly exposed to traumatic
events or material is a workplace psychosocial hazard... includes vicarious
exposure... and cumulative trauma."

`traumatic events or material` (43), `vicarious` (8), `fatality/fatal`,
`deceased`, `suicide` (26 incl. self-harm), `self-harm`, `abuse`,
`deceased persons`, `emergency services`, `crime scene`, `mortuary`,
`body`, `autopsy`, `child safety material`, `disturbing images/videos`,
`graphic material`, `incident files`, `case files`, `interviews`.

Handling: tokenise role and event class, never echo the visual detail, and
route to the highest review tier. This is the class where a false negative is
worst and a false positive is cheapest.

### 3.8 Protected attributes (`protected_attribute`)

Already partly in `risk.py::_PROTECTED_RE`. The corpus adds:

`workers from diverse backgrounds` (Code), `CALD` (5), `workers with
disability`, `age, gender, sexuality` (Code's own list), `parental leave`,
`breastfeeding`, `carer responsibilities`, `visa status`, `migrant workers`,
`Aboriginal and Torres Strait Islander`, `LGBTIQA+`, `family and domestic
violence` (83), `pregnant worker`, `neurodivergent`.

Combination rule: attribute + harm behaviour + person marker is a
discrimination-disclosure signal. Attested example structure from the
psychosocial screener items: "I get comments and jokes about my age every
single week. It started when I came back from parental leave."

### 3.9 Third parties (`third_party`)

`my manager/supervisor/team leader/colleague/coworker/HR`, `my caseworker`,
`my case manager`, `the RCM`, `my treating GP/psychologist/psychiatrist`,
`the WRP`, `the union delegate`, `my HSR`, `the person above me`,
`the secretary/branch manager` (public sector), `the claims manager`,
`the investigator`, witnesses, `the person named in the complaint`.

Sensitivity is not the role; it is the role plus attributed behaviour or
health information: "my caseworker was [Name]", "my supervisor told me to
stop taking stress leave", "the case manager said my claim was closed".

### 3.10 Legal and advocacy (`legal_advocacy`)

`lawyer`, `legal advice`, `union`, `Fair Work Commission`, `Fair Work
Ombudsman`, `general protections`, `adverse action`, `Work Health and Safety
entry permit`, `freedom of information request`, `legal professional
privilege (LPP)`, `model litigant`, `settlement/Calderbank offer`,
`delegate`, `instructing officer`, `proceedings`, `hearing`.

Detection note: legal-content words mark a response for restricted handling
because legal privilege and litigation risk attach to them. The Claims Manual
states LPP explicitly ("Legal Professional Privilege ... confidential
communications") and lists the ART implementation machinery; both are flags,
not identifiers.

## 4. Abbreviations and acronyms

Full table with expansions and counts in `abbreviations.json`. Grouped here by
class; **N** counts corpus occurrences. Terms marked ✱ were confirmed in the
fetched corpus; terms marked † are field-standard but were not attested in the
fetched pages — verify locally before enabling.

**Scheme and claim (Comcare/SRC Act)**

| Abbr. | Expansion | N |
| --- | --- | --- |
| SRC Act | Safety, Rehabilitation and Compensation Act 1988 ✱ | 180 |
| WHS Act | Work Health and Safety Act 2011 ✱ | 409 |
| PI | permanent impairment ✱ | — |
| WPI | whole person impairment ✱ | 33 |
| NWE | normal weekly earnings ✱ | 361 |
| NWH | normal weekly hours ✱ | 24 |
| IME | independent medical examiner ✱ | 34 |
| LQMP | legally qualified medical practitioner ✱ | 10 |
| RCM | rehabilitation case manager ✱ | 28 |
| GRTW | graduated return to work ✱ | 41 |
| RTWP | return to work plan ✱ | — |
| FCDR / FFD | Fitness for Continued Duty Report / fitness for duty ✱ | — |
| LPP | legal professional privilege ✱ | — |
| ART | Administrative Review Tribunal ✱ | 224 |
| AAT | Administrative Appeals Tribunal (historical) ✱ | 17 |
| PICS | Parliamentary Injury Compensation Scheme ✱ | 9 |
| SRCOLA | Safety, Rehabilitation and Compensation and Other Legislation Amendment Act 2007 ✱ | — |
| AD(JR) | Administrative Decisions (Judicial Review) Act 1977 ✱ | — |
| CAIS / Pracsys / MART / Content Manager | Comcare systems and business areas ✱ | — |
| NV1 / NV2 / PV | security clearance levels (NV1 names a "sensitive claim") ✱ | 6 |
| Centrelink CRN † | Customer Reference Number | — |
| DVA, ADF, MRCA † | veterans' scheme terms | few |

**WHS and enforcement**

| Abbr. | Expansion | N |
| --- | --- | --- |
| PCBU | person conducting a business or undertaking ✱ | 374 |
| HSR | health and safety representative ✱ | 294 |
| HSC | health and safety committee ✱ | — |
| DWG | designated work group ✱ | — |
| PIN | provisional improvement notice ✱ | — |
| SWMS | safe work method statement ✱ | 1 |
| JSA | job safety analysis † | — |
| PPE | personal protective equipment ✱ | 28 |
| RCS | respirable crystalline silica ✱ | 2 |
| MHF | major hazard facility ✱ | — |
| LTIFR / LTI | lost time injury frequency rate / lost time injury ✱ | 1 |
| MoG | machinery of government (restructure) ✱ | 3 |
| FDV / DFV | family and domestic violence ✱ | 83 |
| AOD | alcohol and other drugs ✱ (expansion) | 2 |
| CALD | culturally and linguistically diverse ✱ | 5 |
| MARAM † | multi-agency risk assessment and management (FDV) | — |
| WFH | working from home ✱ | 5 |
| WHO, SIRA, OAIC, AHPRA, CDPP | external bodies ✱ | mixed |
| APS, EL1/EL2, SES | public service classifications ✱ | 22 |
| NDS | National Dataset for Compensation-based Statistics ✱ | — |
| PAW | People at Work ✱ (now decommissioned in QLD) | — |
| COPSOQ | Copenhagen Psychosocial Questionnaire ✱ | 13 |

Two lessons for the model. First, `ART` and `AAT` are different institutions;
a rule that treats them as one mis-dates the matter. Second, `PI` is two
concepts (permanent impairment; also "personal injury" in casual usage); keep
the label nearby or it is unactionable. The abbreviation file records both
expansions where ambiguous.

## 5. Sentence structures

### 5.1 The five-part sensitivity sentence

The most common composition in scope is not a sentence but a **shape**. Each
part passes a normal scan on its own; the answer becomes sensitive when three
or more co-occur in one response:

```
[self or role]  →  [hazard or behaviour]  →  [time/frequency]  →
[impact or condition]  →  [action or consequence]
```

Attested instances:

| Source | Text | Parts present |
| --- | --- | --- |
| Claims Manual ex. 4 | "The employee tells their general practitioner that they feel bullied by their supervisor... report panicking... anxious distress" | role, behaviour, clinician, condition |
| Held-out probe H003 | "I felt humiliated and undermined for months and nobody in HR would take it seriously. It was discrimination dressed up as feedback." | behaviour, duration, role, characterisation |
| WorkSafe ACT page | "They might say they feel: stressed, burnt-out... anxious or scared about talking to or dealing with an aggressive person, and humiliated, degraded or undermined by the behaviour of others" | condition, behaviour, cause |
| SWA model Code | "Have workers reported feeling stressed or are they showing signs of harm?" | condition, reporting, harm |
| Comcare harassment page | "unwelcome comments about my appearance and requests to go for a drink that I have said no to" | behaviour, frequency, refusal |

The rule engine does not need to parse the shape. It needs the pairs:
`behaviour + person marker`, `condition + cause marker`, `behaviour +
frequency`, and so on — which is what [§6](#6-context-rules) encodes.

### 5.2 Regulator / duty voice

| Template | Attested wording |
| --- | --- |
| `(PCBU/worker/officer) must (ensure/minimise/notify/consult) ...` | "A PCBU must ensure, so far as is reasonably practicable, the health and safety of workers" |
| `so far as is reasonably practicable` | 246 hits |
| `if <trigger>, you must <action> immediately` | "You must notify the regulator immediately after becoming aware" |
| `X means Y` / `X refers to Y` | 127 definition sentences; "Intrusive surveillance refers to..." |
| `a person commits an offence if ...` | prosecutions guide |
| `Comcare may (request/accept/decline/affirm/vary/set aside)` | ART and reconsideration pages |

Not sensitive in themselves. They are useful as **negative context**: text
made entirely of these frames is policy prose, not disclosure, and should not
be flagged. They also supply the cue vocabulary the worker's paraphrase may
echo ("I was told to notify immediately").

### 5.3 Medical / causation voice

- "The employee's general practitioner says that the employee has experienced a
  worsening of their depressive symptoms because of the confrontation with an
  aggressive customer and that they require time off work, an increase in
  antidepressant medication."
  (Comcare aggravation guidance — a single sentence with clinician role,
  condition, cause, capacity and treatment.)

- "My doctor said the stress at work made my anxiety worse."
- "I was diagnosed with adjustment disorder after the restructure."
- "I have been off since the incident and I am on medication now."

Frames: `diagnosed with`, `treating <role>`, `(GP/doctor/psychiatrist) says
that`, `because of / due to / as a result of / caused by`, `worsening /
aggravation / flare-up`, `require time off`, `medication`.

### 5.4 Decision and review voice

`affirm / vary / set aside the decision`; `liability accepted / denied`;
`reconsideration requested`; `apply to the ART`; `within 30/60/90 days`;
`notice of rights`; `reviewable decision`. These mark a case history that
identifies a person to anyone with access to the file.

### 5.5 Worker register (for test corpora)

From the model Code's worker-language section and the regulator examples:

- symptom-first ("I spend most of my shift stressed and burnt out...")
- metaphor ("running on fumes", "the last straw", "walking on eggshells")
- concrete counts ("four on, four off, twelve hours each")
- understatement ("it has been raised and nothing has happened")
- evasion ("I won't name names", see §7)
- affective extremes (all caps, repeated punctuation, profanity)

`pii_redact.generate` and the held-out probes should adopt these shapes so
the rules are measured against the register they will meet.

## 6. Context rules

The operative output. Full machine-readable form in
`sensitive_context_rules.json`; each rule has `triggers`, `proximity`,
`action`, `why`, `evidence` and a ready-to-compile `regex` in the style of
`implication.py`. Actions: `escalate` (raise risk / route to review),
`label` (redact span to a category label), `mask` (remove identifying value),
`generalise` (widen), `suppress` (withhold from wide output), `keep` (do not
alter; detection only).

| ID | Rule | Trigger composition | Action |
| --- | --- | --- | --- |
| SC-01 | Condition caused by work | `psych_condition` + (`causal_marker` or first-person employment link), same sentence | escalate; label condition if bare |
| SC-02 | RAA fact pattern | `admin_action` + `psych_condition`, same paragraph | escalate (highest tier); never mask narrative |
| SC-03 | Behaviour against a person | `harm_behaviour` + (`person_marker` or 1st/3rd person object), same clause | label the clause `[IMPLICATION]`-style; escalate |
| SC-04 | Attribute-based harassment | `harm_behaviour` + `protected_attribute`, same sentence | escalate; generalise attribute only if standalone |
| SC-05 | Repeated behaviour pattern | `harm_behaviour` + `frequency_marker` + `person_marker` | escalate; label identifying clause |
| SC-06 | Claim-linked health info | `claim_marker` + `psych_condition`/`capacity` + first person | mask claim refs; tokenise condition |
| SC-07 | Case history disclosure | `claim_marker` + (`determination`/`reconsideration`/`ART`/`denied`/`accepted`) | mask refs; escalate |
| SC-08 | Third-party capacity/restriction | `capacity_restriction` + (`named_person` or role + possessive) | label; escalate |
| SC-09 | Substance + consequence | (`alcohol`/`drugs`) + (`discipline`/`incident`/`safety-critical role`) | escalate; label; suppress slurs |
| SC-10 | Self-harm content | `suicid`/`self-harm`/`self-inflicted` | suppress verbatim; keep coded mention; highest tier |
| SC-11 | Traumatic material exposure | `trauma_material` + (`role` or `reading/viewing` verb) + first person | tokenise; escalate |
| SC-12 | Surveillance objection | `surveillance` + (`unreasonable`/`without consent`/`monitoring me`) | keep (it is the finding); escalate if person named |
| SC-13 | Protected uniqueness | `protected_attribute` + (`only`/`one of small N`) | escalate; generalise attribute |
| SC-14 | Quasi-combination | ≥2 `identity_combinator` + (`event` or `date` or `condition`) | generalise as per `generalise.py`; escalate |
| SC-15 | Confidentiality / reprisal fear | `confidentiality` + (`named role`/`employer`) | escalate; no masking fixes it — flag and restrict |
| SC-16 | Enforcement exposure | `enforcement` + (`named site`/`role`/first person) | escalate; restrict access |
| SC-17 | FDV disclosure | `FDV` + (`danger`/`threat`/`workplace`) | suppress verbatim; escalate; safety pathway |
| SC-18 | Quoted abuse | quote frame + (`slur`/`threat`/`profanity`) + attributed or unnamed speaker | label/suppress quote; escalate |
| SC-19 | Performance + condition third party | `admin_action` + `psych_condition` where subject is a named worker | escalate; label worker name; do not merge with respondent |
| SC-20 | Adjustment denied | `reasonable_adjustment` + `psych_condition` + (`denied`/`refused`/`not provided`) | escalate (discrimination-risk context) |
| SC-21 | Role plus name | `third_party_role` (caseworker, manager, clinician) + a capitalised name within six words | mask name; escalate |

Every rule carries a precision guard in the JSON: terms that cancel the match
(`topic_talk` like "the Code of Practice says", "workers may be exposed",
"training about", "policy", and negation frames). Without the guards, SC-01
alone would fire on most of the model Code.

### 6.1 Measured behaviour on the repo corpora

Before any integration, the rules were run over the two existing fixtures. This
is a rule-quality measurement, not a claim about detection recall: the survey
fixture was authored to exercise PII categories, not these classes.

| Corpus | Records/probes | Findings | Notes |
| --- | --- | --- | --- |
| `data/survey_responses.ndjson` | 58 | 5 | all five on records the manifest already marks sensitive; **zero findings on the 26 declared-clean negative controls**; one guard cancellation (a service exclusion that used the word "excluded") |
| `data/heldout_probes.ndjson` | 50 | 13 across 7 probes | the expected harm, condition, trauma and uniqueness themes; no fires on the role-clarity and support probes |

The five survey-fixture findings were SC-01 (anxiety after the restructure),
SC-03 (three interpersonal-behaviour disclosures, including one written with
typos and leetspeak), and SC-13 (the only woman in the unit). The held-out
run found the age/harassment probe, the graphic-material probe and the
violence-and-aggression probe. That is the
intended shape: the rules fire on composition, not on the topic vocabulary
that fills the regulator corpus. The scanner is literal, so the misspelled and
leet forms in the survey fixture were caught by the surrounding co-triggers,
not by the harm term itself; normalisation in the real engine extends this.

### 6.2 Worked examples

**SC-02.** Input (Claims Manual shape, adapted):

> "My manager put me on a performance improvement plan in March. Since then I
> have been diagnosed with anxiety and my doctor signed me off for two weeks."

`admin_action` (`performance improvement plan`, `manager`) + `psych_condition`
(`anxiety`, `diagnosed`) in one paragraph → escalate. The output keeps the
narrative (it is the finding), tokenises the condition and the leave date, and
routes the record to the top of the review queue because it is the RAA fact
pattern and a potential claim.

**SC-06.** "Comcare accepted liability for my claim CLM-2026-8812 for
adjustment disorder." → claim reference masked, condition tokenised, record
escalated. The reference is the `TestKnownGaps` item and the corpus confirms
it travels in correspondence.

**SC-03.** "There are only two of us on the night shift and he does this to
both of us." → the clause is the identifier. Label `[IMPLICATION]`, escalate,
keep the fact of a repeated behaviour claim.

## 7. Covert naming and evasion

Respondents often avoid naming deliberately. These pass every name scan and
are identifiable to colleagues. Attested or natural variants:

| Evasion | What it does | Detection handle |
| --- | --- | --- |
| "I won't name names" / "no names" | says a name exists elsewhere | escalate on the frame |
| "the person above me" / "the person in charge" | replaces a role | role + possessive + behaviour rule |
| "a certain individual" / "certain people" | singularises without naming | phrase rule |
| "the usual suspects" / "everyone knows who" | presumes shared knowledge | phrase rule; attach to behaviour |
| "you know who" / "IYKYK" | in-group reference | phrase rule |
| implied by rank: "the EL2 in my branch" | unique level + unit | existing `aps_level` + unit rules |
| implied by event: "the manager who ran the restructure" | event + role | `post_event` + person marker |
| implied by roster: "the TL on my shift" | shift + role + team | shift + role + team-size rules |
| initials only: "S.D. did it" | pseudonym in prose | capitalised initials rule |
| indirect third party: "my caseworker was [Name]" | known gap S043 | role + capitalised name, no possessive requirement |

Recommended new implication rules (repo-style): `no_names_frame`,
`certain_person`, `shared_knowledge`, `initials_only`, `role_plus_event`.
Each should label the clause rather than the noun, per the existing design.

## 8. Precision guards

These are corpus-attested phrases that must **not** fire the rules, because
the source material is itself full of them. They are the WHS-domain analogue
of the `clear`/`support`/`torn` denylist that the README documents.

| False friend | Why it is not sensitive here |
| --- | --- |
| "psychosocial hazards" (825 hits) | topic talk; the Code says it on every page |
| "workers may be exposed to..." | policy voice, no person |
| "The model Code of Practice requires..." | regulator voice |
| "I am clear on my responsibilities" | validated role-clarity item (README documents the old FP) |
| "I have support to work safely" | validated support item |
| "we have a good claims team" | service praise, not a claim status |
| "incident notification training" | process name |
| "our EAP provider offers counselling" | program description, no personal disclosure |
| "I run the toolbox talks" | task, not behaviour |
| "support worker" | job title |
| "performance of the scheme" | corporate register |
| "adjustment of the figure" | financial sense |
| "my condition improved after the return to work plan" | protective; keep, do not escalate as harm |
| "reasonable adjustments were made and it worked" | protective |
| "the regulator inspected and we passed" | no person-specific harm |

Rules must require first-person or person-linked co-triggers, and must respect
the existing rule that placeholder text (`[HEALTH DETAIL]`) is never
re-detected on a second pass (idempotence).

## 9. Handling map

| Class | Default handling | Rationale |
| --- | --- | --- |
| `psych_health` | tokenise (`[HEALTH DETAIL]`), escalate with cause | preserves the analysis, keeps the finding |
| `claim_status` | mask references; escalate status narrative | identifiers plus legal/financial history |
| `admin_action` | keep (it is the finding); escalate when paired with health | masking destroys the RAA evidence |
| `harm_behaviour` | label identifying clauses; keep allegation type | behaviour is the survey's purpose |
| `rehab_rtw` | keeping restrictions tokenised (`[WORK RESTRICTION]`); escalate | both signalled in the manual as confidential |
| `whs_enforcement` | keep narrative; mask names/sites; escalate | liability and HSR protection |
| `trauma_content` | tokenise event class; suppress verbatim detail | avoid reproducing harming material |
| `protected_attribute` | generalise where standalone; keep when it is the complaint | existing `generalise.py` practices |
| `third_party` | name-level masking already in repo; extend role+name | S043 gap |
| `legal_advocacy` | escalate; restrict access | privilege and litigation risk |
| `confidentiality_request` | escalate; restrict access; no mask satisfies it | README already states this |
| `FDV` | suppress verbatim; escalate to safety pathway | immediate-danger guidance (Triple Zero) |
| `self-harm` | suppress verbatim; coded routing only | the most harmful class to mishandle |
| `covert naming` | label clause `[IMPLICATION]`; escalate | clause is the identifier |

## 10. Integration into pii_redact

Concrete, module by module. Nothing here changes the repo's core invariant
(deterministic first, NLP for recall only).

1. **New detector `detectors/context.py`.** Consumes
   `sensitive_context_rules.json`, scans sentence windows with the same
   `normalise_aligned` offsets as other detectors, emits
   `Finding(category=Category.SENSITIVE_CONTEXT, ...)` with metadata
   `rule`, `action`, `class`. A new `Category.SENSITIVE_CONTEXT` (in
   `types.py`) keeps it distinct from `HEALTH_DETAIL` and `IMPLICATION`, and
   joins `_QUASI_IDENTIFIERS` only if the deployment wants it scored; my
   recommendation is to keep it **out** of the quasi set and surface it as a
   review flag, because its action is escalation, not generalisation.
2. **`risk.py`.** Add weights and named signals:
   `psych_admin_action` (high), `harm_against_person`, `claim_status`,
   `trauma_material`, `self_harm` (route separately from scoring),
   `third_party_health`, `legal_advocacy`, `fdv_disclosure`. Follow the
   existing pattern: weights in `WEIGHTS`, reasons on `RiskSignal`, and a test
   pinning that topic-talk guards keep the model Code's own sentences clean.
3. **`detectors/implication.py`.** Add `no_names_frame`, `certain_person`,
   `shared_knowledge`, `initials_only`, `role_plus_event`. Each exposed as a
   `clause` rule so the whole identifying phrase is consumed, and each tested
   against a bare vocabulary word (existing `TestOverGenericPhrasing`
   direction).
4. **`lexicon.json`.** Add Comcare-scheme terms that pin the jurisdiction and
   the employer class, matching the README's "narrows an employer" test —
   `Pracsys`, `CAIS`, `rehabilitation case manager (RCM)`, `FCDR`,
   `reasonable administrative action`, `sensitive claim`, `NV1`. Do **not**
   add generic hazards; those belong to rules, per the README's own argument.
5. **`generate.py` and `data/heldout_probes.ndjson`.** Adopt the five-part
   sentence (§5.1) and worker register (§5.5) in templates; add probes drawn
   from the Claims Manual examples (they are quoted above), from the model
   Code worker-language list, and from the ART/reconsideration pages.
6. **`residual.py`.** The narrative selectivity estimate is already flagged as
   the weakest number; this research suggests splitting it by class
   (`harm`, `claim`, `trauma`, `legal`) rather than one narrative constant.
7. **Evaluation.** `context_scan.py` in this directory is the measurement
   harness: it reports per-rule counts and the guard cancellations over any
   text file. Run it over `data/survey_responses.ndjson` free text before and
   after integrating, and record the review-load delta the way the README
   records the residual-control table.

## 11. Provenance

Retrieved 2026-10-17. Primary sources, by group:

**Comcare** — `comcare.gov.au`

- Psychosocial hazards: overview, about, and 17 hazard topic pages (bullying,
  harassment incl. sexual harassment, violence and aggression, conflict,
  fatigue, job demands, job insecurity, low job control, organisational
  justice, poor physical environment, poor support, remote or isolated work,
  reward and recognition, role clarity, traumatic events or materials,
  change management, intrusive surveillance), plus psychosocial risk
  assessment tools, WHS regulations case studies, who can help.
- `scheme-legislation/whs-act/regulatory-guides/managing-psychosocial-hazards`,
  `.../family-domestic-violence`, duties guides, incident notification guide
  (PDF), responding to an incident, notifications.
- Claims: workers compensation, assessing claims, disagree with a
  determination (reconsideration, appeal to ART), getting you back to work,
  supports and benefits, employer information, roles in the claims process,
  certificate of capacity, clinical panel and clinical reviews.
- `scheme-legislation/src-act/guidance`: definition of injury and disease,
  definition of medical treatment, aggravation of a pre-existing condition,
  applying the clinical framework.
- Claims Manual (4 PDFs: Lodgement/Registration/Triage/Liability; Claims
  management/Closure; Reconsiderations/Appeals; Glossary/Get help/FAQs/Claims
  lifecycle), via the Information Publication Scheme.
- News items used for enforcement register: charges over psychosocial risks,
  workplace bullying prevention toolkit, interventions to reduce workplace
  sexual harassment.

**Safe Work Australia** — `safeworkaustralia.gov.au`

- Model Code of Practice: Managing psychosocial hazards at work (PDF).
- Psychosocial hazards topic pages (12 core hazards and harmful behaviours),
  Mental health, Bullying, Sexual and gender-based harassment, Workplace
  violence and aggression, incident notification fact sheet, glossary,
  Work-related psychological health and safety (PDF), data insights page.

**State and territory regulators**

- SafeWork NSW: psychosocial hazards; bullying.
- WorkSafe Victoria: psychological health; psychosocial hazards contributing
  to work-related stress; workplace bullying; sexual harassment.
- WorkSafe Queensland: psychosocial hazards; Managing the risk of psychosocial
  hazards at work Code of Practice 2022 (identifies 14 hazards); People at Work
  (decommissioning notice).
- WorkSafe WA: Psychosocial hazards in the workplace code of practice (PDF);
  common work-related psychosocial hazards and risk factors.
- SafeWork SA: psychosocial hazards; managing harmful behaviours; Managing
  Psychosocial Hazards at Work Code of Practice (PDF, 2026).
- WorkSafe Tasmania: psychosocial hazards topic pages.
- WorkSafe ACT: psychosocial hazards; managing psychosocial hazards;
  disclosing mental health information; WHS investigations into psychosocial
  hazards.
- NT WorkSafe: psychosocial hazards (page fetch blocked; not relied on).

**Statute and privacy**

- Work Health and Safety Act 2011 (Cth) text (`legislation.gov.au`), for s19
  duties, s35–38 notification, s36 definition, enforcement and entry parts.
- OAIC: what is personal information; what is health information.

**Blocked or unavailable during retrieval** (noted for completeness):
WorkSafe Queensland and WorkSafe Tasmania returned 403 to polite fetches;
NT WorkSafe and part of SafeWork SA returned 403/404. Their content above is
covered from search-result extracts or from the equivalent material in other
jurisdictions; re-fetch with a browser session before quoting them as primary.

## Licence note

Research notes and rule data in this directory are provided to the
`pii_redact` project for detector development. Source quotations remain the
property of their publishers; they are included here in short form for
attestation only.

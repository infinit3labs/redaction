"""Australian public-sector and corporate vocabulary.

Shared across the implication rules, the occupation detector and the corpus
generator, so that "a policy officer in a directorate" is recognised in the same
three places.

Every list is built through :func:`alt` and is therefore a self-contained
group. Interpolating a bare ``"a|b"`` into a larger pattern applies the
alternation to the whole expression, which silently turns the surrounding
capture group into a set of top-level branches. That has happened three times in
this package; ``tests/test_vocabulary.py`` asserts the invariant.

The organising idea is that a workplace can be identified by *type* long before
it is identified by name. "Our directorate", "the machinery of government
review" and "a policy officer at EL3" place a respondent in a small population
inside a large organisation, which for re-identification purposes is nearly as
good as an address.
"""

from __future__ import annotations


def alt(*parts: str) -> str:
    """A grouped alternation, safe to interpolate into a larger pattern."""
    return "(?:" + "|".join(parts) + ")"


# --- Commonwealth machinery -------------------------------------------------
# Structure words that only exist in an Australian government department.
#
# Bare nouns only. The rules consume the determiner outside this group, so an
# entry like "the centre" can never match: by the time the group is reached the
# "the" has already been consumed, and only a bare noun can follow.
DEPARTMENT_STRUCTURE = alt(
    "department", "directorate", "branch", "division", "secretariat",
    "ministry", "minister's office", "ministerial office", "agency",
    "commission", "authority", "council",
)

#: Structure nouns that are only informative behind a qualifier. "the integrity
#: unit" and "the policy centre" place a respondent in a small population;
#: bare "my team" or "my office" says nothing and appears in every positive
#: response, so these require a preceding word.
DEPARTMENT_STRUCTURE_QUALIFIED = alt(
    "unit", "centre", "center", "service", "office", "area", "team",
    "function", "portfolio", "program", "programme", "section", "cell",
)

#: Modifiers that name a *service function* rather than a place inside one
#: organisation. Every insurer, bank and government department has a claims
#: team, a member services team and a customer service centre, so these words
#: do not narrow the employer to anything at all -- and in a survey sent by a
#: provider they describe the surveyor's own unit, which is the one workplace
#: in the response that cannot identify the respondent.
#:
#: This guards the *head* modifier only. "the claims team" is suppressed;
#: "the digital services team" is not, because "digital" is the informative
#: word and "services" is incidental. The trade-off is explicit: an employer
#: whose internal team is literally called the claims team will not be caught by
#: the unit rules. It is the better error to make -- a missed unit name costs an
#: attribution signal, while a redacted "member services" destroys the only
#: sentence that says how the respondent was actually treated.
SERVICE_FUNCTION = alt(
    "claim", "claims", "member", "members", "customer", "customers",
    "client", "clients", "policyholder", "policyholders",
    "sales", "marketing", "enquiry", "enquiries", "inquiry", "inquiries",
    "support", "contact", "reception", "billing", "accounts",
    "administration", "administrative", "service", "services",
    "underwriting", "retail", "wholesale", "fulfilment", "fulfillment",
)

# Administrative acts that only occur in a government office.
GOVERNMENT_PROCESS = alt(
    "machinery of government", "ministerial submission", "cabinet submission",
    "parliamentary estimates", "budget papers", "outcome roadmap",
    "operational plan", "workplan", "intelligence product",
    "freedom of information request", r"foi request", "round trip",
    "gateway review", "compliance review", "internal review",
    "the annual report", "the delegate", "the signatory", "the approver",
    "the secretariat", "the inspectorate", "the audit office",
    "the enabling legislation", "the legislative instrument",
)

# --- APS grades --------------------------------------------------------------
# Officer level shorthand. EL1-EL9, SOG and SES are APS specific, and combined
# with a branch they usually identify one person.
APS_LEVEL = alt(
    r"el\s?[1-9]", r"aps\s?[1-9]", r"aps\s?band\s?[1-4]", r"so[0-3]",
    r"ses\s?[1-3]", r"aps\s?2\s?i\b", r"aps\s?3\s?c\b", "officer level",
    "executive officer", "secretary level", "general manager level",
    r"o[1-9]", "directorate level", "grade 1", "grade 2", "grade 3",
)

# Roles whose combination with a branch or directorate is near-unique inside an
# agency. Government is small at the top: there is one director, sometimes one
# EL9, and a handful of branch managers.
SENIOR_ROLE = alt(
    "the secretary", "the chief executive", "the commissioner", "the ombudsman",
    "the auditor-general", "the inspector-general", r"the ceo\b", "the cfo\b",
    "the coo\b", "the director-general", "the director", "branch manager",
    "the dean", "the provost", "the comptroller", "the registrar",
    "the national manager", "the state manager", "the regional manager",
    "assistant director", "directorate manager",
)

# --- Occupational families ---------------------------------------------------
# Deliberately specific. Abstract nouns such as "officer", "manager" or
# "specialist" are excluded: they match almost any sentence and would destroy
# the signal.
OCCUPATION_FAMILY = alt(
    # General public service
    "policy officer", "policy adviser", "policy analyst", "assistant director",
    "senior policy officer", "principal analyst", "senior analyst",
    "branch manager", "directorate manager", "general manager",
    "program manager", "project manager", "portfolio manager",
    "eligibility specialist", "entitlement specialist", "case officer",
    "contact officer", "contact centre agent", "service centre operator",
    "grants administrator", "grant administrator", "funding officer",
    "contracts officer", "procurement officer", "payroll officer",
    "recruiter", "aps assessor", "people and culture advisor",
    # Law, audit and investigation
    "lawyer", "paralegal", "prosecutor", "legal officer", "in-house counsel",
    "auditor", "internal auditor", "investigator", "inspector",
    "compliance officer", "enforcement officer", "ombudsman",
    # Analytical and technical
    "statistician", "economist", "actuary", "data analyst", "data scientist",
    "researcher", "research officer", "modeller", "forecaster",
    "actuary", "geoscientist", "hydrologist", "toxicologist", "veterinarian",
    "laboratory technician", "science officer",
    # Records, content and enabling functions
    "archivist", "curator", "librarian", "records manager",
    "communications officer", "media officer", "content designer",
    "policy writer", "editor", "secretariat officer",
    # Health agencies
    "registrar", "clinician", "nurse practitioner", "allied health officer",
    "sanitation officer", "environmental health officer", "food inspector",
    # Education
    "principal", "deputy principal", "head teacher", "classroom teacher",
    "student services officer",
    # Law enforcement and emergency services
    "detective", "investigator", "custodial officer", "probation officer",
    "firefighter", "station officer", "triage nurse",
    # Corporate
    "trader", "underwriter", "actuary", "relationship manager",
    "store manager", "depot manager", "site supervisor", "shift supervisor",
    "mine supervisor", "production supervisor", "quality inspector",
    "senior trader", "fund manager", "investment analyst",
    # Traineeships are small cohorts, so an apprentice is close to nameable.
    "apprentice", "trainee", "cadet", "graduate", "intern",
)

# Roles that identify a *government* workplace when combined with structure.
PUBLIC_SECTOR_ROLE = alt(
    "policy officer", "policy adviser", "eligibility specialist", "case officer",
    "aps officer", "public servant", "regulator", "adjudicator",
    "tribunal member", "hearing officer", "programme manager",
    "assistant director", "senior policy officer", "branch manager",
    "directorate manager", "senior investigator", "principal analyst",
)

# --- Workplace type markers ---------------------------------------------------
# Industry and setting nouns. These do not name an employer, but a small one
# that uses a distinctive combination gives the sector away immediately.
#: Workplace nouns that name a sector even with a bare "the". "The depot" or
#: "the correctional centre" is a workplace; "the store" in "declined at the
#: store" is a transaction, and redacting it destroys the meaning of the
#: sentence without reducing risk.
WORKPLACE_TYPE_STRONG = alt(
    "depot", "correctional centre", "prison", "detention centre", "remand centre",
    "refinery", "smelter", "quarry", "open cut", "underground mine",
    "distribution centre", "freight terminal", "container yard", "interchange",
    "sidings", "water authority", "waste facility", "sewage plant",
    "treatment plant", "landfill", "gas plant", "processing plant",
    "residential aged care", "disability accommodation", "foster placement",
    "substation", "transmission line", "power station", "ambulance station",
    "police station", "fire station", "tafe", "remand",
)

#: The general list. Requires a possessive determiner.
WORKPLACE_TYPE = alt(
    # Government and services
    "department", "agency", "statutory authority", "regulator", "council",
    "shire", "borough", "tribunal", "parliament", "embassy", "consulate",
    "correctional centre", "prison", "detention centre", "remand centre",
    "public hospital", "community health centre", "ambulance station",
    "fire station", "police station", "school", "tafe", "university",
    # Utilities and regulated infrastructure
    "water authority", "power station", "substation", "transmission line",
    "waste facility", "sewage plant", "treatment plant", "landfill",
    # Resources and heavy industry
    "mine", "pit", "open cut", "underground mine", "refinery", "smelter",
    "quarry", "gas plant", "processing plant",
    # Transport and logistics
    "depot", "terminal", "interchange", "sidings", "yard", "distribution centre",
    "freight terminal", "container yard", "airport", "aerodrome",
    # Retail and services
    "store", "supermarket", "shopping centre", "drive-through", "franchise",
    # Construction and trades
    "construction site", "scaffold", "fit-out", "civil works site",
    # Accommodation and social care
    "shelter", "hostel", "refugee centre", "store", "supermarket",
    "shopping centre", "drive-through", "franchise", "facility",
    "campus", "workshop", "clinic", "hospital", "school", "university",
    "council", "airport", "aerodrome",
)

# --- Corporate markers -------------------------------------------------------
CORPORATE_MARKER = alt(
    "graduate program", "graduate cohort", "our eba", "enterprise agreement",
    "values and behaviours", "the shareholder return", "asx announcement",
    "the board", "the executive team", "our shareholders", "the pit",
    "the trading floor", "our site", "shop floor", "the ceo", "the cfo",
    "annual bonus", "performance scorecard", "the graduate intake",
)

# --- Cohort, programme and clearance markers ---------------------------------
# Anything that places a respondent in a small named population. For a survey
# shared outside the organisation these are among the most dangerous phrases
# in the data.
COHORT_MARKER = alt(
    "the graduate cohort", "the graduate intake", r"the \d{4} intake",
    r"the \d{4} cohort", "the pilot cohort", "the review team",
    "the trial site", "the reference site", "the secondment",
    "the acting manager", "the relief manager", "the transition team",
)

# Security clearance. Naming a clearance level identifies a person as surely as
# naming a project, and in a survey about bullying it can be career-ending if
# published.
CLEARANCE_MARKER = alt(
    "cleared to", "security clearance", "clearance level", "clearance",
    "ahptc", "vetting",
    "national security vetting", "nvl", "baseline clearance",
    "protected clearance", "top secret clearance",
)

# The clearance levels themselves, for the pattern that requires one.
CLEARANCE_LEVEL = alt("secret", "top secret", "protected", "baseline", "pv")

# --- Retention signals -------------------------------------------------------
# Past-tense verbs that follow "I have" without naming a condition. A
# cybersecurity or workplace-safety survey is full of "I have raised", "I have
# reported" and "I have raised it twice", and none of those are a diagnosis.
NON_CONDITION_VERBS = alt(
    "put", "got", "gotten", "taken", "done", "seen", "noticed", "heard",
    "read", "written", "said", "told", "asked", "found", "used", "left",
    "been", "going", "being", "doing", "working", "trying", "hoping",
    "raised", "worked", "let", "set", "kept", "gave", "made", "reported",
    "logged", "escalated", "followed up", "complained", "already", "also",
    "just", "only", "still", "even", "never", "always", "twice", "thrice",
    "since", "before", "after", "again", "repeatedly", "previously",
)

__all__ = [
    "APS_LEVEL",
    "CLEARANCE_LEVEL",
    "CLEARANCE_MARKER",
    "COHORT_MARKER",
    "CORPORATE_MARKER",
    "DEPARTMENT_STRUCTURE",
    "DEPARTMENT_STRUCTURE_QUALIFIED",
    "GOVERNMENT_PROCESS",
    "NON_CONDITION_VERBS",
    "OCCUPATION_FAMILY",
    "PUBLIC_SECTOR_ROLE",
    "SENIOR_ROLE",
    "SERVICE_FUNCTION",
    "WORKPLACE_TYPE",
    "WORKPLACE_TYPE_STRONG",
    "alt",
]
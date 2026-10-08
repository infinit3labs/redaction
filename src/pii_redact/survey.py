"""Survey profile: the detector set and policy for WHS free text.

The defaults in :mod:`pii_redact.engine` are tuned for documents. Survey free
text is a different problem:

* Respondents rarely write anything in a canonical format, so the
  quasi-identifier and organisation detectors matter more than the phone and
  TFN detectors.
* Employer attribution is a first-class risk, not a footnote.
* Some things should be *generalised* rather than removed, or the survey
  loses the signal it exists to measure.

:func:`survey_redactor` returns a redactor configured for that job, and
:func:`survey_policy` returns the policy alone for callers that supply their
own detector list.
"""

from __future__ import annotations

from dataclasses import replace

from .detectors.base import Detector
from .detectors.implication import default_implication_detector
from .detectors.org import default_organisation_detectors
from .detectors.quasi import default_quasi_detectors
from .engine import DEFAULT_DETECTORS, Policy, Redactor
from .lexicon import Lexicon, load_builtin_lexicon
from .types import Category, MaskStyle

#: Values that are generalised rather than masked by default. Ages, exact
#: dates, tenure and team size all carry survey signal that masking would
#: destroy, while the precision that makes them identifying is exactly what a
#: band removes. Direct identifiers are masked, not generalised.
DEFAULT_GENERALISED = frozenset(
    {
        Category.AGE,
        Category.DATE_AU,
        Category.DATE_ISO,
        Category.TENURE,
        Category.TEAM_SIZE,
    }
)

#: Health detail is labelled, never masked. "I developed anxiety after the
#: restructure" is the finding; masking it would destroy the analysis while
#: barely reducing risk. ``[HEALTH DETAIL]`` keeps the record analysable as a
#: health-related response without naming the condition.
HEALTH_DETAIL_STYLE: dict[Category, MaskStyle] = {
    Category.HEALTH_DETAIL: MaskStyle.LABEL,
}

#: Employer attribution is labelled: it is an identifier, not a measure.
ATTRIBUTION_STYLE: dict[Category, MaskStyle] = {
    Category.ORGANISATION_NAME: MaskStyle.LABEL,
    Category.INTERNAL_TERM: MaskStyle.LABEL,
    Category.AWARD: MaskStyle.LABEL,
}

#: Implicating sequences are labelled. The clause is the identifier, and the
#: fact that a uniqueness claim was made is analytically important, so the
#: response keeps a marker rather than losing the clause entirely.
IMPLICATION_STYLE: dict[Category, MaskStyle] = {
    Category.IMPLICATION: MaskStyle.LABEL,
}

#: Quasi-identifiers are generalised where a band exists (see
#: ``DEFAULT_GENERALISED``) and tokenised where it does not. Tokenising a job
#: title rather than masking it preserves the occupational breakdown that makes
#: a WHS survey actionable: responses can still be grouped by "nurse" without
#: the title being readable.
QUASI_STYLE: dict[Category, MaskStyle] = {
    Category.JOB_TITLE: MaskStyle.LABEL,
    Category.EMPLOYMENT_STATUS: MaskStyle.LABEL,
}


def survey_detectors(
    lexicon: Lexicon | None = None,
    *,
    include_builtin_lexicon: bool = True,
) -> tuple[Detector, ...]:
    """The full detector list for survey free text.

    Order is the engine's tie-break order, so the strong, high-confidence
    checksummed identifiers come first and the noisier heuristics come last.

    A caller-supplied ``lexicon`` is *merged over* the bundled seed rather than
    replacing it, so adding your internal project names never loses the generic
    Australian WHS vocabulary. Pass ``include_builtin_lexicon=False`` to opt out.
    """
    base: tuple[Detector, ...] = tuple(cls() for cls in DEFAULT_DETECTORS)  # type: ignore[call-arg]
    resolved = load_builtin_lexicon() if include_builtin_lexicon else Lexicon()
    if lexicon is not None:
        resolved = resolved.merged(lexicon)
    organisation = default_organisation_detectors(resolved)
    quasi = default_quasi_detectors()
    implication = default_implication_detector()
    return base + organisation + quasi + (implication,)


def survey_policy(
    *,
    salt: bytes | None = None,
    nlp_enabled: bool = True,
    generalise_categories: frozenset[Category] = DEFAULT_GENERALISED,
    **overrides: object,
) -> Policy:
    """A policy for WHS and bullying survey free text.

    ``salt`` should be a secret held in Databricks secret scope. Without it,
    ``TOKEN`` style still works but the tokens are only stable within a single
    executor process, which defeats the purpose of using them.
    """
    styles: dict[Category | str, MaskStyle] = {
        # Direct identifiers keep a full mask: there is nothing analytical in
        # the shape of a phone number.
        "default": MaskStyle.MASK,
        **ATTRIBUTION_STYLE,
        **QUASI_STYLE,
        **HEALTH_DETAIL_STYLE,
        **IMPLICATION_STYLE,
    }
    policy = Policy(
        styles=styles,  # type: ignore[arg-type]
        generalise_categories=generalise_categories,
        salt=salt,
        nlp_enabled=nlp_enabled,
        nlp_backend="gazetteer",
    )
    if overrides:
        policy = replace(policy, **overrides)  # type: ignore[arg-type]
    return policy


def survey_redactor(
    *,
    lexicon: Lexicon | None = None,
    salt: bytes | None = None,
    nlp_enabled: bool = True,
    generalise_categories: frozenset[Category] = DEFAULT_GENERALISED,
    include_builtin_lexicon: bool = True,
    **policy_overrides: object,
) -> Redactor:
    """A redactor wired for survey free text.

    ``lexicon`` is merged over the bundled generic seed, so passing your own
    internal term list adds to the baseline rather than replacing it.

    >>> redactor = survey_redactor(salt=b"...")
    >>> redactor.redact_text("I am a Grade 4 nurse at Acme Pty Ltd")
    """
    return Redactor(
        survey_policy(
            salt=salt,
            nlp_enabled=nlp_enabled,
            generalise_categories=generalise_categories,
            **policy_overrides,
        ),
        detectors=survey_detectors(lexicon, include_builtin_lexicon=include_builtin_lexicon),
    )


__all__ = [
    "ATTRIBUTION_STYLE",
    "DEFAULT_GENERALISED",
    "HEALTH_DETAIL_STYLE",
    "IMPLICATION_STYLE",
    "QUASI_STYLE",
    "survey_detectors",
    "survey_policy",
    "survey_redactor",
]
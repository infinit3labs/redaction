"""Spark integration for redacting survey data before it is persisted.

Target: Databricks. The NDJSON extract is read into a DataFrame, redacted, and
written to Delta. Redaction therefore happens inside the cluster, on executors,
thousands of times per run.

Three things that matter on this path:

**Do not ship the redactor in the closure.** Compiled regexes and the lexicon
delete index are expensive to build and should not be relied on to serialise
cleanly. :class:`RedactorSpec` is a small frozen config; each worker process
builds and caches its own redactor on first use, so a task that retries or lands
on a new executor pays the build cost once per JVM rather than once per task.

**Use a pandas UDF, not a scalar UDF.** A scalar UDF pays Python serialisation
per row. The batch UDF below amortises that across a whole Arrow batch.

**Offsets in findings are against the decoded field value**, not the raw JSON
line. Qualtrics escapes text aggressively, so offsets taken from the raw line
would not line up with the string a reviewer sees.

Every Spark import is deferred, so this module imports and its non-Spark
helpers are testable in an environment without pyspark installed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .lexicon import Lexicon
from .records import SurveyRedactor
from .risk import Audience, RiskBand
from .survey import DEFAULT_GENERALISED
from .types import Category

#: Output column suffixes. Suffix rather than overwrite, so a trial run can
#: still compare the redacted output against the raw extract.
REDACTED_SUFFIX = "_redacted"
FINDINGS_SUFFIX = "_findings"
RISK_SUFFIX = "_risk"
RISK_BAND_SUFFIX = "_risk_band"
REVIEW_SUFFIX = "_needs_review"

#: Field names inside the UDF's struct, mapped to their column suffix.
_STRUCT_FIELDS = {
    "redacted": REDACTED_SUFFIX,
    "findings": FINDINGS_SUFFIX,
    "risk": RISK_SUFFIX,
    "risk_band": RISK_BAND_SUFFIX,
    "needs_review": REVIEW_SUFFIX,
}

#: Per-JVM redactor cache. Spark reuses executors, so this turns a per-task
#: rebuild into a per-executor one.
_WORKER_CACHE: dict[str, SurveyRedactor] = {}


@dataclass(frozen=True, slots=True)
class RedactorSpec:
    """A small, serialisable description of a redactor.

    Everything expensive is derived from this in the worker: the detector list,
    the compiled patterns, the lexicon delete index and the NLP backend.
    """

    salt_hex: str | None = None
    nlp_enabled: bool = True
    generalise_categories: tuple[str, ...] = ()
    audience: str = Audience.INTERNAL.value
    review_threshold: str = RiskBand.HIGH.value
    #: Terms supplied by the caller, as lexicon term descriptions.
    lexicon_terms: tuple[dict[str, Any], ...] = ()
    policy_overrides: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        salt: bytes | str | None = None,
        nlp_enabled: bool = True,
        generalise_categories: frozenset[Category] | None = None,
        audience: Audience = Audience.INTERNAL,
        review_threshold: RiskBand = RiskBand.HIGH,
        lexicon: Lexicon | None = None,
        **policy_overrides: Any,
    ) -> RedactorSpec:
        """Build a spec, taking the salt as raw bytes or a hex string.

        The salt should come from a Databricks secret. It is carried as hex so
        the spec stays JSON-serialisable and can be logged or stored without
        leaking the raw value.

        ``generalise_categories`` defaults to the survey profile's set. An empty
        set is a valid choice, but it must be passed explicitly: silently
        falling back to masking everything would differ from
        ``survey_redactor()`` on the same text, which is a very hard difference
        to notice in a Delta table.
        """
        return cls(
            salt_hex=_encode_salt(salt),
            nlp_enabled=nlp_enabled,
            generalise_categories=tuple(
                sorted(
                    c.value
                    for c in (
                        DEFAULT_GENERALISED
                        if generalise_categories is None
                        else generalise_categories
                    )
                )
            ),
            audience=audience.value,
            review_threshold=review_threshold.value,
            lexicon_terms=tuple(lexicon.describe()) if lexicon is not None else (),
            policy_overrides=tuple(sorted(policy_overrides.items(), key=lambda kv: kv[0])),
        )

    def salt_bytes(self) -> bytes | None:
        return bytes.fromhex(self.salt_hex) if self.salt_hex else None

    def categories(self) -> frozenset[Category]:
        return frozenset(Category(v) for v in self.generalise_categories)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for shipping to workers or storing in a table."""
        return {
            "salt_hex": self.salt_hex,
            "nlp_enabled": self.nlp_enabled,
            "generalise_categories": list(self.generalise_categories),
            "audience": self.audience,
            "review_threshold": self.review_threshold,
            "lexicon_terms": [dict(t) for t in self.lexicon_terms],
            "policy_overrides": dict(self.policy_overrides),
        }

    def cache_key(self) -> str:
        """Stable identity for the per-process cache.

        Derived from the full config so two specs that differ in any way that
        affects output get separate redactors.
        """
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


def _encode_salt(salt: bytes | str | None) -> str | None:
    if isinstance(salt, bytes):
        return salt.hex()
    if isinstance(salt, str):
        stripped = salt.removeprefix("0x")
        if stripped and len(stripped) % 2 == 0 and _is_hex(stripped):
            return stripped.lower()
    return None


def _is_hex(value: str) -> bool:
    return all(c in "0123456789abcdefABCDEF" for c in value)


def build_worker_redactor(spec: RedactorSpec) -> SurveyRedactor:
    """Build, or reuse, the redactor for ``spec`` in this process."""
    key = spec.cache_key()
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        return cached

    from .engine import Redactor
    from .survey import survey_detectors, survey_policy

    lexicon = (
        Lexicon.from_spec({"terms": [dict(t) for t in spec.lexicon_terms]})
        if spec.lexicon_terms
        else None
    )
    policy = survey_policy(
        salt=spec.salt_bytes(),
        nlp_enabled=spec.nlp_enabled,
        generalise_categories=spec.categories() or DEFAULT_GENERALISED,
        **dict(spec.policy_overrides),
    )
    survey = SurveyRedactor(
        Redactor(policy, detectors=survey_detectors(lexicon)),
        audience=Audience(spec.audience),
        review_threshold=RiskBand(spec.review_threshold),
    )
    _WORKER_CACHE[key] = survey
    return survey


# --- Row-level helpers, usable without Spark ---------------------------------
def redact_text(spec: RedactorSpec, text: str | None, record_id: str | None = None):
    """Redact a single string. The unit the UDFs below wrap."""
    return build_worker_redactor(spec).redact_record(text, record_id=record_id)


def summarise_batch(
    spec: RedactorSpec,
    texts: Sequence[str | None],
) -> dict[str, list[Any]]:
    """Redact a batch, returning column-oriented arrays.

    Written against the Python sequence protocol so it works with a pandas
    ``Series`` and with a plain list, which keeps it testable without pandas.
    """
    survey = build_worker_redactor(spec)
    redacted: list[Any] = []
    findings: list[Any] = []
    risks: list[Any] = []
    bands: list[Any] = []
    reviews: list[Any] = []
    for text in texts:
        record = survey.redact_record(text)
        redacted.append(record.text)
        findings.append(record.findings_json())
        risks.append(record.risk.score)
        bands.append(record.risk.band.value)
        reviews.append(survey.needs_review(record))
    return {
        "redacted": redacted,
        "findings": findings,
        "risk": risks,
        "risk_band": bands,
        "needs_review": reviews,
    }


# --- Spark surface ------------------------------------------------------------
def redact_batch_udf(spec: RedactorSpec):
    """A pandas UDF returning redacted text, findings JSON, risk and a flag.

    Arrow batch serialisation is amortised across the batch, which is the
    difference between minutes and hours on a full survey extract.
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("redacted", StringType(), True),
            StructField("findings", StringType(), True),
            StructField("risk", DoubleType(), True),
            StructField("risk_band", StringType(), True),
            StructField("needs_review", BooleanType(), True),
        ]
    )

    @pandas_udf(schema)
    def _apply(series):
        import pandas as pd

        out = summarise_batch(spec, series.tolist())
        return pd.DataFrame(out, columns=list(_STRUCT_FIELDS))

    return _apply


def redact_column_udf(spec: RedactorSpec):
    """A scalar UDF returning only the redacted text.

    The simplest option and the slowest. Prefer :func:`redact_frame` for
    anything over a trivial row count.
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    return udf(lambda text: redact_text(spec, text).text, StringType())


def redact_frame(df, columns: Sequence[str], spec: RedactorSpec):
    """Redact several free-text columns on a Spark DataFrame.

    For each column in ``columns`` this adds ``<column>_redacted``,
    ``<column>_findings``, ``<column>_risk``, ``<column>_risk_band`` and
    ``<column>_needs_review``. The original columns are left in place, so
    select only the redacted text and the non-text columns when writing the
    Delta table: keeping the originals alongside would defeat the purpose.
    """
    from pyspark.sql.functions import col

    udf_fn = redact_batch_udf(spec)
    result = df
    for column in columns:
        expanded = udf_fn(col(column)).alias("_redaction")
        for field_name, suffix in _STRUCT_FIELDS.items():
            result = result.withColumn(f"{column}{suffix}", expanded[field_name])
        result = result.drop("_redaction")
    return result


def redact_dataframe(
    df,
    spec: RedactorSpec,
    *,
    columns: Sequence[str] | None = None,
    auto_detect: bool = True,
):
    """Redact a survey DataFrame, detecting the free-text columns if needed.

    Qualtrics exports name question columns ``Q1``, ``Q12`` and so on, with
    metadata such as ``StartDate`` and ``IPAddress`` alongside. With
    ``auto_detect`` every string column is redacted and non-string columns are
    left alone, which is the safe default: it never silently skips an answer.

    Note that ``IPAddress`` is itself an identifier, and Redactor already
    detects IPv4, so auto-detect covers it.
    """
    if columns is not None:
        return redact_frame(df, columns, spec)
    if not auto_detect:
        raise ValueError("provide columns= or enable auto_detect")
    string_columns = [
        field.name for field in df.schema.fields if field.dataType.typeName() == "StringType"
    ]
    return redact_frame(df, string_columns, spec)


__all__ = [
    "FINDINGS_SUFFIX",
    "REDACTED_SUFFIX",
    "REVIEW_SUFFIX",
    "RISK_BAND_SUFFIX",
    "RISK_SUFFIX",
    "RedactorSpec",
    "build_worker_redactor",
    "redact_batch_udf",
    "redact_column_udf",
    "redact_dataframe",
    "redact_frame",
    "redact_text",
    "summarise_batch",
]
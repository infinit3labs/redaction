# Databricks: redact a survey DataFrame before writing Delta

This notebook is the intended entry point. It assumes:

* the Qualtrics NDJSON extract has landed in a table or a volume path
* the DataFrame is materialised with `spark.read.json` (or from a staging table)
* redaction happens **before** the Delta write

```python
raw = spark.read.json("/Volumes/main/surveys/extract/2026-03.ndjson")
```

## 1. Install the package on the cluster

```python
%pip install --upgrade pii-redact
```

For a cluster library, build a wheel and attach it. Do not `%pip install` in a
long-lived production job; it mutates the environment per session.

## 2. Build the spec on the driver

The spec is the only thing that crosses to executors. The salt comes from a
secret, so it never appears in the job definition or in the query log.

```python
from pii_redact import Audience, Category, RiskBand
from pii_redact.lexicon import Lexicon, Term
from pii_redact.spark import RedactorSpec

salt = dbutils.secrets.get(scope="pii", key="redaction_salt")  # hex string

internal = Lexicon.from_terms([
    Term("Project Kestrel", Category.INTERNAL_TERM,
         aliases=["kestrel", "proj kestrel"], label="internal project"),
    Term("the Beacon rollout", Category.INTERNAL_TERM, label="internal program"),
])

spec = RedactorSpec.build(
    salt=salt,
    audience=Audience.INTERNAL,        # change per sharing decision
    review_threshold=RiskBand.HIGH,
    lexicon=internal,
)
```

The lexicon you pass is **merged over** the bundled generic seed, so you get the
baseline Australian WHS vocabulary as well as your own terms. Pass
`include_builtin_lexicon=False` to `survey_detectors` if you want only yours.

## 3. Preview before you commit

Run the same code locally on a sample so you can see what the risk bands look
like and tune thresholds before writing anything.

```python
sample = [row["Q7"] for row in raw.select("Q7").limit(200).collect()]
for record in summarise_batch(spec, sample)["redacted"][:5]:
    print(record)
```

A useful check: read a few of the highest-risk rows and confirm they *should* be
high. A model that scores your own data as all-low is broken.

## 4. Redact

```python
from pii_redact.spark import redact_dataframe, redact_frame

free_text_columns = [c for c in raw.columns if c.startswith("Q")]

clean = redact_frame(raw, free_text_columns, spec)
```

Each column gains five new ones:

| column | contents |
| --- | --- |
| `Q7_redacted` | the rewritten text — **this is what you persist** |
| `Q7_findings` | JSON, offsets against the original `Q7` |
| `Q7_risk` | 0.0 to 1.0 |
| `Q7_risk_band` | low / moderate / high / critical |
| `Q7_needs_review` | boolean, for triage |

To redact every string column instead of naming them:

```python
clean = redact_dataframe(raw, spec)
```

That covers Qualtrics metadata too, which matters: `IPAddress` is itself an
identifier and the IPv4 detector picks it up.

## 5. Triage before you publish

```python
review = clean.filter(
    reduce(lambda c, col: c | (col[col + "_needs_review"]),
           [F.lit(False)], *[F.col(c) for c in free_text_columns])
)
display(review.select("ResponseID", "Q7_redacted", "Q7_risk", "Q7_risk_band"))
```

`Q7_findings` explains every decision. A finding looks like:

```json
{"category":"internal_term","start":30,"end":37,"detector":"lexicon_term",
 "stage":"deterministic","confidence":0.85,
 "metadata":{"canonical":"Project Kestrel","distance":1,"fuzzy":true,
             "label":"internal project"}}
```

Offsets are against the **original** column value, so you can join back to the
raw extract and verify any redaction.

## 6. Write Delta without the originals

This is the step people get wrong. Selecting the redacted columns only is what
actually protects the data; a table that keeps `Q7` next to `Q7_redacted` is
still a table full of PII.

```python
persist = [
    "ResponseID", "StartDate", "EndDate",          # non-free-text columns
] + [c + suffix for c in free_text_columns
     for suffix in ("_redacted", "_risk", "_risk_band", "_needs_review")]

(
    clean
    .select(*persist)
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.surveys.whs_redacted")
)
```

Keep `_findings` out of the Delta table. It contains offsets into data you no
longer have, and it is audit material, not analysis material. Park it in a
separate access-controlled location if you need it.

## Things that will bite you

**Tokens are only stable with a fixed salt.** Without a salt from a secret
scope, each executor process generates its own, so `job_title_ab12cd34` in
today's table is a different token from tomorrow's. Any grouping you do across
runs will be silently wrong. Persist the salt; rotate it only when you accept
breaking linkage with historical data.

**Lexicons with common words misfire.** A term like "Care", "Grade" or "Shift"
will match ordinary prose. Fuzzy matching is disabled below six characters for
exactly this reason, but an exact match on a short term is still a match. List
an internal term only when it actually narrows the employer.

**Free-text names need the NLP stage.** "Jane Citizen" with no label or
honorific is not detected by default. `nlp_enabled=True` (the default for
`survey_redactor`) catches label and honorific forms; a spaCy model catches
more but needs `pip install 'pii-redact[nlp]'` and a model on the cluster.

**A pandas UDF still serialises every row.** It is much cheaper than a scalar
UDF but not free. On a very large extract, filter to non-empty free-text
columns before the UDF, or partition so each task handles a sane batch.

**Test on a sample of your own data.** The weights in `risk.py` are starting
points, not measurements. `WEIGHTS` is a plain dict; adjust it with evidence
from your reviews and redeploy.
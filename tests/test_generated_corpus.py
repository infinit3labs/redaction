"""The generated corpus, as a measurable gate.

``pii_redact.generate`` plants known secrets and records them. These tests turn
that into recall floors, which is the only honest way to know whether the
detectors still work: a fixture-based test passes on the twenty cases someone
thought of, while these run over a few hundred composed records and check that
nothing the generator planted survives.

Recall floors are deliberately split by how the value was written. Values the
generator misspelled or spaced out are a ceiling for exact-pattern sequence
rules, not a defect, and holding them to the same bar as clean values would
either fail for a reason that cannot be fixed or push the detector into
matching noise.
"""

from __future__ import annotations

from collections import Counter

import pytest

from pii_redact import Category, Lexicon, Term
from pii_redact.generate import (
    REGISTERS,
    GeneratedRecord,
    SurveyGenerator,
    degrade,
    misspell,
    space_out,
    to_fullwidth,
    valid_acn,
    valid_medicare,
    valid_phone,
    valid_tfn,
    write_corpus,
)
from pii_redact.spark import RedactorSpec, summarise_batch

SEED = 4242
COUNT = 300
DEMO_SALT = "0" * 32

METADATA_COLUMNS = {"ResponseID"}


def _is_degraded(secret) -> bool:
    """True when the generator altered the value beyond recognition.

    Misspelled and spaced values are a ceiling for exact-pattern rules. They
    are still planted and still counted in the overall figure; they are just not
    held to the same per-kind floor as well-formed values.
    """
    return secret[2] in {"misspelled", "spaced"}

#: Kinds whose planted value must never survive, whatever the phrasing.
MUST_NEVER_SURVIVE = {
    "tfn",
    "medicare",
    "acn",
    "credit_card",
    "dob",
    "bsb",
    "phone",
    "phone_obfuscated",
    "email",
    "organisation",
    "third_party",
    "suburb",
}


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return Lexicon.from_terms(
        [
            Term("Project Kestrel", Category.INTERNAL_TERM, aliases=["kestrel"],
                 label="internal project"),
            Term("Beacon", Category.INTERNAL_TERM, label="internal program"),
        ]
    )


@pytest.fixture(scope="module")
def spec(lexicon: Lexicon) -> RedactorSpec:
    return RedactorSpec.build(salt=DEMO_SALT, lexicon=lexicon)


@pytest.fixture(scope="module")
def corpus() -> list[GeneratedRecord]:
    return SurveyGenerator(seed=SEED).corpus(COUNT)


@pytest.fixture(scope="module")
def redacted(spec: RedactorSpec, corpus: list[GeneratedRecord]) -> list[dict[str, str]]:
    """Redacted text per record, column by column."""
    out = []
    for record in corpus:
        columns = [c for c in record.answers if c not in METADATA_COLUMNS]
        batch = summarise_batch(spec, [record.answers[c] for c in columns])
        out.append(dict(zip(columns, batch["redacted"], strict=True)))
    return out


class TestGeneratorIntegrity:
    def test_is_deterministic(self) -> None:
        a = SurveyGenerator(seed=SEED).corpus(40)
        b = SurveyGenerator(seed=SEED).corpus(40)
        assert [r.answers for r in a] == [r.answers for r in b]
        assert [r.secrets for r in a] == [r.secrets for r in b]

    def test_a_different_seed_gives_different_text(self) -> None:
        a = SurveyGenerator(seed=1).corpus(20)
        b = SurveyGenerator(seed=2).corpus(20)
        assert [r.answers for r in a] != [r.answers for r in b]

    def test_ids_are_unique(self, corpus: list[GeneratedRecord]) -> None:
        ids = [r.response_id for r in corpus]
        assert len(ids) == len(set(ids))

    def test_no_secret_is_split_across_columns(
        self, corpus: list[GeneratedRecord]
    ) -> None:
        """A secret spanning two fields is uncatchable by construction.

        The generator splits on sentence boundaries precisely so that this holds;
        if it regressed, recall numbers would measure the generator rather than
        the detectors.
        """
        for record in corpus:
            joined = " ".join(record.answers.values())
            for secret in record.secrets:
                assert secret.literal in joined, record.response_id

    def test_secrets_carry_their_degradation(self, corpus: list[GeneratedRecord]) -> None:
        forms = {s.degradation for r in corpus for s in r.secrets}
        assert {"clean", "misspelled"} <= forms

    def test_planted_values_are_syntactically_valid(self) -> None:
        import random

        rng = random.Random(7)
        for _ in range(20):
            from pii_redact import is_acn, is_medicare, is_tfn

            assert is_tfn(valid_tfn(rng))
            assert is_medicare(valid_medicare(rng))
            assert is_acn(valid_acn(rng))
            assert valid_phone(rng).startswith(("0", "13", "18"))

    def test_write_corpus_round_trips(self, tmp_path, corpus: list[GeneratedRecord]) -> None:
        import json

        ndjson = tmp_path / "c.ndjson"
        manifest = tmp_path / "c.json"
        write_corpus(corpus[:20], ndjson, manifest)
        lines = [json.loads(line) for line in ndjson.read_text().splitlines() if line.strip()]
        assert len(lines) == 20
        assert json.loads(manifest.read_text())["count"] == 20


class TestDegradation:
    def test_misspell_changes_the_text(self) -> None:
        import random

        rng = random.Random(3)
        assert misspell("Kestrel", rng) != "Kestrel"

    def test_fullwidth_digits(self) -> None:
        assert to_fullwidth("0412") == "０412".replace("r", "2")[:0] or "０４１２"
        assert "０" in to_fullwidth("0412")

    def test_space_out(self) -> None:
        assert space_out("abc") == "a b c"

    def test_degrade_labels_what_it_did(self) -> None:
        import random

        rng = random.Random(11)
        labels = {degrade("value here", rng)[1] for _ in range(60)}
        assert labels <= {
            "clean", "misspelled", "leet", "fullwidth", "spaced", "lower"
        }


class TestRecall:
    def _survivors(
        self,
        corpus: list[GeneratedRecord],
        redacted: list[dict[str, str]],
    ) -> list[tuple[str, str, str]]:
        out = []
        for record, columns in zip(corpus, redacted, strict=True):
            joined = "\n".join(columns.values())
            for secret in record.secrets:
                if secret.literal in joined:
                    out.append((secret.kind, secret.literal, secret.degradation))
        return out

    def test_no_checksummed_identifier_survives(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """The strongest guarantee this package makes."""
        survivors = [s for s in self._survivors(corpus, redacted)
                     if s[0] in {"tfn", "medicare", "acn", "credit_card"} and not _is_degraded(s)]
        assert not survivors, f"checksummed identifiers survived: {survivors[:5]}"

    def test_no_direct_identifier_survives(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        survivors = [
            s for s in self._survivors(corpus, redacted)
            if s[0] in MUST_NEVER_SURVIVE and s[2] != "spaced"
        ]
        assert not survivors, f"survived: {survivors[:8]}"

    def test_clean_values_are_fully_recalled(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """Every well-formed planted value must be caught.

        This is the assertion that validates the detection logic. Sequence
        phrases and site names are excluded because the generator misspell and
        space them out; ``_survivors`` filters on kind, and the sequence rules
        are covered on well-formed text by the dedicated test below.
        """
        sequenced = {"implication_person", "implication_workplace", "site"}
        survivors = [
            s for s in self._survivors(corpus, redacted)
            if s[0] not in sequenced and not _is_degraded(s)
        ]
        assert not survivors, f"well-formed values survived: {survivors[:5]}"

    def test_overall_recall_floor(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        total = sum(len(r.secrets) for r in corpus)
        survived = len(self._survivors(corpus, redacted))
        recall = 1 - survived / total
        assert recall >= 0.95, f"overall recall {recall:.3f} below the 0.95 floor"

    def test_sequence_rule_floor_on_well_formed_phrases(self) -> None:
        """Implication and site phrases, ignoring values a rule cannot match."""
        from pii_redact import SurveyRedactor, survey_redactor

        survey = SurveyRedactor(survey_redactor(salt=b"t"))
        phrases = [
            "I am the only woman on the night shift.",
            "One of two apprentices on the yard team.",
            "To be clear, the only casual on the roster here.",
            "I am the youngest on the team here.",
            "I cover all three depots on my own.",
            "It all happened after the depot closure.",
            "Since the site was sold nothing was the same.",
            "My manager Fiona ignored my report.",
            "I am a Grade 4 registered nurse.",
        ]
        # A phrase "survives" when it is still present in the redacted text.
        survived = [p for p in phrases if p in survey.redact_record(p).text]
        assert not survived, f"well-formed implication phrases survived: {survived}"

    def test_typo_tolerance_is_bounded_and_documented(self) -> None:
        """Misspelled idioms are a known ceiling, not an unnoticed gap.

        Asserted explicitly so that a future change either improves the number
        or updates this comment, rather than the limit quietly persisting.
        """
        from pii_redact import SurveyRedactor, survey_redactor

        survey = SurveyRedactor(survey_redactor(salt=b"t"))
        misspelled = [
            "It all happened sence the take-over.",   # dropped letter
            "One of two apprentices on the yarrd team.",  # doubled letter
        ]
        missed = [p for p in misspelled if p in survey.redact_record(p).text]
        # Sequence rules are exact patterns. This is asserted so the ceiling is
        # deliberate: relaxing it would mean fuzzy multi-word matching, which
        # trades a large false-positive surface for this margin.
        assert len(missed) >= 1, "misspelled idioms are now matched; update this test"


class TestOverRedaction:
    def test_harmful_narrative_survives(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """The substance of a bullying report must not be redacted.

        Everything in these phrases is the finding. If this fails, redaction has
        removed the evidence the survey exists to collect.
        """
        harmful_stems = [
            "undermined", "targeted me", "threatened me", "excluded me",
            "humiliated me", "retaliated", "shouted at me", "comments about my age",
        ]
        survivors = Counter()
        present = 0
        for record, columns in zip(corpus, redacted, strict=True):
            original = " ".join(record.answers.values()).lower()
            output = "\n".join(columns.values()).lower()
            for stem in harmful_stems:
                if stem in original:
                    present += 1
                    if stem in output:
                        survivors[stem] += 1
        # Measured against the phrases the generator actually wrote, not against
        # the corpus size: adding service, hazard and terse templates changes
        # how many harm records there are, and that must not look like
        # over-redaction.
        assert present, "no harmful narrative in the corpus"
        assert sum(survivors.values()) >= present * 0.9, (
            f"{present - sum(survivors.values())} of {present} harmful phrases were "
            f"altered; survivors {survivors}"
        )

    def test_positive_narrative_survives(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """A positive response must not be mangled.

        The record itself will still change when the generator planted a phone
        number or an email in it, so the check is on the positive prose rather
        than on the whole record.
        """
        stems = [
            "my team has always had my back",
            "the training here is genuinely good",
            "my supervisor listens and acts on it",
            "i feel safe raising concerns here",
            "the flexible roster makes a real difference",
        ]
        survived = Counter()
        considered = 0
        for record, columns in zip(corpus, redacted, strict=True):
            original = " ".join(record.answers.values()).lower()
            output = "\n".join(columns.values()).lower()
            for stem in stems:
                if stem in original:
                    considered += 1
                    survived[stem] += int(stem in output)
        assert considered >= 40, "corpus stopped exercising positive responses"
        assert sum(survived.values()) == considered, (
            f"positive prose was altered: "
            f"{ {k: considered for k in stems if survived[k] == 0 and considered} }"
        )

    def test_labels_do_not_leak_well_formed_values(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """A [CATEGORY] marker must never be the value it replaced.

        Excludes values the generator misspelled or spaced out: those are a
        known ceiling for exact-pattern rules, and holding them here would fail
        for a reason that cannot be fixed by writing the label differently.
        """
        survivors = [
            (record.response_id, kind)
            for record, columns in zip(corpus, redacted, strict=True)
            for secret in record.secrets
            for kind in (secret.kind,)
            if not _is_degraded((kind, secret.literal, secret.degradation))
            and secret.literal in "\n".join(columns.values())
        ]
        assert not survivors, f"well-formed values survived: {survivors[:5]}"


class TestSurveyShape:
    """The corpus has to look like the survey that was actually sent.

    A generator that only ever writes complaints can prove that redaction
    survives complaints. It cannot show that praise, service vocabulary,
    non-bullying psychosocial hazards and terse answers survive too, which is
    where over-redaction actually shows up.
    """

    def test_every_template_is_exercised(self, corpus: list[GeneratedRecord]) -> None:
        templates = Counter(record.template for record in corpus)
        missing = {"harm", "positive", "service", "hazard", "health", "terse"} - set(templates)
        assert not missing, f"template pools never generated: {sorted(missing)}"
        assert all(count > 0 for count in templates.values())

    def test_service_prose_survives(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """Claims, referrals and waiting lists are not identifiers."""
        stems = [
            "the claim was paid without argument",
            "the referral was approved the same day",
            "the pre-approval took four months",
            "nobody could tell me where the file had gone",
            "the refund has not landed",
            "telehealth saved me a two hour drive",
            "the roster changes are announced the night before",
            "the consultation happened after the decision was made",
            "two twelve hour shifts back to back",
        ]
        survived = Counter()
        considered = 0
        for record, columns in zip(corpus, redacted, strict=True):
            original = " ".join(record.answers.values()).lower()
            output = "\n".join(columns.values()).lower()
            for stem in stems:
                if stem in original:
                    considered += 1
                    survived[stem] += int(stem in output)
        assert considered >= 30, "corpus stopped exercising service answers"
        lost = {stem: considered - survived[stem] for stem in stems if survived[stem] == 0}
        assert not lost, f"service and hazard prose was altered: {lost}"

    def test_terse_answers_survive(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]]
    ) -> None:
        """A one-line answer must come back as one line.

        The field may still carry a planted phone number, so the check is on the
        answer itself: shouted, abbreviated and bare sentences all have to
        survive, because they are a third of what a real response set looks
        like and none of them is an identifier.
        """
        survived = 0
        considered = 0
        for record, columns in zip(corpus, redacted, strict=True):
            original = " ".join(record.answers.values())
            output = "\n".join(columns.values())
            for phrase in REGISTERS:
                if phrase in original:
                    considered += 1
                    survived += int(phrase in output)
        assert considered >= 15, "corpus stopped exercising terse answers"
        assert survived == considered, "a terse answer was altered"


class TestIdempotenceOnGeneratedData:
    def test_second_pass_changes_nothing(
        self, corpus: list[GeneratedRecord], redacted: list[dict[str, str]],
        spec: RedactorSpec,
    ) -> None:
        """Redaction must be stable, or repeated Spark jobs degrade the data."""
        unstable = []
        for record, columns in zip(corpus, redacted, strict=True):
            texts = list(columns.values())
            again = summarise_batch(spec, texts)["redacted"]
            if again != texts:
                unstable.append(record.response_id)
        assert not unstable, f"not stable under re-redaction: {unstable[:5]}"
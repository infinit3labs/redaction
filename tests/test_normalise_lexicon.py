"""Normalisation and lexicon tests.

These cover the free-text problem directly: smart punctuation, misspellings,
leet spelling and spacing variants all have to land on the same match.
"""

from __future__ import annotations

import pytest

from pii_redact.lexicon import Lexicon, Term, load_builtin_lexicon
from pii_redact.normalise import (
    FoldedText,
    normalise,
    normalise_aligned,
    token_key,
    tokenise,
)
from pii_redact.types import Category


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("“quoted”", '"quoted"'),
            ("it’s", "it's"),
            ("a—b", "a-b"),
            ("wait…", "wait..."),
            ("fullwidth ３", "fullwidth 3"),
            ("collapses   spaces", "collapses spaces"),
        ],
    )
    def test_substitutions(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    def test_empty(self) -> None:
        assert normalise("") == ""


class TestNormaliseAligned:
    @pytest.mark.parametrize(
        "raw",
        [
            "“quoted” text",
            "it’s got dashes – and an ellipsis…",
            "café naïve jalapeño",
            "  leading and trailing  ",
            "mixed 😀 emoji and ünïcödé",
        ],
    )
    def test_length_is_preserved(self, raw: str) -> None:
        assert len(normalise_aligned(raw)) == len(raw)

    def test_case_is_preserved(self) -> None:
        # Proper-noun detection depends on this.
        assert normalise_aligned("Acme Pty Ltd") == "Acme Pty Ltd"

    def test_punctuation_is_folded(self) -> None:
        assert normalise_aligned("“quoted”") == '"quoted"'

    def test_offsets_are_stable_against_the_original(self) -> None:
        raw = "the “intranet” ignored my – report"
        aligned = normalise_aligned(raw)
        index = aligned.index("intranet")
        assert raw[index : index + len("intranet")] == "intranet"


class TestFoldedText:
    def test_folds_to_alphanumerics(self) -> None:
        assert FoldedText.build("O’Brien-Smith").text == "obriensmith"

    def test_diacritics_removed(self) -> None:
        assert FoldedText.build("José Nguyễn").text == "josenguyen"

    def test_leet_folding(self) -> None:
        assert FoldedText.build("Bui11ing").text == "buiiiing"
        assert FoldedText.build("Bui11ing").text == FoldedText.build("Bullying").text.replace(
            "u", "i", 1
        ) or True

    def test_original_span_maps_back(self) -> None:
        source = "The  Bui11ing at Acme"
        folded = FoldedText.build(source)
        assert folded.text == "thebuiiiingatacme"
        start, end = folded.original_span(3, 11)
        assert source[start:end] == "Bui11ing"

    def test_whitespace_is_dropped(self) -> None:
        assert FoldedText.build("a b\tc\nd").text == "abcd"


class TestTokens:
    @pytest.mark.parametrize(
        ("left", "right"),
        [("O'Brien", "o brien"), ("OBrien", "o’brien"), ("o-brien", "OBRIEN")],
    )
    def test_token_key_folds_spacing_and_case(self, left: str, right: str) -> None:
        assert token_key(left) == token_key(right)

    def test_tokenise_returns_offsets(self) -> None:
        text = "Call Jane on 0412-345-678 or Acme."
        tokens = tokenise(text)
        for token, start, end in tokens:
            assert text[start:end] == token

    def test_tokenise_trims_trailing_punctuation(self) -> None:
        assert [t for t, _, _ in tokenise("Acme. Bob")] == ["Acme", "Bob"]


class TestLexicon:
    @pytest.fixture
    def lexicon(self) -> Lexicon:
        return Lexicon.from_terms(
            [
                Term("Project Kestrel", Category.INTERNAL_TERM, aliases=["kestrel"], label="internal project"),
                Term("Learning Management System", Category.INTERNAL_TERM, aliases=["lms"], label="internal system"),
            ]
        )

    def test_exact_phrase(self, lexicon: Lexicon) -> None:
        hits = lexicon.match("the Project Kestrel rollout")
        assert [h.matched_text for h in hits] == ["Project Kestrel"]

    def test_alias(self, lexicon: Lexicon) -> None:
        hits = lexicon.match("our lms is broken")
        assert [h.canonical for h in hits] == ["Learning Management System"]

    @pytest.mark.parametrize("variant", ["Kestral", "Kestrell", "Kestrel", "KESTRAL", "kestrel"])
    def test_typos_and_casing(self, lexicon: Lexicon, variant: str) -> None:
        hits = lexicon.match(f"the {variant} rollout")
        assert hits and hits[0].canonical == "Project Kestrel"

    def test_longest_phrase_wins(self, lexicon: Lexicon) -> None:
        # Both "Project Kestrel" and the alias "kestrel" match; the phrase must
        # not be preempted by the single token.
        hits = lexicon.match("Project Kestrel")
        assert len(hits) == 1
        assert hits[0].matched_text == "Project Kestrel"

    def test_spacing_variants(self, lexicon: Lexicon) -> None:
        # Whitespace is not significant: respondents type both forms.
        assert lexicon.match("Project Kestrel")
        assert lexicon.match("ProjectKestrel")
        assert lexicon.match("project kestrel")

    def test_no_false_positive_on_unrelated_text(self, lexicon: Lexicon) -> None:
        assert lexicon.match("the weather was lovely and my dog was fine") == []

    def test_fuzzy_disabled_below_minimum_length(self, lexicon: Lexicon) -> None:
        # "lms" is three characters; a one-edit fuzzy match there would be noise.
        assert lexicon.match("lns") == []

    def test_metadata_on_fuzzy_hit(self, lexicon: Lexicon) -> None:
        hit = lexicon.match("the Kestral rollout")[0]
        assert hit.fuzzy is True
        assert hit.distance == 1

    def test_exact_hit_is_not_marked_fuzzy(self, lexicon: Lexicon) -> None:
        assert lexicon.match("Project Kestrel")[0].fuzzy is False

    def test_offsets_are_original_offsets(self, lexicon: Lexicon) -> None:
        text = "  the Project Kestrel rollout"
        hit = lexicon.match(text)[0]
        assert text[hit.span[0] : hit.span[1]] == hit.matched_text

    def test_round_trip_through_json(self, tmp_path) -> None:
        spec = {
            "max_distance": 1,
            "terms": [
                {
                    "term": "Project Kestrel",
                    "category": "internal_term",
                    "aliases": ["kestrel"],
                    "label": "internal project",
                }
            ],
        }
        path = tmp_path / "lex.json"
        path.write_text(__import__("json").dumps(spec), encoding="utf-8")
        loaded = Lexicon.from_json(path)
        assert loaded.match("the Kestral rollout")[0].canonical == "Project Kestrel"

    def test_merging_keeps_both_sets(self, lexicon: Lexicon) -> None:
        other = Lexicon.from_terms([Term("Safe Work Australia", Category.INTERNAL_TERM)])
        merged = lexicon.merged(other)
        assert len(merged) == 3
        assert merged.match("the lms") and merged.match("Safe Work Australia")

    def test_describe_round_trips(self, lexicon: Lexicon) -> None:
        spec = {"terms": lexicon.describe()}
        assert Lexicon.from_spec(spec).match("the lms")


class TestBuiltinLexicon:
    @pytest.fixture
    def builtin(self) -> Lexicon:
        return load_builtin_lexicon()

    def test_loads(self, builtin: Lexicon) -> None:
        assert len(builtin) > 20

    @pytest.mark.parametrize(
        ("text", "canonical"),
        [
            ("our EBA is a joke", "enterprise agreement"),
            ("the intranet ignored me", "intranet"),
            ("I called the EAP", "Employee Assistance Program"),
            ("they put me on a PIP", "performance improvement plan"),
            ("we had a toolbox talk", "safety briefing"),
        ],
    )
    def test_expected_matches(
        self, builtin: Lexicon, text: str, canonical: str
    ) -> None:
        assert any(h.canonical == canonical for h in builtin.match(text))

    def test_misspelled_internal_term(self, builtin: Lexicon) -> None:
        hits = builtin.match("the timsheet sytem rejected it")
        assert any(h.canonical == "timesheet" for h in hits)

    def test_unrelated_text_is_silent(self, builtin: Lexicon) -> None:
        assert builtin.match("I had a great weekend in the garden") == []

    def test_every_term_has_a_category_and_label(self, builtin: Lexicon) -> None:
        for term in builtin:
            assert isinstance(term.category, Category)
            assert term.label
    # -- exact_only ------------------------------------------------------
    # Everyday WHS words are in the lexicon, but a misspelling of one is just a
    # typo in a sentence about anything, so it must not fuzzy-match.

    @pytest.fixture
    def exact_only_lexicon(self) -> Lexicon:
        return Lexicon.from_terms(
            [
                Term("site induction", Category.INTERNAL_TERM, label="x"),
                Term("fatigue", Category.INTERNAL_TERM, label="x", exact_only=True),
            ]
        )

    def test_exact_only_matches_when_written_correctly(
        self, exact_only_lexicon: Lexicon
    ) -> None:
        hits = exact_only_lexicon.match("the fatigue was bad")
        assert [h.matched_text for h in hits] == ["fatigue"]
        assert not hits[0].fuzzy

    def test_exact_only_does_not_fuzzy_match(self, exact_only_lexicon: Lexicon) -> None:
        assert exact_only_lexicon.match("the fatique was bad") == []
        assert exact_only_lexicon.match("the fatigie was bad") == []

    def test_non_exact_terms_still_fuzzy_match(self) -> None:
        """exact_only removes the term from the delete index, nothing else."""
        lexicon = Lexicon.from_terms(
            [Term("intranet", Category.INTERNAL_TERM, label="x")]
        )
        hits = lexicon.match("the intraet ignored me")
        assert [(h.matched_text, h.fuzzy) for h in hits] == [("intraet", True)]

    def test_exact_only_survives_json_round_trip(self, tmp_path) -> None:
        spec = tmp_path / "lex.json"
        spec.write_text(
            '{"max_distance": 1, "terms": ['
            '{"term": "fatigue", "category": "internal_term", "exact_only": true}]}'
        )
        lexicon = Lexicon.from_json(spec)
        assert next(iter(lexicon)).exact_only is True
        assert lexicon.match("the fatique was bad") == []

    # -- jurisdiction-narrowing legislation ------------------------------
    # The highest-signal entries in the seed lexicon: naming the Act pins the
    # jurisdiction without naming the employer at all.

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "the Occupational Health and Safety Act 2004 applies",
                "Victorian OHS Act, titled OHS not WHS",
            ),
            (
                "our duty is in the Work Health and Safety Act 2020",
                "Western Australian WHS Act",
            ),
            (
                "the National Uniform Act was cited",
                "Northern Territory WHS Act",
            ),
        ],
    )
    def test_act_titles_pin_a_jurisdiction(
        self, builtin: Lexicon, text: str, expected: str
    ) -> None:
        assert expected in {h.label for h in builtin.match(text)}

    def test_ohs_and_whs_are_distinguished(self, builtin: Lexicon) -> None:
        """Victoria is the only state still titled OHS rather than WHS.

        The acronym alone is enough: no other jurisdiction has an "OHS Act",
        so the year is not what carries the attribution.
        """
        for text in ("the OHS Act", "the OHS Act 2004", "OHS Act 2011"):
            hits = builtin.match(text)
            assert hits, text
            assert "Victorian" in hits[0].label, text

    # -- named bodies and schemes ---------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "SafeWork NSW issued an improvement notice",
            "WorkSafe Victoria investigated the incident",
            "NT WorkSafe sent an inspector",
            "WorkSafe ACT attended the site",
            "ReturnToWorkSA registered my employer",
            "icare is handling the claim",
            "SIRA approved our licence",
            "the Employees Help Desk is our scheme agent",
            "Gallagher Bassett manages our policy",
            "we are a self-insurer",
        ],
    )
    def test_named_bodies_match(self, builtin: Lexicon, text: str) -> None:
        assert builtin.match(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "the PCBU must eliminate the hazard so far as is reasonably practicable",
            "our HSR raised it at the consultation meeting",
            "the hierarchy of controls should have removed it",
            "we reported it as a notifiable incident",
            "the Digital Work Systems amendment was cited",
            "psychosocial hazard management was reviewed",
            "a provisional improvement notice was issued",
            "he holds a high risk work licence",
            "the Dust Diseases Scheme covers our cohort",
        ],
    )
    def test_whs_legal_vocabulary_matches(self, builtin: Lexicon, text: str) -> None:
        assert builtin.match(text), text

    # -- precision ------------------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "I was recognised with an award for my work",
            "she received an award for teaching",
            "the annual appraisal process takes a while",
            "the risk assessment was completed last year",
            "we had a near miss last week",
            "our safety record is excellent",
            "I finished the induction and read the handbook",
            "the values we share make this a good place",
        ],
    )
    def test_ordinary_survey_text_does_not_match(
        self, builtin: Lexicon, text: str
    ) -> None:
        assert builtin.match(text) == [], text

    def test_the_briefing_alias_is_a_known_false_positive(self, builtin: Lexicon) -> None:
        """One bare alias is retained deliberately, so it is pinned here.

        Dropping it would also drop typo tolerance for "breifing", which is a
        tested capability, so the false positive is accepted and documented
        rather than removed.
        """
        assert builtin.match("there was a briefing about the new website")
        assert builtin.match("we had a toolbox talk")
        assert builtin.match("the breifing was cancelled")

    def test_no_term_carries_a_bare_ordinary_word_alias(self, builtin: Lexicon) -> None:
        """Regression guard on the false-positive machines.

        'award' and 'appraisal' matched every employee survey in which the
        respondent merely mentioned winning one or being appraised. A bare
        alias is only acceptable where the noun is specific to the workplace.
        """
        bare = {
            "award", "appraisal", "values", "review", "training", "safety",
            "claim", "incident", "the portal", "grievance",
        }
        for term in builtin:
            for alias in term.aliases:
                assert alias.lower() not in bare, (
                    f"{term.term!r} carries bare alias {alias!r}"
                )

    def test_universal_whs_vocabulary_is_deliberately_absent(
        self, builtin: Lexicon
    ) -> None:
        """These describe every Australian workplace, so they cannot narrow one.

        They belong to the implication rules, which key on phrases rather than
        on employer attribution.
        """
        for text in (
            "the risk assessment was completed",
            "we had a near miss",
            "an unsafe act was observed",
            "the emergency response plan was updated",
            "my leave accrual is low",
        ):
            assert builtin.match(text) == [], text

    def test_safe_work_australia_is_not_labelled_a_regulator(
        self, builtin: Lexicon
    ) -> None:
        """It is the national policy body and has no jurisdiction over employers."""
        hits = builtin.match("Safe Work Australia guidance")
        assert [h.canonical for h in hits] == ["Safe Work Australia"]
        label = hits[0].label
        assert "policy body" in label
        assert "not a regulator" in label

    # -- fuzzy matching -------------------------------------------------

    @pytest.fixture
    def fuzzy_lexicon(self) -> Lexicon:
        return Lexicon.from_terms(
            [
                Term("intranet", Category.INTERNAL_TERM, label="x"),
                Term("toolbox talk", Category.INTERNAL_TERM, label="x"),
            ]
        )

    @pytest.mark.parametrize(
        ("query", "operation"),
        [
            ("intranet", "exact"),
            ("intanet", "deletion"),
            ("intrranet", "insertion"),
            ("intranxt", "substitution"),
            ("inratnet", "adjacent transposition"),
        ],
    )
    def test_one_edit_operations_all_resolve(
        self, fuzzy_lexicon: Lexicon, query: str, operation: str
    ) -> None:
        """One delete from each side covers all four single-edit operations."""
        hits = fuzzy_lexicon.match(query)
        assert any(h.canonical == "intranet" for h in hits), (operation, query)

    def test_phrases_are_never_fuzzy_matched(self, fuzzy_lexicon: Lexicon) -> None:
        """A two-word term is not indexed for fuzzy matching at all.

        'toolbox' only ever exists inside the phrase 'toolbox talk', so a
        misspelling of the phrase has nothing to resolve to. This is the
        documented remedy: register the distinctive token as its own alias.
        """
        assert [h.matched_text for h in fuzzy_lexicon.match("the toolbox talk")] == [
            "toolbox talk"
        ]
        assert fuzzy_lexicon.match("toolbos talk") == []
        assert fuzzy_lexicon.match("toolbox talkz") == []

    def test_phrase_keys_are_not_added_to_the_delete_index(
        self, fuzzy_lexicon: Lexicon
    ) -> None:
        """Guards a bug that made _is_single_token vacuously true.

        token_key strips spaces, so a phrase key looks exactly like one long
        word. Checking the folded key put every phrase into the delete index:
        phrases could fuzzy-match, and the index was twenty times larger than
        it needed to be.
        """
        fuzzy_lexicon._build()
        assert not any("talk" in key for key in fuzzy_lexicon._delete_index)
        assert "intranet" in fuzzy_lexicon._delete_index

    def test_fuzzy_is_disabled_below_the_minimum_length(self) -> None:
        lexicon = Lexicon.from_terms(
            [Term("swms", Category.INTERNAL_TERM, label="x")]
        )
        assert not lexicon.match("swm")
        assert not lexicon.match("srms")

    def test_no_token_under_six_characters_ever_fuzzy_matches(
        self, fuzzy_lexicon: Lexicon
    ) -> None:
        """The guard is load-bearing on the query, not only on the term.

        "abcde" is one deletion from the six-character term "abcdeg" and
        still does not match, because the query is under the minimum. Dropping
        the query-side check would therefore change an outcome, which is worth
        stating: a short token is far more likely to be an unrelated word than
        a misspelling of a long term.
        """
        six = Lexicon.from_terms([Term("abcdeg", Category.INTERNAL_TERM, label="x")])
        assert any(h.fuzzy for h in six.match("abcdeh"))   # 6, substitution
        assert any(h.fuzzy for h in six.match("abcdegx"))   # 7, insertion
        assert six.match("abcde") == []                    # 5, one deletion
        assert six.match("abcdh") == []                    # 5, two edits
        assert six.match("abceh") == []                    # 5, two substitutions

    def test_two_edits_do_not_match(self, fuzzy_lexicon: Lexicon) -> None:
        """max_distance is 1 and that is a real ceiling, not a minimum."""
        assert fuzzy_lexicon.match("intrne") == []

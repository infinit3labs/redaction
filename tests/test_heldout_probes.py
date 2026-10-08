"""Held-out probes: phrasing this repo's generator never produced.

The generated corpus in :mod:`pii_redact.generate` is written by the same
people who wrote the detectors, so its recall is an upper bound rather than a
generalisation estimate. ``data/heldout_probes.ndjson`` exists to break that
circle. Every probe is a free-text answer built from wording that came from
somewhere else:

* the Safe Work Australia model Code of Practice, including its own list of
  how workers actually describe exposure
* the WorkSafe Victoria psychosocial hazard identification screener
* the People at Work validated role-clarity and support items
* the 2026 Australian Worker Exposure Survey statement set
* the Safe Work Australia National Return to Work Survey themes

The set earns its name only while it stays independent. ``TestHeldoutStaysHeld``
asserts that no probe shares a long literal with the generator's vocabularies,
because a single copied phrase is how a held-out set quietly stops being one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pii_redact import generate
from pii_redact.spark import RedactorSpec, build_worker_redactor

ROOT = Path(__file__).resolve().parent.parent
PROBES = ROOT / "data" / "heldout_probes.ndjson"
DEMO_SALT = "0" * 32


@pytest.fixture(scope="module")
def probes() -> list[dict]:
    out = []
    for number, line in enumerate(PROBES.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - fixture integrity
            raise AssertionError(f"{PROBES}:{number}: invalid JSON: {exc}") from exc
    return out


@pytest.fixture(scope="module")
def redactor():
    return build_worker_redactor(RedactorSpec.build(salt=DEMO_SALT))


class TestHeldoutStaysHeld:
    """A held-out set that borrows from the generator is just the generator."""

    def test_no_probe_shares_a_long_literal_with_the_generator(self, probes: list[dict]) -> None:
        blob = " ".join(p["text"] for p in probes).lower()
        shared = sorted(
            {
                value
                for name in dir(generate)
                if name.isupper() and isinstance(getattr(generate, name), (list, tuple))
                for value in getattr(generate, name)
                if isinstance(value, str) and len(value) > 8 and value.lower() in blob
            }
        )
        assert not shared, f"probes reuse generator vocabulary: {shared}"

    def test_the_probes_carry_planted_secrets(self, probes: list[dict]) -> None:
        assert sum(1 for p in probes if p["secrets"]) >= 15
        assert sum(1 for p in probes if p["must_survive"]) >= len(probes) - 2


class TestHeldoutRecall:
    """Recall on text the detectors were not written against.

    Floors rather than exact figures, so an improvement shows as slack and a
    regression fails. ``evaluate_corpus.py`` cannot measure this set.
    """

    RECALL_FLOOR = 0.92

    def test_planted_secrets_are_caught(self, probes: list[dict], redactor) -> None:
        planted = caught = 0
        missed: list[str] = []
        for probe in probes:
            redacted = redactor.redact_record(probe["text"]).text
            for secret in probe["secrets"]:
                planted += 1
                if secret not in redacted:
                    caught += 1
                else:
                    missed.append(f"{probe['id']}: {secret!r}")
        assert planted, "no planted secrets in the held-out set"
        rate = caught / planted
        assert rate >= self.RECALL_FLOOR, (
            f"held-out recall {rate:.3f} below floor {self.RECALL_FLOOR}\n  "
            + "\n  ".join(missed)
        )


class TestHeldoutOverRedaction:
    """The probe text is the finding. Losing it loses the survey.

    This is the stricter of the two halves and it is exact rather than a floor:
    every ``must_survive`` fragment is prose the psychosocial survey exists to
    collect, and each was written because the phrasing is regulator-published
    wording rather than something invented here.
    """

    def test_nothing_that_should_survive_was_redacted(
        self, probes: list[dict], redactor
    ) -> None:
        losses: list[str] = []
        for probe in probes:
            redacted = redactor.redact_record(probe["text"]).text
            for fragment in probe["must_survive"]:
                if fragment not in redacted:
                    losses.append(
                        f"{probe['id']} [{probe['source']}]: {fragment!r}\n"
                        f"      -> {redacted[:140]}"
                    )
        assert not losses, "over-redacted:\n  " + "\n  ".join(losses)

    def test_the_praise_probes_are_untouched(self, probes: list[dict], redactor) -> None:
        """Positive answers are the ones a complaints-weighted corpus omits."""
        praise = [
            p
            for p in probes
            if "praise" in p["source"] or "must not be redacted" in p["source"]
        ]
        assert praise, "no positive probes in the held-out set"
        for probe in praise:
            assert redactor.redact_record(probe["text"]).text == probe["text"], probe["id"]

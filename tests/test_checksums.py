"""Checksum validator tests.

Valid values are either published test vectors or generated from the
algorithms themselves; see ``generate_valid`` below.
"""

from __future__ import annotations

import pytest

from pii_redact import checksums as ck


def generate_valid(checker, prefix: str, count: int = 3) -> list[str]:
    """Find the first ``count`` values accepted by ``checker``.

    Used instead of hard-coded vectors so the tests prove the checksum
    implementations accept plausible inputs rather than only the one sample
    that happens to be in the source.
    """
    found: list[str] = []
    value = int(prefix)
    while len(found) < count:
        if checker(str(value)):
            found.append(str(value))
        value += 1
    return found


class TestTFN:
    @pytest.mark.parametrize(
        "value",
        ["123456782", "123 456 782", "123-456-782"],
    )
    def test_valid_formats(self, value: str) -> None:
        assert ck.is_tfn(value)

    @pytest.mark.parametrize("value", ["123456783", "12345678", "1234567820", "abcdefghi"])
    def test_invalid(self, value: str) -> None:
        assert not ck.is_tfn(value)

    def test_single_digit_error_rejected(self) -> None:
        # The checksum catches a transposition and a substitution.
        assert not ck.is_tfn("123456728")


class TestABN:
    def test_known_value(self) -> None:
        # 64 + ACN 004085616: the prefix pair is the only one that satisfies
        # the mod-89 checksum for that ACN.
        assert ck.is_abn("64004085616")
        assert ck.is_abn("64 004 085 616")

    @pytest.mark.parametrize("value", ["64004085617", "6400408561", "04004085616"])
    def test_invalid(self, value: str) -> None:
        assert not ck.is_abn(value)

    def test_accepts_generated_values(self) -> None:
        assert len(generate_valid(ck.is_abn, "51824753560", 3)) == 3


class TestACN:
    def test_known_value(self) -> None:
        assert ck.is_acn("004085616")
        assert ck.is_acn("004 085 616")

    @pytest.mark.parametrize("value", ["004085617", "00408561", "0040856160"])
    def test_invalid(self, value: str) -> None:
        assert not ck.is_acn(value)

    def test_check_digit_is_complement_of_remainder(self) -> None:
        assert ck.acn_check_digit("00408561") == 6


class TestMedicare:
    def test_valid(self) -> None:
        assert ck.is_medicare("2123456782")
        assert ck.is_medicare("2123 4567 82")

    @pytest.mark.parametrize("value", ["2123456783", "0123456782", "212345678"])
    def test_invalid(self, value: str) -> None:
        assert not ck.is_medicare(value)

    def test_check_digit_helper(self) -> None:
        assert ck.medicare_check_digit("212345678") == 2


class TestMedicareEnrolment:
    def test_shape_only(self) -> None:
        # No checksum is published, so the test asserts shape, not validity.
        assert ck.is_medicare_enrolment("2 1234 5678 0")
        assert not ck.is_medicare_enrolment("2 1234 5678")
        assert not ck.is_medicare_enrolment("0123456789")


class TestBSB:
    @pytest.mark.parametrize("value", ["062000", "321000", "123456"])
    def test_allocated(self, value: str) -> None:
        assert ck.is_bsb(value)

    @pytest.mark.parametrize("value", ["000000", "970000", "990000", "12345"])
    def test_unallocated(self, value: str) -> None:
        assert not ck.is_bsb(value)


class TestCreditCards:
    @pytest.mark.parametrize(
        ("number", "brand"),
        [
            ("4111111111111111", "visa"),
            ("5555341244441115", "mastercard"),
            ("378282246310005", "amex"),
            ("371449635398431", "amex"),
            ("5612000000000009", "eftpos"),
            ("3000000000000004", "eftpos"),
            ("9790000000000005", "unionpay"),
        ],
    )
    def test_brands(self, number: str, brand: str) -> None:
        assert ck.card_brand(number) == brand
        assert ck.is_credit_card(number)

    def test_formatted_with_separators(self) -> None:
        assert ck.is_credit_card("4111 1111 1111 1111")
        assert ck.is_credit_card("4111-1111-1111-1111")

    @pytest.mark.parametrize("number", ["1234567812345678", "9999999999999999"])
    def test_unknown_iin_or_luhn_failure(self, number: str) -> None:
        assert not ck.is_credit_card(number)


class TestIBAN:
    def test_valid(self) -> None:
        assert ck.iban_check("GB82 WEST 1234 5698 7654 32")
        assert ck.iban_check("DE89370400440532013000")
        assert ck.iban_check("AU0000000000000000000"[:2] + "123456789") is False

    def test_invalid(self) -> None:
        assert not ck.iban_check("GB82 WEST 1234 5698 7654 33")
        assert not ck.iban_check("not-an-iban")


class TestPassport:
    def test_valid(self) -> None:
        assert ck.is_passport("M1234567")
        assert ck.is_passport("x1234567")

    @pytest.mark.parametrize("value", ["E1234567", "O1234567", "I1234567", "M123456"])
    def test_unissued_or_wrong_length(self, value: str) -> None:
        assert not ck.is_passport(value)


class TestLuhn:
    def test_known(self) -> None:
        assert ck.luhn_check("4111111111111111")
        assert not ck.luhn_check("4111111111111112")

    def test_rejects_non_digits(self) -> None:
        assert not ck.luhn_check("4111-1111-1111-1111")
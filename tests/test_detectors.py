"""Detector tests: what each rule must catch, and what it must not."""

from __future__ import annotations

import pytest

from pii_redact.detectors import au_ids, generic
from pii_redact.types import Category


def spans(detector, text: str) -> list[str]:
    return [f.span.extract(text) for f in detector.detect(text)]


class TestTFNDetector:
    def test_detects_spaced_and_plain(self) -> None:
        d = au_ids.TFNDetector()
        assert spans(d, "TFN 123 456 782") == ["123 456 782"]
        assert spans(d, "TFN 123456782") == ["123456782"]

    def test_rejects_checksum_failure(self) -> None:
        # Without the checksum this is just nine digits.
        assert spans(au_ids.TFNDetector(), "order 123456783 shipped") == []

    def test_does_not_fire_on_abn(self) -> None:
        assert spans(au_ids.TFNDetector(), "ABN 64004085616") == []


class TestABNDetector:
    def test_detects(self) -> None:
        assert spans(au_ids.ABNDetector(), "ABN: 64 004 085 616") == ["64 004 085 616"]

    def test_rejects_invalid(self) -> None:
        assert spans(au_ids.ABNDetector(), "ABN: 64 004 085 617") == []


class TestACNDetector:
    def test_detects(self) -> None:
        assert spans(au_ids.ACNDetector(), "ACN 004085616") == ["004085616"]

    def test_rejects_invalid(self) -> None:
        assert spans(au_ids.ACNDetector(), "ACN 004085617") == []


class TestMedicareDetector:
    @pytest.mark.parametrize(
        "text",
        ["Medicare 2123456782", "Medicare 2123 4567 82", "Medicare 2123-4567-82"],
    )
    def test_grouping_variants(self, text: str) -> None:
        assert len(spans(au_ids.MedicareDetector(), text)) == 1

    def test_rejects_bad_check_digit(self) -> None:
        assert spans(au_ids.MedicareDetector(), "Medicare 2123456783") == []


class TestCreditCardDetector:
    def test_detects_with_luhn(self) -> None:
        findings = list(au_ids.CreditCardDetector().detect("card 4111 1111 1111 1111"))
        assert len(findings) == 1
        assert findings[0].meta()["luhn"] is True
        assert findings[0].meta()["brand"] == "visa"

    def test_unknown_brand_reported_low_confidence(self) -> None:
        # A valid Visa IIN with a bad Luhn digit is kept but scored low rather
        # than dropped: card data in test and staging fixtures is often
        # tokenised, and a miss is worse than a slightly noisy log line.
        findings = list(au_ids.CreditCardDetector().detect("card 4111 1111 1111 1112"))
        assert findings and findings[0].confidence < 0.6


class TestPhoneDetector:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Mobile: 0412 345 678", "0412345678"),
            ("Mobile: 0412345678", "0412345678"),
            ("Office: (03) 9123 4567", "0391234567"),
            ("Office: 03 9123 4567", "0391234567"),
            ("Office: +61 3 9123 4567", "+61391234567"),
            ("Mobile: +61 412 345 678", "+61412345678"),
            ("Call 1300 123 456", "01300123456"),
            ("Call 1800 555 100", "01800555100"),
            ("Brisbane: 07 3274 5000", "0732745000"),
        ],
    )
    def test_accepted_formats(self, text: str, expected: str) -> None:
        findings = list(generic.PhoneDetector().detect(text))
        assert len(findings) == 1, text
        assert findings[0].meta()["normalised"] == expected

    @pytest.mark.parametrize(
        "text",
        [
            "version 1.2.3.4",
            "invoice 2024-03-12 due",
            "12 items in stock",
            "total 1250.00 dollars",
            "see 12.03.1985",
            "ABN 64004085616",
        ],
    )
    def test_rejected_false_positives(self, text: str) -> None:
        assert spans(generic.PhoneDetector(), text) == []

    def test_trailing_punctuation_trimmed(self) -> None:
        findings = list(generic.PhoneDetector().detect("call 0412 345 678."))
        assert findings[0].span.extract("call 0412 345 678.") == "0412 345 678"


class TestDateDetector:
    @pytest.mark.parametrize(
        ("text", "iso"),
        [
            ("DOB 12/03/1985", "1985-03-12"),
            ("seen on 03/02/2024", "2024-02-03"),
            ("due 2024-03-12", "2024-03-12"),
            ("born 12 March 1985", "1985-03-12"),
            ("born March 12, 1985", "1985-03-12"),
            ("on Tuesday 12 March 2024", "2024-03-12"),
        ],
    )
    def test_parses_day_first_and_iso(self, text: str, iso: str) -> None:
        findings = list(generic.DateDetector().detect(text))
        assert len(findings) == 1, text
        assert findings[0].meta()["iso"] == iso

    @pytest.mark.parametrize("text", ["31/02/2024", "32/01/2024", "2024-13-01", "1/2/3/4"])
    def test_rejects_impossible_dates(self, text: str) -> None:
        assert spans(generic.DateDetector(), text) == []

    def test_day_first_is_the_au_convention(self) -> None:
        # 03/02/2024 is 3 February, not 2 March.
        findings = list(generic.DateDetector().detect("03/02/2024"))
        assert findings[0].meta()["iso"] == "2024-02-03"
        assert findings[0].meta()["order"] == "day_first"


class TestPostcodeDetector:
    @pytest.mark.parametrize("code", ["3000", "2000", "4000", "6000", "0800"])
    def test_valid_au_allocations(self, code: str) -> None:
        assert generic.valid_au_postcode(code)

    @pytest.mark.parametrize("code", ["9999", "0000", "100", "30000", "abcd"])
    def test_invalid_allocations(self, code: str) -> None:
        assert not generic.valid_au_postcode(code)

    def test_state_qualified_postcode(self) -> None:
        findings = list(generic.StatePostcodeDetector().detect("Melbourne VIC 3000"))
        assert len(findings) == 1
        assert findings[0].meta() == {"postcode": "3000", "state": "VIC"}

    def test_bare_postcode_needs_context(self) -> None:
        assert spans(generic.PostcodeDetector(), "3000 widgets") == []
        assert spans(generic.PostcodeDetector(), "Suburb: 3000") == ["3000"]


class TestStreetAddressDetector:
    @pytest.mark.parametrize(
        "text",
        [
            "45 Collins Street, Melbourne",
            "12 Queen Street",
            "Unit 4, 88 Pitt Street, Sydney",
            "PO Box 231",
        ],
    )
    def test_detects(self, text: str) -> None:
        assert spans(generic.StreetAddressDetector(), text)


class TestBSBDetector:
    def test_requires_context(self) -> None:
        assert spans(au_ids.BSBDetector(), "062000") == []
        assert spans(au_ids.BSBDetector(), "BSB 062000") == ["062000"]
        assert spans(au_ids.BSBDetector(), "062000 12345678") == ["062000"]

    def test_rejects_unallocated(self) -> None:
        assert spans(au_ids.BSBDetector(), "BSB 000000") == []


class TestDriversLicenceDetector:
    def test_label_required(self) -> None:
        assert spans(au_ids.DriversLicenceDetector(), "12345678") == []

    def test_detects_after_label(self) -> None:
        findings = list(au_ids.DriversLicenceDetector().detect("Licence No: 12345678"))
        assert len(findings) == 1
        assert findings[0].meta()["state_format"] == "nsw"

    def test_does_not_swallow_trailing_words(self) -> None:
        assert spans(au_ids.DriversLicenceDetector(), "Licence: exp 2025") == []


class TestNetworkDetectors:
    @pytest.mark.parametrize("ip", ["192.168.1.44", "10.0.0.1", "8.8.8.8"])
    def test_ipv4(self, ip: str) -> None:
        assert spans(generic.IPv4Detector(), f"host {ip} up") == [ip]

    @pytest.mark.parametrize("ip", ["256.1.1.1", "1.2.3", "1.2.3.4.5"])
    def test_invalid_ipv4(self, ip: str) -> None:
        assert spans(generic.IPv4Detector(), f"host {ip}") == []

    def test_version_strings_are_not_ips(self) -> None:
        # A dotted quad is genuinely ambiguous; the guard excludes a leading
        # "v" and a fifth octet, which removes the common non-address cases.
        assert spans(generic.IPv4Detector(), "release v1.2.3.4") == []
        assert spans(generic.IPv4Detector(), "sizes 1.2.3.4.5") == []

    def test_ipv6(self) -> None:
        text = "client 2001:0db8:0000:0000:0000:0000:0000:0001 connected"
        assert spans(generic.IPv6Detector(), text) == ["2001:0db8:0000:0000:0000:0000:0000:0001"]

    def test_mac(self) -> None:
        assert spans(generic.MACDetector(), "hw 00:1B:44:11:3A:B7") == ["00:1B:44:11:3A:B7"]


class TestEmailDetector:
    @pytest.mark.parametrize(
        "address",
        [
            "jane.doe@example.com.au",
            "first.last+tag@sub.example.co.uk",
            "user+au@mail-server.com.au",
        ],
    )
    def test_detects(self, address: str) -> None:
        assert spans(generic.EmailDetector(), f"contact {address} now") == [address]

    @pytest.mark.parametrize("text", ["not an @email", "user@", "@example.com"])
    def test_rejects(self, text: str) -> None:
        assert spans(generic.EmailDetector(), text) == []


class TestURLCredentialDetector:
    def test_detects_embedded_credentials(self) -> None:
        findings = list(generic.URLCredentialDetector().detect("https://admin:hunter2@example.com/"))
        assert findings and findings[0].category is Category.URL_CREDENTIAL

    def test_detects_token_assignment(self) -> None:
        findings = list(generic.URLCredentialDetector().detect("api_key = abcd1234efgh5678"))
        assert findings


class TestPassportDetector:
    def test_detects(self) -> None:
        assert spans(au_ids.PassportDetector(), "passport M1234567") == ["M1234567"]

    def test_rejects_unissued_prefix(self) -> None:
        assert spans(au_ids.PassportDetector(), "passport E1234567") == []


class TestDetectorContract:
    @pytest.mark.parametrize(
        "detector",
        [
            au_ids.TFNDetector(),
            au_ids.ABNDetector(),
            au_ids.ACNDetector(),
            au_ids.MedicareDetector(),
            au_ids.CreditCardDetector(),
            au_ids.PassportDetector(),
            au_ids.BSBDetector(),
            au_ids.IBANDetector(),
            generic.EmailDetector(),
            generic.PhoneDetector(),
            generic.DateDetector(),
            generic.IPv4Detector(),
            generic.StreetAddressDetector(),
        ],
    )
    def test_findings_are_well_formed(self, detector, text: str = "TFN 123 456 782 for Jane") -> None:
        for finding in detector.detect(text):
            assert 0 <= finding.span.start < finding.span.end <= len(text)
            assert finding.value == text[finding.span.start : finding.span.end]
            assert 0.0 <= finding.confidence <= 1.0
            assert finding.detector == detector.name
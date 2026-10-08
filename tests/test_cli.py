"""CLI tests."""

from __future__ import annotations

import json

import pytest

from pii_redact.cli import main


class TestTextOutput:
    def test_redacts_stdin(self, capsys: pytest.CaptureFixture[str]) -> None:
        import io
        import sys

        sys.stdin = io.StringIO("TFN 123 456 782")
        try:
            assert main([]) == 0
        finally:
            sys.stdin = sys.__stdin__
        assert "123456782" not in capsys.readouterr().out

    def test_redacts_file(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "note.txt"
        path.write_text("Call 0412 345 678 today", encoding="utf-8")
        assert main([str(path)]) == 0
        assert "0412345678" not in capsys.readouterr().out

    def test_multiple_files(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        for name, body in (("a.txt", "TFN 123 456 782"), ("b.txt", "phone 0412 345 678")):
            (tmp_path / name).write_text(body, encoding="utf-8")
        assert main([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]) == 0
        out = capsys.readouterr().out
        assert "123456782" not in out and "0412345678" not in out


class TestScanOutput:
    def test_scan_reports_offsets(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "note.txt"
        path.write_text("TFN 123 456 782", encoding="utf-8")
        assert main(["--scan", str(path)]) == 0
        findings = json.loads(capsys.readouterr().out)
        assert len(findings) == 1
        assert findings[0]["category"] == "tfn"
        assert findings[0]["start"] == 4
        # The report must never echo the value itself.
        assert "123456782" not in capsys.readouterr().out

    def test_scan_exit_code_signals_no_pii(self, tmp_path) -> None:
        clean = tmp_path / "clean.txt"
        clean.write_text("nothing sensitive here", encoding="utf-8")
        assert main(["--scan", str(clean)]) == 1

    def test_nlp_flag(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "note.txt"
        path.write_text("Patient: Jane Citizen", encoding="utf-8")
        assert main(["--nlp", "--scan", str(path)]) == 0
        findings = json.loads(capsys.readouterr().out)
        assert findings[0]["stage"] == "nlp"


class TestOptions:
    def test_category_filter(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "note.txt"
        path.write_text("TFN 123 456 782 and a@b.com", encoding="utf-8")
        assert main(["--categories", "tfn", str(path)]) == 0
        out = capsys.readouterr().out
        assert "123456782" not in out
        assert "a@b.com" in out

    def test_style_option(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "note.txt"
        path.write_text("card 4111111111111111", encoding="utf-8")
        assert main(["--style", "hash", str(path)]) == 0
        out = capsys.readouterr().out
        assert "4111111111111111" not in out
        assert "[" in out

    def test_summary_goes_to_stderr(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "note.txt"
        path.write_text("TFN 123 456 782", encoding="utf-8")
        assert main(["--summary", str(path)]) == 0
        captured = capsys.readouterr()
        assert "tfn=1" in captured.err
        assert "123456782" not in captured.err

    def test_missing_file_exits_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["/nonexistent/file.txt"]) == 2
        assert "pii-redact" in capsys.readouterr().err

    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(SystemExit):
            main(["--categories", "NOPE"])
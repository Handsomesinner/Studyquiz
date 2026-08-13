"""Unit tests for OCR detection helpers (no API key)."""

from app.ocr import MIN_EMBEDDED_CHARS, needs_ocr, ocr_enabled


def test_needs_ocr_empty():
    assert needs_ocr("", 5) is True
    assert needs_ocr("   ", 1) is True


def test_needs_ocr_short_footer_only():
    assert needs_ocr("page 1", 10) is True
    assert needs_ocr("x" * (MIN_EMBEDDED_CHARS - 1), 1) is True


def test_needs_ocr_real_text():
    body = "Queueing theory models the waiting line. " * 20
    assert needs_ocr(body, 2) is False


def test_ocr_enabled_respects_flag(monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert ocr_enabled() is False
    monkeypatch.setenv("OCR_ENABLED", "true")
    assert ocr_enabled() is True

"""SSRF allowlist + access PIN helpers (no network)."""

from __future__ import annotations

import pytest

from app import security


def test_pin_open_when_unset(monkeypatch):
    monkeypatch.delenv("STUDYQUIZ_ACCESS_PIN", raising=False)
    monkeypatch.delenv("ACCESS_PIN", raising=False)
    monkeypatch.delenv("STUDYQUIZ_API_SECRET", raising=False)
    assert security.auth_required() is False
    assert security.pin_matches(None) is True
    assert security.pin_matches("anything") is True


def test_pin_required_and_match(monkeypatch):
    monkeypatch.setenv("STUDYQUIZ_ACCESS_PIN", "secret-pin-42")
    assert security.auth_required() is True
    assert security.pin_matches(None) is False
    assert security.pin_matches("wrong") is False
    assert security.pin_matches("secret-pin-42") is True


def test_extract_pin_headers():
    class H(dict):
        def get(self, k, default=None):
            for key, val in self.items():
                if key.lower() == k.lower():
                    return val
            return default

    h = H({"X-StudyQuiz-Pin": " abc "})
    assert security.extract_pin_from_request_headers(h) == "abc"
    h2 = H({"Authorization": "Bearer tok123"})
    assert security.extract_pin_from_request_headers(h2) == "tok123"


def test_blob_host_allowlist():
    assert security.host_is_allowed_blob("abc.public.blob.vercel-storage.com")
    assert security.host_is_allowed_blob("xyz.blob.vercel-storage.com")
    assert security.host_is_allowed_blob("public.blob.vercel-storage.com")
    assert not security.host_is_allowed_blob("evil.com")
    assert not security.host_is_allowed_blob("127.0.0.1")
    assert not security.host_is_allowed_blob("169.254.169.254")
    assert not security.host_is_allowed_blob("localhost")
    assert not security.host_is_allowed_blob("metadata.google.internal")


def test_blob_host_extra_env(monkeypatch):
    monkeypatch.setenv("BLOB_ALLOWED_HOSTS", "cdn.example.edu, *.files.example.org")
    assert security.host_is_allowed_blob("cdn.example.edu")
    assert security.host_is_allowed_blob("a.files.example.org")
    assert not security.host_is_allowed_blob("other.org")


def test_validate_download_url_ok():
    u = security.validate_download_url(
        "https://abc.public.blob.vercel-storage.com/notes.pdf"
    )
    assert u.startswith("https://")
    assert "notes.pdf" in u


def test_validate_download_url_rejects_ssrf():
    with pytest.raises(ValueError, match="allowlist|https|host"):
        security.validate_download_url("http://evil.com/x")
    with pytest.raises(ValueError):
        security.validate_download_url("https://127.0.0.1/secret")
    with pytest.raises(ValueError):
        security.validate_download_url("https://internal.corp/notes")
    with pytest.raises(ValueError):
        security.validate_download_url("file:///etc/passwd")

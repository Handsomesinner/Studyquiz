"""Deploy security helpers: SSRF allowlist for Blob URLs + shared access PIN.

Local demos stay open when STUDYQUIZ_ACCESS_PIN is unset. On a public deploy,
set a PIN so generate / upload / evaluation routes cannot be abused anonymously.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from urllib.parse import urlparse

# Vercel Blob hosts (public + store-prefixed). Extra hosts via BLOB_ALLOWED_HOSTS.
_DEFAULT_BLOB_HOST_SUFFIXES = (
    "public.blob.vercel-storage.com",
    "blob.vercel-storage.com",
)


def access_pin() -> str | None:
    """Shared deploy PIN/secret, or None if auth is disabled (local open mode)."""
    raw = (
        os.getenv("STUDYQUIZ_ACCESS_PIN")
        or os.getenv("ACCESS_PIN")
        or os.getenv("STUDYQUIZ_API_SECRET")
        or ""
    ).strip()
    return raw or None


def auth_required() -> bool:
    return access_pin() is not None


def pin_matches(provided: str | None) -> bool:
    expected = access_pin()
    if expected is None:
        return True
    if not provided:
        return False
    a = provided.encode("utf-8")
    b = expected.encode("utf-8")
    # compare_digest requires equal length; unequal → not a match
    if len(a) != len(b):
        return False
    return hmac.compare_digest(a, b)


def extract_pin_from_request_headers(headers) -> str | None:
    """Read PIN from X-StudyQuiz-Pin or Authorization: Bearer <pin>."""
    # Starlette Headers are case-insensitive
    pin = headers.get("x-studyquiz-pin") or headers.get("x-access-pin")
    if pin:
        return pin.strip()
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def allowed_blob_host_suffixes() -> list[str]:
    extras = [
        h.strip().lower().lstrip("*.")
        for h in (os.getenv("BLOB_ALLOWED_HOSTS") or "").split(",")
        if h.strip()
    ]
    out: list[str] = []
    for h in list(_DEFAULT_BLOB_HOST_SUFFIXES) + extras:
        h = h.lower().lstrip("*.")
        if h and h not in out:
            out.append(h)
    return out


def host_is_allowed_blob(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower().rstrip(".")
    # Reject obvious local / metadata names even if somehow listed.
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return False
        # Bare IPs are never Blob hosts we want.
        return False
    except ValueError:
        pass

    for suffix in allowed_blob_host_suffixes():
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def validate_download_url(url: str) -> str:
    """Return cleaned URL or raise ValueError with a user-facing reason.

    Only https Blob hosts (allowlisted) may be fetched — blocks classic SSRF
    to internal IPs / metadata endpoints.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("URL is required.")
    parsed = urlparse(raw)
    if parsed.scheme != "https":
        raise ValueError("Only https URLs are allowed for remote file ingest.")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed.")
    host = parsed.hostname
    if not host_is_allowed_blob(host):
        allowed = ", ".join(f"*.{s}" for s in allowed_blob_host_suffixes())
        raise ValueError(
            "URL host is not on the allowlist. "
            f"Only Vercel Blob hosts are accepted ({allowed})."
        )
    # Rebuild without fragment; keep query (Blob may use signed query params).
    cleaned = parsed._replace(fragment="").geturl()
    return cleaned


def is_blob_url_for_auth_header(url: str) -> bool:
    """Whether to attach BLOB_READ_WRITE_TOKEN when downloading."""
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    return host_is_allowed_blob(host)

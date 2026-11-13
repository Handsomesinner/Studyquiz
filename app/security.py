"""Deploy security helpers: SSRF allowlist for Blob URLs + shared access PIN.

Local demos stay open when STUDYQUIZ_ACCESS_PIN is unset. On a public deploy,
set a PIN so generate / upload / evaluation routes cannot be abused anonymously.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from contextvars import ContextVar
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


def public_deployment() -> bool:
    return bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def auth_required() -> bool:
    return public_deployment() or access_pin() is not None


def pin_matches(provided: str | None) -> bool:
    expected = access_pin()
    if expected is None:
        return not public_deployment()
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
    if parsed.port not in (None, 443):
        raise ValueError("Only the standard HTTPS port is allowed.")
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
    """Never send the store credential to a custom host or another store."""
    token = os.getenv("BLOB_READ_WRITE_TOKEN", "")
    # Vercel tokens identify their store: vercel_blob_rw_<store>_<secret>.
    parts = token.split("_")
    if len(parts) < 5 or parts[:3] != ["vercel", "blob", "rw"]:
        return False
    host = (urlparse(url).hostname or "").lower()
    store_id = parts[3].lower()
    return host in {f"{store_id}.public.blob.vercel-storage.com", f"{store_id}.private.blob.vercel-storage.com", f"{store_id}.blob.vercel-storage.com"}


workspace: ContextVar[str | None] = ContextVar("workspace", default=None)
source_path: ContextVar[str | None] = ContextVar("source_path", default=None)


class WorkspaceAccessError(Exception):
    """A resource is absent or belongs to another browser workspace."""


def reserve_model_call() -> None:
    """Conservative hard limits on calls (including OCR, batches and retries)."""
    if workspace.get() is None:
        return  # offline use/tests do not consume a deployed student's allowance
    from datetime import datetime, timezone
    from fastapi import HTTPException
    from . import store
    day = datetime.now(timezone.utc).date().isoformat()
    for name, limit in (("global", int(os.getenv("AI_DAILY_GLOBAL_CALLS", "100"))),
                        (workspace.get(), int(os.getenv("AI_DAILY_WORKSPACE_CALLS", "30")))):
        if not store.consume_budget(f"ai:{name}:{day}", limit):
            raise HTTPException(429, "Daily AI allowance reached. No further AI call was made.")


def password_hash(password: str) -> str:
    import hashlib
    import secrets
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200000).hex()
    return f"pbkdf2${salt}${digest}"


def password_matches(password: str, encoded: str) -> bool:
    import hashlib
    try:
        scheme, salt, digest = encoded.split("$")
        if scheme != "pbkdf2":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200000).hex()
        return hmac.compare_digest(actual, digest)
    except (ValueError, TypeError):
        return False  # Legacy unsalted links must be recreated, not silently downgraded.

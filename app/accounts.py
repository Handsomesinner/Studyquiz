"""Verified email identities and revocable, HttpOnly StudyQuiz sessions.

Supabase verifies inbox ownership. Provider tokens are never returned to the
browser or stored here; StudyQuiz issues its own bounded opaque session.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import security, store

router = APIRouter(prefix="/api/account")
COOKIE = "sq_student"
SESSION_SECONDS = 12 * 60 * 60


def auth_url() -> str:
    return os.getenv("STUDYQUIZ_AUTH_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""


def public_key() -> str:
    return (os.getenv("STUDYQUIZ_AUTH_PUBLIC_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY") or "")


def configured() -> bool:
    url = urlparse(auth_url())
    return bool(url.scheme == "https" and re.fullmatch(r"[a-z0-9-]+\.supabase\.co", url.hostname or "")
                and not url.username and not url.password and url.port in (None, 443)
                and url.path in ("", "/") and not url.query and not url.fragment
                and public_key())


def enabled() -> bool:
    return os.getenv("STUDYQUIZ_AUTH_MODE", "pin") == "accounts"


def provider(path: str, payload: dict | None = None, *, token: str | None = None) -> dict:
    if not configured():
        raise HTTPException(503, "Student sign-in is not configured yet.")
    headers = {"apikey": public_key()}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with httpx.Client(timeout=15, follow_redirects=False) as client:
            response = client.request("GET" if payload is None else "POST",
                auth_url().rstrip("/") + "/auth/v1/" + path,
                headers=headers, json=payload)
        if response.status_code == 429:
            raise HTTPException(429, "Please wait before trying sign-in again.")
        if response.status_code >= 500 or response.status_code in (401, 403):
            raise HTTPException(503, "Email sign-in is temporarily unavailable.")
        if not response.is_success:
            raise HTTPException(400, "The email code is invalid or expired. Request a new code.")
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid provider response")
        return data
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Could not reach email sign-in. Please try again.") from None


def resolve_session(raw: str) -> dict | None:
    if not configured() or not re.fullmatch(r"[A-Za-z0-9_-]{43}", raw):
        return None
    with store.connection() as conn:
        row = conn.execute("SELECT a.* FROM student_sessions s JOIN student_accounts a ON a.workspace = s.workspace "
            "WHERE s.token_hash = ? AND s.expires_at > ?", (hashlib.sha256(raw.encode()).hexdigest(), int(time.time()))).fetchone()
    return dict(row) if row else None


def limit(request: Request, email: str, purpose: str):
    hour = int(time.time()) // 3600
    # Global budget remains effective even when a caller spoofs an IP header.
    address = request.client.host if request.client else "unknown"
    if os.getenv("VERCEL"):
        address = request.headers.get("x-vercel-forwarded-for", address).split(",")[0].strip()
    hashed = lambda value: hashlib.sha256(value.encode()).hexdigest()
    for bucket, maximum in ((f"account:{purpose}:email:{hashed(email)}:{hour}", 4 if purpose == "send" else 12),
                            (f"account:{purpose}:ip:{hashed(address)}:{hour}", 15 if purpose == "send" else 40),
                            (f"account:{purpose}:global:{hour}", 100 if purpose == "send" else 500)):
        if not store.consume_budget(bucket, maximum):
            raise HTTPException(429, "Too many sign-in attempts. Please try again later.")


class EmailBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)

    def normalized(self):
        email = self.email.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            raise HTTPException(422, "Enter a valid email address.")
        return email


class VerifyBody(EmailBody):
    code: str = Field(pattern=r"^[0-9]{6,10}$")


@router.get("")
def status(request: Request):
    student = getattr(request.state, "student", None)
    legacy = 0
    if student:
        with store.connection() as conn:
            legacy = conn.execute("SELECT COUNT(*) AS n FROM resource_owners WHERE kind = 'doc' AND workspace = ?",
                (request.state.browser_workspace,)).fetchone()["n"]
    return {"configured": configured(), "required": enabled(),
            "browser_notes_to_import": legacy,
            "user": {"email": student["email"]} if student else None}


@router.post("/send-code")
def send_code(body: EmailBody, request: Request):
    if not configured():
        raise HTTPException(503, "Student sign-in is not configured yet.")
    email = body.normalized()
    limit(request, email, "send")
    provider("otp", {"email": email, "create_user": True})
    return {"message": "Check your inbox for your sign-in code. It may take a minute; check spam too."}


@router.post("/verify")
def verify(body: VerifyBody, request: Request):
    email = body.normalized()
    limit(request, email, "verify")
    result = provider("verify", {"type": "email", "email": email, "token": body.code})
    access = result.get("access_token")
    if not isinstance(access, str) or not access:
        raise HTTPException(401, "Could not verify your sign-in. Please request a new code.")
    # Never trust a browser-supplied identity or a decoded-but-unverified JWT.
    user = provider("user", token=access)
    if not user.get("email_confirmed_at") or str(user.get("email", "")).lower() != email or not user.get("id"):
        raise HTTPException(401, "Verify your email before continuing.")
    workspace = hashlib.sha256((auth_url().rstrip("/") + ":" + str(user["id"])).encode()).hexdigest()
    raw = secrets.token_urlsafe(32)
    now = int(time.time())
    with store.connection() as conn:
        conn.execute("INSERT INTO student_accounts (workspace, email, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(workspace) DO UPDATE SET email = excluded.email", (workspace, email, now))
        old = request.cookies.get(COOKIE, "")
        if old:
            conn.execute("DELETE FROM student_sessions WHERE token_hash = ?", (hashlib.sha256(old.encode()).hexdigest(),))
        conn.execute("DELETE FROM student_sessions WHERE expires_at <= ?", (now,))
        conn.execute("INSERT INTO student_sessions VALUES (?, ?, ?)", (hashlib.sha256(raw.encode()).hexdigest(), workspace, now + SESSION_SECONDS))
    response = JSONResponse({"user": {"email": email}})
    response.set_cookie(COOKIE, raw, httponly=True, secure=security.public_deployment(), samesite="lax", max_age=SESSION_SECONDS)
    return response


@router.post("/logout")
def logout(request: Request):
    raw = request.cookies.get(COOKIE, "")
    if raw:
        with store.connection() as conn:
            conn.execute("DELETE FROM student_sessions WHERE token_hash = ?", (hashlib.sha256(raw.encode()).hexdigest(),))
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, secure=security.public_deployment(), httponly=True, samesite="lax")
    return response


@router.post("/logout-all")
def logout_all(request: Request):
    student = getattr(request.state, "student", None)
    if not student:
        raise HTTPException(401, "Sign in first.")
    with store.connection() as conn:
        conn.execute("DELETE FROM student_sessions WHERE workspace = ?", (student["workspace"],))
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, secure=security.public_deployment(), httponly=True, samesite="lax")
    return response


@router.get("/dashboard")
def dashboard(request: Request):
    student = getattr(request.state, "student", None)
    if not student:
        raise HTTPException(401, "Sign in to see your dashboard.")
    return store.student_dashboard(student["workspace"])


class ImportBody(BaseModel):
    previous_code: str = Field(min_length=1, max_length=256)


@router.post("/import-browser-notes")
def import_browser_notes(body: ImportBody, request: Request):
    student = getattr(request.state, "student", None)
    if not student:
        raise HTTPException(401, "Sign in first.")
    limit(request, student["email"], "import")
    if not security.pin_matches(body.previous_code.strip()):
        raise HTTPException(401, "Your previous workspace code was not accepted.")
    # No source workspace ID is accepted from the browser. Only its HttpOnly
    # legacy cookie identifies the old notes being explicitly moved.
    old = request.state.browser_workspace
    with store.connection() as conn:
        conn.execute("UPDATE resource_owners SET workspace = ? WHERE workspace = ?", (student["workspace"], old))
        conn.execute("UPDATE generation_runs SET workspace = ? WHERE workspace = ?", (student["workspace"], old))
        conn.execute("UPDATE upload_permits SET workspace = ? WHERE workspace = ?", (student["workspace"], old))
    return {"ok": True}

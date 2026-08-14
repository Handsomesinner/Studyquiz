"""Spaced-repetition helpers (light SM-2) for weak MCQ / flashcards."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fingerprint_question(text: str) -> str:
    """Stable id for a question stem (dedupe across quizzes)."""
    norm = re.sub(r"\s+", " ", (text or "").strip().casefold())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def parse_due(iso: str | None) -> datetime:
    if not iso:
        return utc_now()
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return utc_now()


def sm2_schedule(
    *,
    grade: str,
    ease: float = 2.5,
    interval_days: float = 0.0,
    reps: int = 0,
    lapses: int = 0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return updated ease/interval/reps/lapses/due_at for a review grade.

    Grades: again | hard | good | easy
    """
    now = now or utc_now()
    g = (grade or "good").strip().lower()
    if g not in ("again", "hard", "good", "easy"):
        g = "good"

    ease = max(1.3, float(ease or 2.5))
    interval = float(interval_days or 0)
    reps = int(reps or 0)
    lapses = int(lapses or 0)

    if g == "again":
        lapses += 1
        reps = 0
        interval = 0
        ease = max(1.3, ease - 0.2)
        due = now  # due immediately
    elif g == "hard":
        reps += 1
        ease = max(1.3, ease - 0.15)
        if interval <= 0:
            interval = 0.5
        else:
            interval = max(0.5, interval * 1.2)
        due = now + timedelta(days=interval)
    elif g == "easy":
        reps += 1
        ease = ease + 0.15
        if interval <= 0:
            interval = 3
        else:
            interval = interval * ease * 1.3
        due = now + timedelta(days=interval)
    else:  # good
        reps += 1
        if interval <= 0:
            interval = 1
        elif reps == 1:
            interval = 1
        elif reps == 2:
            interval = 3
        else:
            interval = interval * ease
        due = now + timedelta(days=interval)

    return {
        "ease": round(ease, 3),
        "interval_days": round(interval, 3),
        "reps": reps,
        "lapses": lapses,
        "due_at": due.isoformat(),
    }


def is_due(due_at: str | None, now: datetime | None = None) -> bool:
    now = now or utc_now()
    return parse_due(due_at) <= now

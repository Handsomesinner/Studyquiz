"""Topic map extraction from lecture notes (headings + frequent phrases).

Used to suggest focus-topic chips in the UI without an extra LLM call.
"""

from __future__ import annotations

import re
from collections import Counter

# Common function words (English + a few academic stop tokens).
_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "for", "with", "by", "from", "as", "at", "into",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "shall", "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "we", "our", "you", "your", "he", "she", "his",
    "her", "not", "no", "yes", "also", "than", "such", "via", "per", "using",
    "used", "use", "based", "figure", "table", "page", "chapter", "section",
    "example", "examples", "note", "notes", "lecture", "course", "unit",
    "introduction", "overview", "summary", "conclusion", "references",
    "et", "al", "etc", "ie", "eg", "see", "shown", "show", "shows",
}


_HEADING_LINE = re.compile(
    r"(?m)^(?:#{1,3}\s+|"
    r"(?:chapter|section|unit|module|topic|part)\s+\d+[.:)\s]+|"
    r"\d+(?:\.\d+){0,3}\s+|"
    r"[A-Z][A-Z0-9 /&\-]{4,60}$)",
    re.IGNORECASE,
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _normalize_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"^[\d.#)\-\s]+", "", s)
    return s.strip(" :-–—")


def extract_heading_topics(text: str, *, limit: int = 12) -> list[str]:
    """Pull candidate topics from markdown / numbered / ALL-CAPS headings."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for m in _HEADING_LINE.finditer(text[:200_000]):
        # Prefer the full line containing the match
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        if end < 0:
            end = min(len(text), m.end() + 80)
        line = _normalize_phrase(text[start:end])
        if len(line) < 4 or len(line) > 72:
            continue
        # Skip pure numbers / very short
        words = [w for w in _TOKEN.findall(line) if w.lower() not in _STOP]
        if len(words) < 1:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        # Prefer Title Case-ish display
        found.append(line[:72])
        if len(found) >= limit:
            break
    return found


def extract_phrase_topics(text: str, *, limit: int = 12) -> list[str]:
    """Frequent content bigrams/trigrams as soft topic suggestions."""
    if not text:
        return []
    tokens = [
        t.lower()
        for t in _TOKEN.findall(text[:300_000])
        if t.lower() not in _STOP and not t.isdigit()
    ]
    if len(tokens) < 8:
        return []

    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a == b:
            continue
        bigrams[f"{a} {b}"] += 1
        if i + 2 < len(tokens):
            c = tokens[i + 2]
            if c != b:
                trigrams[f"{a} {b} {c}"] += 1

    # Prefer trigrams that appear often, then bigrams
    scored: list[tuple[int, str]] = []
    for phrase, n in trigrams.items():
        if n >= 3:
            scored.append((n * 3, phrase))
    for phrase, n in bigrams.items():
        if n >= 4:
            scored.append((n * 2, phrase))
    scored.sort(key=lambda x: (-x[0], x[1]))

    out: list[str] = []
    seen: set[str] = set()
    for _, phrase in scored:
        # Title-case lightly for chips
        display = " ".join(w.capitalize() if w.islower() else w for w in phrase.split())
        key = phrase.casefold()
        if key in seen:
            continue
        # Avoid nesting: skip if phrase is substring of already chosen
        if any(key in s or s in key for s in seen):
            continue
        seen.add(key)
        out.append(display)
        if len(out) >= limit:
            break
    return out


def topic_map(
    text: str,
    *,
    max_topics: int = 16,
) -> list[dict]:
    """Return ranked topic suggestions: [{topic, source: heading|phrase}]."""
    max_topics = max(4, min(int(max_topics or 16), 24))
    headings = extract_heading_topics(text, limit=max_topics)
    phrases = extract_phrase_topics(text, limit=max_topics)

    out: list[dict] = []
    seen: set[str] = set()
    for h in headings:
        k = h.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append({"topic": h, "source": "heading"})
        if len(out) >= max_topics:
            return out
    for p in phrases:
        k = p.casefold()
        if k in seen or any(k in s or s in k for s in seen):
            continue
        seen.add(k)
        out.append({"topic": p, "source": "phrase"})
        if len(out) >= max_topics:
            break
    return out

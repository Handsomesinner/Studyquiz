"""Topic map from structural headings in lecture notes.

Chips come from real section structure — not random frequent phrases:
  - Markdown headings (# …)
  - Bold standalone headings (**…** / __…__)
  - Chapter / Section / Unit / Module / Topic / Part labels
  - Numbered outline headings (1.2 Title, 3. Title)
  - Short ALL-CAPS or Title-Case lines standing alone
  - Short “headline” lines that are then explained in following paragraphs
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")

# Labels that mark structure but are not useful as the chip alone.
_STRUCT_PREFIX = re.compile(
    r"^(?:"
    r"chapter|section|unit|module|topic|part|lesson|week|"
    r"ch\.?|sec\.?|mod\.?"
    r")\s*"
    r"(?:\d+[A-Za-z]?|[IVXLC]+)?\s*[:.\-–—)]?\s*",
    re.IGNORECASE,
)

_MD_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_MD_BOLD_LINE = re.compile(
    r"(?m)^(?:\*\*(.+?)\*\*|__(.+?)__)\s*$"
)
_CHAPTER_LINE = re.compile(
    r"(?m)^(?:chapter|section|unit|module|topic|part|lesson|week)\s+"
    r"(?:\d+[A-Za-z]?|[IVXLC]+)(?:\s*[:.\-–—)]\s*|\s+)(.+?)\s*$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(
    r"(?m)^(\d+(?:\.\d+){0,4})\.?\s+([A-Za-z].{2,70})\s*$"
)
_ALL_CAPS = re.compile(
    r"(?m)^([A-Z][A-Z0-9][A-Z0-9 /&\-,:]{2,68})\s*$"
)
# Short title-case line: e.g. "Process Scheduling" (2–8 words, capitalised starts)
_TITLE_CASE_LINE = re.compile(
    r"(?m)^((?:[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,7}))\s*$"
)

# Noise lines that look like headings but aren't topics.
_NOISE = re.compile(
    r"^(?:"
    r"figure|table|appendix|references?|bibliography|contents|"
    r"table of contents|index|abstract|acknowledgements?|"
    r"page\s+\d+|slide\s+\d+"
    r")\b",
    re.IGNORECASE,
)


def _normalize_phrase(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"_+", "", s)
    s = re.sub(r"\s+", " ", s)
    # Drop leading numeric outline markers only (not letters of the title)
    s = re.sub(r"^\d+(?:\.\d+)*\s*[:.)\-–—]?\s*", "", s)
    # Roman numerals only when followed by punctuation/space (e.g. "IV. Title")
    s = re.sub(r"^[IVXLC]{1,6}\s*[:.)\-–—]\s+", "", s, flags=re.I)
    s = re.sub(r"^[.#)\-\s]+", "", s)
    s = _STRUCT_PREFIX.sub("", s).strip()
    return s.strip(" :-–—.|")


def _is_good_topic(line: str) -> bool:
    line = _normalize_phrase(line)
    if len(line) < 3 or len(line) > 72:
        return False
    if _NOISE.match(line):
        return False
    words = _TOKEN.findall(line)
    if len(words) < 1 or len(words) > 12:
        return False
    # Reject pure numbers / symbols
    if not any(c.isalpha() for c in line):
        return False
    # Reject long prose sentences
    if line.endswith((".", "?", "!")) and len(words) > 6:
        return False
    if line.count(",") >= 3:
        return False
    return True


def _add(out: list[dict], seen: set[str], topic: str, source: str, limit: int) -> bool:
    topic = _normalize_phrase(topic)
    if not _is_good_topic(topic):
        return len(out) >= limit
    key = topic.casefold()
    if key in seen:
        return len(out) >= limit
    # Avoid near-subsumption / reverse containment
    for s in list(seen):
        if key == s:
            return len(out) >= limit
        if key in s or s in key:
            return len(out) >= limit
    seen.add(key)
    out.append({"topic": topic[:72], "source": source})
    return len(out) >= limit


def extract_markdown_headings(text: str, *, limit: int = 24) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for m in _MD_HEADING.finditer(text):
        if _add(out, seen, m.group(2), "heading", limit):
            break
    return out


def extract_bold_headings(text: str, *, limit: int = 24) -> list[dict]:
    """Standalone bold lines: **Process Scheduling** or __Virtual Memory__."""
    out: list[dict] = []
    seen: set[str] = set()
    for m in _MD_BOLD_LINE.finditer(text):
        body = m.group(1) or m.group(2) or ""
        if _add(out, seen, body, "bold", limit):
            break
    return out


def extract_chapter_headings(text: str, *, limit: int = 24) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for m in _CHAPTER_LINE.finditer(text):
        if _add(out, seen, m.group(1), "chapter", limit):
            break
    # Also keep "Chapter 3: Title" full form cleaned
    for m in re.finditer(
        r"(?mi)^(chapter|section|unit|module|topic|part)\s+"
        r"(\d+[A-Za-z]?|[IVXLC]+)\s*[:.\-–—)]\s*(.+)$",
        text,
    ):
        title = m.group(3)
        if _add(out, seen, title, "chapter", limit):
            break
    return out


def extract_numbered_headings(text: str, *, limit: int = 24) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for m in _NUMBERED_HEADING.finditer(text):
        title = m.group(2)
        # Skip if it looks like a long bullet sentence
        if len(title.split()) > 10:
            continue
        if _add(out, seen, title, "numbered", limit):
            break
    return out


def extract_caps_and_title_lines(text: str, *, limit: int = 24) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for m in _ALL_CAPS.finditer(text):
        line = m.group(1)
        # Skip single acronyms like HTTP
        if len(line.split()) < 2 and len(line) < 8:
            continue
        if _add(out, seen, line.title() if line.isupper() else line, "caps", limit):
            break
    for m in _TITLE_CASE_LINE.finditer(text):
        line = m.group(1)
        words = line.split()
        if len(words) < 2:
            continue
        # Require majority of words capitalised (heading style)
        caps = sum(1 for w in words if w[:1].isupper())
        if caps < max(2, len(words) // 2 + 1):
            continue
        if _add(out, seen, line, "title_case", limit):
            break
    return out


def extract_standalone_explained_headings(
    text: str, *, limit: int = 24
) -> list[dict]:
    """Short lines that stand alone, then a longer paragraph follows (explained later)."""
    if not text:
        return []
    # Work on first ~250k chars for speed
    sample = text[:250_000]
    lines = sample.splitlines()
    out: list[dict] = []
    seen: set[str] = set()

    i = 0
    n = len(lines)
    while i < n and len(out) < limit:
        raw = lines[i].strip()
        i += 1
        if not raw or not _is_good_topic(raw):
            continue
        # Standalone: blank or end before/after, short line
        prev_blank = i < 2 or not lines[i - 2].strip()
        # Look ahead for explanatory body (not another short heading)
        j = i
        while j < n and not lines[j].strip():
            j += 1
        if j >= n:
            continue
        body = lines[j].strip()
        # Body should be longer prose that starts explaining
        body_words = body.split()
        head_words = raw.split()
        if len(body_words) < 8:
            continue
        if len(head_words) > 8:
            continue
        # Prefer cases where next content is longer than the heading
        if len(body) < len(raw) * 1.5:
            continue
        # Body shouldn't be another heading pattern
        if _MD_HEADING.match(body) or _CHAPTER_LINE.match(body):
            continue
        # Soft signal: heading tokens reappear in nearby text
        head_tokens = {
            t.lower()
            for t in _TOKEN.findall(raw)
            if len(t) > 3
        }
        nearby = " ".join(lines[j : min(n, j + 6)]).casefold()
        overlap = sum(1 for t in head_tokens if t in nearby)
        if head_tokens and overlap < 1 and not prev_blank:
            # still allow blank-isolated short lines followed by long prose
            if not prev_blank:
                continue
        if _add(out, seen, raw, "standalone", limit):
            break
    return out


def topic_map(
    text: str,
    *,
    max_topics: int = 16,
) -> list[dict]:
    """Structural topic chips only (no frequency-phrase noise).

    Priority: markdown → bold → chapter/section → numbered → caps/title-case
    → standalone lines later explained.
    """
    max_topics = max(4, min(int(max_topics or 16), 24))
    if not text or not text.strip():
        return []

    sample = text[:400_000]
    buckets = [
        extract_markdown_headings(sample, limit=max_topics),
        extract_bold_headings(sample, limit=max_topics),
        extract_chapter_headings(sample, limit=max_topics),
        extract_numbered_headings(sample, limit=max_topics),
        extract_caps_and_title_lines(sample, limit=max_topics),
        extract_standalone_explained_headings(sample, limit=max_topics),
    ]

    out: list[dict] = []
    seen: set[str] = set()
    for bucket in buckets:
        for item in bucket:
            t = item["topic"]
            k = t.casefold()
            if k in seen:
                continue
            if any(k in s or s in k for s in seen if abs(len(k) - len(s)) < 8):
                # skip near-duplicate nested titles
                if any(k in s for s in seen):
                    continue
            seen.add(k)
            out.append(item)
            if len(out) >= max_topics:
                return out
    return out

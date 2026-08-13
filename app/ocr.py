"""OCR fallback for scanned / image-only PDFs via Claude document vision.

When pypdf finds little or no embedded text, we send page batches of the PDF
to Anthropic as document blocks and ask for plain-text extraction. This works
on Vercel (no Tesseract binary) using the existing ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import base64
import io
import os
import re

from pypdf import PdfReader, PdfWriter

# If embedded text is shorter than this, treat the PDF as needing OCR.
MIN_EMBEDDED_CHARS = 80
# Average chars/page below this → likely image-only (even if a footer exists).
MIN_CHARS_PER_PAGE = 40

OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "30"))
OCR_BATCH_PAGES = int(os.getenv("OCR_BATCH_PAGES", "4"))
# Prefer a strong vision model for scans; override with OCR_MODEL.
OCR_MODEL = os.getenv("OCR_MODEL", "claude-sonnet-4-5")

OCR_PROMPT = (
    "Extract ALL readable text from this PDF exactly as it appears. "
    "Preserve reading order (top to bottom, left to right). "
    "Keep headings, numbered items, and formulas as plain text. "
    "Do not summarize, translate, or add commentary. "
    "If a page is blank or unreadable, skip it silently. "
    "Output only the extracted text."
)


class OcrError(Exception):
    """User-presentable OCR failure."""


def ocr_enabled() -> bool:
    flag = (os.getenv("OCR_ENABLED") or "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def needs_ocr(embedded_text: str, page_count: int) -> bool:
    """Return True when embedded PDF text is too thin to be useful."""
    text = (embedded_text or "").strip()
    if not text:
        return True
    if len(text) < MIN_EMBEDDED_CHARS:
        return True
    pages = max(page_count, 1)
    if len(text) / pages < MIN_CHARS_PER_PAGE:
        return True
    return False


def pdf_page_count(data: bytes) -> int:
    try:
        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return 0


def _slice_pdf_bytes(data: bytes, start: int, end: int) -> bytes:
    """Return a new PDF containing pages [start, end)."""
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    for i in range(start, min(end, len(reader.pages))):
        writer.add_page(reader.pages[i])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _client():
    import anthropic

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise OcrError(
            "This looks like a scanned PDF (no extractable text). "
            "OCR needs ANTHROPIC_API_KEY to read the pages."
        )
    return anthropic.Anthropic()


def _ocr_pdf_batch(pdf_bytes: bytes, *, page_label: str) -> str:
    """Send one PDF slice to Claude as a document and return extracted text."""
    import anthropic

    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
    client = _client()
    try:
        # PDF document blocks (vision). Falls back cleanly if model rejects.
        response = client.messages.create(
            model=OCR_MODEL,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{OCR_PROMPT}\n\n(Pages: {page_label})",
                        },
                    ],
                }
            ],
        )
    except anthropic.BadRequestError as e:
        raise OcrError(
            f"OCR request rejected for pages {page_label}: {e.message or e}"
        ) from e
    except anthropic.AuthenticationError as e:
        raise OcrError("Anthropic API key rejected during OCR.") from e
    except anthropic.RateLimitError as e:
        raise OcrError("Rate limited during OCR. Wait a moment and try again.") from e
    except anthropic.APIStatusError as e:
        raise OcrError(f"OCR API error ({e.status_code}). Try again shortly.") from e
    except anthropic.APIConnectionError as e:
        raise OcrError("Could not reach Anthropic for OCR. Check internet.") from e

    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "\n".join(parts).strip()


def ocr_pdf(data: bytes, *, filename: str = "scan.pdf") -> str:
    """OCR a scanned PDF via Claude. Raises OcrError on failure."""
    if not ocr_enabled():
        raise OcrError(
            "No text could be extracted and OCR is disabled. "
            "Set OCR_ENABLED=true and ANTHROPIC_API_KEY, or upload a text-based PDF."
        )

    reader = PdfReader(io.BytesIO(data))
    n_pages = len(reader.pages)
    if n_pages == 0:
        raise OcrError("This PDF has no pages.")

    limit = min(n_pages, OCR_MAX_PAGES)
    texts: list[str] = []
    for start in range(0, limit, OCR_BATCH_PAGES):
        end = min(start + OCR_BATCH_PAGES, limit)
        slice_bytes = _slice_pdf_bytes(data, start, end)
        label = f"{start + 1}-{end} of {n_pages}"
        batch_text = _ocr_pdf_batch(slice_bytes, page_label=label)
        if batch_text:
            texts.append(batch_text)

    combined = "\n\n".join(texts).strip()
    combined = re.sub(r"[ \t]+", " ", combined)
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()

    if len(combined) < MIN_EMBEDDED_CHARS:
        raise OcrError(
            "OCR ran but recovered almost no text. The scan may be too blurry, "
            "handwritten, or empty. Try a clearer scan or a text-based PDF."
        )

    if n_pages > OCR_MAX_PAGES:
        combined += (
            f"\n\n[Note: Only the first {OCR_MAX_PAGES} of {n_pages} pages "
            f"were OCR'd due to size limits.]"
        )
    return combined

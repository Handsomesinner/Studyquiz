"""Vercel serverless entry point.

Vercel runs Python apps as serverless functions rather than a long-lived
`uvicorn` server, so it needs a module that exposes the ASGI `app` object.
This file re-exports the FastAPI app from the project package; `vercel.json`
routes every incoming request to it.
"""

import sys
from pathlib import Path

# Make the repo root importable so the `app` package resolves when Vercel
# bundles this function.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402  (import after sys.path setup)

__all__ = ["app"]

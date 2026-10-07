"""
data/loader.py — Stage 1: load a PDF from disk or URL and extract page text.

Keeps page numbers so every chunk (and eventually every answer) can be
traced back to an exact page — essential for citations and debugging.
"""

import os
import re
import logging
import requests
import fitz   # PyMuPDF


def download_pdf(url: str, save_path: str) -> None:
    """Download a PDF from a URL. No-op if the file already exists."""
    if os.path.exists(save_path):
        logging.info(f"PDF already cached at '{save_path}'. Skipping download.")
        return

    logging.info(f"Downloading PDF from {url} ...")
    r = requests.get(url, timeout=30)
    if not r.ok:
        raise ValueError(f"Download failed — HTTP {r.status_code}: {url}")

    with open(save_path, "wb") as f:
        f.write(r.content)
    logging.info(f"Saved to '{save_path}'.")


def load_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Open a PDF and return one dict per non-blank page:
        { "page": int, "text": str }

    Text is whitespace-normalised (collapsed to single spaces).
    Blank and image-only pages are silently skipped.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

    doc   = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue                                         # skip blank/image pages
        pages.append({
            "page": i,
            "text": re.sub(r"\s+", " ", text).strip(),      # collapse whitespace
        })

    logging.info(f"Loaded {len(pages)} pages from '{pdf_path}'.")
    return pages

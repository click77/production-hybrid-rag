# src/ingestion/loader.py

import os
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalize raw text into clean plaintext:
    - Collapse multiple blank lines into one
    - Strip leading/trailing whitespace per line
    - Remove null bytes and non-printable chars
    """
    text = text.replace("\x00", "")                    # remove null bytes
    text = re.sub(r"\r\n|\r", "\n", text)             # normalize line endings
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)            # max 2 blank lines
    return text.strip()


def _extract_heading(text: str) -> Optional[str]:
    """
    Extract the first heading from plaintext.
    Looks for Markdown headings (# Heading) or
    ALL CAPS lines as a fallback.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line.isupper() and len(line) > 3:
            return line
    return None


def _make_doc_id(source_path: str) -> str:
    """Stable unique ID based on the file path."""
    return hashlib.md5(source_path.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────
# PER-FORMAT LOADERS
# ─────────────────────────────────────────────

def _load_txt(path: Path) -> list[dict]:
    """Load a plain .txt or .md file as a single document."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    processed = _clean_text(raw)

    return [{
        "doc_id":        _make_doc_id(str(path)),
        "source_file":   str(path),
        "file_type":     path.suffix.lower().lstrip("."),
        "page_number":   1,
        "section_heading": _extract_heading(processed),
        "raw_text":      raw,
        "text":          processed,
        "char_count":    len(processed),
        "loaded_at":     datetime.utcnow().isoformat(),
    }]


def _load_html(path: Path) -> list[dict]:
    """
    Load an HTML file. Extracts visible text only —
    strips <script>, <style>, and nav boilerplate.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

    processed = _clean_text(soup.get_text(separator="\n"))

    # Try to grab a heading from <h1>–<h3>
    heading = None
    for level in ["h1", "h2", "h3"]:
        tag = soup.find(level)
        if tag:
            heading = tag.get_text(strip=True)
            break

    return [{
        "doc_id":          _make_doc_id(str(path)),
        "source_file":     str(path),
        "file_type":       "html",
        "page_number":     1,
        "section_heading": heading or _extract_heading(processed),
        "raw_text":        raw,
        "text":            processed,
        "char_count":      len(processed),
        "loaded_at":       datetime.utcnow().isoformat(),
    }]


def _load_pdf(path: Path) -> list[dict]:
    """
    Load a PDF file. Returns one document dict PER PAGE
    so page_number metadata is always accurate.
    Each page stores its own raw + processed text.
    """
    doc = fitz.open(str(path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")           # raw text from the PDF layer
        processed = _clean_text(raw)

        if not processed:                     # skip blank/image-only pages
            continue

        pages.append({
            "doc_id":          _make_doc_id(f"{path}::page{page_num}"),
            "source_file":     str(path),
            "file_type":       "pdf",
            "page_number":     page_num,
            "total_pages":     len(doc),
            "section_heading": _extract_heading(processed),
            "raw_text":        raw,
            "text":            processed,
            "char_count":      len(processed),
            "loaded_at":       datetime.utcnow().isoformat(),
        })

    doc.close()
    return pages


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}

def load_document(path: str | Path) -> list[dict]:
    """
    Load a single document and return a list of document dicts.
    PDFs return one dict per page; all others return one dict total.

    Each dict contains:
        doc_id          – stable unique ID
        source_file     – original file path
        file_type       – txt / md / html / pdf
        page_number     – always set (1 for non-PDF)
        section_heading – first heading found, or None
        raw_text        – unmodified original text
        text            – cleaned, normalized plaintext  ← used for indexing
        char_count      – length of cleaned text
        loaded_at       – UTC timestamp
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext in {".txt", ".md"}:
        return _load_txt(path)
    elif ext in {".html", ".htm"}:
        return _load_html(path)
    elif ext == ".pdf":
        return _load_pdf(path)


def load_directory(directory: str | Path) -> list[dict]:
    """
    Recursively load all supported documents from a directory.
    Skips hidden files and files starting with '.'.

    Returns a flat list of all document dicts across all files.
    """
    directory = Path(directory)
    all_docs = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs = load_document(file_path)
                    all_docs.extend(docs)
                    print(f"  ✓ Loaded {file_path.name} → {len(docs)} section(s)")
                except Exception as e:
                    print(f"  ✗ Skipped {file_path.name}: {e}")

    return all_docs


def save_to_raw_store(docs: list[dict], output_dir: str | Path) -> None:
    """
    Save loaded documents to disk as JSON so you can re-index later
    without re-reading the original files.

    Each doc is saved as:  data/raw/<doc_id>.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        out_path = output_dir / f"{doc['doc_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(docs)} documents to {output_dir}/")


def load_from_raw_store(raw_dir: str | Path) -> list[dict]:
    """
    Re-load previously saved documents from the raw store.
    Use this to re-index without re-reading original files.
    """
    raw_dir = Path(raw_dir)
    docs = []

    for json_file in sorted(raw_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            docs.append(json.load(f))

    print(f"  Loaded {len(docs)} documents from raw store.")
    return docs
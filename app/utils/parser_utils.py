import logging
import re
from pathlib import Path
from typing import Optional

import fitz
from docx import Document

from app.config.settings import MAX_LINE_LENGTH, MAX_PDF_CHARS, _JUNK_LINE_RE


logger = logging.getLogger(__name__)


def parse_pdf(path: str) -> str:
    logger.info(f"Parsing PDF: {path}")
    pages = []
    total_chars = 0
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if not text.strip():
                blocks = page.get_text("blocks")
                if isinstance(blocks, list):
                    text = "\n".join(
                        block[4]
                        for block in blocks
                        if isinstance(block[4], str) and block[4].strip()
                    )
            if text.strip():
                total_chars += len(text)

                if total_chars > MAX_PDF_CHARS:
                    logger.warning(f"PDF too large after extraction: {path}")
                    raise ValueError("PDF content too large after extraction")

                pages.append(text)
            else:
                logger.warning(f"Empty page detected: page {i}")
    if not pages:
        raise ValueError("No extractable text (possibly scanned PDF)")

    return "\n\n".join(pages)


def parse_docx(path: str) -> str:
    doc = Document(path)
    text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return text


def parse_txt(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = Path(path).read_text(encoding="latin-1")
    return text


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    # Normalize
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Fix soft hyphens & broken words
    text = text.replace("\xad", "")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Normalize unicode punctuation
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _JUNK_LINE_RE.match(stripped):
            continue
        # Trim long lines
        if len(stripped) > MAX_LINE_LENGTH:
            stripped = stripped[:MAX_LINE_LENGTH]
        lines.append(stripped.rstrip())
    result = "\n".join(lines)
    # Collapse excessive newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return str(result.strip())

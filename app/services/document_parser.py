import logging
from pathlib import Path
import trafilatura
from app.utils.parser_utils import clean_text, parse_docx, parse_pdf, parse_txt

logger = logging.getLogger(__name__)


def parse_file(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    logger.info(f"Parsing file: {filename}")
    if ext == ".pdf":
        text = parse_pdf(file_path)
    elif ext == ".docx":
        text = parse_docx(file_path)
    elif ext == ".txt":
        text = parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    cleaned = clean_text(text)
    if not cleaned:
        logger.warning(f"Empty content after cleaning: {filename}")
        raise ValueError("Parsed content is empty after cleaning")
    return str(cleaned)


def parse_url_from_content(html: str, url: str) -> str:
    logger.info(f"Parsing URL content: {url}")
    raw = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if not raw:
        logger.error(f"URL extraction failed or returned no content: {url}")
        raise ValueError("Content extraction failed")
    cleaned = clean_text(raw)
    if not cleaned:
        logger.warning(f"Empty content after cleaning URL: {url}")
        raise ValueError("URL content is empty after extraction")
    return str(cleaned)

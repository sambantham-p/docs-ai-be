from pathlib import Path
import re


UPLOAD_DIR = Path("storage/uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_EXTRACTED_TEXT = 150000
ALLOWED_MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": None,
}
MAX_PDF_CHARS = 500_000
MAX_LINE_LENGTH = 2000
_JUNK_LINE_RE = re.compile(r"^[\W_]+$")

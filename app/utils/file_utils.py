import logging
from pathlib import Path
import filetype
from app.config.settings import ALLOWED_MIME_TYPES


logger = logging.getLogger(__name__)


# Removes temporary file safely from the disk
def delete_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Temp file removed: {path}")
    except Exception:
        logger.warning(f"Could not remove temp file: {path}")


# Validate file content against its extension via magic-byte detection
def validate_mime(file_path: Path, ext: str) -> None:
    expected_mime = ALLOWED_MIME_TYPES.get(ext)
    if expected_mime is None:
        return
    # reads the file header to guess the MIME type, not relying on the extension
    detected = filetype.guess(str(file_path))
    if detected is None:
        logger.warning(f"MIME detection failed: {file_path}")
        raise ValueError("Could not verify file type")
    if detected.mime != expected_mime:
        logger.warning(
            f"MIME mismatch: declared={ext}, detected={detected.mime}, file={file_path}"
        )
        raise ValueError("File content does not match its extension")
    logger.debug(f"MIME validated: {detected.mime}")

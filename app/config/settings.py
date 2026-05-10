import os
import re
from pathlib import Path
from typing import Final
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
# MongoDB
MONGO_URI: Final[str] = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB:  Final[str] = os.getenv("MONGO_DB",  "rag_db")

# File Upload
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR: Final[Path] = Path(os.getenv("UPLOAD_DIR", BASE_DIR/"storage/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024
MAX_EXTRACTED_TEXT: Final[int] = 200_000
ALLOWED_EXTENSIONS: Final[set[str]] = {"pdf", "txt", "docx"}
ALLOWED_MIME_TYPES: Final[dict[str, str]] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}
# Parsing Constraints
MAX_PDF_CHARS:  Final[int] = 500_000
MAX_LINE_LENGTH: Final[int] = 2000
_JUNK_LINE_RE: Final[re.Pattern] = re.compile(r"^[\W_]+$")

# Vector DB (Qdrant)
EMBEDDING_DIM:    Final[int] = 3072
EMBEDDING_MODEL = "gemini-embedding-001"
QDRANT_PATH: Final[Path] = Path(
    os.getenv("QDRANT_PATH", BASE_DIR / "data/qdrant")
)
QDRANT_URL: Final[str] = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API: Final[str] = os.getenv("QDRANT_API", "")
QDRANT_PATH.mkdir(parents=True, exist_ok=True)
QDRANT_COLLECTION: Final[str] = os.getenv("QDRANT_COLLECTION", "rag_chunks")

# Gemini
MAX_CONTEXT_CHARS = 9_000
GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "")
LLM_TIMEOUT: Final[int] = int(os.getenv("LLM_TIMEOUT", "30"))

# Retrieval Config
QDRANT_CANDIDATE_CAP: Final[int] = 100
RETRIEVAL_BASE_FETCH: Final[int] = 10
RETRIEVAL_FILTERED_FETCH: Final[int] = 14

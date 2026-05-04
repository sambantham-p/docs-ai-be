import logging
import numpy as np
import google.generativeai as genai
from app.config.settings import EMBEDDING_DIM, GEMINI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add it to your environment or .env file."
    )

_MODEL = EMBEDDING_MODEL


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    """Call Gemini embedding API and return (N, EMBEDDING_DIM) float32 array."""
    if not texts:
        raise ValueError("embed called with empty text list")
    result = genai.embed_content(
        model=_MODEL,
        content=texts,
        task_type=task_type,
    )
    vectors = np.array(result["embedding"], dtype="float32")
    if vectors.ndim != 2 or vectors.shape != (len(texts), EMBEDDING_DIM):
        raise ValueError(
            f"Gemini returned shape {vectors.shape}, "
            f"expected ({len(texts)}, {EMBEDDING_DIM})"
        )
    return vectors


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed document chunks for indexing."""
    logger.debug(f"Embedding {len(texts)} document chunks")
    return _embed(texts, task_type="retrieval_document")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query for retrieval (returns shape (1, EMBEDDING_DIM))."""
    logger.debug("Embedding query")
    return _embed([query], task_type="retrieval_query")
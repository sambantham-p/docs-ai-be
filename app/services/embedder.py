import logging
import numpy as np
import google.generativeai as genai
from app.config.settings import (
    EMBEDDING_DIM,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Configure Gemini once at startup
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add it to your environment or .env file."
    )

genai.configure(api_key=GEMINI_API_KEY)


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    """Call Gemini embedding API and return (N, EMBEDDING_DIM) float32 array."""
    if not texts:
        raise ValueError("embed called with empty text list")

    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type=task_type,
        )
    except Exception:
        logger.exception("Gemini embedding request failed")
        raise

    embeddings = result.get("embedding")
    if not embeddings:
        raise ValueError("No embeddings returned from Gemini")

    # If it's a single flat list of floats, wrap it in a list to make it a batch of 1
    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], float):
        embeddings = [embeddings]

    vectors = np.array(embeddings, dtype=np.float32)
    logger.info(f"[DEBUG] Gemini returned vectors with shape: {vectors.shape}")

    # Shape validation
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        logger.error(f"[DEBUG] Shape mismatch! Expected {EMBEDDING_DIM}, got {vectors.shape[1]}")
        raise ValueError(
            f"Gemini returned shape {vectors.shape}, "
            f"expected ({len(texts)}, {EMBEDDING_DIM})"
        )
    # L2 normalization
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors


def embed_documents(texts: list[str]) -> np.ndarray:
    """
    Embed document chunks for indexing.
    """
    logger.debug(f"Embedding {len(texts)} document chunks")
    return _embed(
        texts=texts,
        task_type="retrieval_document",
    )


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query for retrieval.
    Returns shape: (1, EMBEDDING_DIM)
    """

    logger.debug("Embedding query")
    return _embed(
        texts=[query],
        task_type="retrieval_query",
    )
# app/db/faiss_store.py

import os
import logging
import threading
import faiss
import numpy as np
from app.config.settings import EMBEDDING_DIM, FAISS_INDEX_PATH, FAISS_SAVE_EVERY

logger = logging.getLogger(__name__)

faiss_lock = threading.Lock()
_vectors_since_last_save: int = 0


# Index construction
def _build_fresh_index() -> faiss.IndexIDMap:
    """Inner product index — cosine similarity when vectors are L2-normalised."""
    base = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(base)


def load_or_create_index() -> faiss.IndexIDMap:
    """
    Load index from disk if it exists and is valid. On corruption: rename for inspection, start fresh.
    Run POST /admin/rebuild-index to restore vectors from MongoDB.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"FAISS index loaded — {index.ntotal} vectors from {FAISS_INDEX_PATH}")
            return index
        except Exception:
            corrupt_path = FAISS_INDEX_PATH + ".corrupt"
            os.rename(FAISS_INDEX_PATH, corrupt_path)
            logger.exception(
                f"FAISS index corrupted — renamed to {corrupt_path}. "
                "Starting fresh. Run POST /admin/rebuild-index to restore."
            )
    logger.info("No FAISS index found — creating fresh index")
    return _build_fresh_index()


# Single shared instance
faiss_index: faiss.IndexIDMap = load_or_create_index()


# Write operations
def add_vectors(vectors: np.ndarray, faiss_ids: np.ndarray) -> None:
    """
    Normalize vectors, add them to FAISS with IDs under lock, and conditionally persist the index.
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    # Normalize to ensure cosine similarity works correctly, and to match query normalization
    faiss.normalize_L2(vectors)
    with faiss_lock:
        faiss_index.add_with_ids(vectors, faiss_ids)
        save_index_if_needed(len(vectors))


def remove_vectors(faiss_ids: np.ndarray) -> None:
    """
    Remove vectors by ID under lock and immediately flush to disk.
    """
    id_selector = faiss.IDSelectorArray(faiss_ids)
    with faiss_lock:
        removed = faiss_index.remove_ids(id_selector)
        _flush_index()
    logger.info(f"FAISS: removed {removed} vectors and flushed to disk")


# Read operations
def search_vectors(query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize query, perform FAISS similarity search under lock, and return (distances, ids).
    """
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    # Normalize query to match ingestion normalization
    faiss.normalize_L2(query)
    with faiss_lock:
        distances, ids = faiss_index.search(query, k)
    return distances, ids


# Persistence
def save_index_if_needed(vectors_added: int) -> None:
    """
    Increment counter and save index if threshold reached. Must be called while faiss_lock is held.
    """
    global _vectors_since_last_save
    _vectors_since_last_save += vectors_added
    if _vectors_since_last_save >= FAISS_SAVE_EVERY:
        _flush_index()
        _vectors_since_last_save = 0


def _flush_index() -> None:
    """
    Save the FAISS index to disk atomically. Must be called while faiss_lock is held.
    """
    tmp_path = FAISS_INDEX_PATH + ".tmp"
    faiss.write_index(faiss_index, tmp_path)
    os.replace(tmp_path, FAISS_INDEX_PATH)   # atomic on POSIX, best-effort on Windows
    logger.info(f"FAISS index saved — {faiss_index.ntotal} total vectors")


def force_save_index() -> None:
    """Flush unconditionally, call on graceful shutdown."""
    with faiss_lock:
        _flush_index()

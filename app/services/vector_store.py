import logging
import numpy as np
from pymongo import UpdateOne
from app.config.settings import EMBEDDING_DIM
from app.db.mongodb import chunk_collection
from app.db.qdrant_store import upsert_vectors
from app.services.embedder import embed_documents


logger = logging.getLogger(__name__)


# Embed chunks, insert into Qdrant, and mark indexed in Mongo.
async def add_chunks_to_vector_store(chunk_docs: list[dict]) -> None:
    if not chunk_docs:
        return
    texts = [c["text"] for c in chunk_docs]
    chunk_ids = [c["chunk_id"] for c in chunk_docs]
    doc_ids = [c["doc_id"] for c in chunk_docs]
    
    vectors = embed_documents(texts)
    if vectors.ndim != 2 or vectors.shape != (len(texts), EMBEDDING_DIM):
        raise ValueError(
            f"Embedder returned shape {vectors.shape}, "
            f"expected ({len(texts)}, {EMBEDDING_DIM})"
        )
    
    payloads = [{"doc_id": did} for did in doc_ids]

    bridge_ops = [
        UpdateOne(
            {"chunk_id": chunk_id},
            {"$set": {"index_status": "pending"}},
        )
        for chunk_id in chunk_ids
    ]
    await chunk_collection.bulk_write(bridge_ops, ordered=False)
    
    # Add to Qdrant
    try:
        upsert_vectors(vectors, chunk_ids, payloads)
    except Exception:
        logger.exception(
            f"upsert_vectors failed for {len(chunk_ids)} chunks — "
            "index_status remains 'pending'."
        )
        raise
        
    status_ops = [
        UpdateOne(
            {"chunk_id": chunk_id},
            {"$set": {"index_status": "indexed"}},
        )
        for chunk_id in chunk_ids
    ]
    await chunk_collection.bulk_write(status_ops, ordered=False)
    logger.info(
        f"Indexed {len(chunk_ids)} chunks into Qdrant, "
        f"index_status=indexed"
    )
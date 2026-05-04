import logging
import numpy as np
from pymongo import UpdateOne, ReturnDocument
from app.config.settings import EMBEDDING_DIM
from app.db.mongodb import chunk_collection, counter_collection
from app.db.faiss_store import add_vectors
from app.services.embedder import embed_documents
logger = logging.getLogger(__name__)


async def _reserve_faiss_ids(count: int) -> list[int]:
    """Atomically reserve a contiguous range of FAISS integer IDs."""
    result = await counter_collection.find_one_and_update(
        {"_id": "faiss_id_counter"},
        {"$inc": {"seq": count}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    end   = result.get("seq", count)
    start = end - count
    return list(range(start, end))


async def add_chunks_to_faiss(chunk_docs: list[dict]) -> None:
    """
    Embed chunks, assign FAISS IDs, insert into FAISS, mark indexed in Mongo.
    """
    if not chunk_docs:
        return
    texts = [c["text"] for c in chunk_docs]
    chunk_ids = [c["chunk_id"] for c in chunk_docs]
    vectors = embed_documents(texts)
    if vectors.ndim != 2 or vectors.shape != (len(texts), EMBEDDING_DIM):
        raise ValueError(
            f"Embedder returned shape {vectors.shape}, "
            f"expected ({len(texts)}, {EMBEDDING_DIM})"
        )
    # Reserve atomic IDs
    faiss_ids    = await _reserve_faiss_ids(len(vectors))
    faiss_ids_np = np.array(faiss_ids, dtype="int64")
    bridge_ops = [
        UpdateOne(
            {"chunk_id": chunk_id},
            {"$set": {"faiss_id": fid, "faiss_status": "pending"}},
        )
        for chunk_id, fid in zip(chunk_ids, faiss_ids)
    ]
    await chunk_collection.bulk_write(bridge_ops, ordered=False)
    # Add to FAISS
    try:
        add_vectors(vectors, faiss_ids_np)
    except Exception:
        logger.exception(
            f"add_vectors failed for {len(chunk_ids)} chunks — "
            "faiss_status remains 'pending'. Run /admin/rebuild-index to recover."
        )
        raise
    status_ops = [
        UpdateOne(
            {"chunk_id": chunk_id},
            {"$set": {"faiss_status": "indexed"}},
        )
        for chunk_id in chunk_ids
    ]
    await chunk_collection.bulk_write(status_ops, ordered=False)
    logger.info(
        f"Indexed {len(chunk_ids)} chunks — "
        f"FAISS IDs {faiss_ids[0]}–{faiss_ids[-1]}, "
        f"faiss_status=indexed"
    )
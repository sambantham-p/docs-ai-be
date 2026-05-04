import logging
from app.config.settings import FAISS_CANDIDATE_CAP, RETRIEVAL_BASE_FETCH, RETRIEVAL_FILTERED_FETCH
from app.db.faiss_store import search_vectors
from app.db.mongodb import chunk_collection
from app.services.embedder import embed_query
from app.services.mmr import mmr_select


logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    doc_id: str | None = None,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    query_vec = embed_query(query)
    multiplier  = RETRIEVAL_FILTERED_FETCH if doc_id else RETRIEVAL_BASE_FETCH
    fetch_count = min(top_k * multiplier, FAISS_CANDIDATE_CAP)
    distances, faiss_int_ids = search_vectors(query_vec, fetch_count)
    scores = distances[0]
    ranked_pairs: list[tuple[int, float]] = [
        (int(fid), float(score))
        for fid, score in zip(faiss_int_ids[0], scores)
        if fid != -1
    ]
    if not ranked_pairs:
        logger.warning("FAISS returned no results")
        return []
    ranked_faiss_ids   = [fid   for fid, _ in ranked_pairs]
    score_by_faiss_id  = {fid:  score for fid, score in ranked_pairs}
    mongo_filter: dict = {
        "faiss_id":     {"$in": ranked_faiss_ids},
        "faiss_status": "indexed",
    }
    if doc_id:
        mongo_filter["doc_id"] = doc_id
    raw_chunks = await chunk_collection.find(
        mongo_filter,
        {
            "_id":           0,
            "chunk_id":      1,
            "doc_id":        1,
            "text":          1,
            "section_title": 1,
            "chunk_index":   1,
            "faiss_id":      1,
        },
    ).to_list(length=fetch_count)
    if not raw_chunks:
        logger.warning(f"No indexed chunks found (doc_id={doc_id})")
        return []
    seen: set[int] = set()
    deduped_chunks: list[dict] = []
    for c in raw_chunks:
        fid = c["faiss_id"]
        if fid not in seen:
            seen.add(fid)
            deduped_chunks.append(c)
        else:
            logger.warning(f"Duplicate faiss_id {fid} in Mongo — skipping")
    chunk_by_faiss_id = {
        c["faiss_id"]: {
            **c,
            "score": score_by_faiss_id.get(c["faiss_id"], 0.0),
        }
        for c in deduped_chunks
    }
    available_ids = [
        fid for fid in ranked_faiss_ids
        if fid in chunk_by_faiss_id
    ]
    mmr_ids = mmr_select(
        query_vec=query_vec[0],  # shape (EMBEDDING_DIM,) — MMR needs 1D
        ranked_faiss_ids=available_ids,
        top_k=top_k,
        lambda_param=lambda_param,
    )
    ordered = [
        chunk_by_faiss_id[fid]
        for fid in mmr_ids
        if fid in chunk_by_faiss_id
    ]
    missing = set(ranked_faiss_ids) - set(chunk_by_faiss_id.keys())
    if missing:
        logger.warning(
            f"FAISS/Mongo mismatch — {len(missing)} faiss_ids not found in Mongo "
            f"(first 5: {list(missing)[:5]}). "
            "Possible cause: chunk deleted or faiss_status not indexed."
        )
    if len(ordered) < top_k:
        logger.warning(
            f"Result starvation: {len(ordered)}/{top_k} "
            f"(doc_id={doc_id}, cap={FAISS_CANDIDATE_CAP})"
        )
    return ordered[:top_k]
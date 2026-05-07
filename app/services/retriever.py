import logging
from app.config.settings import QDRANT_CANDIDATE_CAP, RETRIEVAL_BASE_FETCH, RETRIEVAL_FILTERED_FETCH
from app.db.mongodb import chunk_collection
from app.db.qdrant_store import query_vectors
from app.services.embedder import embed_query
from app.utils.chunker_utils import is_valid_chunk
from app.services.mmr import mmr_select
from app.services.rerank import rerank_async

logger = logging.getLogger(__name__)





async def retrieve(
    query: str,
    doc_id: str | None = None,
    top_k: int = 5,
    lambda_param: float = 0.6,
) -> tuple[list[dict], bool]:
    query_vec = embed_query(query)
    multiplier  = RETRIEVAL_FILTERED_FETCH if doc_id else RETRIEVAL_BASE_FETCH
    fetch_count = max(
        min(top_k * multiplier, QDRANT_CANDIDATE_CAP),
        20  # minimum pool
    )
    scores, ranked_chunk_ids = query_vectors(query_vec, fetch_count, doc_id=doc_id)
    if not ranked_chunk_ids:
        logger.warning("Qdrant returned no results")
        return [], True
    score_by_chunk_id = {cid: score for cid, score in zip(ranked_chunk_ids, scores)}
    mongo_filter: dict = {
        "chunk_id": {"$in": ranked_chunk_ids},
        "index_status": "indexed",
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
        },
    ).to_list(length=None)
    if not raw_chunks:
        logger.warning(f"No indexed chunks found in Mongo (doc_id={doc_id})")
        return [], True
    chunk_by_id = {}
    for c in raw_chunks:
        cid = c["chunk_id"]
        if cid not in chunk_by_id:
            chunk_by_id[cid] = {
                **c,
                "score": score_by_chunk_id.get(cid, 0.0),
            }
        else:
            logger.warning(f"Duplicate chunk_id {cid} in Mongo — skipping")
    
    available_ids = [
        cid for cid in ranked_chunk_ids
        if cid in chunk_by_id
    ]
    if not available_ids:
        logger.warning("No overlap between Qdrant results and Mongo documents")
        return [], True
    filtered_ids = [
        cid for cid in available_ids
        if is_valid_chunk(chunk_by_id[cid]["text"])
    ]
    if not filtered_ids:
        logger.warning("All chunks filtered — fallback to available_ids")
        filtered_ids = available_ids
    elif len(filtered_ids) < top_k:
        logger.warning("Low filtered pool — mixing fallback")
        filtered_ids = filtered_ids + [
            cid for cid in available_ids if cid not in filtered_ids
        ]
    MMR_POOL = min(len(filtered_ids), max(top_k * 5, 15))
    mmr_ids = mmr_select(
        query_vec=query_vec[0],
        ranked_chunk_ids=filtered_ids,
        top_k=MMR_POOL,
        lambda_param=lambda_param,
    )
    ordered = [chunk_by_id[cid] for cid in mmr_ids if cid in chunk_by_id]
    reranked, allow_general = await rerank_async(query, ordered, top_k)
    missing = set(ranked_chunk_ids) - set(chunk_by_id)
    if missing:
        logger.warning(
            f"Qdrant/Mongo mismatch — {len(missing)} missing "
            f"(first {top_k}: {list(missing)[:top_k]})"
        )
    if len(reranked) < top_k:
        logger.warning(
            f"Result starvation after rerank: {len(reranked)}/{top_k} "
            f"(doc_id={doc_id}, cap={QDRANT_CANDIDATE_CAP})"
        )
    return reranked, allow_general
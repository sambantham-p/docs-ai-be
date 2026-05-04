import logging
import numpy as np
from app.db.faiss_store import faiss_index, faiss_lock

logger = logging.getLogger(__name__)


def mmr_select(
    query_vec: np.ndarray,
    ranked_faiss_ids: list[int],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[int]:
    if not ranked_faiss_ids:
        return []
    vectors: dict[int, np.ndarray] = {}
    failed: list[int] = []

    with faiss_lock:
        for fid in ranked_faiss_ids:
            try:
                vectors[fid] = faiss_index.reconstruct(fid)
            except Exception:
                failed.append(fid)
    if failed:
        logger.warning(
            f"MMR: could not reconstruct {len(failed)} vectors from FAISS "
            f"(first 5: {failed[:5]}) — skipping them"
        )
    candidates = [fid for fid in ranked_faiss_ids if fid in vectors]
    if not candidates:
        logger.warning("MMR: no reconstructable candidates — returning empty")
        return []
    selected: list[int] = []
    while candidates and len(selected) < top_k:
        mmr_scores: list[tuple[int, float]] = []
        for fid in candidates:
            relevance = float(np.dot(query_vec, vectors[fid]))
            diversity_penalty = (
                0.0 if not selected
                else max(
                    float(np.dot(vectors[fid], vectors[s]))
                    for s in selected
                )
            )
            mmr_score = (
                    lambda_param * relevance
                    - (1.0 - lambda_param) * diversity_penalty
            )
            mmr_scores.append((fid, mmr_score))
        best_id = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_id)
        candidates.remove(best_id)
    return selected
import logging
import numpy as np
from app.db.qdrant_store import fetch_vectors_by_ids


logger = logging.getLogger(__name__)


def mmr_select(
    query_vec: np.ndarray,
    ranked_chunk_ids: list[str],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[str]:
    if not ranked_chunk_ids:
        return []
    vectors = fetch_vectors_by_ids(ranked_chunk_ids)
    failed = [cid for cid in ranked_chunk_ids if cid not in vectors]
    if failed:
        logger.warning(
            f"MMR: could not reconstruct {len(failed)} vectors from Qdrant "
            f"(first 5: {failed[:5]}) — skipping them"
        )
    candidates = [cid for cid in ranked_chunk_ids if cid in vectors]
    if not candidates:
        logger.warning("MMR: no reconstructable candidates — returning empty")
        return []
    selected: list[str] = []
    while candidates and len(selected) < top_k:
        mmr_scores: list[tuple[str, float]] = []
        for cid in candidates:
            relevance = float(np.dot(query_vec, vectors[cid]))
            diversity_penalty = (
                0.0 if not selected
                else max(
                    float(np.dot(vectors[cid], vectors[s]))
                    for s in selected
                )
            )
            mmr_score = (
                    lambda_param * relevance
                    - (1.0 - lambda_param) * diversity_penalty
            )
            mmr_scores.append((cid, mmr_score))
        best_id = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_id)
        candidates.remove(best_id)
    return selected
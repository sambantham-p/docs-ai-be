import asyncio
import logging
import threading
from typing import List, Dict, Any

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

# NOTE:
# No hard thresholds — model is relative scorer

_ranker: Ranker | None = None
_init_lock = threading.Lock()


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        with _init_lock:
            if _ranker is None:
                logger.info("Loading FlashRank model (MiniLM reranker)...")
                _ranker = Ranker(
                    model_name="ms-marco-MiniLM-L-12-v2",
                    cache_dir=".cache/flashrank",
                )
    return _ranker


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    # 🚨 DO NOT FILTER — cross-encoder needs full candidate pool
    passages = [
        {"id": i, "text": c.get("text", "")}
        for i, c in enumerate(chunks)
    ]

    request = RerankRequest(query=query, passages=passages)

    try:
        results = _get_ranker().rerank(request)
    except Exception:
        logger.exception("Rerank failed — fallback to original order")
        return [{**c, "rerank_score": 0.0} for c in chunks][:top_k]

    all_scored: List[Dict[str, Any]] = []

    for r in results:
        score = float(r.get("score", 0.0))
        base = chunks[r["id"]]

        all_scored.append({
            **base,
            "rerank_score": score
        })

    if not all_scored:
        logger.warning("Empty rerank output — fallback")
        return [{**c, "rerank_score": 0.0} for c in chunks][:top_k]

    # ✅ ONLY operation: sorting
    all_scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 🔍 Observability (relative, not absolute)
    top_score = all_scored[0]["rerank_score"]
    logger.debug(f"Top rerank score: {top_score:.4f}")

    return all_scored[:top_k]


async def rerank_async(query: str, chunks: List[Dict[str, Any]], top_k: int):
    return await asyncio.to_thread(rerank, query, chunks, top_k)
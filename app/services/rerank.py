import asyncio
import logging
import threading
from typing import List, Dict, Any, Tuple

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

_ranker: Ranker | None = None
_init_lock = threading.Lock()

# Reranking logic disabled for troubleshooting


# ── Thresholds ────────────────────────────────────────────────────────────────
# Tune these based on your corpus and domain.
#
# RELEVANCE_THRESHOLD:
#   FlashRank top score below this → retrieval likely failed.
#   Start at 0.4, raise if hybrid triggers too often on partial hits.
#
# MIN_CONTEXT_CHARS:
#   Even if score is low, if we retrieved enough text, stay strict.
#   Prevents weak partial retrieval from incorrectly triggering hybrid mode.
#   200 chars ≈ 1-2 meaningful sentences minimum.
#
RELEVANCE_THRESHOLD = 0.4
MIN_CONTEXT_CHARS = 200


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


# def _should_allow_general(
#     top_score: float,
#     reranked_chunks: List[Dict[str, Any]],
# ) -> bool:
#     """
#     Decides whether to allow general knowledge fallback.
#
#     Hybrid mode triggers ONLY when retrieval essentially failed — not when
#     it partially succeeded. Both conditions must be true:
#
#       1. Top rerank score is below threshold (low relevance signal)
#       2. Total retrieved context is too small to be useful (low content signal)
#
#     This prevents:
#       - Partial hits incorrectly triggering hybrid mode
#       - Model extending or overwriting real document content
#       - Provenance contamination (doc + general knowledge blending)
#     """
#     if not reranked_chunks:
#         logger.debug("allow_general=True — no chunks retrieved")
#         return True
#
#     total_context_chars = sum(
#         len(c.get("text", "")) for c in reranked_chunks
#     )
#
#     score_too_low = top_score < RELEVANCE_THRESHOLD
#     context_too_thin = total_context_chars < MIN_CONTEXT_CHARS
#
#     allow = score_too_low and context_too_thin
#
#     logger.debug(
#         f"Grounding gate | top_score={top_score:.4f} "
#         f"(threshold={RELEVANCE_THRESHOLD}) | "
#         f"context_chars={total_context_chars} "
#         f"(min={MIN_CONTEXT_CHARS}) | "
#         f"allow_general={allow}"
#     )
#
#     return allow


# def rerank(
#     query: str,
#     chunks: List[Dict[str, Any]],
#     top_k: int,
# ) -> Tuple[List[Dict[str, Any]], bool]:
#     """
#     Reranks chunks using FlashRank and determines grounding policy.
#
#     Returns:
#         reranked_chunks: top_k chunks sorted by relevance score
#         allow_general:   True only when retrieval essentially failed
#                          (low score AND thin context)
#     """
#     if not chunks:
#         return [], True
#
#     passages = [
#         {"id": i, "text": c.get("text", "")}
#         for i, c in enumerate(chunks)
#         ]
#
#     request = RerankRequest(query=query, passages=passages)
#
#     try:
#         results = _get_ranker().rerank(request)
#     except Exception:
#         logger.exception("Rerank failed — falling back to original order")
#         return [{**c, "rerank_score": 0.0} for c in chunks][:top_k], True
#
#     if not results:
#         logger.warning("Empty rerank output — falling back to original order")
#         return [{**c, "rerank_score": 0.0} for c in chunks][:top_k], True
#
#     all_scored: List[Dict[str, Any]] = [
#         {**chunks[r["id"]], "rerank_score": float(r.get("score", 0.0))}
#         for r in results
#     ]
#
#     all_scored.sort(key=lambda x: x["rerank_score"], reverse=True)
#
#     top_score = all_scored[0]["rerank_score"]
#     reranked_top_k = all_scored[:top_k]
#     allow_general = _should_allow_general(top_score, reranked_top_k)
#
#     return reranked_top_k, allow_general


# async def rerank_async(
#     query: str,
#     chunks: List[Dict[str, Any]],
#     top_k: int,
# ) -> Tuple[List[Dict[str, Any]], bool]:
#     return await asyncio.to_thread(rerank, query, chunks, top_k)
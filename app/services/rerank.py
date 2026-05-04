# app/services/reranker.py

import asyncio
import logging
import threading
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model      = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
_model_lock = threading.Lock()


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    with _model_lock:
        scores = _model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


async def rerank_async(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    return await asyncio.to_thread(rerank, query, chunks, top_k)
import logging
from app.prompt import SYSTEM_PROMPT, build_prompt
from app.services.confidence import compute_confidence, confidence_label
from app.services.context_builder import build_context
from app.services.llm import generate_answer, MAX_CONTEXT_CHARS
from app.services.retriever import retrieve

logger = logging.getLogger(__name__)

_FALLBACK = "I don't have enough information to answer this question."


def _truncate_chunks_by_chars(
    chunks: list[dict],
    max_chars: int,
) -> list[dict]:
    kept:        list[dict] = []
    total_chars: int        = 0
    for chunk in chunks:
        chunk_len = len(chunk.get("text", ""))
        if total_chars + chunk_len > max_chars:
            logger.warning(
                f"Context budget reached at chunk {len(kept)} — "
                f"dropping {len(chunks) - len(kept)} lower-priority chunks"
            )
            break
        kept.append(chunk)
        total_chars += chunk_len
    return kept


def _fallback_response(chunks: list[dict] | None = None) -> dict:
    return {
        "answer":           _FALLBACK,
        "sources":          [],
        "chunks":           chunks or [],
        "confidence":       0.0,
        "confidence_label": "low",
    }


async def answer_query(
    query:  str,
    doc_id: str | None = None,
    top_k:  int = 5,
) -> dict:
    chunks = await retrieve(query, doc_id=doc_id, top_k=top_k)
    if not chunks:
        logger.warning(f"No chunks retrieved for query: {query!r}")
        return _fallback_response()
    chunks = _truncate_chunks_by_chars(chunks, MAX_CONTEXT_CHARS)
    if not chunks:
        return _fallback_response()
    context = build_context(chunks)
    prompt  = SYSTEM_PROMPT + "\n" + build_prompt(query, context)
    answer  = await generate_answer(prompt)
    if not answer:
        logger.error("LLM returned empty response")
        return _fallback_response(chunks)
    seen:    set[str]  = set()
    sources: list[str] = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            sources.append(c["doc_id"])
    confidence = compute_confidence(chunks, top_k=top_k)
    return {
        "answer":           answer,
        "sources":          sources,
        "chunks":           chunks,
        "confidence":       confidence,
        "confidence_label": confidence_label(confidence),
    }


async def retrieve_and_build_prompt(
    query:  str,
    doc_id: str | None = None,
    top_k:  int = 5,
) -> tuple[list[dict], str, float]:
    chunks = await retrieve(query, doc_id=doc_id, top_k=top_k)
    if not chunks:
        return [], "", 0.0
    chunks  = _truncate_chunks_by_chars(chunks, MAX_CONTEXT_CHARS)
    if not chunks:
        return [], "", 0.0
    context    = build_context(chunks)
    prompt     = SYSTEM_PROMPT + "\n" + build_prompt(query, context)
    confidence = compute_confidence(chunks, top_k=top_k)
    return chunks, prompt, confidence
import json
import logging
import re

from fastapi import Request
from app.config.settings import MAX_CONTEXT_CHARS
from app.prompt import build_prompt, get_system_prompt
from app.services.confidence import compute_confidence, confidence_label
from app.services.context_builder import build_context
from app.services.llm import generate_answer, stream_answer
from app.services.retriever import retrieve
from app.utils.stream_utils import split_text


logger = logging.getLogger(__name__)
_FALLBACK = "I don't have enough information to answer this question."


def clean_answer(text: str) -> str:
    text = re.sub(r"\[Doc:.*?\]", "", text)
    return (
        text.replace("\\n", "\n")
            .replace("\n\n\n", "\n\n")
            .strip()
    )


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
        "grounding_mode":   "strict",
    }


async def answer_query(
    query:  str,
    top_k:  int = 5,
) -> dict:
    chunks, allow_general = await retrieve(query, top_k=top_k)
    if not chunks:
        logger.warning(f"No chunks retrieved for query: {query!r}")
        return _fallback_response()

    chunks = _truncate_chunks_by_chars(chunks, MAX_CONTEXT_CHARS)
    if not chunks:
        return _fallback_response()

    context       = build_context(chunks)
    system_prompt = get_system_prompt(allow_general=allow_general)
    user_prompt   = build_prompt(query, context, allow_general=allow_general)
    prompt        = system_prompt + "\n" + user_prompt

    answer = await generate_answer(prompt)
    if not answer:
        logger.error("LLM returned empty response")
        return _fallback_response(chunks)

    answer = clean_answer(answer)

    seen: set[str] = set()
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
        "grounding_mode":   "hybrid" if allow_general else "strict",
    }


async def retrieve_and_build_prompt(
    query:  str,
    top_k:  int = 5,
) -> tuple[list[dict], str, float, bool]:          # ← added allow_general to return type
    chunks, allow_general = await retrieve(query, top_k=top_k)  # ← unpack
    if not chunks:
        return [], "", 0.0, False

    chunks = _truncate_chunks_by_chars(chunks, MAX_CONTEXT_CHARS)
    if not chunks:
        return [], "", 0.0, False

    context       = build_context(chunks)
    system_prompt = get_system_prompt(allow_general=allow_general)
    user_prompt   = build_prompt(query, context, allow_general=allow_general)
    prompt        = system_prompt + "\n" + user_prompt
    confidence    = compute_confidence(chunks, top_k=top_k)

    return chunks, prompt, confidence, allow_general  # ← pass it up


async def empty_stream():
    yield "data: I don't have enough information to answer this question.\n\n"
    yield "data: [DONE]\n\n"


async def generate_qa_stream(
    prompt:       str,
    http_request: Request,
    confidence:   float,
    sources:      list[str],
    allow_general: bool = False,                   # ← added
):
    async for token in stream_answer(prompt):
        if await http_request.is_disconnected():
            logger.info("Client disconnected — stopping stream")
            return
        for part in split_text(token):
            yield f"event: token\ndata: {part}\n\n"

    meta = json.dumps({
        "confidence":       confidence,
        "confidence_label": confidence_label(confidence),
        "sources":          sources,
        "grounding_mode":   "hybrid" if allow_general else "strict",  # ← added
    })
    yield f"event: meta\ndata: {meta}\n\n"
    yield "event: done\ndata: [DONE]\n\n"
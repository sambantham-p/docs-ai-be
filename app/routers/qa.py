import json
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.services.confidence import confidence_label
from app.services.llm import stream_answer
from app.services.qa_service import answer_query, retrieve_and_build_prompt
from app.utils.stream_utils import split_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/qa", tags=["qa"])


class QARequest(BaseModel):
    query:  str        = Field(..., min_length=1)
    doc_id: str | None = None
    top_k:  int        = Field(default=5, ge=1, le=20)


class QAResponse(BaseModel):
    answer:           str
    sources:          list[str]
    confidence:       float = 0.0
    confidence_label: str   = "low"


@router.post("/", response_model=QAResponse)
async def qa_endpoint(request: QARequest) -> QAResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = await answer_query(
        query=request.query,
        doc_id=request.doc_id,
        top_k=request.top_k,
    )
    return QAResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result.get("confidence", 0.0),
        confidence_label=result.get("confidence_label", "low"),
    )


@router.post("/stream")
async def qa_stream_endpoint(
    request: QARequest,
    http_request: Request,
) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    chunks, prompt, confidence = await retrieve_and_build_prompt(
        query=request.query,
        doc_id=request.doc_id,
        top_k=request.top_k,
    )
    if not chunks:
        async def _empty():
            yield "data: I don't have enough information to answer this question.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")
    seen:    set[str]  = set()
    sources: list[str] = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            sources.append(c["doc_id"])
    async def _generate_stream():
        async for token in stream_answer(prompt):
            if await http_request.is_disconnected():
                logger.info("Client disconnected — stopping stream")
                return
            for part in split_text(token):
                yield f"event: token\ndata: {part}\n\n"
        meta = json.dumps({
            "confidence": confidence,
            "confidence_label": confidence_label(confidence),
            "sources": sources,
        })
        yield f"event: meta\ndata: {meta}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        _generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
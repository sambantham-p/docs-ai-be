import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.schemas.requests import QARequest
from app.schemas.responses import QAResponse
from app.services.qa_service import answer_query, retrieve_and_build_prompt, empty_stream, generate_qa_stream


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/qa", tags=["qa"])


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
        return StreamingResponse(empty_stream(), media_type="text/event-stream")
    seen:    set[str]  = set()
    sources: list[str] = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            sources.append(c["doc_id"])
    return StreamingResponse(
        generate_qa_stream(prompt, http_request, confidence, sources),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
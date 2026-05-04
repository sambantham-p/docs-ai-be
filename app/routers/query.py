import logging
from fastapi import APIRouter, HTTPException
from app.services.retriever import retrieve
from app.schemas.requests import QueryRequest
from app.schemas.responses import QueryResponse, ChunkResult


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    chunks = await retrieve(
        query=request.query,
        doc_id=request.doc_id,
        top_k=request.top_k,
    )
    return QueryResponse(
        query=request.query,
        doc_id=request.doc_id,
        total=len(chunks),
        chunks=[ChunkResult(**c) for c in chunks],
    )
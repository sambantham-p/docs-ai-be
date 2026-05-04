import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.retriever import retrieve

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query:  str        = Field(..., min_length=1)
    doc_id: str | None = None
    top_k:  int        = Field(default=5, ge=1, le=20)


class ChunkResult(BaseModel):
    chunk_id:      str
    doc_id:        str
    text:          str
    section_title: str | None
    chunk_index:   int
    faiss_id:      int
    score:         float



class QueryResponse(BaseModel):
    query:  str
    doc_id: str | None
    total:  int
    chunks: list[ChunkResult]


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
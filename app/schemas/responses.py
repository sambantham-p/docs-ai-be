from pydantic import BaseModel


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


class QAResponse(BaseModel):
    answer:           str
    sources:          list[str]
    confidence:       float = 0.0
    confidence_label: str   = "low"


class UploadResponse(BaseModel):
    status: str
    doc_id: str

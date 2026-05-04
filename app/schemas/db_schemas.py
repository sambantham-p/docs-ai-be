from datetime import datetime
from pydantic import BaseModel


class DocumentModel(BaseModel):
    doc_id: str
    text_length: int
    status: str
    created_at: datetime
    source_type: str | None = None
    filename: str | None = None
    file_size: int | None = None
    url: str | None = None


class ChunkModel(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_title: str | None = None
    section_index: int | None = None
    text: str
    token_start: int | None = None
    token_end: int | None = None
    faiss_id: int | None = None
    faiss_status: str
    created_at: datetime

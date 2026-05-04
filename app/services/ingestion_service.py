import asyncio
from datetime import datetime, timezone
import logging
import uuid
import httpx
from pathlib import Path
from urllib.parse import urlparse
from fastapi import BackgroundTasks, HTTPException, UploadFile
from app.config.settings import ALLOWED_EXTENSIONS, MAX_EXTRACTED_TEXT, MAX_FILE_SIZE, UPLOAD_DIR
from app.constants.db_constants import (
    STATUS_PROCESSING,
    STATUS_READY,
    STATUS_INDEX_FAILED,
    FAISS_STATUS_UNINDEXED,
    FIELD_DOC_ID,
    FIELD_CHUNK_ID,
    FIELD_STATUS,
    FIELD_CREATED_AT,
    FIELD_CHUNK_INDEX,
    FIELD_FAISS_ID,
    FIELD_FAISS_STATUS
)
from app.db.mongodb import doc_collection, chunk_collection
from app.services.document_parser import parse_file, parse_url_from_content
from app.services.hybrid_chunker import hybrid_chunk
from app.services.vector_store import add_chunks_to_faiss
from app.utils.file_utils import delete_file, validate_mime
from app.utils.http_client_util import http_client


logger = logging.getLogger(__name__)


# Background task to index chunks and mark doc ready.
async def _index_and_mark_ready(chunk_ids: list[str], doc_id: str) -> None:
    try:
        chunk_docs = await chunk_collection.find(
            {FIELD_CHUNK_ID: {"$in": chunk_ids}},
            {"_id": 0},
        ).to_list(length=None)
        if not chunk_docs:
            await doc_collection.update_one(
                {FIELD_DOC_ID: doc_id},
                {"$set": {FIELD_STATUS: STATUS_INDEX_FAILED}},
            )
            return
        await add_chunks_to_faiss(chunk_docs)
        await doc_collection.update_one(
            {FIELD_DOC_ID: doc_id},
            {"$set": {FIELD_STATUS: STATUS_READY}},
        )
    except Exception:
        logger.exception(f"Indexing failed for doc {doc_id}")
        await doc_collection.update_one(
            {FIELD_DOC_ID: doc_id},
            {"$set": {FIELD_STATUS: STATUS_INDEX_FAILED}},
        )


# Store doc and chunks in MongoDB, then trigger background indexing.
async def _store_document_and_chunks(
    text: str,
    metadata: dict,
    background_tasks: BackgroundTasks,
) -> str:
    doc_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc)
    doc_record = {
        FIELD_DOC_ID: doc_id,
        "text_length": len(text),
        FIELD_STATUS: STATUS_PROCESSING,
        FIELD_CREATED_AT: now,
        **metadata,
    }
    await doc_collection.insert_one(doc_record)
    raw_chunks = hybrid_chunk(text, doc_id)
    if not raw_chunks:
        await doc_collection.delete_one({FIELD_DOC_ID: doc_id})
        raise HTTPException(status_code=400, detail="Text could not be chunked")

    chunk_docs = [
        {
            FIELD_CHUNK_ID: str(uuid.uuid4()),
            FIELD_DOC_ID: doc_id,
            FIELD_CHUNK_INDEX: chunk["chunk_index"],
            "section_title": chunk.get("section_title"),
            "section_index": chunk.get("section_index"),
            "text": chunk["text"],
            "token_start": chunk.get("token_start"),
            "token_end": chunk.get("token_end"),
            FIELD_FAISS_ID: None,
            FIELD_FAISS_STATUS: FAISS_STATUS_UNINDEXED,
            FIELD_CREATED_AT: now,
        }
        for chunk in raw_chunks
    ]
    await chunk_collection.insert_many(chunk_docs)
    chunk_ids = [c[FIELD_CHUNK_ID] for c in chunk_docs]
    background_tasks.add_task(_index_and_mark_ready, chunk_ids, doc_id)
    return doc_id


def parse_uploaded_file(file_path: str, filename: str) -> str:
    return parse_file(file_path, filename)


def parse_uploaded_url(html: str, url: str) -> str:
    return parse_url_from_content(html, url)


async def process_uploaded_file(file: UploadFile, background_tasks: BackgroundTasks) -> dict:
    file_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file name")
        logger.info(f"Upload started: {file.filename}")
        # Extension check
        ext = Path(file.filename).suffix.lower().replace(".", "")
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}",
            )
        # write uploaded file to disk in chunks with size validation
        safe_name = Path(file.filename).name
        unique_name = f"{uuid.uuid4()}_{safe_name}"
        file_path = UPLOAD_DIR / unique_name
        total_bytes = 0
        try:
            with open(file_path, "wb") as buffer:
                # read the file in 1 MB chunks to avoid loading large files into memory
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=400,
                            detail="File too large (max 10 MB)",
                        )
                    buffer.write(chunk)

        except HTTPException:
            raise
        except Exception:
            logger.exception("Error while streaming file")
            raise HTTPException(
                status_code=400,
                detail="Failed to process uploaded file",
            )
        # Empty file check
        if total_bytes == 0:
            logger.warning(f"Empty file uploaded: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )
        logger.info(f"File saved ({total_bytes} bytes): {file_path}")
        # MIME type validation using magic bytes
        validate_mime(file_path, ext)
        text = await asyncio.to_thread(
            parse_uploaded_file, str(file_path), file.filename
        )
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable content")
        if len(text) > MAX_EXTRACTED_TEXT:
            raise HTTPException(status_code=400, detail="Text too large")
        doc_id = await _store_document_and_chunks(
            text,{"source_type": "file", "filename": file.filename, "file_size": total_bytes},background_tasks,
        )
        background_tasks.add_task(delete_file, file_path)
        return {
            "status": STATUS_PROCESSING,
            "doc_id": doc_id,
        }
    except HTTPException:
        if file_path:
            delete_file(file_path)
        raise
    except Exception:
        logger.exception("File upload failed")
        if file_path:
            delete_file(file_path)
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_uploaded_url(
    url: str,
    background_tasks: BackgroundTasks,
) -> dict:
    try:
        url = url.strip() if url else None
        if not url:
            raise HTTPException(status_code=400, detail="URL cannot be empty")
        logger.info(f"URL upload started: {url}")
        # Auto-add scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL")
        parts = (parsed.hostname or "").split(".")
        if len(parts) < 2 or any(len(p) == 0 for p in parts):
            raise HTTPException(status_code=400, detail="Invalid domain")
        try:
            response = await http_client.get(
                url,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning(f"URL inaccessible: {url} (HTTP {e.response.status_code})")
            raise HTTPException(
                status_code=400,
                detail=f"URL not accessible (HTTP {e.response.status_code})",
            )
        except httpx.RequestError:
            logger.error(f"Network error fetching URL: {url}")
            raise HTTPException(status_code=400, detail="Failed to reach URL")
        # Content-Type validation
        content_type = response.headers.get("content-type", "").lower()
        if "text" not in content_type and "html" not in content_type:
            logger.warning(f"Unsupported content-type: {content_type} for URL: {url}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {content_type}",
            )
        if not response.text or not response.text.strip():
            logger.warning(f"Empty response body from URL: {url}")
            raise HTTPException(
                status_code=400,
                detail="Empty response body — nothing to extract",
            )
        text = await asyncio.to_thread(parse_url_from_content, response.text, url)
        if not text or not text.strip():
            logger.warning(f"No content extracted from: {url}")
            raise HTTPException(
                status_code=400,
                detail="No extractable content found",
            )
        if len(text) > MAX_EXTRACTED_TEXT:
            raise HTTPException(status_code=400, detail="Text too large")
        # Store and index
        doc_id = await _store_document_and_chunks(
            text,
            {"source_type": "url", "url": url},
            background_tasks,
        )
        logger.info(f"URL parsed and stored: {url}")
        return {
            "status": STATUS_PROCESSING,
            "doc_id": doc_id,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("URL processing failed")
        raise HTTPException(status_code=500, detail="Internal server error")

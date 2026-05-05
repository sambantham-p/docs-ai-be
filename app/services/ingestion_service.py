import asyncio
from datetime import datetime, timezone
import logging
import uuid
import httpx
from pathlib import Path
from urllib.parse import urlparse
from fastapi import BackgroundTasks, HTTPException, UploadFile
from pymongo.errors import DuplicateKeyError
from app.config.settings import ALLOWED_EXTENSIONS, MAX_EXTRACTED_TEXT, MAX_FILE_SIZE, UPLOAD_DIR
from app.constants.db_constants import (
    STATUS_PROCESSING,
    STATUS_READY,
    STATUS_INDEX_FAILED,
    INDEX_STATUS_UNINDEXED,
    FIELD_DOC_ID,
    FIELD_CHUNK_ID,
    FIELD_STATUS,
    FIELD_CREATED_AT,
    FIELD_CHUNK_INDEX,
    FIELD_INDEX_STATUS,
)
from app.db.mongodb import doc_collection, chunk_collection
from app.services.document_parser import parse_file, parse_url_from_content
from app.services.hybrid_chunker import hybrid_chunk
from app.services.vector_store import add_chunks_to_vector_store
from app.utils.file_utils import delete_file, validate_mime
from app.utils.chunker_utils import hash_text
from app.utils.http_client_util import http_client


logger = logging.getLogger(__name__)
CHUNK_DEDUP_BATCH = 200


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
        await add_chunks_to_vector_store(chunk_docs)
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


# Store document and its chunks in MongoDB.
# Returns doc_id and boolean indicating if it was a duplicate.
async def _store_document_and_chunks(
    text: str,
    metadata: dict,
    background_tasks: BackgroundTasks,
) -> tuple[str, bool]:
    content_hash = hash_text(text)
    # deduplicate check
    existing_doc = await doc_collection.find_one(
        {"content_hash": content_hash},
        {"doc_id": 1},
    )
    if existing_doc:
        logger.info(f"[DEDUP] Document hit: hash={content_hash[:12]}, doc_id={existing_doc['doc_id']}")
        return existing_doc["doc_id"], True
    #New document insert
    doc_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc)
    doc_record = {
        FIELD_DOC_ID:     doc_id,
        "content_hash":   content_hash,
        "text_length":    len(text),
        FIELD_STATUS:     STATUS_PROCESSING,
        FIELD_CREATED_AT: now,
        **metadata,
    }
    try:
        await doc_collection.insert_one(doc_record)
    except DuplicateKeyError:
        # Race condition (find_one and insert_one)
        existing = await doc_collection.find_one(
            {"content_hash": content_hash},
            {"doc_id": 1},
        )
        logger.info(f"[DEDUP] Race condition caught: hash={content_hash[:12]}")
        return existing["doc_id"], True
    # Chunking
    raw_chunks = hybrid_chunk(text, doc_id)
    if not raw_chunks:
        await doc_collection.delete_one({FIELD_DOC_ID: doc_id})
        raise HTTPException(status_code=400, detail="Text could not be chunked")

    all_hashes = [c["chunk_hash"] for c in raw_chunks]
    # Chunk-level dedup
    existing_hashes: set[str] = set()
    for i in range(0, len(all_hashes), CHUNK_DEDUP_BATCH):
        batch = all_hashes[i:i + CHUNK_DEDUP_BATCH]
        cursor = chunk_collection.find(
            {
                "chunk_hash":       {"$in": batch},
                FIELD_INDEX_STATUS: "indexed",
            },
            {"chunk_hash": 1, "_id": 0},
        )
        async for doc in cursor:
            existing_hashes.add(doc["chunk_hash"])
    # Build new chunk docs
    seen_hashes = set(existing_hashes)
    chunk_docs = []
    for chunk in raw_chunks:
        h = chunk["chunk_hash"]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        chunk_docs.append({
            FIELD_CHUNK_ID:    str(uuid.uuid4()),
            FIELD_DOC_ID:      doc_id,
            FIELD_CHUNK_INDEX: chunk["chunk_index"],
            "section_title":   chunk.get("section_title"),
            "section_index":   chunk.get("section_index"),
            "text":            chunk["text"],
            "chunk_hash":      chunk["chunk_hash"],
            "token_start":     chunk.get("token_start"),
            "token_end":       chunk.get("token_end"),
            FIELD_INDEX_STATUS: INDEX_STATUS_UNINDEXED,
            FIELD_CREATED_AT:  now,
        })
    # Accurate skip count
    skipped_count = len(all_hashes) - len(chunk_docs)
    logger.info(
        f"[DEDUP] doc={doc_id} | total={len(all_hashes)} "
        f"| skipped={skipped_count} | inserting={len(chunk_docs)}"
    )
    if not chunk_docs:
        logger.info(f"[DEDUP] All chunks duplicate — marking ready: doc_id={doc_id}")
        await doc_collection.update_one(
            {FIELD_DOC_ID: doc_id},
            {"$set": {FIELD_STATUS: STATUS_READY}},
        )
        return doc_id, False
    await chunk_collection.insert_many(chunk_docs, ordered=False)
    chunk_ids = [c[FIELD_CHUNK_ID] for c in chunk_docs]
    background_tasks.add_task(_index_and_mark_ready, chunk_ids, doc_id)
    return doc_id, False


# Process an uploaded file, extract text, and initiate indexing.
async def process_uploaded_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> dict:
    file_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file name")
        logger.info(f"Upload started: {file.filename}")
        ext = Path(file.filename).suffix.lower().replace(".", "")
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}",
            )
        safe_name   = Path(file.filename).name
        unique_name = f"{uuid.uuid4()}_{safe_name}"
        file_path   = UPLOAD_DIR / unique_name
        total_bytes = 0
        try:
            with open(file_path, "wb") as buffer:
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
            raise HTTPException(status_code=400, detail="Failed to process uploaded file")
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        logger.info(f"File saved ({total_bytes} bytes): {file_path}")
        validate_mime(file_path, ext)
        text = await asyncio.to_thread(parse_file, str(file_path), file.filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable content")
        if len(text) > MAX_EXTRACTED_TEXT:
            raise HTTPException(status_code=400, detail="Text too large")
        doc_id, is_duplicate = await _store_document_and_chunks(
            text,
            {"source_type": "file", "filename": file.filename, "file_size": total_bytes},
            background_tasks,
        )
        background_tasks.add_task(delete_file, file_path)
        return {
            "status": STATUS_READY if is_duplicate else STATUS_PROCESSING,
            "doc_id": doc_id,
            "duplicate": is_duplicate,
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


# Process an uploaded URL, extract text, and initiate indexing.
async def process_uploaded_url(
    url: str,
    background_tasks: BackgroundTasks,
) -> dict:
    try:
        url = url.strip() if url else None
        if not url:
            raise HTTPException(status_code=400, detail="URL cannot be empty")
        logger.info(f"URL upload started: {url}")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL")
        parts = (parsed.hostname or "").split(".")
        if len(parts) < 2 or any(len(p) == 0 for p in parts):
            raise HTTPException(status_code=400, detail="Invalid domain")
        try:
            response = await http_client.get(url, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=400,
                detail=f"URL not accessible (HTTP {e.response.status_code})",
            )
        except httpx.RequestError:
            raise HTTPException(status_code=400, detail="Failed to reach URL")
        content_type = response.headers.get("content-type", "").lower()
        if "text" not in content_type and "html" not in content_type:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {content_type}",
            )
        if not response.text or not response.text.strip():
            raise HTTPException(status_code=400, detail="Empty response body — nothing to extract")

        text = await asyncio.to_thread(parse_url_from_content, response.text, url)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No extractable content found")
        if len(text) > MAX_EXTRACTED_TEXT:
            raise HTTPException(status_code=400, detail="Text too large")
        doc_id, is_duplicate = await _store_document_and_chunks(
            text,
            {"source_type": "url", "url": url},
            background_tasks,
        )
        logger.info(f"URL parsed and stored: {url}")
        return {
            "status":    STATUS_READY if is_duplicate else STATUS_PROCESSING,
            "doc_id":    doc_id,
            "duplicate": is_duplicate,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("URL processing failed")
        raise HTTPException(status_code=500, detail="Internal server error")
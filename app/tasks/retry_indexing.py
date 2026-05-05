import logging
from app.constants.db_constants import (
    FIELD_DOC_ID,
    FIELD_CHUNK_ID,
    FIELD_STATUS,
    FIELD_INDEX_STATUS,
    STATUS_INDEX_FAILED,
    INDEX_STATUS_UNINDEXED,
)
from app.db.mongodb import doc_collection, chunk_collection
from app.services.ingestion_service import _index_and_mark_ready
import asyncio

logger = logging.getLogger(__name__)


_RETRY_SEM = asyncio.Semaphore(5)

async def _retry_one(doc_id: str) -> None:
    async with _RETRY_SEM:
        chunk_ids = await chunk_collection.distinct(
            FIELD_CHUNK_ID,
            {
                FIELD_DOC_ID:       doc_id,
                FIELD_INDEX_STATUS: INDEX_STATUS_UNINDEXED,
            },
        )
        if chunk_ids:
            logger.info(f"[RETRY] Re-indexing {len(chunk_ids)} chunks for doc_id={doc_id}")
            await _index_and_mark_ready(chunk_ids, doc_id)


async def retry_stuck_chunks() -> None:
    stuck_docs = await doc_collection.distinct(
        FIELD_DOC_ID,
        {FIELD_STATUS: STATUS_INDEX_FAILED},
    )
    if not stuck_docs:
        logger.info("[RETRY] No stuck documents found")
        return
    await asyncio.gather(*[_retry_one(doc_id) for doc_id in stuck_docs])
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel
from app.config.settings import MONGO_URI, MONGO_DB

from app.constants.db_constants import (
    MONGO_DOC_COLLECTION,
    MONGO_CHUNK_COLLECTION,
    MONGO_COUNTER_COLLECTION,
    FIELD_DOC_ID,
    FIELD_CHUNK_ID,
    FIELD_STATUS,
    FIELD_CREATED_AT,
    FIELD_CHUNK_INDEX,
    FIELD_FAISS_ID,
    FIELD_FAISS_STATUS
)

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
db: AsyncIOMotorDatabase   = client[MONGO_DB]

# Collections
doc_collection     = db[MONGO_DOC_COLLECTION]
chunk_collection   = db[MONGO_CHUNK_COLLECTION]
counter_collection = db[MONGO_COUNTER_COLLECTION]

# Create all MongoDB indexes on startup.
async def setup_indexes() -> None:
    # documents
    await doc_collection.create_indexes([
        IndexModel([(FIELD_DOC_ID,     ASCENDING)], unique=True),
        IndexModel([(FIELD_STATUS,     ASCENDING)]),
        IndexModel([(FIELD_CREATED_AT, ASCENDING)]),
    ])

    # chunks
    await chunk_collection.create_indexes([
        IndexModel([(FIELD_CHUNK_ID, ASCENDING)], unique=True),
        IndexModel([(FIELD_DOC_ID, ASCENDING)]),
        IndexModel([(FIELD_FAISS_ID, ASCENDING)]),
        IndexModel([(FIELD_FAISS_STATUS, ASCENDING)]),
        IndexModel([(FIELD_DOC_ID, ASCENDING), (FIELD_CHUNK_INDEX, ASCENDING)]),  
        IndexModel([(FIELD_FAISS_ID, ASCENDING), (FIELD_DOC_ID, ASCENDING)]), 
    ])

    logger.info("MongoDB indexes ensured")
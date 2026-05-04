import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel
from app.config.settings import MONGO_URI, MONGO_DB

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
db: AsyncIOMotorDatabase   = client[MONGO_DB]

# Collections
doc_collection     = db["documents"]
chunk_collection   = db["chunks"]
counter_collection = db["counters"]

async def setup_indexes() -> None:
    """
    Create all MongoDB indexes on startup.
    """
    # documents
    await doc_collection.create_indexes([
        IndexModel([("doc_id",     ASCENDING)], unique=True),
        IndexModel([("status",     ASCENDING)]),
        IndexModel([("created_at", ASCENDING)]),
    ])

    # chunks
    await chunk_collection.create_indexes([
        IndexModel([("chunk_id", ASCENDING)], unique=True),
        IndexModel([("doc_id", ASCENDING)]),
        IndexModel([("faiss_id", ASCENDING)]),
        IndexModel([("faiss_status", ASCENDING)]),
        IndexModel([("doc_id", ASCENDING), ("chunk_index", ASCENDING)]),  # range queries within doc
        IndexModel([("faiss_id", ASCENDING), ("doc_id", ASCENDING)]),  # filtered retrieval
    ])

    logger.info("MongoDB indexes ensured")
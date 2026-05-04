# Database collection names
MONGO_DOC_COLLECTION = "documents"
MONGO_CHUNK_COLLECTION = "chunks"
MONGO_COUNTER_COLLECTION = "counters"

# Document status
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_INDEX_FAILED = "index_failed"

# Chunk status
FAISS_STATUS_UNINDEXED = "unindexed"
FAISS_STATUS_INDEXED = "indexed"

# Field names
FIELD_DOC_ID = "doc_id"
FIELD_CHUNK_ID = "chunk_id"
FIELD_STATUS = "status"
FIELD_CREATED_AT = "created_at"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_FAISS_ID = "faiss_id"
FIELD_FAISS_STATUS = "faiss_status"

import logging
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config.settings import EMBEDDING_DIM, QDRANT_PATH, QDRANT_COLLECTION


logger = logging.getLogger(__name__)
client = QdrantClient(path=str(QDRANT_PATH))


if not client.collection_exists(QDRANT_COLLECTION):
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")
else:
    # Validate dimension (important for safety)
    info = client.get_collection(QDRANT_COLLECTION)
    existing_dim = info.config.params.vectors.size
    if existing_dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dim mismatch: Qdrant={existing_dim}, Expected={EMBEDDING_DIM}"
        )
    logger.info(f"Loaded Qdrant collection: {QDRANT_COLLECTION}")


# Insert or update vectors in Qdrant.
def upsert_vectors(
    vectors: np.ndarray,
    chunk_ids: list[str],
    payloads: list[dict] | None = None,
) -> None:
    if len(vectors) == 0:
        logger.warning("No vectors to insert")
        return
    if len(vectors) != len(chunk_ids):
        raise ValueError("Vectors and chunk_ids length mismatch")
    if payloads and len(payloads) != len(chunk_ids):
        raise ValueError("Payloads length mismatch")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Invalid embedding dimension: got {vectors.shape[1]}, expected {EMBEDDING_DIM}"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    if payloads is None:
        payloads = [{} for _ in chunk_ids]
    for i, payload in enumerate(payloads):
        if "doc_id" not in payload:
            raise ValueError(f"Missing 'doc_id' in payload at index {i}")
    points = [
        PointStruct(
            id=cid,
            vector=vec.tolist(),
            payload=payload,
        )
        for cid, vec, payload in zip(chunk_ids, vectors, payloads)
    ]
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )
    logger.info(f"Upserted {len(points)} vectors into Qdrant")


# Search top-k similar vectors.
def query_vectors(
    query_vec: np.ndarray,
    k: int,
    doc_id: str | None = None,
) -> tuple[list[float], list[str]]:
    if query_vec.ndim != 2 or query_vec.shape[0] != 1:
        raise ValueError("query_vec must be shape (1, EMBEDDING_DIM)")
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32)
    query_list = query_vec[0].tolist()
    query_filter = None
    if doc_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )
    try:
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_list,
            query_filter=query_filter,
            limit=k,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        return [], []
    if not results.points:
        return [], []
    scores = [p.score for p in results.points]
    ids = [str(p.id) for p in results.points]
    return scores, ids


# Fetch stored vectors by IDs (used for MMR).
def fetch_vectors_by_ids(
    chunk_ids: list[str],
) -> dict[str, np.ndarray]:
    if not chunk_ids:
        return {}
    try:
        points = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=chunk_ids,
            with_vectors=True,
        )
    except Exception:
        logger.exception("Qdrant retrieve failed")
        return {}
    return {
        str(p.id): np.array(p.vector, dtype=np.float32)
        for p in points
        if p.vector is not None
    }


# Delete all vectors belonging to a document.
def delete_vectors_by_doc(doc_id: str) -> None:
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted vectors for doc_id={doc_id}")
    except Exception:
        logger.exception(f"Failed to delete vectors for doc_id={doc_id}")
import numpy as np


def compute_confidence(chunks: list[dict], top_k: int = 5) -> float:
    if not chunks:
        return 0.0

    def _sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    # Fix 5: use max over all chunks — not assumed sorted
    rerank_scores = [c.get("rerank_score", 0.0) for c in chunks]
    top_rerank    = float(max(_sigmoid(s) for s in rerank_scores)) if rerank_scores else 0.0

    # FAISS cosine similarity — shift -1..1 → 0..1
    faiss_scores = [c.get("score", 0.0) for c in chunks]
    mean_cosine  = float(np.mean([(s + 1) / 2 for s in faiss_scores])) if faiss_scores else 0.0

    # Fix 4: use actual top_k, not hardcoded 5
    coverage = min(len(chunks) / max(top_k, 1), 1.0)

    confidence = (
        0.4 * top_rerank  +
        0.4 * mean_cosine +
        0.2 * coverage
    )

    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"
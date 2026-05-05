import numpy as np


def compute_confidence(chunks: list[dict], top_k: int = 5) -> float:
    if not chunks:
        return 0.0
    # RERANK SCORE
    rerank_scores = [c.get("rerank_score", 0.0) for c in chunks]
    top_rerank = sorted(rerank_scores, reverse=True)[:3]
    mean_rerank = float(np.mean(top_rerank)) if top_rerank else 0.0
    # adaptive scaling
    if mean_rerank < 0.4:
        rerank_norm = min(mean_rerank * 2.5, 1.0)
    else:
        rerank_norm = min(mean_rerank, 1.0)
    vector_scores = [c.get("score", 0.0) for c in chunks]
    top_cosine = sorted(vector_scores, reverse=True)[:3]
    mean_cosine = float(np.mean(top_cosine)) if top_cosine else 0.0
    cosine_norm = max(min(mean_cosine, 1.0), 0.0)
    coverage = min(len(chunks) / max(top_k, 1), 1.0)
    # FINAL SCORE
    confidence = (
        0.4 * rerank_norm +
        0.4 * cosine_norm +
        0.2 * coverage
    )
    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


def confidence_label(score: float) -> str:
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"
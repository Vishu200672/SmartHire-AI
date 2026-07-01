"""
similarity.py  v5
-----------------
Final calibrated scoring engine.

Calibration window: LOW=0.62  HIGH=0.87  (empirical range of all-MiniLM-L6-v2)

Tier thresholds (v5 — tuned to fine-tuned model output distribution):
  TIER_HR  = 60   raw >= 0.770  Highly Recommended
  TIER_REC = 38   raw >= 0.715  Recommended
  TIER_CON = 18   raw >= 0.665  Consider
  NR       < 18   raw <  0.665  Not Recommended

Key change from v4: TIER_HR lowered 70 → 60 because the fine-tuned model
assigns raw 0.73-0.79 to strong matches (gold 0.78-0.88), calibrating to
44-68%. Old threshold of 70% was dropping these into Recommended incorrectly.

Author: SmartHire AI
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

# ── Calibration ───────────────────────────────────────────────
CALIBRATION_LOW  = 0.62
CALIBRATION_HIGH = 0.87

# ── Tier thresholds (calibrated 0-100 scale) ──────────────────
TIER_HR  = 60   # Highly Recommended  →  raw >= 0.770
TIER_REC = 38   # Recommended         →  raw >= 0.715
TIER_CON = 18   # Consider            →  raw >= 0.665
# below TIER_CON  Not Recommended     →  raw <  0.665


def calibrate_score(raw_cosine: float) -> float:
    """Map raw cosine [0,1] → calibrated percentage [0,100]."""
    span = CALIBRATION_HIGH - CALIBRATION_LOW
    return round(min(100.0, max(0.0, (raw_cosine - CALIBRATION_LOW) / span * 100.0)), 2)


def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(f"Shape mismatch: {embedding_a.shape} vs {embedding_b.shape}")
    return float(max(0.0, min(1.0, torch.dot(embedding_a.float(), embedding_b.float()).item())))


def batch_similarity(resume_embeddings: torch.Tensor, jd_embedding: torch.Tensor) -> List[float]:
    if resume_embeddings.dim() != 2:
        raise ValueError(f"resume_embeddings must be 2-D, got {resume_embeddings.shape}")
    if jd_embedding.dim() != 1:
        raise ValueError(f"jd_embedding must be 1-D, got {jd_embedding.shape}")
    jd_vec      = jd_embedding.float().unsqueeze(0)
    resume_vecs = resume_embeddings.float()
    scores      = torch.mm(resume_vecs, jd_vec.t()).squeeze(1)
    scores      = torch.clamp(scores, min=0.0, max=1.0)
    logger.info(
        f"Batch similarity: {len(scores)} resumes | "
        f"max={scores.max().item():.4f}  min={scores.min().item():.4f}  "
        f"mean={scores.mean().item():.4f}"
    )
    return scores.tolist()


def similarity_to_percentage(score: float) -> float:
    return round(score * 100, 2)


def get_recommendation(score_pct: float) -> Tuple[str, str]:
    """
    Map calibrated score → recommendation tier and UI colour.

        >= 60%  Highly Recommended  (neon green)   raw >= 0.770
        >= 38%  Recommended         (cyan)          raw >= 0.715
        >= 18%  Consider            (amber)         raw >= 0.665
        <  18%  Not Recommended     (red)           raw <  0.665
    """
    if score_pct >= TIER_HR:
        return "Highly Recommended", "#4ade80"
    elif score_pct >= TIER_REC:
        return "Recommended", "#00d4ff"
    elif score_pct >= TIER_CON:
        return "Consider", "#fbbf24"
    else:
        return "Not Recommended", "#f87171"


def compute_confidence(score_pct: float, num_candidates: int) -> str:
    if num_candidates >= 5:
        return "High" if (score_pct >= 55 or score_pct < 20) else "Moderate"
    elif num_candidates >= 3:
        return "Moderate"
    elif num_candidates == 2:
        return "Low"
    return "Uncertain"


def compute_percentile_ranks(scores: List[float]) -> List[float]:
    n = len(scores)
    if n == 1:
        return [100.0]
    sorted_scores = sorted(scores)
    return [round(sum(1 for s in sorted_scores if s <= score) / n * 100, 1) for score in scores]


def compute_match_results(
    resume_names: List[str],
    resume_embeddings: torch.Tensor,
    jd_embedding: torch.Tensor,
) -> List[Dict]:
    if len(resume_names) != len(resume_embeddings):
        raise ValueError(f"Mismatch: {len(resume_names)} names vs {len(resume_embeddings)} embeddings.")
    raw_scores = batch_similarity(resume_embeddings, jd_embedding)
    results    = []
    for name, raw in zip(resume_names, raw_scores):
        calibrated            = calibrate_score(raw)
        recommendation, color = get_recommendation(calibrated)
        results.append({
            "name"                : name,
            "score"               : round(raw, 4),
            "score_pct"           : calibrated,
            "recommendation"      : recommendation,
            "recommendation_color": color,
        })
    return results

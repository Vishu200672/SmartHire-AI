"""
model_ensemble.py
-----------------
Multi-model ensemble for improved accuracy and robustness.

Features:
  - Load multiple embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, bge-base-en)
  - Weighted voting across models
  - Individual model scores + ensemble score
  - Model comparison mode

Author: SmartHire AI
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Supported models with their characteristics
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "weight": 0.5,  # Higher weight - more reliable
        "dim": 384,
        "speed": "fast",
        "accuracy": "high",
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2",
        "weight": 0.3,  # Medium weight
        "dim": 768,
        "speed": "medium",
        "accuracy": "very_high",
    },
    "bge-base-en": {
        "name": "bge-base-en",
        "weight": 0.2,  # Lower weight - newer model
        "dim": 768,
        "speed": "fast",
        "accuracy": "high",
    },
}


class ModelEnsemble:
    """
    Ensemble of multiple embedding models for robust matching.
    """

    def __init__(self, model_names: Optional[List[str]] = None, use_weights: bool = True):
        """
        Initialize ensemble with specified models.
        
        Args:
            model_names: List of model names. If None, uses primary only.
            use_weights: Whether to use weighted voting (vs equal voting).
        """
        self.use_weights = use_weights
        self.models = {}
        self.model_names = model_names or ["all-MiniLM-L6-v2"]
        self._load_models()

    def _load_models(self) -> None:
        """Load all specified models."""
        from sentence_transformers import SentenceTransformer
        import torch

        for model_name in self.model_names:
            try:
                logger.info(f"Loading ensemble model: {model_name}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SentenceTransformer(model_name, device=device)
                self.models[model_name] = model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")

        if not self.models:
            raise RuntimeError("No models loaded successfully")
        logger.info(f"Ensemble ready with {len(self.models)} model(s)")

    def encode_all(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode texts with all models.
        
        Returns:
            Dict mapping model_name -> embeddings tensor [N, dim]
        """
        embeddings = {}
        for model_name, model in self.models.items():
            logger.debug(f"Encoding with {model_name}")
            embs = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            embeddings[model_name] = embs.cpu()
        return embeddings

    def ensemble_similarity(
        self,
        resume_embeddings: Dict[str, torch.Tensor],
        jd_embedding: Dict[str, torch.Tensor],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted ensemble similarity score.
        
        Returns:
            (ensemble_score, individual_scores_dict)
        """
        individual_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for model_name in self.models.keys():
            resume_emb = resume_embeddings[model_name]
            jd_emb = jd_embedding[model_name]

            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                resume_emb, jd_emb, dim=1
            )
            sim_score = float(sim.mean())
            individual_scores[model_name] = round(sim_score, 4)

            # Weighted sum
            weight = AVAILABLE_MODELS[model_name]["weight"] if self.use_weights else 1.0
            weighted_sum += sim_score * weight
            total_weight += weight

        ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return round(ensemble_score, 4), individual_scores

    def get_model_info(self) -> Dict:
        """Return info about loaded models."""
        return {
            "num_models": len(self.models),
            "models": {
                name: {
                    "dim": AVAILABLE_MODELS.get(name, {}).get("dim", "unknown"),
                    "weight": AVAILABLE_MODELS.get(name, {}).get("weight", 0),
                    "speed": AVAILABLE_MODELS.get(name, {}).get("speed", "unknown"),
                    "accuracy": AVAILABLE_MODELS.get(name, {}).get("accuracy", "unknown"),
                }
                for name in self.models.keys()
            },
            "use_weights": self.use_weights,
        }


def compare_models(
    resume_text: str,
    jd_text: str,
) -> Dict:
    """
    Compare all available models on a single match.
    
    Returns:
        {
            "all-MiniLM-L6-v2": 0.845,
            "all-mpnet-base-v2": 0.862,
            "bge-base-en": 0.851,
            "ensemble": 0.853,
        }
    """
    from sentence_transformers import SentenceTransformer
    import torch

    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Individual models
    for model_name in AVAILABLE_MODELS.keys():
        try:
            model = SentenceTransformer(model_name, device=device)
            resume_emb = model.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
            jd_emb = model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
            sim = float(torch.nn.functional.cosine_similarity(resume_emb.unsqueeze(0), jd_emb.unsqueeze(0)))
            results[model_name] = round(sim, 4)
        except Exception as e:
            logger.warning(f"Failed to compare with {model_name}: {e}")
            results[model_name] = None

    # Ensemble
    valid_scores = [s for s in results.values() if s is not None]
    if valid_scores:
        ensemble_score = sum(valid_scores) / len(valid_scores)
        results["ensemble"] = round(ensemble_score, 4)

    return results

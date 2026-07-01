"""
advanced_analytics.py
---------------------
Advanced analytics for hiring benchmarking and insights.

Features:
  - Score calibration analysis
  - A/B testing framework
  - Sensitivity analysis (how score changes with weight adjustments)
  - Hiring correlation analysis

Author: SmartHire AI
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calibration_analysis(match_scores: List[float], hiring_outcomes: List[bool]) -> Dict:
    """
    Analyze if scores are well-calibrated with actual outcomes.
    
    Args:
        match_scores: List of predicted match scores (0-100)
        hiring_outcomes: List of booleans (hired=True, rejected=False)
    
    Returns:
        Calibration analysis with confidence intervals
    """
    df = pd.DataFrame({"score": match_scores, "hired": hiring_outcomes})
    
    # Bin scores and compute actual hire rate per bin
    df["score_bin"] = pd.cut(df["score"], bins=[0, 30, 50, 70, 90, 100], right=False)
    calibration = df.groupby("score_bin", observed=True).agg({
        "hired": ["sum", "count", "mean"]
    }).round(3)
    
    # Expected vs actual
    df["expected_hire_pct"] = (df["score"] / 100).round(1) * 100
    correlation = np.corrcoef(df["score"], df["hired"].astype(int))[0, 1]
    
    return {
        "correlation": round(correlation, 3),
        "calibration_by_bin": calibration.to_dict(),
        "well_calibrated": abs(correlation) > 0.6,
    }


def sensitivity_analysis(
    base_score: float,
    similarity_pct: float,
    skill_coverage_pct: float,
) -> Dict:
    """
    Show how score changes with different weight distributions.
    
    Returns:
        {
            "current": 75.5,
            "if_sim_weight_90": 82.3,
            "if_skill_weight_90": 68.5,
            "if_equal": 74.2,
        }
    """
    results = {"current": round(base_score, 2)}
    
    # Extreme emphasis on similarity
    score_sim_heavy = similarity_pct * 0.9 + skill_coverage_pct * 0.1
    results["if_sim_weight_90"] = round(score_sim_heavy, 2)
    
    # Extreme emphasis on skill
    score_skill_heavy = similarity_pct * 0.1 + skill_coverage_pct * 0.9
    results["if_skill_weight_90"] = round(score_skill_heavy, 2)
    
    # Equal weights
    score_equal = (similarity_pct + skill_coverage_pct) / 2
    results["if_equal"] = round(score_equal, 2)
    
    # Range of scores
    all_scores = [results[k] for k in ["current", "if_sim_weight_90", "if_skill_weight_90", "if_equal"]]
    results["score_range"] = (min(all_scores), max(all_scores))
    results["recommendation_stable"] = max(all_scores) - min(all_scores) < 15
    
    return results


def a_b_testing_framework(
    control_scores: List[float],
    treatment_scores: List[float],
) -> Dict:
    """
    Compare two scoring approaches (e.g., original vs ensemble).
    
    Returns:
        Statistical comparison and effect size
    """
    control_mean = np.mean(control_scores)
    treatment_mean = np.mean(treatment_scores)
    
    # T-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(control_scores) + np.var(treatment_scores)) / 2)
    cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        "control_mean": round(control_mean, 2),
        "treatment_mean": round(treatment_mean, 2),
        "improvement": round(treatment_mean - control_mean, 2),
        "p_value": round(p_value, 4),
        "statistically_significant": p_value < 0.05,
        "effect_size_cohens_d": round(cohens_d, 3),
        "effect_size_label": (
            "small" if abs(cohens_d) < 0.5 else
            "medium" if abs(cohens_d) < 0.8 else
            "large"
        ),
    }


def score_distribution_analysis(scores: List[float]) -> Dict:
    """
    Analyze score distribution and identify patterns.
    """
    df = pd.Series(scores)
    
    return {
        "mean": round(df.mean(), 2),
        "median": round(df.median(), 2),
        "std": round(df.std(), 2),
        "min": round(df.min(), 2),
        "max": round(df.max(), 2),
        "q25": round(df.quantile(0.25), 2),
        "q75": round(df.quantile(0.75), 2),
        "skewness": round(df.skew(), 2),
        "kurtosis": round(df.kurtosis(), 2),
        "highly_skewed": abs(df.skew()) > 1,
    }


def hiring_quality_metrics(match_scores: List[float], hiring_outcomes: List[bool]) -> Dict:
    """
    Compute quality metrics: precision, recall, ROC-AUC at different thresholds.
    """
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
    
    scores = np.array(match_scores) / 100  # Normalize to 0-1
    outcomes = np.array(hiring_outcomes, dtype=int)
    
    # ROC-AUC
    if len(set(outcomes)) > 1:
        roc_auc = roc_auc_score(outcomes, scores)
    else:
        roc_auc = None
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(outcomes, scores)
    pr_auc = auc(recall, precision)
    
    # Threshold analysis
    thresholds_to_test = [0.3, 0.5, 0.7, 0.85]
    threshold_metrics = {}
    
    for thresh in thresholds_to_test:
        predictions = (scores >= thresh).astype(int)
        if sum(predictions) > 0:
            precision_at_thresh = sum((predictions == 1) & (outcomes == 1)) / sum(predictions)
            threshold_metrics[f"threshold_{thresh}"] = {
                "precision": round(precision_at_thresh, 3),
                "positive_rate": round(sum(predictions) / len(predictions), 3),
            }
    
    return {
        "roc_auc": round(roc_auc, 3) if roc_auc else None,
        "pr_auc": round(pr_auc, 3),
        "threshold_analysis": threshold_metrics,
        "total_samples": len(outcomes),
        "positive_samples": sum(outcomes),
    }

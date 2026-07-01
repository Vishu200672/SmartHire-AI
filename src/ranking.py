"""
ranking.py
----------
Production candidate ranking engine.

Improvements over v1:
  - Calibrated similarity scores (no score clustering at 70-80%)
  - Weighted skill scoring (critical skills count 3x)
  - Confidence intervals per candidate
  - Percentile rank across the pool
  - AI-generated insight text per candidate
  - important_missing field in CandidateResult
  - Full structured CandidateResult dataclass

Author: SmartHire AI
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.similarity import (
    calibrate_score,
    compute_confidence,
    compute_percentile_ranks,
    get_recommendation,
    similarity_to_percentage,
)
from src.skills import full_skill_analysis, get_skill_weight

logger = logging.getLogger(__name__)


# -- CandidateResult Dataclass ----------------------------------------

@dataclass
class CandidateResult:
    """
    Full result record for one candidate against one job description.
    """
    name                 : str
    resume_text          : str
    similarity_score     : float
    calibrated_sim_pct   : float
    score_pct            : float
    recommendation       : str
    recommendation_color : str
    matching_skills      : List[str] = field(default_factory=list)
    missing_skills       : List[str] = field(default_factory=list)
    critical_missing     : List[str] = field(default_factory=list)
    important_missing    : List[str] = field(default_factory=list)
    resume_only_skills   : List[str] = field(default_factory=list)
    skill_coverage_pct   : float = 0.0
    weighted_coverage_pct: float = 0.0
    skills_by_category   : Dict[str, List[str]] = field(default_factory=dict)
    confidence           : str = "Moderate"
    percentile_rank      : float = 0.0
    rank                 : int = 0
    ai_insight           : str = ""

    def to_dict(self) -> Dict:
        """Serialize to flat dictionary for DataFrame/CSV export."""
        return {
            "Candidate"               : self.name,
            "Rank"                    : self.rank,
            "Match Score (%)"         : self.score_pct,
            "Semantic Similarity (%)" : self.calibrated_sim_pct,
            "Skill Coverage (%)"      : self.skill_coverage_pct,
            "Weighted Coverage (%)"   : self.weighted_coverage_pct,
            "Recommendation"          : self.recommendation,
            "Confidence"              : self.confidence,
            "Percentile Rank"         : self.percentile_rank,
            "Matched Skills"          : ", ".join(self.matching_skills) or "--",
            "Missing Skills"          : ", ".join(self.missing_skills) or "--",
            "Critical Missing"        : ", ".join(self.critical_missing) or "None",
            "AI Insight"              : self.ai_insight,
        }


# -- AI Insight Generator ---------------------------------------------

def generate_insight(result_data: dict, jd_skill_count: int) -> str:
    """Generate a concise human-readable insight. Rule-based, no API needed."""
    sim          = result_data["calibrated_sim_pct"]
    coverage     = result_data["skill_coverage_pct"]
    critical_miss= result_data["critical_missing"]
    important_miss=result_data["important_missing"]
    resume_extra = result_data["resume_only"]
    parts        = []

    if sim >= 80:
        parts.append(f"Strong contextual alignment with the JD (semantic similarity {sim:.0f}%).")
    elif sim >= 60:
        parts.append(f"Moderate contextual alignment with the JD (semantic similarity {sim:.0f}%).")
    else:
        parts.append(f"Limited contextual alignment with the JD (semantic similarity {sim:.0f}%).")

    if coverage >= 80:
        parts.append(f"Covers {coverage:.0f}% of required skills -- excellent match.")
    elif coverage >= 60:
        parts.append(f"Covers {coverage:.0f}% of required skills -- solid foundation.")
    elif coverage >= 40:
        parts.append(f"Covers {coverage:.0f}% of required skills -- some gaps present.")
    else:
        parts.append(f"Covers only {coverage:.0f}% of required skills -- significant gaps.")

    if critical_miss:
        top = ", ".join(critical_miss[:3])
        parts.append(f"Missing critical skills: {top}{'...' if len(critical_miss) > 3 else ''}.")
    elif important_miss:
        top = ", ".join(important_miss[:2])
        parts.append(f"Gaps in important skills: {top}.")
    else:
        parts.append("No critical skill gaps detected.")

    if len(resume_extra) >= 5:
        parts.append(f"Brings {len(resume_extra)} additional skills beyond JD requirements.")

    return " ".join(parts)


# -- Core Ranking Function --------------------------------------------

def rank_candidates(
    candidates: List[Dict],
    jd_text: str,
    similarity_weight: float = 0.7,
    skill_weight: float = 0.3,
) -> List["CandidateResult"]:
    """
    Rank candidates using weighted composite of calibrated semantic similarity
    and weighted skill coverage.

    Composite = (similarity_weight * calibrated_semantic_pct)
              + (skill_weight * weighted_skill_coverage_pct)

    Args:
        candidates        : List of dicts with 'name', 'text', 'score'.
        jd_text           : Preprocessed JD text.
        similarity_weight : Weight for semantic score (default 0.7).
        skill_weight      : Weight for skill coverage (default 0.3).

    Returns:
        List of CandidateResult sorted by composite score (descending).
    """
    if not candidates:
        raise ValueError("Candidates list is empty.")
    if abs(similarity_weight + skill_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0. Got {similarity_weight + skill_weight:.2f}")

    logger.info(f"Ranking {len(candidates)} candidates | sim={similarity_weight} skill={skill_weight}")

    raw_results = []
    for candidate in candidates:
        name       = candidate["name"]
        raw_cosine = candidate["score"]

        calibrated_sim = calibrate_score(raw_cosine)
        skill_data     = full_skill_analysis(candidate["text"], jd_text)
        weighted_cov   = skill_data["weighted_coverage_pct"]
        simple_cov     = skill_data["skill_coverage_pct"]

        composite = round(min(100.0, max(0.0,
            calibrated_sim * similarity_weight + weighted_cov * skill_weight
        )), 2)

        recommendation, color = get_recommendation(composite)

        raw_results.append({
            "name"              : name,
            "resume_text"       : candidate["text"],
            "similarity_score"  : round(raw_cosine, 4),
            "calibrated_sim_pct": calibrated_sim,
            "score_pct"         : composite,
            "recommendation"    : recommendation,
            "color"             : color,
            "matching"          : skill_data["matching"],
            "missing"           : skill_data["missing"],
            "critical_missing"  : skill_data["critical_missing"],
            "important_missing" : skill_data["important_missing"],
            "resume_only"       : skill_data["resume_only"],
            "skill_coverage_pct"    : simple_cov,
            "weighted_coverage_pct" : weighted_cov,
            "skills_by_category"    : skill_data["skills_by_category"],
            "jd_skill_count"        : len(skill_data["jd_skills"]),
        })

    raw_results.sort(key=lambda x: x["score_pct"], reverse=True)

    scores      = [r["score_pct"] for r in raw_results]
    percentiles = compute_percentile_ranks(scores)
    n           = len(raw_results)

    results: List[CandidateResult] = []
    for rank_idx, (r, pct) in enumerate(zip(raw_results, percentiles), start=1):
        confidence = compute_confidence(r["score_pct"], n)
        ai_insight = generate_insight(r, r["jd_skill_count"])

        result = CandidateResult(
            name                  = r["name"],
            resume_text           = r["resume_text"],
            similarity_score      = r["similarity_score"],
            calibrated_sim_pct    = r["calibrated_sim_pct"],
            score_pct             = r["score_pct"],
            recommendation        = r["recommendation"],
            recommendation_color  = r["color"],
            matching_skills       = r["matching"],
            missing_skills        = r["missing"],
            critical_missing      = r["critical_missing"],
            important_missing     = r["important_missing"],
            resume_only_skills    = r["resume_only"],
            skill_coverage_pct    = r["skill_coverage_pct"],
            weighted_coverage_pct = r["weighted_coverage_pct"],
            skills_by_category    = r["skills_by_category"],
            confidence            = confidence,
            percentile_rank       = pct,
            rank                  = rank_idx,
            ai_insight            = ai_insight,
        )
        results.append(result)
        logger.debug(
            f"  #{rank_idx} {r['name']}: composite={r['score_pct']:.1f}% "
            f"sim={r['calibrated_sim_pct']:.1f}% skill={r['weighted_coverage_pct']:.1f}%"
        )

    logger.info(f"Top candidate: {results[0].name} ({results[0].score_pct:.1f}%)")
    return results


# -- Export Helpers ---------------------------------------------------

def results_to_dataframe(results: List[CandidateResult]) -> pd.DataFrame:
    """Convert ranked results to a Pandas DataFrame indexed by Rank."""
    rows = [r.to_dict() for r in results]
    df   = pd.DataFrame(rows)
    df   = df.set_index("Rank")
    return df


def export_to_csv(results: List[CandidateResult], filepath: str) -> str:
    """Export results to CSV and return filepath."""
    df = results_to_dataframe(results)
    df.to_csv(filepath)
    logger.info(f"Exported to: {filepath}")
    return filepath


def summarize_rankings(results: List[CandidateResult]) -> Dict:
    """Compute aggregate summary statistics."""
    if not results:
        return {}
    scores = [r.score_pct for r in results]
    return {
        "total_candidates"  : len(results),
        "average_score"     : round(sum(scores) / len(scores), 2),
        "highest_score"     : round(max(scores), 2),
        "lowest_score"      : round(min(scores), 2),
        "score_std"         : round(pd.Series(scores).std(), 2) if len(scores) > 1 else 0.0,
        "highly_recommended": sum(1 for r in results if r.recommendation == "Highly Recommended"),
        "recommended"       : sum(1 for r in results if r.recommendation == "Recommended"),
        "consider"          : sum(1 for r in results if r.recommendation == "Consider"),
        "not_recommended"   : sum(1 for r in results if r.recommendation == "Not Recommended"),
    }

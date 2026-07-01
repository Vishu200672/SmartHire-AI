"""
skill_ner.py
-----------
Named Entity Recognition for skills using pattern matching and rule-based extraction.

Features:
  - Advanced skill taxonomy with domains
  - Contextual skill detection (avoid false positives)
  - Skill confidence scores
  - Skill evolution tracking

Author: SmartHire AI
"""

import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Skill domains with context
SKILL_DOMAINS = {
    "backend": {
        "skills": ["python", "java", "go", "rust", "node.js", "flask", "django", "fastapi", "spring"],
        "frameworks": ["fastapi", "flask", "django", "spring boot"],
        "keywords": ["api", "rest", "backend", "server"],
    },
    "frontend": {
        "skills": ["javascript", "typescript", "react", "vue", "angular", "css", "html"],
        "frameworks": ["react", "vue", "angular", "svelte"],
        "keywords": ["ui", "ux", "frontend", "client", "web app"],
    },
    "data_science": {
        "skills": ["python", "r", "sql", "pandas", "scikit-learn", "tensorflow", "pytorch"],
        "frameworks": ["tensorflow", "pytorch", "keras", "scikit-learn"],
        "keywords": ["data", "model", "analysis", "statistics", "ml"],
    },
    "devops": {
        "skills": ["docker", "kubernetes", "ci/cd", "terraform", "aws", "gcp", "azure"],
        "frameworks": ["docker", "kubernetes", "terraform"],
        "keywords": ["deployment", "infrastructure", "cloud", "monitoring"],
    },
}


def detect_skill_context(text: str, skill: str, window: int = 50) -> str:
    """
    Extract context window around skill mention.
    
    Returns:
        Text snippet containing the skill with surrounding context.
    """
    pattern = rf'\b{re.escape(skill)}\b'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if not matches:
        return ""
    
    match = matches[0]
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return text[start:end]


def get_skill_confidence(skill: str, context: str, domain: str = None) -> float:
    """
    Compute confidence score for a skill detection (0-1).
    
    Factors:
      - Frequency in context
      - Domain relevance
      - Surrounding keywords
    """
    confidence = 0.5  # Base
    
    # Frequency boost
    count = len(re.findall(rf'\b{re.escape(skill)}\b', context, re.IGNORECASE))
    confidence += min(0.3, count * 0.1)
    
    # Domain boost
    if domain and domain in SKILL_DOMAINS:
        if skill.lower() in [s.lower() for s in SKILL_DOMAINS[domain].get("skills", [])]:
            confidence += 0.1
        
        domain_keywords = SKILL_DOMAINS[domain].get("keywords", [])
        if any(kw.lower() in context.lower() for kw in domain_keywords):
            confidence += 0.1
    
    return min(1.0, confidence)


def extract_skills_with_confidence(
    text: str,
    skill_list: List[str],
    domain: str = None,
) -> List[Tuple[str, float]]:
    """
    Extract skills with confidence scores.
    
    Returns:
        List of (skill, confidence) tuples sorted by confidence.
    """
    results = []
    seen = set()
    
    for skill in skill_list:
        if skill.lower() in seen:
            continue
        seen.add(skill.lower())
        
        if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE):
            context = detect_skill_context(text, skill)
            confidence = get_skill_confidence(skill, context, domain)
            results.append((skill, round(confidence, 2)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def extract_domain_skills(text: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract skills organized by domain.
    
    Returns:
        {
            "backend": [("python", 0.95), ("flask", 0.88), ...],
            "frontend": [...],
            ...
        }
    """
    domain_skills = {}
    
    for domain, domain_config in SKILL_DOMAINS.items():
        skills = extract_skills_with_confidence(
            text,
            domain_config["skills"],
            domain=domain
        )
        if skills:
            domain_skills[domain] = skills
    
    return domain_skills


def track_skill_evolution(
    historical_extractions: List[Dict],
) -> Dict:
    """
    Analyze skill acquisition over time (if timestamps available).
    
    Args:
        historical_extractions: List of {"timestamp", "skills"} dicts.
    
    Returns:
        {
            "new_skills": [...],
            "mastered_skills": [...],
            "declining_skills": [...],
            "trends": {...}
        }
    """
    if len(historical_extractions) < 2:
        return {"status": "insufficient_data"}
    
    earliest = set(s[0] for s in historical_extractions[0].get("skills", []))
    latest = set(s[0] for s in historical_extractions[-1].get("skills", []))
    
    new_skills = latest - earliest
    lost_skills = earliest - latest
    stable_skills = latest & earliest
    
    return {
        "new_skills": sorted(list(new_skills)),
        "mastered_skills": sorted(list(stable_skills)),
        "declining_skills": sorted(list(lost_skills)),
        "total_skill_growth": len(new_skills),
        "skill_retention": len(stable_skills) / len(earliest) if earliest else 0.0,
    }


def get_skill_recommendations(
    current_skills: List[str],
    target_domain: str,
) -> Dict:
    """
    Recommend skills to acquire based on target domain.
    """
    if target_domain not in SKILL_DOMAINS:
        return {"error": f"Unknown domain: {target_domain}"}
    
    target_skills = set(SKILL_DOMAINS[target_domain]["skills"])
    current_set = set(s.lower() for s in current_skills)
    
    missing = [s for s in target_skills if s.lower() not in current_set]
    complementary = []
    
    # Find complementary skills from other domains
    for other_domain, config in SKILL_DOMAINS.items():
        if other_domain != target_domain:
            overlap = set(config["skills"]) & current_set
            if overlap:
                complementary.extend(list(set(config["skills"]) - current_set)[:2])
    
    return {
        "target_domain": target_domain,
        "skill_gaps": missing[:10],
        "complementary_skills": list(set(complementary))[:10],
        "priority_level": "critical" if len(missing) > 5 else "moderate" if len(missing) > 2 else "low",
    }

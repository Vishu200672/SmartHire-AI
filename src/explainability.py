"""
explainability.py
-----------------
Resume matching explainability -- identify key phrases and sections that drove the score.

Features:
  - Extract high-impact phrases from resume matching JD
  - Section-level importance (experience, skills, education)
  - Attention-based highlighting
  - SHAP-inspired local explanations (without external models)

Author: SmartHire AI
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def extract_sections(resume_text: str) -> Dict[str, str]:
    """Parse resume into common sections (Education, Experience, Skills, etc)."""
    sections = {
        "experience": "",
        "education": "",
        "skills": "",
        "projects": "",
        "certifications": "",
        "summary": "",
        "other": "",
    }
    
    # Section headers (flexible matching)
    section_patterns = {
        "experience": r"(?i)(professional\s+experience|work\s+experience|employment|career)",
        "education": r"(?i)(education|academic|degree|university|college)",
        "skills": r"(?i)(technical\s+skills|skills|competencies|expertise)",
        "projects": r"(?i)(projects?|portfolio|achievements)",
        "certifications": r"(?i)(certifications?|licenses|credentials)",
        "summary": r"(?i)(professional\s+summary|objective|executive\s+summary|about)",
    }
    
    lines = resume_text.split('\n')
    current_section = "summary"
    
    for line in lines:
        matched = False
        for section_key, pattern in section_patterns.items():
            if re.search(pattern, line):
                current_section = section_key
                matched = True
                break
        
        if matched or not line.strip():
            continue
        
        sections[current_section] += line + "\n"
    
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def extract_key_phrases(text: str, jd_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Extract phrases from resume that appear in JD (TF-IDF style scoring).
    Returns list of (phrase, importance_score) tuples.
    """
    # Simple phrase extraction (2-4 word chunks)
    phrases = []
    words = re.findall(r'\w+', text.lower())
    
    for i in range(len(words) - 1):
        for j in range(i + 2, min(i + 5, len(words) + 1)):
            phrase = ' '.join(words[i:j])
            phrases.append(phrase)
    
    # Score each phrase by frequency in JD
    jd_lower = jd_text.lower()
    scored_phrases = []
    seen = set()
    
    for phrase in set(phrases):
        if phrase in seen:
            continue
        seen.add(phrase)
        
        count_jd = len(re.findall(rf'\b{re.escape(phrase)}\b', jd_lower))
        count_resume = len(re.findall(rf'\b{re.escape(phrase)}\b', text.lower()))
        
        if count_jd > 0:
            score = count_jd * count_resume
            scored_phrases.append((phrase, score))
    
    scored_phrases.sort(key=lambda x: x[1], reverse=True)
    return scored_phrases[:top_k]


def compute_section_importance(
    resume_sections: Dict[str, str],
    jd_text: str,
) -> Dict[str, float]:
    """
    Score each section by how much it overlaps with JD.
    Returns dict of section -> importance_score (0-100).
    """
    scores = {}
    jd_lower = jd_text.lower()
    
    for section_name, section_text in resume_sections.items():
        if not section_text:
            scores[section_name] = 0.0
            continue
        
        section_words = set(re.findall(r'\b\w+\b', section_text.lower()))
        jd_words = set(re.findall(r'\b\w+\b', jd_lower))
        
        if not section_words:
            scores[section_name] = 0.0
            continue
        
        overlap = len(section_words & jd_words)
        score = min(100.0, (overlap / len(section_words)) * 100)
        scores[section_name] = round(score, 2)
    
    return scores


def highlight_matching_phrases(
    resume_text: str,
    jd_text: str,
    top_k: int = 10,
) -> Dict:
    """
    Generate a comprehensive explainability report.
    
    Returns:
        {
            "key_phrases": [(phrase, score), ...],
            "sections": { "experience": 75.5, ... },
            "highlight_text": "Resume with highlighted phrases",
            "summary": "Human-readable explanation"
        }
    """
    sections = extract_sections(resume_text)
    key_phrases = extract_key_phrases(resume_text, jd_text, top_k=top_k)
    section_scores = compute_section_importance(sections, jd_text)
    
    # Generate highlight text
    highlight_text = resume_text
    for phrase, score in key_phrases:
        if score > 0:
            highlight_text = re.sub(
                rf'\b{re.escape(phrase)}\b',
                f'**{phrase}**',
                highlight_text,
                flags=re.IGNORECASE
            )
    
    # Generate summary
    top_section = max(section_scores.items(), key=lambda x: x[1]) if section_scores else ("", 0)
    summary = (
        f"Key drivers: {', '.join(p[0] for p in key_phrases[:3])}. "
        f"Strongest section: {top_section[0]} ({top_section[1]:.0f}% alignment). "
    )
    
    return {
        "key_phrases": key_phrases,
        "sections": section_scores,
        "highlight_text": highlight_text,
        "summary": summary,
        "top_section": top_section[0],
        "top_section_score": top_section[1],
    }


def generate_explainability_report(
    candidate_name: str,
    resume_text: str,
    jd_text: str,
    match_score: float,
    skill_data: Dict,
) -> Dict:
    """
    Generate a full explainability report for a candidate match.
    
    Includes:
      - Why the score is X% (key drivers)
      - Which resume sections matter most
      - Top matching phrases
      - Recommendations for improvement
    """
    explainability = highlight_matching_phrases(resume_text, jd_text, top_k=7)
    
    # Compute improvement recommendations
    improvements = []
    if skill_data.get("critical_missing"):
        improvements.append(
            f"Learn/gain experience in: {', '.join(skill_data['critical_missing'][:2])}"
        )
    if skill_data.get("important_missing"):
        improvements.append(
            f"Strengthen: {', '.join(skill_data['important_missing'][:2])}"
        )
    if match_score < 70 and explainability["top_section_score"] < 50:
        improvements.append("Restructure resume to emphasize relevant experience")
    
    return {
        "candidate": candidate_name,
        "match_score": match_score,
        "key_drivers": explainability["key_phrases"],
        "section_alignment": explainability["sections"],
        "strongest_section": explainability["top_section"],
        "summary": explainability["summary"],
        "improvements": improvements,
        "highlight_text": explainability["highlight_text"],
    }

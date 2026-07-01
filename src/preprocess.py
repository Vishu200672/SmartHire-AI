"""
preprocess.py
-------------
Production-grade text preprocessing for resumes and job descriptions.

Key improvements over v1:
  - Preserves C++, C#, .NET, F# -- no longer stripped
  - Section-aware parsing (extracts structured sections)
  - Experience-year extraction (regex: "5+ years Python")
  - Cleans PDF artifacts (ligatures, bullet symbols, page numbers)
  - UTF-8 safe with smart unicode normalization

Author: SmartHire AI
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------
MAX_TOKENS    = 512
CHARS_PER_TOK = 3.8  # tighter estimate for technical text

# Resume section headers
SECTION_HEADERS = [
    "experience", "work experience", "employment", "professional experience",
    "education", "skills", "technical skills", "core competencies",
    "projects", "certifications", "achievements", "summary", "objective",
    "publications", "languages", "interests", "volunteer",
]

# PDF ligature replacements
LIGATURE_MAP = {
    "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
    "\ufb03": "ffi", "\ufb04": "ffl", "\u2019": "'",
    "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2022": " ",
    "\u25cf": " ", "\u25aa": " ", "\u2023": " ",
    "\xa0": " ",  # non-breaking space
}

# Tech terms that must NOT be stripped
PRESERVE_TERMS = {
    "c++": "cplusplus", "c#": "csharp", ".net": "dotnet",
    "f#": "fsharp", "node.js": "nodejs", "vue.js": "vuejs",
    "next.js": "nextjs", "express.js": "expressjs",
    "asp.net": "aspnet", "ado.net": "adonet",
}

RESTORE_TERMS = {v: k for k, v in PRESERVE_TERMS.items()}


def fix_ligatures(text: str) -> str:
    """Replace common PDF ligature artifacts with ASCII equivalents."""
    for char, replacement in LIGATURE_MAP.items():
        text = text.replace(char, replacement)
    return text


def protect_tech_terms(text: str) -> str:
    """Temporarily replace C++, C#, .NET etc. before lowercasing."""
    for term, placeholder in PRESERVE_TERMS.items():
        text = re.sub(re.escape(term), placeholder, text, flags=re.IGNORECASE)
    return text


def restore_tech_terms(text: str) -> str:
    """Restore protected tech terms after preprocessing."""
    for placeholder, term in RESTORE_TERMS.items():
        text = text.replace(placeholder, term)
    return text


def normalize_unicode(text: str) -> str:
    """NFKC normalization -- preserves more chars than NFKD."""
    return unicodedata.normalize("NFKC", text)


def remove_urls_emails(text: str) -> str:
    """Strip URLs and email addresses."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    return text


def remove_pdf_artifacts(text: str) -> str:
    """Remove common PDF-extraction noise: page numbers, divider lines."""
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[-_=]{4,}", " ", text)
    text = re.sub(r"\|{2,}", " ", text)
    return text


def clean_special_characters(text: str) -> str:
    """Remove non-useful special chars while preserving tech punctuation."""
    text = re.sub(r"[^\w\s\.\,\-\+\#\/\(\)\@\%\&]", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces/newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_experience_years(text: str) -> Dict[str, int]:
    """
    Extract experience mentions like '5+ years Python', '3 years of AWS'.

    Returns:
        Dict mapping skill -> years, e.g. {"python": 5, "aws": 3}
    """
    pattern = re.compile(
        r"(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+([a-zA-Z][a-zA-Z0-9\+\#\.\/\s]{1,30})",
        re.IGNORECASE
    )
    results = {}
    for match in pattern.finditer(text):
        years = int(match.group(1))
        skill = match.group(2).strip().lower().rstrip(".,;:")
        if len(skill) >= 2:
            results[skill] = years
    return results


def extract_sections(text: str) -> Dict[str, str]:
    """
    Identify resume sections and return section_name -> content dict.
    """
    sections: Dict[str, str] = {}
    current_section = "header"
    current_lines: List[str] = []

    header_pattern = re.compile(
        r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?\s*$",
        re.IGNORECASE
    )

    for line in text.split("\n"):
        stripped = line.strip()
        if header_pattern.match(stripped):
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped.lower().rstrip(":")
            current_lines   = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def truncate_to_token_budget(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate to token budget using character heuristic."""
    max_chars = int(max_tokens * CHARS_PER_TOK)
    if len(text) <= max_chars:
        return text

    logger.warning(f"Truncating text from {len(text)} chars to ~{max_chars}")
    truncated = text[:max_chars]

    for sep in [". ", ".\n", "! ", "? "]:
        last = truncated.rfind(sep)
        if last > max_chars * 0.85:
            return truncated[:last + 1].strip()

    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.9:
        return truncated[:last_space].strip()

    return truncated.strip()


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    fix_pdf: bool = True,
    truncate: bool = True,
    max_tokens: int = MAX_TOKENS,
    preserve_sections: bool = False,
) -> str:
    """
    Full production preprocessing pipeline.

    Steps:
      1. Fix PDF ligatures & artifacts
      2. Unicode normalize (NFKC)
      3. Protect tech terms (C++, C#, .NET)
      4. Remove URLs / emails
      5. Lowercase
      6. Clean special characters
      7. Restore tech terms
      8. Collapse whitespace
      9. Truncate to token budget

    Args:
        text             : Raw input text.
        lowercase        : Lowercase the text (default True).
        remove_urls      : Strip URLs and emails (default True).
        fix_pdf          : Fix PDF extraction artifacts (default True).
        truncate         : Truncate to token limit (default True).
        max_tokens       : Token budget (default 512).
        preserve_sections: Skip truncation for section parsing.

    Returns:
        Cleaned, normalized text.

    Raises:
        ValueError: If text is empty after processing.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    if fix_pdf:
        text = fix_ligatures(text)
        text = remove_pdf_artifacts(text)

    text = normalize_unicode(text)
    text = protect_tech_terms(text)

    if remove_urls:
        text = remove_urls_emails(text)

    if lowercase:
        text = text.lower()

    text = clean_special_characters(text)
    text = restore_tech_terms(text)
    text = collapse_whitespace(text)

    if not text.strip():
        raise ValueError("Text is empty after preprocessing.")

    if truncate and not preserve_sections:
        text = truncate_to_token_budget(text, max_tokens)

    logger.debug(f"Preprocessed: {len(text)} chars")
    return text


def tokenize_words(text: str) -> List[str]:
    """Tokenize into word list (min length 2)."""
    return [t for t in re.findall(r"\b[a-z][a-z0-9\+\#\.]*\b", text.lower()) if len(t) >= 2]


def clean_skill_token(skill: str) -> str:
    """Normalize a skill string for comparison."""
    return re.sub(r"\s+", " ", skill.strip().lower())

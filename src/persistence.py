"""
persistence.py
--------------
Persistence layer for candidate results, feedback, and analytics.

Features:
  - SQLite for local storage (no external DB needed)
  - Store results, feedback, hiring outcomes
  - Query historical matches
  - Analytics on hiring success

Author: SmartHire AI
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path("smarthire_data.db")


class CandidateDatabase:
    """
    SQLite-backed candidate and result persistence.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Candidates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                email TEXT,
                phone TEXT,
                resume_text TEXT,
                skills TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Job descriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY,
                title TEXT,
                jd_text TEXT,
                skills_required TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Match results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
                id INTEGER PRIMARY KEY,
                candidate_id INTEGER,
                jd_id INTEGER,
                match_score REAL,
                semantic_score REAL,
                skill_coverage REAL,
                recommendation TEXT,
                result_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(candidate_id) REFERENCES candidates(id),
                FOREIGN KEY(jd_id) REFERENCES job_descriptions(id)
            )
        """)

        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                match_id INTEGER,
                outcome TEXT,
                notes TEXT,
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(match_id) REFERENCES match_results(id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def add_candidate(self, name: str, email: str, phone: str, resume_text: str) -> int:
        """Add a candidate. Returns candidate_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO candidates (name, email, phone, resume_text) VALUES (?, ?, ?, ?)",
                (name, email, phone, resume_text),
            )
            conn.commit()
            candidate_id = cursor.lastrowid
            logger.info(f"Candidate added: {name} (ID: {candidate_id})")
            return candidate_id
        except sqlite3.IntegrityError:
            logger.warning(f"Candidate already exists: {name}")
            return self.get_candidate_id(name)
        finally:
            conn.close()

    def get_candidate_id(self, name: str) -> Optional[int]:
        """Get candidate ID by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM candidates WHERE name = ?", (name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def add_job_description(self, title: str, jd_text: str) -> int:
        """Add a job description. Returns jd_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO job_descriptions (title, jd_text) VALUES (?, ?)",
            (title, jd_text),
        )
        conn.commit()
        jd_id = cursor.lastrowid
        conn.close()
        return jd_id

    def save_match_result(
        self,
        candidate_name: str,
        jd_id: int,
        match_score: float,
        semantic_score: float,
        skill_coverage: float,
        recommendation: str,
        result_dict: Dict,
    ) -> int:
        """Save a match result."""
        candidate_id = self.get_candidate_id(candidate_name)
        if not candidate_id:
            logger.warning(f"Candidate not found: {candidate_name}")
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO match_results (
                candidate_id, jd_id, match_score, semantic_score,
                skill_coverage, recommendation, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                jd_id,
                match_score,
                semantic_score,
                skill_coverage,
                recommendation,
                json.dumps(result_dict),
            ),
        )
        conn.commit()
        result_id = cursor.lastrowid
        conn.close()
        logger.info(f"Match result saved (ID: {result_id})")
        return result_id

    def add_feedback(self, match_id: int, outcome: str, rating: int, notes: str = "") -> int:
        """Add feedback for a match. outcome: 'hired'|'rejected'|'in_progress'"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (match_id, outcome, rating, notes) VALUES (?, ?, ?, ?)",
            (match_id, outcome, rating, notes),
        )
        conn.commit()
        feedback_id = cursor.lastrowid
        conn.close()
        logger.info(f"Feedback added for match {match_id}")
        return feedback_id

    def get_analytics(self) -> Dict:
        """Get overall hiring analytics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total matches
        cursor.execute("SELECT COUNT(*) FROM match_results")
        total_matches = cursor.fetchone()[0]

        # Hiring outcomes
        cursor.execute(
            "SELECT outcome, COUNT(*) FROM feedback GROUP BY outcome"
        )
        outcomes = {row[0]: row[1] for row in cursor.fetchall()}

        # Average scores by outcome
        cursor.execute(
            """
            SELECT f.outcome, AVG(m.match_score)
            FROM feedback f
            JOIN match_results m ON f.match_id = m.id
            GROUP BY f.outcome
            """
        )
        avg_scores = {row[0]: round(row[1], 2) for row in cursor.fetchall()}

        # Success rate (hired / total)
        hired = outcomes.get("hired", 0)
        total_with_feedback = sum(outcomes.values())
        success_rate = (
            (hired / total_with_feedback * 100)
            if total_with_feedback > 0
            else 0.0
        )

        conn.close()
        return {
            "total_matches": total_matches,
            "outcomes": outcomes,
            "average_scores_by_outcome": avg_scores,
            "hiring_success_rate": round(success_rate, 2),
            "total_with_feedback": total_with_feedback,
        }

    def get_match_history(self, candidate_name: str) -> List[Dict]:
        """Get all matches for a candidate."""
        candidate_id = self.get_candidate_id(candidate_name)
        if not candidate_id:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, match_score, semantic_score, skill_coverage,
                   recommendation, created_at
            FROM match_results
            WHERE candidate_id = ?
            ORDER BY created_at DESC
            """,
            (candidate_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "match_score": row[1],
                "semantic_score": row[2],
                "skill_coverage": row[3],
                "recommendation": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

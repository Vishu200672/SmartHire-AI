"""
main.py — CLI entry point for SmartHire AI
Usage:
    python main.py --demo
    python main.py --resume resume.pdf --jd job.txt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("SmartHireAI")


def run_demo() -> None:
    sample_dir = Path("datasets")
    if not sample_dir.exists():
        logger.error("datasets/ directory not found. Run from the SmartHireAI/ root.")
        sys.exit(1)

    resume_files = list(sample_dir.glob("*.txt"))
    jd_file = sample_dir / "sample_jd.txt"

    if not jd_file.exists():
        logger.error(f"Sample JD not found: {jd_file}")
        sys.exit(1)

    jd_text = jd_file.read_text(encoding="utf-8")
    logger.info(f"Job Description loaded: {jd_file.name}")

    resume_candidates = []
    for rf in resume_files:
        if rf.name == "sample_jd.txt":
            continue
        text = rf.read_text(encoding="utf-8")
        resume_candidates.append({"name": rf.stem, "raw_text": text})

    logger.info(f"Loaded {len(resume_candidates)} resume(s)")
    _run_pipeline(resume_candidates, jd_text)


def run_custom(resume_paths: list, jd_path: str) -> None:
    from src.parser import parse_resume, parse_job_description

    jd_path = Path(jd_path)
    jd_raw = jd_path.read_bytes()
    jd_text = parse_job_description(jd_raw, filename=jd_path.name)

    resume_candidates = []
    for rp in resume_paths:
        rp = Path(rp)
        raw = rp.read_bytes()
        text = parse_resume(raw, filename=rp.name)
        resume_candidates.append({"name": rp.stem, "raw_text": text})

    _run_pipeline(resume_candidates, jd_text)


def _run_pipeline(resume_candidates: list, jd_text: str) -> None:
    from src.preprocess import preprocess_text
    from src.model import get_model
    from src.similarity import batch_similarity
    from src.ranking import rank_candidates, summarize_rankings
    import torch

    logger.info("Step 1/4: Preprocessing text...")
    jd_clean = preprocess_text(jd_text)
    for c in resume_candidates:
        c["clean_text"] = preprocess_text(c["raw_text"])

    logger.info("Step 2/4: Loading DistilBERT model and encoding texts...")
    t0 = time.time()
    model = get_model()

    resume_texts = [c["clean_text"] for c in resume_candidates]
    resume_embeddings = model.encode(resume_texts, show_progress=True)
    jd_embedding = model.encode_single(jd_clean)
    elapsed = time.time() - t0
    logger.info(f"Encoding complete in {elapsed:.2f}s for {len(resume_candidates)} resume(s).")

    logger.info("Step 3/4: Computing cosine similarities...")
    scores = batch_similarity(resume_embeddings, jd_embedding)
    for c, score in zip(resume_candidates, scores):
        c["score"] = score

    logger.info("Step 4/4: Ranking candidates...")
    candidates_input = [{"name": c["name"], "text": c["clean_text"], "score": c["score"]} for c in resume_candidates]
    results = rank_candidates(candidates_input, jd_clean)

    print("\n" + "=" * 65)
    print("  SmartHire AI — Candidate Ranking Results")
    print("=" * 65)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank #{rank}: {result.name}")
        print(f"  Match Score    : {result.score_pct:.1f}%")
        print(f"  Recommendation : {result.recommendation}")
        print(f"  Skill Coverage : {result.skill_coverage_pct:.1f}%")
        if result.matching_skills:
            print(f"  Matching Skills: {', '.join(result.matching_skills[:8])}")
        if result.critical_missing:
            print(f"  Critical Missing: {', '.join(result.critical_missing)}")

    summary = summarize_rankings(results)
    print("\n" + "-" * 65)
    print(f"Summary: {summary['total_candidates']} candidates | Avg: {summary['average_score']}% | Top: {summary['highest_score']}%")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartHire AI — Transformer-Based Resume & Job Matching System")
    parser.add_argument("--demo", action="store_true", help="Run with bundled sample dataset")
    parser.add_argument("--resume", nargs="+", metavar="FILE", help="Path(s) to resume file(s)")
    parser.add_argument("--jd", metavar="FILE", help="Path to job description file")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.resume and args.jd:
        run_custom(args.resume, args.jd)
    else:
        parser.print_help()
        print("\nError: Provide --demo or both --resume and --jd flags.")
        sys.exit(1)


if __name__ == "__main__":
    main()
